"""Prompt construction and supervised-example encoding.

Two layers, split on purpose:

* Everything above `build_supervised_example` is **pure text** and has no tokenizer
  dependency, so it can be tested and inspected without downloading a model.
* `mask_prompt_labels` is a **pure list operation** with no tokenizer either. Completion-
  only loss masking is the single easiest thing to get silently wrong in SFT -- get it
  backwards and the model trains to reproduce the schema instead of the SQL, and nothing
  crashes; the loss curve looks fine and the eval is a disaster. Isolating it as a pure
  function makes it directly unit-testable.

The same prompt builder is used for the base model and the fine-tuned model. That is not
an incidental detail: if the two systems saw different prompts, the measured delta would
be confounded by prompt engineering rather than attributable to fine-tuning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite queries. "
    "Given a database schema and a question, reply with a single SQL query that answers "
    "the question. Output only the SQL, with no explanation and no markdown fences."
)

USER_TEMPLATE = """Database schema:
{schema}

Question: {question}

SQL:"""


@dataclass(frozen=True)
class Example:
    """One Spider row, already paired with its rendered schema."""

    db_id: str
    question: str
    query: str
    schema_text: str


def build_user_message(schema_text: str, question: str) -> str:
    return USER_TEMPLATE.format(schema=schema_text.strip(), question=question.strip())


def build_messages(
    example: Example,
    few_shot: list[Example] | None = None,
) -> list[dict[str, str]]:
    """Build the chat message list for one example.

    Few-shot examples are injected as completed user/assistant turns rather than pasted
    into the user message, because that is how an instruct model was trained to consume
    demonstrations -- and giving the few-shot baseline its best honest shot is the whole
    point of including it.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for shot in few_shot or []:
        messages.append(
            {"role": "user", "content": build_user_message(shot.schema_text, shot.question)}
        )
        messages.append({"role": "assistant", "content": shot.query.strip()})

    messages.append(
        {"role": "user", "content": build_user_message(example.schema_text, example.question)}
    )
    return messages


def select_few_shot(pool: list[Example], n: int, seed: int) -> list[Example]:
    """Pick a fixed set of in-context demonstrations.

    Sampled once from the TRAINING pool with a fixed seed and reused for every test
    question -- not retrieved per-question. Per-question retrieval would be a stronger
    baseline but a different system (it adds a retriever), and the comparison here is
    meant to isolate fine-tuning.
    """
    if n <= 0:
        return []
    if n > len(pool):
        raise ValueError(f"asked for {n} few-shot examples from a pool of {len(pool)}")
    return random.Random(seed).sample(pool, n)


def mask_prompt_labels(
    prompt_ids: list[int],
    target_ids: list[int],
    max_seq_len: int,
) -> tuple[list[int], list[int], bool]:
    """Concatenate prompt and target, masking the prompt out of the loss.

    Returns `(input_ids, labels, truncated)`, where `labels` is `-100` over every prompt
    position and the true token id over every target position.

    Truncation cuts from the LEFT of the prompt, never from the target. A right-truncated
    example would lose the end of the SQL and teach the model to emit unterminated
    queries; left-truncating drops the top of the schema, which costs accuracy on that
    example but keeps the supervision signal intact.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if not target_ids:
        raise ValueError("target_ids is empty; there would be nothing to learn from")

    truncated = False
    budget = max_seq_len - len(target_ids)

    if budget <= 0:
        # Pathological: the target alone exceeds the window. Keep the target's tail and
        # drop the prompt entirely rather than returning something unusable.
        truncated = True
        kept_prompt: list[int] = []
        kept_target = target_ids[-max_seq_len:]
    else:
        kept_target = target_ids
        if len(prompt_ids) > budget:
            truncated = True
            kept_prompt = prompt_ids[-budget:]
        else:
            kept_prompt = list(prompt_ids)

    input_ids = kept_prompt + kept_target
    labels = [-100] * len(kept_prompt) + list(kept_target)
    return input_ids, labels, truncated


def build_supervised_example(
    tokenizer,
    example: Example,
    max_seq_len: int,
    few_shot: list[Example] | None = None,
) -> dict:
    """Tokenize one example into `input_ids` / `labels` / `attention_mask`.

    The prompt is rendered with `add_generation_prompt=True` so it ends exactly where the
    model is expected to start writing at inference time -- identical framing at train and
    test time, which is what makes the training signal transfer.
    """
    messages = build_messages(example, few_shot=few_shot)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    target_text = example.query.strip() + tokenizer.eos_token
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    input_ids, labels, truncated = mask_prompt_labels(prompt_ids, target_ids, max_seq_len)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "truncated": truncated,
        "n_prompt_tokens": len(prompt_ids),
        "n_target_tokens": len(target_ids),
    }


def build_inference_prompt(
    tokenizer,
    example: Example,
    few_shot: list[Example] | None = None,
) -> str:
    """The exact text handed to the model at eval time."""
    messages = build_messages(example, few_shot=few_shot)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
