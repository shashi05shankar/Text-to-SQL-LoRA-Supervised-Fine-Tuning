"""Prompt construction for context-grounded financial QA.

The same builder serves the base model and the tuned model, so the measured delta is
attributable to fine-tuning and not to prompt differences.
"""

from __future__ import annotations

from finqa_lora.data import QAExample
from text2sql_lora.data.prompts import mask_prompt_labels

SYSTEM_PROMPT = (
    "You are a financial analyst. Answer the question using the passage from the company's "
    "10-K filing. Reply with a concise answer in one or two sentences."
)

USER_TEMPLATE = """Passage:
{context}

Question: {question}

Answer:"""


def build_messages(example: QAExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                context=example.context.strip(), question=example.question.strip()
            ),
        },
    ]


def build_inference_prompt(tokenizer, example: QAExample) -> str:
    return tokenizer.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True
    )


def build_supervised_example(tokenizer, example: QAExample, max_seq_len: int) -> dict:
    """Tokenize to `input_ids` / `labels` with the prompt masked out of the loss."""
    prompt_ids = tokenizer(
        build_inference_prompt(tokenizer, example), add_special_tokens=False
    )["input_ids"]
    target_ids = tokenizer(
        example.answer.strip() + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    input_ids, labels, truncated = mask_prompt_labels(prompt_ids, target_ids, max_seq_len)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "truncated": truncated,
    }


class QADataset:
    """List-backed tokenized dataset (the shape `text2sql_lora.train.sft.train_lora` expects)."""

    def __init__(self, examples: list[QAExample], tokenizer, max_seq_len: int) -> None:
        self.records: list[dict] = []
        self.n_truncated = 0
        for example in examples:
            record = build_supervised_example(tokenizer, example, max_seq_len)
            self.n_truncated += int(record.pop("truncated"))
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]

    def truncation_rate(self) -> float:
        return self.n_truncated / len(self.records) if self.records else 0.0
