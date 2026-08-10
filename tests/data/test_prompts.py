"""Prompt construction and completion-only loss masking.

`mask_prompt_labels` is pure list arithmetic, so these tests need no tokenizer and no
model. That is the point of the split: if the masking is backwards, the model trains to
reproduce schemas instead of writing SQL, nothing raises, and the loss curve looks
perfectly healthy. It has to be caught here or not at all.
"""

from __future__ import annotations

import pytest

from text2sql_lora.data.prompts import (
    SYSTEM_PROMPT,
    Example,
    build_messages,
    build_user_message,
    mask_prompt_labels,
    select_few_shot,
)

pytestmark = pytest.mark.unit

IGNORE = -100


def make_example(n: int = 0) -> Example:
    return Example(
        db_id=f"db_{n}",
        question=f"question {n}?",
        query=f"SELECT {n}",
        schema_text=f"CREATE TABLE t{n} (a TEXT);",
    )


# --- masking -----------------------------------------------------------------


def test_prompt_is_masked_and_target_is_not():
    prompt_ids = [1, 2, 3, 4]
    target_ids = [10, 11]

    input_ids, labels, truncated = mask_prompt_labels(prompt_ids, target_ids, max_seq_len=16)

    assert input_ids == [1, 2, 3, 4, 10, 11]
    assert labels == [IGNORE, IGNORE, IGNORE, IGNORE, 10, 11]
    assert truncated is False


def test_labels_align_positionally_with_input_ids():
    """Every non-masked label must equal the token at the same index -- an off-by-one
    here trains the model against a shifted target."""
    input_ids, labels, _ = mask_prompt_labels([1, 2, 3], [7, 8, 9], max_seq_len=32)

    assert len(input_ids) == len(labels)
    for token, label in zip(input_ids, labels, strict=True):
        assert label in (IGNORE, token)
    assert [label for label in labels if label != IGNORE] == [7, 8, 9]


def test_exactly_the_target_tokens_contribute_to_loss():
    _, labels, _ = mask_prompt_labels(list(range(1, 51)), [90, 91, 92], max_seq_len=128)
    assert sum(1 for label in labels if label != IGNORE) == 3


def test_truncation_drops_the_front_of_the_prompt_and_keeps_the_whole_target():
    """Right-truncating would cut the end off the SQL and teach the model to emit
    unterminated queries. Left-truncating costs schema context instead."""
    prompt_ids = list(range(100))
    target_ids = [900, 901, 902]

    input_ids, labels, truncated = mask_prompt_labels(prompt_ids, target_ids, max_seq_len=10)

    assert truncated is True
    assert len(input_ids) == 10
    # Whole target survives, at the end.
    assert input_ids[-3:] == target_ids
    assert labels[-3:] == target_ids
    # The prompt kept is its TAIL (93..99), not its head.
    assert input_ids[:7] == list(range(93, 100))


def test_no_truncation_when_everything_fits_exactly():
    input_ids, _, truncated = mask_prompt_labels([1, 2, 3], [4, 5], max_seq_len=5)
    assert truncated is False
    assert len(input_ids) == 5


def test_target_longer_than_window_keeps_the_target_tail():
    input_ids, labels, truncated = mask_prompt_labels([1, 2, 3], [7, 8, 9, 10], max_seq_len=3)

    assert truncated is True
    assert input_ids == [8, 9, 10]
    assert labels == [8, 9, 10]  # nothing masked; there is no prompt left


def test_empty_target_is_rejected():
    with pytest.raises(ValueError):
        mask_prompt_labels([1, 2, 3], [], max_seq_len=16)


def test_non_positive_max_seq_len_is_rejected():
    with pytest.raises(ValueError):
        mask_prompt_labels([1], [2], max_seq_len=0)


# --- prompt text -------------------------------------------------------------


def test_user_message_carries_schema_and_question():
    message = build_user_message("CREATE TABLE t (a TEXT);", "How many rows?")
    assert "CREATE TABLE t (a TEXT);" in message
    assert "How many rows?" in message
    assert message.rstrip().endswith("SQL:")


def test_zero_shot_messages_are_system_then_user():
    messages = build_messages(make_example())

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_few_shot_demonstrations_become_completed_turns():
    """Demonstrations go in as real user/assistant turns, which is how an instruct model
    was trained to read them."""
    shots = [make_example(1), make_example(2)]
    messages = build_messages(make_example(0), few_shot=shots)

    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == "SELECT 1"
    assert messages[4]["content"] == "SELECT 2"
    # The question under test is last, and its answer is NOT present.
    assert "question 0?" in messages[-1]["content"]
    assert not any(m["content"] == "SELECT 0" for m in messages)


def test_few_shot_selection_is_seeded_and_reproducible():
    pool = [make_example(i) for i in range(50)]
    first = select_few_shot(pool, n=3, seed=42)
    second = select_few_shot(pool, n=3, seed=42)

    assert [e.db_id for e in first] == [e.db_id for e in second]
    assert len(first) == 3
    assert select_few_shot(pool, n=3, seed=7) != first


def test_few_shot_selection_rejects_an_oversized_request():
    with pytest.raises(ValueError):
        select_few_shot([make_example(0)], n=3, seed=42)


def test_zero_few_shot_returns_nothing():
    assert select_few_shot([make_example(0)], n=0, seed=42) == []
