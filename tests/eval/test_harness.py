"""The harness, end to end, driven by stubs against a real SQLite database.

No model is loaded anywhere in this file.
"""

from __future__ import annotations

import pytest

from text2sql_lora.config import EvalConfig, Paths
from text2sql_lora.data.prompts import Example
from text2sql_lora.data.spider import SpiderData
from text2sql_lora.eval.harness import (
    GoldCache,
    compare,
    evaluate_system,
    gold_self_consistency,
    score_predictions,
)
from text2sql_lora.models.stub import EchoGoldGenerator, StubGenerator

pytestmark = pytest.mark.unit


@pytest.fixture
def mini_data(mini_db, monkeypatch):
    """A three-question SpiderData pointing at the fixture database.

    `Paths.db_path` is redirected to the tmp fixture so the harness runs unmodified.
    """
    examples = [
        Example("mini_school", "How many students?", "SELECT COUNT(*) FROM student", "schema"),
        Example(
            "mini_school",
            "List student names by age.",
            "SELECT name FROM student ORDER BY age",
            "schema",
        ),
        Example("mini_school", "Which courses exist?", "SELECT course FROM enrolment", "schema"),
    ]
    monkeypatch.setattr(Paths, "db_path", lambda self, db_id: mini_db)
    return SpiderData(split="test", examples=examples, schemas={})


def test_perfect_predictions_score_one(mini_data):
    """The harness must report 100% for a system that is correct by construction."""
    system = score_predictions(
        "perfect", mini_data, [e.query for e in mini_data.examples], progress_every=0
    )
    summary = system.summary()

    assert summary["execution_accuracy"] == 1.0
    assert summary["invalid_sql_rate"] == 0.0
    assert summary["n"] == 3


def test_gold_self_consistency_is_the_harness_self_test(mini_data):
    """This is the gate that runs before any GPU time is spent."""
    system = gold_self_consistency(mini_data)
    assert system.summary()["execution_accuracy"] == 1.0


def test_gold_cache_reports_no_failures_on_valid_gold(mini_data):
    cache = GoldCache.build(mini_data, progress_every=0)
    assert cache.failures() == []


def test_invalid_predictions_score_zero_and_are_counted(mini_data):
    system = score_predictions(
        "broken", mini_data, ["SELCT bad", "DROP TABLE student", ""], progress_every=0
    )
    summary = system.summary()

    assert summary["execution_accuracy"] == 0.0
    assert summary["invalid_sql_rate"] == 1.0


def test_a_semantically_equivalent_query_is_credited(mini_data):
    """Execution accuracy's whole point: different SQL, same answer, still correct."""
    predictions = [
        "SELECT COUNT(student_id) FROM student",  # different from gold COUNT(*)
        "SELECT name FROM student ORDER BY age",
        "SELECT course FROM enrolment",
    ]
    system = score_predictions("equivalent", mini_data, predictions, progress_every=0)

    assert system.summary()["execution_accuracy"] == 1.0
    # ...and exact match does NOT credit it, which is why EX is the headline.
    assert system.scores[0].exact_match is False


def test_order_by_question_requires_the_right_order(mini_data):
    """Example 1's gold has ORDER BY, so a differently-ordered result is wrong."""
    predictions = [
        "SELECT COUNT(*) FROM student",
        "SELECT name FROM student ORDER BY age DESC",  # reversed
        "SELECT course FROM enrolment",
    ]
    system = score_predictions("reversed", mini_data, predictions, progress_every=0)

    assert system.scores[1].execution_correct is False
    assert system.scores[1].invalid_sql is False  # it ran fine, it was just wrong
    assert system.summary()["execution_accuracy"] == pytest.approx(2 / 3)


def test_unordered_question_tolerates_a_different_order(mini_data):
    """Example 2's gold has no ORDER BY, so any row order is acceptable."""
    predictions = [
        "SELECT COUNT(*) FROM student",
        "SELECT name FROM student ORDER BY age",
        "SELECT course FROM enrolment ORDER BY enrolment_id DESC",
    ]
    system = score_predictions("shuffled", mini_data, predictions, progress_every=0)

    assert system.scores[2].execution_correct is True
    assert system.summary()["execution_accuracy"] == 1.0


def test_markdown_wrapped_predictions_are_still_scored(mini_data):
    """The base model habitually fences its output; that is formatting, not error."""
    predictions = [f"```sql\n{e.query}\n```" for e in mini_data.examples]
    system = score_predictions("fenced", mini_data, predictions, progress_every=0)

    assert system.summary()["execution_accuracy"] == 1.0


def test_evaluate_system_passes_prompts_to_the_generator(mini_data):
    generator = EchoGoldGenerator([e.query for e in mini_data.examples])
    prompts = [f"prompt {i}" for i in range(3)]

    system = evaluate_system("echo", mini_data, prompts, generator.generate)

    assert generator.calls == 1
    assert system.summary()["execution_accuracy"] == 1.0


def test_stub_generator_records_the_prompts_it_saw(mini_data):
    generator = StubGenerator(["SELECT COUNT(*) FROM student"])
    prompts = ["p0", "p1", "p2"]

    evaluate_system("stub", mini_data, prompts, generator.generate)

    assert generator.seen_prompts == prompts


def test_prediction_count_mismatch_is_rejected(mini_data):
    """Silently zipping a short prediction list would misalign every score after the gap."""
    with pytest.raises(ValueError):
        score_predictions("short", mini_data, ["SELECT 1"], progress_every=0)


def test_gold_cache_is_reused_across_systems(mini_data):
    """Three systems score against the same gold; executing it three times is waste."""
    cache = GoldCache.build(mini_data, progress_every=0)

    first = score_predictions(
        "a", mini_data, [e.query for e in mini_data.examples], gold_cache=cache, progress_every=0
    )
    second = score_predictions(
        "b", mini_data, ["SELCT bad"] * 3, gold_cache=cache, progress_every=0
    )

    assert first.summary()["execution_accuracy"] == 1.0
    assert second.summary()["execution_accuracy"] == 0.0


def test_compare_computes_the_headline_delta(mini_data):
    base = score_predictions("base_zero_shot", mini_data, ["SELCT bad"] * 3, progress_every=0)
    tuned = score_predictions(
        "lora_zero_shot",
        mini_data,
        [e.query for e in mini_data.examples],
        progress_every=0,
    )

    payload = compare([base, tuned])

    assert payload["headline"]["base_zero_shot_execution_accuracy"] == 0.0
    assert payload["headline"]["lora_zero_shot_execution_accuracy"] == 1.0
    assert payload["headline"]["absolute_gain"] == 1.0
    # Relative gain is undefined against a zero baseline; it must be None, not inf.
    assert payload["headline"]["relative_gain"] is None


def test_compare_includes_a_hardness_breakdown(mini_data):
    system = score_predictions(
        "lora_zero_shot",
        mini_data,
        [e.query for e in mini_data.examples],
        progress_every=0,
    )
    payload = compare([system])

    assert payload["by_hardness"]
    assert all("execution_accuracy" in row for row in payload["by_hardness"])


def test_compare_on_nothing_is_an_error():
    with pytest.raises(ValueError):
        compare([])


def test_timeout_config_is_respected(mini_data):
    """A per-query deadline that scored as "skipped" would be gameable; check it lands as
    an ordinary wrong answer."""
    runaway = (
        "WITH RECURSIVE forever(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM forever) "
        "SELECT COUNT(*) FROM forever"
    )
    system = score_predictions(
        "slow",
        mini_data,
        [runaway] * 3,
        config=EvalConfig(timeout_seconds=1.0),
        progress_every=0,
    )

    assert system.summary()["execution_accuracy"] == 0.0
    assert system.summary()["invalid_sql_rate"] == 1.0
