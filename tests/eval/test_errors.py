"""Failure classification, against hand-constructed execution outcomes."""

from __future__ import annotations

import pytest

from text2sql_lora.eval.errors import (
    FailureKind,
    build_profile,
    classify,
    classify_error_message,
)
from text2sql_lora.eval.execute import ExecutionResult

pytestmark = pytest.mark.unit


def ok(rows) -> ExecutionResult:
    return ExecutionResult(rows=rows, error=None, elapsed_seconds=0.01)


def failed(error: str, timed_out: bool = False) -> ExecutionResult:
    return ExecutionResult(
        rows=None, error=error, elapsed_seconds=0.01, timed_out=timed_out
    )


# --- error message classification --------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("OperationalError: no such column: pet_type", FailureKind.INVALID_NO_SUCH_COLUMN),
        ("OperationalError: no such table: Studens", FailureKind.INVALID_NO_SUCH_TABLE),
        ("OperationalError: ambiguous column name: name", FailureKind.INVALID_AMBIGUOUS_COLUMN),
        (
            "OperationalError: misuse of aggregate function COUNT()",
            FailureKind.INVALID_AGGREGATE_MISUSE,
        ),
        ('OperationalError: near "SELCT": syntax error', FailureKind.INVALID_SYNTAX),
        ("OperationalError: no such function: STRFTIME_X", FailureKind.INVALID_SYNTAX),
        ("timeout", FailureKind.INVALID_TIMEOUT),
        ("empty query", FailureKind.INVALID_EMPTY),
        ("DatabaseError: something unusual", FailureKind.INVALID_OTHER),
        ("", FailureKind.INVALID_OTHER),
    ],
)
def test_error_messages_are_classified(message, expected):
    assert classify_error_message(message) == expected


def test_specific_patterns_win_over_generic_ones():
    """'no such function' mentions neither column nor table but is a syntax-family fault."""
    assert classify_error_message("no such function: FOO") == FailureKind.INVALID_SYNTAX


# --- correct -----------------------------------------------------------------


def test_a_correct_prediction_is_classified_correct():
    kind = classify(ok([("a",)]), ok([("a",)]), execution_correct=True)
    assert kind == FailureKind.CORRECT
    assert not kind.is_invalid and not kind.is_semantic


def test_the_taxonomy_defers_to_the_headline_metric():
    """If the harness says correct, the taxonomy must not disagree -- even when the rows
    look different to a naive comparison."""
    kind = classify(ok([("b",), ("a",)]), ok([("a",), ("b",)]), execution_correct=True)
    assert kind == FailureKind.CORRECT


# --- invalid -----------------------------------------------------------------


def test_invalid_sql_is_classified_from_its_error():
    kind = classify(ok([("a",)]), failed("no such column: xyz"), execution_correct=False)
    assert kind == FailureKind.INVALID_NO_SUCH_COLUMN
    assert kind.is_invalid


def test_timeout_is_its_own_kind_regardless_of_message():
    kind = classify(ok([("a",)]), failed("interrupted", timed_out=True), execution_correct=False)
    assert kind == FailureKind.INVALID_TIMEOUT


# --- executed but wrong ------------------------------------------------------


def test_empty_result_when_rows_were_expected():
    kind = classify(ok([("a",)]), ok([]), execution_correct=False)
    assert kind == FailureKind.WRONG_EMPTY
    assert kind.is_semantic


def test_rows_returned_when_none_were_expected():
    kind = classify(ok([]), ok([("a",)]), execution_correct=False)
    assert kind == FailureKind.WRONG_SHOULD_BE_EMPTY


def test_wrong_column_count():
    kind = classify(ok([("a", 1)]), ok([("a",)]), execution_correct=False)
    assert kind == FailureKind.WRONG_COLUMN_COUNT


def test_wrong_row_count():
    kind = classify(ok([("a",), ("b",)]), ok([("a",)]), execution_correct=False)
    assert kind == FailureKind.WRONG_ROW_COUNT


def test_right_rows_wrong_order():
    """Same multiset, same size -- so only the ordering can be at fault. This is only
    reachable when the gold query had ORDER BY, since otherwise the harness would have
    scored it correct."""
    kind = classify(ok([("a",), ("b",)]), ok([("b",), ("a",)]), execution_correct=False)
    assert kind == FailureKind.WRONG_ORDER


def test_same_shape_different_values():
    kind = classify(ok([("a",), ("b",)]), ok([("c",), ("d",)]), execution_correct=False)
    assert kind == FailureKind.WRONG_VALUES


def test_float_rounding_is_applied_before_comparing():
    """Values differing only past the rounding point must not read as WRONG_VALUES."""
    kind = classify(
        ok([(1.00000001,), (2.0,)]), ok([(1.0,), (2.0,)]), execution_correct=False
    )
    assert kind == FailureKind.WRONG_ORDER  # multisets match after canonicalization


# --- profiles ----------------------------------------------------------------


def test_profile_counts_and_rates():
    kinds = (
        [FailureKind.CORRECT] * 6
        + [FailureKind.INVALID_NO_SUCH_COLUMN] * 2
        + [FailureKind.WRONG_VALUES] * 2
    )
    profile = build_profile("test", kinds)

    assert profile.n == 10
    assert profile.n_correct == 6
    assert profile.n_invalid == 2
    assert profile.n_semantic == 2
    assert profile.rate(FailureKind.CORRECT) == 0.6


def test_semantic_share_of_errors_tracks_error_migration():
    """The headline diagnostic: of the questions a system gets wrong, how many at least
    ran? Higher means the model has learned to write SQL and now has a harder problem."""
    mostly_invalid = build_profile(
        "before", [FailureKind.INVALID_SYNTAX] * 8 + [FailureKind.WRONG_VALUES] * 2
    )
    mostly_semantic = build_profile(
        "after", [FailureKind.INVALID_SYNTAX] * 2 + [FailureKind.WRONG_VALUES] * 8
    )

    assert mostly_invalid.semantic_share_of_errors() == pytest.approx(0.2)
    assert mostly_semantic.semantic_share_of_errors() == pytest.approx(0.8)


def test_semantic_share_is_zero_when_nothing_is_wrong():
    """No errors at all must not divide by zero."""
    profile = build_profile("perfect", [FailureKind.CORRECT] * 5)
    assert profile.semantic_share_of_errors() == 0.0


def test_profile_serializes_sorted_by_frequency():
    kinds = (
        [FailureKind.CORRECT] * 3
        + [FailureKind.INVALID_SYNTAX] * 5
        + [FailureKind.WRONG_VALUES]
    )
    payload = build_profile("test", kinds).as_dict()

    assert next(iter(payload["by_kind"])) == FailureKind.INVALID_SYNTAX.value
    assert payload["by_kind"][FailureKind.INVALID_SYNTAX.value]["n"] == 5
    assert payload["invalid_rate"] == pytest.approx(5 / 9)


def test_empty_profile_does_not_divide_by_zero():
    profile = build_profile("nothing", [])
    assert profile.as_dict()["invalid_rate"] == 0.0
    assert profile.rate(FailureKind.CORRECT) == 0.0
