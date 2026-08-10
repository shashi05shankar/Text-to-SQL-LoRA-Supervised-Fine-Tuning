"""The metric definition, against hand-checkable cases.

If any of these is wrong, the headline number is wrong, and no amount of correct training
would reveal it. This is the most important test file in the repo.
"""

from __future__ import annotations

import pytest

from text2sql_lora.eval.metrics import (
    ExampleScore,
    SystemScores,
    canonicalize_value,
    exact_match,
    extract_sql,
    has_order_by,
    normalize_sql,
    results_match,
    sql_hardness,
)

pytestmark = pytest.mark.unit


# --- ORDER BY detection ------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "SELECT name FROM student ORDER BY age",
        "select name from student order by age desc",
        "SELECT name FROM student\nORDER   BY\n  age",
    ],
)
def test_detects_order_by(query):
    assert has_order_by(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "SELECT name FROM student",
        "SELECT COUNT(*) FROM student WHERE name = 'order by'",
        "SELECT ordering FROM t",  # 'order' must not match inside a longer identifier
    ],
)
def test_does_not_detect_order_by(query):
    assert has_order_by(query) is False


# --- the central rule: when does row order matter ----------------------------


def test_row_order_is_ignored_without_order_by():
    gold = [("Asha",), ("Bo",)]
    pred = [("Bo",), ("Asha",)]
    assert results_match(gold, pred, order_matters=False) is True


def test_row_order_is_enforced_with_order_by():
    """Same rows, different order. Correct without ORDER BY, wrong with it -- this single
    pair of assertions is the rule the whole metric turns on."""
    gold = [("Asha",), ("Bo",)]
    pred = [("Bo",), ("Asha",)]

    assert results_match(gold, pred, order_matters=False) is True
    assert results_match(gold, pred, order_matters=True) is False


def test_identical_results_match_either_way():
    rows = [("Asha", 21), ("Bo", 23)]
    assert results_match(rows, list(rows), order_matters=True) is True
    assert results_match(rows, list(rows), order_matters=False) is True


# --- multiset, not set -------------------------------------------------------


def test_duplicate_rows_are_part_of_the_answer():
    """A set comparison would call these equal. They are not: the counts differ, and a
    COUNT/GROUP BY question would get a different answer."""
    gold = [("a",), ("a",), ("b",)]
    pred = [("a",), ("b",)]
    assert results_match(gold, pred, order_matters=False) is False


def test_duplicates_match_when_counts_agree():
    gold = [("a",), ("a",), ("b",)]
    pred = [("b",), ("a",), ("a",)]
    assert results_match(gold, pred, order_matters=False) is True


# --- shape mismatches --------------------------------------------------------


def test_column_order_is_significant():
    assert results_match([("Asha", 21)], [(21, "Asha")], order_matters=False) is False


def test_different_row_counts_never_match():
    assert results_match([("a",)], [("a",), ("b",)], order_matters=False) is False


def test_two_empty_results_match():
    """A question whose correct answer is "no rows" must be scorable."""
    assert results_match([], [], order_matters=False) is True
    assert results_match([], [], order_matters=True) is True


def test_empty_does_not_match_non_empty():
    assert results_match([], [("a",)], order_matters=False) is False


def test_failed_execution_never_matches():
    """`None` means the query did not run. It is wrong even against another failure."""
    assert results_match(None, [("a",)], order_matters=False) is False
    assert results_match([("a",)], None, order_matters=False) is False
    assert results_match(None, None, order_matters=False) is False


# --- value canonicalization --------------------------------------------------


def test_floats_compare_after_rounding():
    """SQLite's AVG can differ in the last digits by summation order; neither model
    controls that."""
    assert results_match([(1.0000001,)], [(1.0,)], order_matters=False) is True


def test_genuinely_different_numbers_still_differ():
    assert results_match([(1.5,)], [(2.5,)], order_matters=False) is False


def test_integer_and_float_of_equal_value_match():
    assert results_match([(3,)], [(3.0,)], order_matters=False) is True


def test_booleans_normalize_to_integers():
    assert canonicalize_value(True) == 1
    assert canonicalize_value(False) == 0


def test_surrounding_whitespace_in_text_is_ignored():
    assert results_match([("Asha",)], [(" Asha ",)], order_matters=False) is True


def test_none_cells_are_preserved_not_coerced():
    assert canonicalize_value(None) is None
    assert results_match([(None,)], [(None,)], order_matters=False) is True
    assert results_match([(None,)], [("",)], order_matters=False) is False


# --- extracting SQL from raw model output ------------------------------------


def test_plain_sql_passes_through():
    assert extract_sql("SELECT * FROM t") == "SELECT * FROM t"


def test_markdown_fences_are_stripped():
    assert extract_sql("```sql\nSELECT * FROM t\n```") == "SELECT * FROM t"


def test_unterminated_fence_is_stripped():
    """Happens when the model hits the token limit mid-answer."""
    assert extract_sql("```sql\nSELECT * FROM t") == "SELECT * FROM t"


def test_leading_label_is_removed():
    assert extract_sql("SQL: SELECT 1") == "SELECT 1"
    assert extract_sql("Answer: SELECT 1") == "SELECT 1"


def test_only_the_first_statement_is_kept():
    assert extract_sql("SELECT 1; SELECT 2") == "SELECT 1"


def test_trailing_commentary_after_a_semicolon_is_dropped():
    assert extract_sql("SELECT 1; -- this counts students") == "SELECT 1"


def test_empty_output_extracts_to_empty():
    assert extract_sql("") == ""
    assert extract_sql("   ") == ""


# --- exact match (secondary) -------------------------------------------------


def test_exact_match_ignores_formatting():
    assert exact_match("SELECT a, b FROM t", "select   a,b   from t") is True
    assert exact_match("SELECT a FROM t;", "SELECT a FROM t") is True


def test_exact_match_is_sensitive_to_real_differences():
    assert exact_match("SELECT a FROM t", "SELECT b FROM t") is False


def test_normalize_sql_handles_empty():
    assert normalize_sql("") == ""


def test_exact_match_undercredits_equivalent_queries():
    """Documents the known weakness that makes exact match the secondary metric: these
    two queries are semantically identical and exact match still says no."""
    assert exact_match("SELECT a FROM t WHERE x = 1", "SELECT a FROM t WHERE 1 = x") is False


# --- hardness bucketing ------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("SELECT name FROM student", "easy"),
        ("SELECT name FROM student WHERE age > 20", "medium"),
        ("SELECT COUNT(*) FROM student", "medium"),
        ("SELECT name FROM student ORDER BY age", "medium"),
        ("SELECT COUNT(*) FROM student GROUP BY age", "hard"),
        ("SELECT s.name FROM student s JOIN enrolment e ON s.id = e.sid", "hard"),
        ("SELECT a FROM t1 JOIN t2 ON x JOIN t3 ON y", "extra"),
        ("SELECT name FROM student WHERE id IN (SELECT sid FROM enrolment)", "extra"),
        ("SELECT a FROM t1 UNION SELECT a FROM t2", "extra"),
    ],
)
def test_hardness_buckets(query, expected):
    assert sql_hardness(query) == expected


# --- aggregation -------------------------------------------------------------


def make_score(correct: bool, invalid: bool = False, hardness: str = "easy") -> ExampleScore:
    return ExampleScore(
        index=0,
        db_id="db",
        question="q",
        gold="SELECT 1",
        prediction="SELECT 1",
        execution_correct=correct,
        exact_match=correct,
        invalid_sql=invalid,
        error=None,
        hardness=hardness,
    )


def test_summary_computes_rates():
    system = SystemScores(
        name="test",
        scores=[make_score(True), make_score(True), make_score(False), make_score(False, True)],
    )
    summary = system.summary()

    assert summary["n"] == 4
    assert summary["execution_accuracy"] == 0.5
    assert summary["invalid_sql_rate"] == 0.25


def test_summary_on_no_examples_is_an_error_not_a_zero():
    """Silently reporting 0.0 for an empty run would put a fabricated number in the
    README."""
    with pytest.raises(ValueError):
        SystemScores(name="empty").summary()


def test_by_hardness_splits_and_omits_empty_buckets():
    system = SystemScores(
        name="test",
        scores=[
            make_score(True, hardness="easy"),
            make_score(False, hardness="easy"),
            make_score(True, hardness="extra"),
        ],
    )
    rows = system.by_hardness()

    assert [row["hardness"] for row in rows] == ["easy", "extra"]
    assert rows[0]["execution_accuracy"] == 0.5
    assert rows[1]["execution_accuracy"] == 1.0
