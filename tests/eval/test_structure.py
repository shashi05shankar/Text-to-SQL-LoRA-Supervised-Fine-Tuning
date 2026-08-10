"""Schema grounding and component matching, on the fixture schema."""

from __future__ import annotations

import pytest

from text2sql_lora.data.schema import parse_db_entry
from text2sql_lora.eval.structure import (
    ComponentMatch,
    SchemaIndex,
    check_grounding,
    compare_shapes,
    parse,
    query_shape,
    summarize_components,
    summarize_grounding,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def index(mini_tables_entry) -> SchemaIndex:
    return SchemaIndex.from_schema(parse_db_entry(mini_tables_entry))


# --- parsing -----------------------------------------------------------------


def test_valid_sql_parses():
    assert parse("SELECT a FROM t") is not None


@pytest.mark.parametrize("sql", ["", "   ", "SELCT nonsense FROM", "this is not sql at all ((("])
def test_unparseable_input_returns_none_rather_than_raising(sql):
    assert parse(sql) is None


# --- schema grounding --------------------------------------------------------


def test_index_knows_the_schema(index):
    assert "student" in index.tables
    assert "enrolment" in index.tables
    assert "student_id" in index.columns
    assert "student.name" in index.qualified
    # The synthetic "*" column is not a real column.
    assert "*" not in index.columns


def test_a_fully_grounded_query(index):
    result = check_grounding("SELECT name FROM student WHERE age > 20", index)

    assert result.parsed and result.grounded
    assert result.tables_used == {"student"}
    assert not result.hallucinated_columns


def test_hallucinated_column_is_caught(index):
    """The exact failure seen in the real run: pet_type instead of petType."""
    result = check_grounding("SELECT nickname FROM student", index)

    assert not result.grounded
    assert "nickname" in result.hallucinated_columns


def test_hallucinated_table_is_caught(index):
    result = check_grounding("SELECT name FROM teachers", index)

    assert not result.grounded
    assert "teachers" in result.hallucinated_tables


def test_aliases_are_resolved_not_flagged(index):
    """`T1` is an alias for `student`, not a hallucinated table."""
    result = check_grounding(
        "SELECT T1.name FROM student AS T1 JOIN enrolment AS T2 "
        "ON T1.student_id = T2.student_id",
        index,
    )

    assert result.grounded
    assert result.hallucinated_tables == set()
    assert result.tables_used == {"student", "enrolment"}


def test_column_on_the_wrong_table_is_caught(index):
    """`course` exists, but on `enrolment` -- not on `student`."""
    result = check_grounding("SELECT student.course FROM student", index)

    assert not result.grounded
    assert result.hallucinated_columns == {"student.course"}


def test_star_is_not_a_hallucination(index):
    assert check_grounding("SELECT * FROM student", index).grounded


def test_double_quoted_string_literals_are_not_hallucinated_columns(index):
    """SQLite treats a double-quoted token that is not a column as a STRING literal, and
    Spider's gold SQL relies on it constantly. A parser returns those as Column nodes;
    counting them as invented identifiers would badly overstate hallucination -- these
    queries run fine and are often correct."""
    result = check_grounding('SELECT name FROM student WHERE name = "Asha"', index)

    assert result.grounded
    assert result.hallucinated_columns == set()


def test_a_quoted_identifier_that_is_a_real_column_still_counts_as_one(index):
    result = check_grounding('SELECT "name" FROM student', index)

    assert result.grounded
    assert "name" in result.columns_used


def test_unquoted_unknown_columns_are_still_caught(index):
    """The literal exemption must not become a blanket amnesty."""
    result = check_grounding("SELECT name FROM student WHERE nickname = 1", index)

    assert not result.grounded
    assert "nickname" in result.hallucinated_columns


def test_unparseable_query_is_not_grounded(index):
    result = check_grounding("SELCT bad FROM", index)

    assert not result.parsed
    assert not result.grounded


# --- query shape -------------------------------------------------------------


def test_shape_of_a_simple_query():
    shape = query_shape("SELECT name FROM student")

    assert shape.parsed
    assert shape.select_width == 1
    assert shape.n_joins == 0
    assert shape.clause_set() == frozenset()


def test_shape_captures_every_clause():
    shape = query_shape(
        "SELECT DISTINCT T1.name, count(*) FROM student AS T1 "
        "JOIN enrolment AS T2 ON T1.student_id = T2.student_id "
        "WHERE T1.age > 20 GROUP BY T1.name HAVING count(*) > 1 "
        "ORDER BY count(*) DESC LIMIT 5"
    )

    assert shape.n_joins == 1
    assert shape.select_width == 2
    assert "count" in shape.aggregates
    assert shape.clause_set() >= {
        "where", "group_by", "having", "order_by", "limit", "join", "distinct"
    }


def test_shape_detects_nesting():
    shape = query_shape(
        "SELECT name FROM student WHERE student_id IN (SELECT student_id FROM enrolment)"
    )
    assert shape.has_nested
    assert "nested" in shape.clause_set()


def test_shape_detects_set_operations():
    shape = query_shape("SELECT a FROM t1 UNION SELECT a FROM t2")
    assert shape.has_set_op
    assert "set_op" in shape.clause_set()


def test_unparseable_shape_is_marked_not_parsed():
    shape = query_shape("SELCT bad FROM")
    assert not shape.parsed
    assert shape.clause_set() == frozenset()


# --- component matching ------------------------------------------------------


def test_identical_queries_match_perfectly():
    sql = "SELECT name FROM student WHERE age > 20 ORDER BY age"
    match = compare_shapes(query_shape(sql), query_shape(sql))

    assert match.clause_f1 == 1.0
    assert match.exact_shape_match
    assert not match.missing_clauses and not match.spurious_clauses


def test_formatting_differences_do_not_change_the_shape():
    a = query_shape("SELECT name,age FROM student WHERE age>20")
    b = query_shape("select   name , age\nfrom student\nwhere age > 20")
    assert compare_shapes(a, b).exact_shape_match


def test_a_missing_clause_is_named():
    gold = query_shape("SELECT name FROM student WHERE age > 20 GROUP BY name")
    pred = query_shape("SELECT name FROM student WHERE age > 20")
    match = compare_shapes(gold, pred)

    assert "group_by" in match.missing_clauses
    assert match.clause_recall < 1.0
    assert match.clause_precision == 1.0  # nothing spurious was added
    assert not match.exact_shape_match


def test_a_spurious_clause_is_named():
    """The real regression pattern: the base model added a JOIN nobody asked for."""
    gold = query_shape("SELECT DISTINCT country FROM singer WHERE age > 20")
    pred = query_shape(
        "SELECT DISTINCT T1.country FROM singer AS T1 "
        "JOIN singer_in_concert AS T2 ON T1.id = T2.id WHERE T1.age > 20"
    )
    match = compare_shapes(gold, pred)

    assert "join" in match.spurious_clauses
    assert match.clause_precision < 1.0
    assert not match.join_count_match


def test_partial_credit_lands_between_zero_and_one():
    gold = query_shape("SELECT name FROM student WHERE age > 20 GROUP BY name ORDER BY name")
    pred = query_shape("SELECT name FROM student WHERE age > 20")
    match = compare_shapes(gold, pred)

    assert 0.0 < match.clause_f1 < 1.0


def test_wrong_aggregate_is_detected_even_when_clauses_agree():
    """Same shape, different aggregate -- execution accuracy would score this 0 with no
    explanation; component matching says exactly what differed."""
    gold = query_shape("SELECT count(*) FROM student")
    pred = query_shape("SELECT sum(age) FROM student")
    match = compare_shapes(gold, pred)

    assert not match.aggregate_match
    assert match.clause_f1 == 1.0  # the clause structure really is identical


def test_unparseable_prediction_scores_zero_not_skipped():
    """Dropping unparseable predictions would flatter the model."""
    match = compare_shapes(query_shape("SELECT a FROM t"), query_shape("SELCT bad"))

    assert not match.both_parsed
    assert match.clause_f1 == 0.0
    assert not match.exact_shape_match


# --- aggregation -------------------------------------------------------------


def test_summarize_grounding(index):
    results = [
        check_grounding("SELECT name FROM student", index),
        check_grounding("SELECT nickname FROM student", index),
        check_grounding("SELCT bad", index),
    ]
    summary = summarize_grounding(results)

    assert summary["n"] == 3
    assert summary["parse_rate"] == pytest.approx(2 / 3)
    assert summary["grounded_rate"] == pytest.approx(1 / 3)
    assert summary["hallucinated_column_rate"] == pytest.approx(1 / 3)


def test_summarize_components_averages():
    matches = [
        ComponentMatch(both_parsed=True, clause_f1=1.0, exact_shape_match=True),
        ComponentMatch(both_parsed=True, clause_f1=0.0, exact_shape_match=False),
    ]
    summary = summarize_components(matches)

    assert summary["clause_f1"] == pytest.approx(0.5)
    assert summary["exact_shape_match_rate"] == pytest.approx(0.5)


def test_summaries_of_nothing_are_empty_not_crashes():
    assert summarize_grounding([]) == {}
    assert summarize_components([]) == {}
