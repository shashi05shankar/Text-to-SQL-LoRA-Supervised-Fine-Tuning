"""SQL execution against a real (tiny) SQLite database built in the fixture."""

from __future__ import annotations

import sqlite3

import pytest

from text2sql_lora.eval.execute import execute_query

pytestmark = pytest.mark.unit


def test_returns_rows_for_a_valid_query(mini_db):
    result = execute_query(mini_db, "SELECT name FROM student ORDER BY student_id")

    assert result.ok
    assert result.error is None
    assert result.rows == [("Asha",), ("Bo",), ("Chen",)]


def test_join_across_tables(mini_db):
    result = execute_query(
        mini_db,
        "SELECT s.name, e.course FROM student s JOIN enrolment e "
        "ON s.student_id = e.student_id ORDER BY e.enrolment_id",
    )

    assert result.ok
    assert result.rows == [("Asha", "algebra"), ("Asha", "physics"), ("Bo", "algebra")]


def test_empty_result_is_a_success_not_a_failure(mini_db):
    """"No rows" is a legitimate answer and must be distinguishable from an error."""
    result = execute_query(mini_db, "SELECT name FROM student WHERE age > 99")

    assert result.ok
    assert result.rows == []


def test_syntax_error_is_returned_not_raised(mini_db):
    """The eval loop scores thousands of model outputs; one bad query must not kill it."""
    result = execute_query(mini_db, "SELCT nonsense FROM")

    assert not result.ok
    assert result.rows is None
    assert "Error" in result.error


def test_unknown_table_is_an_error(mini_db):
    result = execute_query(mini_db, "SELECT * FROM no_such_table")
    assert not result.ok
    assert "no_such_table" in result.error


def test_empty_query_is_an_error(mini_db):
    result = execute_query(mini_db, "   ")
    assert not result.ok
    assert result.error == "empty query"


def test_database_is_opened_read_only(mini_db):
    """A hallucinated DROP or INSERT must not be able to corrupt the corpus mid-eval."""
    result = execute_query(mini_db, "DELETE FROM student")
    assert not result.ok

    connection = sqlite3.connect(mini_db)
    remaining = connection.execute("SELECT COUNT(*) FROM student").fetchone()[0]
    connection.close()
    assert remaining == 3


def test_drop_table_is_rejected_and_the_table_survives(mini_db):
    execute_query(mini_db, "DROP TABLE enrolment")

    connection = sqlite3.connect(mini_db)
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    connection.close()
    assert "enrolment" in tables


def test_missing_database_file_is_an_error_not_a_crash(tmp_path):
    result = execute_query(tmp_path / "does_not_exist.sqlite", "SELECT 1")
    assert not result.ok


def test_timeout_is_reported_as_a_failure(mini_db):
    """A runaway query must score as WRONG, never as skipped -- otherwise emitting
    non-terminating SQL would be a way to avoid being marked incorrect.

    The recursive CTE runs effectively forever; the deadline has to stop it.
    """
    runaway = """
        WITH RECURSIVE forever(n) AS (
            SELECT 1 UNION ALL SELECT n + 1 FROM forever
        )
        SELECT COUNT(*) FROM forever
    """
    result = execute_query(mini_db, runaway, timeout_seconds=1.0)

    assert not result.ok
    assert result.timed_out is True
    assert result.error == "timeout"
    # Should abort near the deadline, not run for minutes.
    assert result.elapsed_seconds < 15.0


def test_elapsed_time_is_recorded(mini_db):
    result = execute_query(mini_db, "SELECT 1")
    assert result.elapsed_seconds >= 0.0
