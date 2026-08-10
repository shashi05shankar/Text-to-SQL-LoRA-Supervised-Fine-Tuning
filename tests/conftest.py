"""Shared fixtures.

Everything here is built in-process: a hand-written two-table schema and a tiny SQLite
database created at test time. No Spider download, no model download, no network. Tests
that genuinely need the real corpus are marked `integration` and skip themselves.
"""

from __future__ import annotations

import sqlite3

import pytest

from text2sql_lora.config import Paths
from text2sql_lora.data.spider import data_is_available

# A minimal Spider-shaped `tables.json` record: two tables, one foreign key, one primary
# key each. Small enough to assert the rendered DDL character for character.
MINI_TABLES_ENTRY = {
    "db_id": "mini_school",
    "table_names_original": ["student", "enrolment"],
    "table_names": ["student", "enrolment"],
    "column_names_original": [
        [-1, "*"],
        [0, "student_id"],
        [0, "name"],
        [0, "age"],
        [1, "enrolment_id"],
        [1, "student_id"],
        [1, "course"],
    ],
    "column_names": [
        [-1, "*"],
        [0, "student id"],
        [0, "name"],
        [0, "age"],
        [1, "enrolment id"],
        [1, "student id"],
        [1, "course"],
    ],
    "column_types": ["text", "number", "text", "number", "number", "number", "text"],
    "primary_keys": [1, 4],
    "foreign_keys": [[5, 1]],
}


@pytest.fixture
def mini_tables_entry() -> dict:
    return dict(MINI_TABLES_ENTRY)


@pytest.fixture
def mini_db(tmp_path):
    """A real SQLite file matching `MINI_TABLES_ENTRY`, with a handful of rows."""
    path = tmp_path / "mini_school.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE student (student_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        CREATE TABLE enrolment (
            enrolment_id INTEGER PRIMARY KEY,
            student_id INTEGER,
            course TEXT
        );
        INSERT INTO student VALUES (1, 'Asha', 21), (2, 'Bo', 23), (3, 'Chen', 21);
        INSERT INTO enrolment VALUES (10, 1, 'algebra'), (11, 1, 'physics'), (12, 2, 'algebra');
        """
    )
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def paths() -> Paths:
    return Paths()


def requires_spider() -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not data_is_available(),
        reason="Spider not downloaded; run `python scripts/download_data.py`",
    )
