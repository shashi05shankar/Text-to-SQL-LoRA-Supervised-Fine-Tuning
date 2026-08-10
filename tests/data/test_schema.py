"""The rendered schema is what the model actually sees, so it is asserted exactly."""

from __future__ import annotations

import pytest

from text2sql_lora.data.schema import parse_db_entry, render_schema

pytestmark = pytest.mark.unit


def test_parses_tables_and_columns(mini_tables_entry):
    schema = parse_db_entry(mini_tables_entry)

    assert schema.db_id == "mini_school"
    assert schema.table_names == ["student", "enrolment"]
    assert schema.n_tables() == 2
    # Seven entries in tables.json, but "*" is synthetic and not a real column.
    assert schema.n_columns() == 6


def test_renders_expected_ddl(mini_tables_entry):
    rendered = render_schema(parse_db_entry(mini_tables_entry))

    assert rendered == (
        "CREATE TABLE student (\n"
        "  student_id NUMBER PRIMARY KEY,\n"
        "  name TEXT,\n"
        "  age NUMBER\n"
        ");\n"
        "\n"
        "CREATE TABLE enrolment (\n"
        "  enrolment_id NUMBER PRIMARY KEY,\n"
        "  student_id NUMBER,\n"
        "  course TEXT,\n"
        "  FOREIGN KEY (student_id) REFERENCES student(student_id)\n"
        ");"
    )


def test_star_column_is_never_rendered(mini_tables_entry):
    rendered = render_schema(parse_db_entry(mini_tables_entry))
    assert "*" not in rendered


def test_foreign_keys_are_rendered_on_the_referencing_table(mini_tables_entry):
    """The FK belongs to `enrolment`, not `student` -- getting this backwards would tell
    the model to join in the wrong direction."""
    schema = parse_db_entry(mini_tables_entry)

    assert len(schema.foreign_keys) == 1
    foreign_key = schema.foreign_keys[0]
    assert foreign_key.from_table == "enrolment"
    assert foreign_key.to_table == "student"

    student_block, enrolment_block = render_schema(schema).split("\n\n")
    assert "FOREIGN KEY" not in student_block
    assert "FOREIGN KEY" in enrolment_block


def test_rendering_is_deterministic(mini_tables_entry):
    """A prompt that varied between the base run and the tuned run would confound the
    entire comparison."""
    schema = parse_db_entry(mini_tables_entry)
    assert render_schema(schema) == render_schema(schema)


def test_skips_foreign_keys_that_reference_the_star_column(mini_tables_entry):
    """Spider contains a few malformed FK entries; they must be dropped, not rendered
    into broken DDL."""
    mini_tables_entry["foreign_keys"] = [[5, 0]]  # column 0 is "*", table index -1
    schema = parse_db_entry(mini_tables_entry)

    assert schema.foreign_keys == []
    assert "FOREIGN KEY" not in render_schema(schema)


def test_mismatched_column_arrays_are_rejected(mini_tables_entry):
    """`zip(strict=True)` should catch a truncated column_types array rather than
    silently dropping columns off the end of the schema."""
    mini_tables_entry["column_types"] = mini_tables_entry["column_types"][:-1]
    with pytest.raises(ValueError):
        parse_db_entry(mini_tables_entry)
