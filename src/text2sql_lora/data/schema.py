"""Turn Spider's `tables.json` into the schema text the model is prompted with.

Spider describes each database as flat parallel arrays -- table names, `(table_index,
column_name)` pairs, column types, primary keys as column indices, and foreign keys as
`(column_index, column_index)` pairs. That representation is compact but unreadable, and
an LLM prompted with it does markedly worse than one prompted with the `CREATE TABLE`
statements it saw a great deal of during pretraining. So we reconstruct DDL.

Foreign keys are included deliberately: Spider is a *multi-table* benchmark and roughly
half its queries join. Without FK lines the model has no way to know which columns link
which tables, and it guesses join conditions.

Column example values are NOT included in v1. They measurably help (the model learns that
`country` holds `'USA'` rather than `'United States'`), but they inflate prompt length and
require reading every database at prompt-build time. Documented as future work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Column:
    table_index: int
    name: str
    type: str


@dataclass(frozen=True)
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass(frozen=True)
class DatabaseSchema:
    """One Spider database, in a form that can be rendered as DDL."""

    db_id: str
    table_names: list[str]
    columns: list[Column]
    primary_keys: list[int]
    foreign_keys: list[ForeignKey]

    def n_tables(self) -> int:
        return len(self.table_names)

    def n_columns(self) -> int:
        # Column 0 is Spider's synthetic "*" column, which is not a real column.
        return sum(1 for c in self.columns if c.name != "*")


def _sql_type(spider_type: str) -> str:
    """Map Spider's coarse column types onto SQLite type names."""
    return {
        "text": "TEXT",
        "number": "NUMBER",
        "time": "TIME",
        "boolean": "BOOLEAN",
        "others": "TEXT",
    }.get(spider_type, "TEXT")


def parse_tables_json(path: Path) -> dict[str, DatabaseSchema]:
    """Read the whole of `tables.json` into `{db_id: DatabaseSchema}`."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {entry["db_id"]: parse_db_entry(entry) for entry in raw}


def parse_db_entry(entry: dict) -> DatabaseSchema:
    """Parse one `tables.json` record into a `DatabaseSchema`.

    Uses the `_original` name fields throughout. Spider also carries lowercased,
    space-separated "natural" names, but the gold SQL -- and therefore anything we
    execute -- refers to the original identifiers.
    """
    table_names: list[str] = list(entry["table_names_original"])
    columns = [
        Column(table_index=table_index, name=name, type=_sql_type(column_type))
        for (table_index, name), column_type in zip(
            entry["column_names_original"], entry["column_types"], strict=True
        )
    ]

    foreign_keys: list[ForeignKey] = []
    for from_index, to_index in entry.get("foreign_keys", []):
        from_column, to_column = columns[from_index], columns[to_index]
        # Spider contains a handful of FK entries pointing at the synthetic "*" column
        # or at a table index of -1. They are not representable as DDL; skip them
        # rather than emitting a broken REFERENCES clause.
        if from_column.table_index < 0 or to_column.table_index < 0:
            continue
        foreign_keys.append(
            ForeignKey(
                from_table=table_names[from_column.table_index],
                from_column=from_column.name,
                to_table=table_names[to_column.table_index],
                to_column=to_column.name,
            )
        )

    return DatabaseSchema(
        db_id=entry["db_id"],
        table_names=table_names,
        columns=columns,
        primary_keys=list(entry.get("primary_keys", [])),
        foreign_keys=foreign_keys,
    )


def render_schema(schema: DatabaseSchema) -> str:
    """Render a `DatabaseSchema` as `CREATE TABLE` statements.

    The output is deterministic -- tables in `tables.json` order, columns in their
    declared order -- because the prompt is part of the experiment. A prompt that
    varies between the base and fine-tuned runs would make the comparison meaningless.
    """
    primary_key_set = set(schema.primary_keys)
    blocks: list[str] = []

    for table_index, table_name in enumerate(schema.table_names):
        lines: list[str] = []
        for column_index, column in enumerate(schema.columns):
            if column.table_index != table_index or column.name == "*":
                continue
            line = f"  {column.name} {column.type}"
            if column_index in primary_key_set:
                line += " PRIMARY KEY"
            lines.append(line)

        for foreign_key in schema.foreign_keys:
            if foreign_key.from_table != table_name:
                continue
            lines.append(
                f"  FOREIGN KEY ({foreign_key.from_column}) "
                f"REFERENCES {foreign_key.to_table}({foreign_key.to_column})"
            )

        body = ",\n".join(lines)
        blocks.append(f"CREATE TABLE {table_name} (\n{body}\n);")

    return "\n\n".join(blocks)


def render_schema_for_db(schemas: dict[str, DatabaseSchema], db_id: str) -> str:
    """Convenience lookup that fails loudly on an unknown `db_id`."""
    if db_id not in schemas:
        raise KeyError(f"unknown db_id {db_id!r}; {len(schemas)} databases loaded")
    return render_schema(schemas[db_id])
