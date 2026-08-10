"""What the model got structurally right, independent of whether the rows matched.

Execution accuracy is binary and unforgiving: a query that gets the joins, the filter and
the grouping right but picks the wrong aggregate scores exactly the same as one that is
pure gibberish. That is correct as a headline metric and useless as a diagnostic.

Two things measured here, both parsed from a real SQL AST (`sqlglot`) rather than by
pattern-matching strings -- because `T1.name` and `singer.name` and `name` may all refer
to the same column, and a regex cannot know that.

**Schema grounding.** Do the tables and columns the query names actually exist in the
database it was given? A model that invents `pet_type` where the schema says `petType`
has failed at *grounding*, not at SQL. This is the failure that better schema
serialization or constrained decoding attacks, and it is invisible in an accuracy number.

**Component matching.** Does the query have the same shape as gold -- same clauses
present, same number of joins, same aggregates, same nesting? This is the family of
partial-credit metrics the Spider literature reports alongside exact and execution match.
Ours is a simplified reimplementation, labelled as such: it compares clause presence and
counts, not the full component decomposition of the official evaluator.

Unparseable predictions are counted, never silently skipped. A model that emits text
sqlglot cannot parse has told you something, and dropping those rows would flatter it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from text2sql_lora.data.schema import DatabaseSchema

DIALECT = "sqlite"

CLAUSES = ("select", "where", "group_by", "having", "order_by", "limit", "join", "nested")


def parse(sql: str):
    """Parse SQL to an AST, or return None if it is not a query.

    sqlglot is deliberately permissive and will happily parse `SELCT bad` as a column
    aliased to `bad` rather than raising -- so "it parsed" is not the same as "it is a
    query". The result is therefore required to be a `Query` (SELECT, or a set operation
    over SELECTs); anything else counts as unparseable, which is what a model emitting
    prose or a truncated statement deserves.
    """
    if not sql or not sql.strip():
        return None
    try:
        tree = sqlglot.parse_one(sql, dialect=DIALECT)
    except Exception:
        return None
    return tree if isinstance(tree, exp.Query) else None


# --- schema grounding --------------------------------------------------------


@dataclass(frozen=True)
class SchemaIndex:
    """Case-insensitive lookup of what actually exists in one database."""

    tables: frozenset[str]
    columns: frozenset[str]  # bare column names, lowercased
    qualified: frozenset[str]  # "table.column", lowercased

    @classmethod
    def from_schema(cls, schema: DatabaseSchema) -> SchemaIndex:
        tables = {name.lower() for name in schema.table_names}
        columns: set[str] = set()
        qualified: set[str] = set()
        for column in schema.columns:
            if column.name == "*":
                continue
            columns.add(column.name.lower())
            if 0 <= column.table_index < len(schema.table_names):
                table = schema.table_names[column.table_index].lower()
                qualified.add(f"{table}.{column.name.lower()}")
        return cls(frozenset(tables), frozenset(columns), frozenset(qualified))


def _is_quoted(column: exp.Column) -> bool:
    """True when the column's own identifier was written in quotes."""
    identifier = column.this
    return bool(isinstance(identifier, exp.Identifier) and identifier.args.get("quoted"))


@dataclass
class GroundingResult:
    """Which identifiers in a query do not exist in its database."""

    parsed: bool
    tables_used: set[str] = field(default_factory=set)
    columns_used: set[str] = field(default_factory=set)
    hallucinated_tables: set[str] = field(default_factory=set)
    hallucinated_columns: set[str] = field(default_factory=set)

    @property
    def grounded(self) -> bool:
        """True when every identifier the query names exists in the schema."""
        return (
            self.parsed
            and not self.hallucinated_tables
            and not self.hallucinated_columns
        )

    def as_dict(self) -> dict:
        return {
            "parsed": self.parsed,
            "grounded": self.grounded,
            "n_tables_used": len(self.tables_used),
            "n_columns_used": len(self.columns_used),
            "hallucinated_tables": sorted(self.hallucinated_tables),
            "hallucinated_columns": sorted(self.hallucinated_columns),
        }


def check_grounding(sql: str, index: SchemaIndex) -> GroundingResult:
    """Find tables and columns named by `sql` that the schema does not contain.

    Alias-aware: `singer AS T1 ... T1.name` resolves `T1` back to `singer` before looking
    the column up, so aliases are not mistaken for hallucinated tables.
    """
    tree = parse(sql)
    if tree is None:
        return GroundingResult(parsed=False)

    # Map alias -> real table name, so qualified columns can be resolved.
    aliases: dict[str, str] = {}
    tables_used: set[str] = set()
    for table in tree.find_all(exp.Table):
        real = (table.name or "").lower()
        if not real:
            continue
        tables_used.add(real)
        alias = (table.alias or "").lower()
        if alias:
            aliases[alias] = real

    columns_used: set[str] = set()
    hallucinated_columns: set[str] = set()
    for column in tree.find_all(exp.Column):
        name = (column.name or "").lower()
        if not name or name == "*":
            continue

        # SQLite's "double-quoted string misfeature": a double-quoted token that does not
        # resolve to a column is treated as a STRING LITERAL, not an identifier. Spider's
        # gold SQL uses this heavily (`WHERE pettype = "cat"`), so the fine-tuned model
        # learned it -- and a parser hands those back as Column nodes.
        #
        # Counting them as hallucinated identifiers is simply wrong: they execute
        # correctly and are frequently part of a *correct* answer. Without this rule the
        # metric reports `cat`, `dog` and `ben jones` as invented columns and badly
        # overstates the tuned model's hallucination rate.
        if _is_quoted(column) and name not in index.columns:
            continue

        columns_used.add(name)

        qualifier = (column.table or "").lower()
        if qualifier:
            resolved = aliases.get(qualifier, qualifier)
            # A qualified column is grounded if table.column exists, or -- when the
            # qualifier is an unknown alias for a subquery -- if the bare name exists.
            if f"{resolved}.{name}" in index.qualified:
                continue
            if resolved not in index.tables and name in index.columns:
                continue
            if name not in index.columns:
                hallucinated_columns.add(f"{qualifier}.{name}")
            elif resolved in index.tables:
                hallucinated_columns.add(f"{resolved}.{name}")
        elif name not in index.columns:
            hallucinated_columns.add(name)

    hallucinated_tables = {t for t in tables_used if t not in index.tables}

    return GroundingResult(
        parsed=True,
        tables_used=tables_used,
        columns_used=columns_used,
        hallucinated_tables=hallucinated_tables,
        hallucinated_columns=hallucinated_columns,
    )


# --- component matching ------------------------------------------------------


@dataclass(frozen=True)
class QueryShape:
    """The structural fingerprint of a query."""

    parsed: bool
    select_width: int = 0
    n_joins: int = 0
    n_aggregates: int = 0
    aggregates: frozenset[str] = frozenset()
    has_where: bool = False
    has_group_by: bool = False
    has_having: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    has_nested: bool = False
    has_set_op: bool = False
    has_distinct: bool = False

    def clause_set(self) -> frozenset[str]:
        """Which clauses are present, as a set, for F1-style comparison."""
        present = set()
        if self.has_where:
            present.add("where")
        if self.has_group_by:
            present.add("group_by")
        if self.has_having:
            present.add("having")
        if self.has_order_by:
            present.add("order_by")
        if self.has_limit:
            present.add("limit")
        if self.n_joins:
            present.add("join")
        if self.has_nested:
            present.add("nested")
        if self.has_set_op:
            present.add("set_op")
        if self.has_distinct:
            present.add("distinct")
        return frozenset(present)

    def as_dict(self) -> dict:
        return {
            "parsed": self.parsed,
            "select_width": self.select_width,
            "n_joins": self.n_joins,
            "n_aggregates": self.n_aggregates,
            "aggregates": sorted(self.aggregates),
            "clauses": sorted(self.clause_set()),
        }


def query_shape(sql: str) -> QueryShape:
    tree = parse(sql)
    if tree is None:
        return QueryShape(parsed=False)

    selects = list(tree.find_all(exp.Select))
    aggregates = {type(a).__name__.lower() for a in tree.find_all(exp.AggFunc)}

    top_select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    select_width = len(top_select.expressions) if top_select is not None else 0

    return QueryShape(
        parsed=True,
        select_width=select_width,
        n_joins=len(list(tree.find_all(exp.Join))),
        n_aggregates=len(list(tree.find_all(exp.AggFunc))),
        aggregates=frozenset(aggregates),
        has_where=tree.find(exp.Where) is not None,
        has_group_by=tree.find(exp.Group) is not None,
        has_having=tree.find(exp.Having) is not None,
        has_order_by=tree.find(exp.Order) is not None,
        has_limit=tree.find(exp.Limit) is not None,
        # More than one SELECT node means a subquery somewhere.
        has_nested=len(selects) > 1,
        # `find` returns a node or None. `find_all` returns a GENERATOR, which is always
        # truthy -- using it here would mark every query as a set operation.
        has_set_op=any(
            tree.find(t) is not None for t in (exp.Union, exp.Intersect, exp.Except)
        ),
        has_distinct=tree.find(exp.Distinct) is not None,
    )


@dataclass
class ComponentMatch:
    """How closely a prediction's shape matches gold's."""

    both_parsed: bool
    clause_precision: float = 0.0
    clause_recall: float = 0.0
    clause_f1: float = 0.0
    join_count_match: bool = False
    select_width_match: bool = False
    aggregate_match: bool = False
    exact_shape_match: bool = False
    missing_clauses: frozenset[str] = frozenset()
    spurious_clauses: frozenset[str] = frozenset()

    def as_dict(self) -> dict:
        return {
            "both_parsed": self.both_parsed,
            "clause_precision": self.clause_precision,
            "clause_recall": self.clause_recall,
            "clause_f1": self.clause_f1,
            "join_count_match": self.join_count_match,
            "select_width_match": self.select_width_match,
            "aggregate_match": self.aggregate_match,
            "exact_shape_match": self.exact_shape_match,
            "missing_clauses": sorted(self.missing_clauses),
            "spurious_clauses": sorted(self.spurious_clauses),
        }


def compare_shapes(gold: QueryShape, pred: QueryShape) -> ComponentMatch:
    """Partial credit for getting the query's structure right.

    An unparseable prediction scores zero on everything rather than being skipped.
    """
    if not (gold.parsed and pred.parsed):
        return ComponentMatch(both_parsed=False)

    gold_clauses = gold.clause_set()
    pred_clauses = pred.clause_set()
    overlap = len(gold_clauses & pred_clauses)

    precision = overlap / len(pred_clauses) if pred_clauses else 1.0
    recall = overlap / len(gold_clauses) if gold_clauses else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    join_match = gold.n_joins == pred.n_joins
    width_match = gold.select_width == pred.select_width
    agg_match = gold.aggregates == pred.aggregates

    return ComponentMatch(
        both_parsed=True,
        clause_precision=precision,
        clause_recall=recall,
        clause_f1=f1,
        join_count_match=join_match,
        select_width_match=width_match,
        aggregate_match=agg_match,
        exact_shape_match=(
            gold_clauses == pred_clauses and join_match and width_match and agg_match
        ),
        missing_clauses=gold_clauses - pred_clauses,
        spurious_clauses=pred_clauses - gold_clauses,
    )


# --- aggregation -------------------------------------------------------------


def summarize_grounding(results: Iterable[GroundingResult]) -> dict:
    results = list(results)
    n = len(results)
    if n == 0:
        return {}
    parsed = [r for r in results if r.parsed]
    return {
        "n": n,
        "parse_rate": len(parsed) / n,
        "grounded_rate": sum(r.grounded for r in results) / n,
        "hallucinated_column_rate": sum(bool(r.hallucinated_columns) for r in results) / n,
        "hallucinated_table_rate": sum(bool(r.hallucinated_tables) for r in results) / n,
    }


def summarize_components(matches: Iterable[ComponentMatch]) -> dict:
    matches = list(matches)
    n = len(matches)
    if n == 0:
        return {}
    return {
        "n": n,
        "clause_f1": sum(m.clause_f1 for m in matches) / n,
        "clause_precision": sum(m.clause_precision for m in matches) / n,
        "clause_recall": sum(m.clause_recall for m in matches) / n,
        "join_count_match_rate": sum(m.join_count_match for m in matches) / n,
        "select_width_match_rate": sum(m.select_width_match for m in matches) / n,
        "aggregate_match_rate": sum(m.aggregate_match for m in matches) / n,
        "exact_shape_match_rate": sum(m.exact_shape_match for m in matches) / n,
    }
