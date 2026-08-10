"""The metric. Pure functions -- no model, no database, no I/O.

This file defines what "correct" means, so it is the file an interviewer would poke at.
Every rule below is a decision that could be made differently, and each is stated rather
than buried.

**Execution accuracy.** A prediction is correct when executing it returns the same result
as executing the gold query. Not string equality: `SELECT a, b FROM t` and
`SELECT a,b FROM t` are the same query, and two genuinely different queries can both
answer the question correctly.

**Row order matters only when the gold query says it does.** SQL result order is undefined
without `ORDER BY`, so demanding it would fail correct predictions for a reason that has
nothing to do with the model. When the gold query *does* have `ORDER BY`, order is part of
the answer and is compared strictly.

**Column order is significant.** `SELECT name, age` and `SELECT age, name` are treated as
different, because the question usually implies which the user asked for and permuting
columns would let a shotgun prediction match.

**Floats are rounded before comparison.** SQLite's `AVG`/`SUM` produce trailing digits that
vary with the summation order the query planner picks -- a difference neither model
controls and neither should be scored on.

This is a set-comparison reimplementation of execution accuracy, not Yale's official
test-suite EX (which runs against multiple perturbed databases to catch predictions that
coincidentally match on one instance). Ours is therefore slightly more permissive in
absolute terms. It is applied identically to every system compared, so the *delta* -- the
number this project reports -- is unaffected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_FLOAT_DECIMALS = 4

# `ORDER BY` inside a string literal or an identifier must not count. Strip literals
# first, then look for the keyword on a word boundary.
_STRING_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_TRAILING_SEMICOLON = re.compile(r";\s*$")


def has_order_by(query: str) -> bool:
    """True when the query's own result order is meaningful."""
    return bool(_ORDER_BY.search(_STRING_LITERAL.sub("''", query or "")))


def canonicalize_value(value: object, float_decimals: int = DEFAULT_FLOAT_DECIMALS) -> object:
    """Reduce a single cell to a form that compares stably.

    Booleans are normalized to integers because SQLite stores them as 0/1 but different
    drivers hand them back differently; `True == 1` in Python but `("a", True)` and
    `("a", 1)` hash the same only after this conversion.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        # -0.0 and 0.0 are equal but format differently; `+ 0.0` normalizes the sign.
        return round(value, float_decimals) + 0.0
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        return value
    return str(value).strip()


def canonicalize_rows(
    rows: list[tuple] | None,
    float_decimals: int = DEFAULT_FLOAT_DECIMALS,
) -> list[tuple] | None:
    if rows is None:
        return None
    return [tuple(canonicalize_value(cell, float_decimals) for cell in row) for row in rows]


def results_match(
    gold_rows: list[tuple] | None,
    pred_rows: list[tuple] | None,
    order_matters: bool,
    float_decimals: int = DEFAULT_FLOAT_DECIMALS,
) -> bool:
    """Compare two result sets under the rules documented at the top of this module."""
    if gold_rows is None or pred_rows is None:
        return False

    gold = canonicalize_rows(gold_rows, float_decimals)
    pred = canonicalize_rows(pred_rows, float_decimals)

    if len(gold) != len(pred):
        return False
    if order_matters:
        return gold == pred

    # Order-insensitive but multiset-exact: duplicate rows are part of the answer, so
    # comparing sets would wrongly accept a prediction that lost or invented duplicates.
    # Rows may contain unhashable cells, so sort by repr rather than building a Counter.
    return sorted(gold, key=repr) == sorted(pred, key=repr)


# --- exact match (secondary metric) ------------------------------------------


def normalize_sql(query: str) -> str:
    """Normalize a query for string comparison: whitespace, case, and punctuation.

    Deliberately shallow. It collapses formatting differences, not semantic ones -- it
    will not recognize that two differently-aliased joins are the same query. That is
    precisely why exact match is the *secondary* metric here and execution accuracy is
    the headline: exact match systematically under-credits correct predictions.
    """
    if not query:
        return ""
    text = _TRAILING_SEMICOLON.sub("", query.strip())
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r"\s*([(),])\s*", r"\1", text)
    return text.lower().strip()


def exact_match(gold: str, pred: str) -> bool:
    return normalize_sql(gold) == normalize_sql(pred)


# --- prediction cleanup ------------------------------------------------------

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(raw: str) -> str:
    """Pull the SQL out of a raw model completion.

    The base model, in particular, ignores "output only the SQL" and wraps its answer in
    markdown fences or prefixes it with prose. Stripping that is not scoring generosity --
    it is refusing to measure formatting compliance when the question is whether the model
    knows the query. The same cleanup runs on every system, so no system is advantaged.
    """
    if not raw:
        return ""
    text = raw.strip()

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        # An unterminated fence: the model opened ```sql and hit the token limit.
        stripped = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = stripped.strip()

    # Drop a leading label like "SQL:" or "Answer:".
    text = re.sub(r"^(sql|answer|query)\s*:\s*", "", text, flags=re.IGNORECASE)

    # Keep only the first statement. Models sometimes continue with a second query or
    # with commentary after the answer; the first statement is the answer.
    if ";" in text:
        text = text.split(";", 1)[0]

    return _WHITESPACE.sub(" ", text).strip()


# --- hardness bucketing (secondary breakdown) --------------------------------


def sql_hardness(query: str) -> str:
    """Bucket a gold query by structural complexity.

    NOTE: this is *our own* component-count heuristic, not Yale's official Spider hardness
    definition, and it is labelled that way everywhere it is reported. It exists to answer
    "where did fine-tuning actually help" -- if the gain is confined to trivial queries,
    that is worth knowing and worth showing.
    """
    text = (query or "").lower()
    stripped = _STRING_LITERAL.sub("''", text)

    n_joins = len(re.findall(r"\bjoin\b", stripped))
    has_nesting = bool(re.search(r"\bselect\b.*\bselect\b", stripped, re.DOTALL))
    has_set_op = bool(re.search(r"\b(union|intersect|except)\b", stripped))
    n_aggregates = len(re.findall(r"\b(count|sum|avg|min|max)\s*\(", stripped))
    has_grouping = bool(re.search(r"\bgroup\s+by\b", stripped))

    if has_nesting or has_set_op or n_joins >= 2:
        return "extra"
    if n_joins == 1 or has_grouping:
        return "hard"
    if n_aggregates or has_order_by(query) or "where" in stripped:
        return "medium"
    return "easy"


HARDNESS_ORDER = ("easy", "medium", "hard", "extra")


# --- aggregation -------------------------------------------------------------


@dataclass
class ExampleScore:
    """The scored outcome for one test question under one system."""

    index: int
    db_id: str
    question: str
    gold: str
    prediction: str
    execution_correct: bool
    exact_match: bool
    invalid_sql: bool
    error: str | None
    hardness: str
    elapsed_seconds: float = 0.0


@dataclass
class SystemScores:
    """Aggregated results for one system over the whole test set."""

    name: str
    scores: list[ExampleScore] = field(default_factory=list)

    def summary(self) -> dict:
        n = len(self.scores)
        if n == 0:
            raise ValueError(f"no scored examples for system {self.name!r}")
        return {
            "system": self.name,
            "n": n,
            "execution_accuracy": sum(s.execution_correct for s in self.scores) / n,
            "exact_match": sum(s.exact_match for s in self.scores) / n,
            "invalid_sql_rate": sum(s.invalid_sql for s in self.scores) / n,
        }

    def by_hardness(self) -> list[dict]:
        rows: list[dict] = []
        for bucket in HARDNESS_ORDER:
            subset = [s for s in self.scores if s.hardness == bucket]
            if not subset:
                continue
            rows.append(
                {
                    "system": self.name,
                    "hardness": bucket,
                    "n": len(subset),
                    "execution_accuracy": sum(s.execution_correct for s in subset) / len(subset),
                    "exact_match": sum(s.exact_match for s in subset) / len(subset),
                }
            )
        return rows
