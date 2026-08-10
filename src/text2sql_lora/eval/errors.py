"""Classify *how* a prediction failed, not just whether it did.

"Execution accuracy 65%" says a third of answers are wrong. It does not say whether they
are wrong because the model invented a column, because it misunderstood the question, or
because it returned the right rows in the wrong order. Those call for completely
different fixes, and the aggregate hides which one you have.

Two families, because they mean different things:

**Invalid** -- the query did not run. This is a *syntax and schema-grounding* failure: the
model wrote something SQLite rejects. Hallucinated identifiers dominate here, and they are
the failure mode that better schema serialization or constrained decoding would address.

**Executed but wrong** -- the query ran and returned the wrong rows. This is a *semantic*
failure: the model understood SQL fine and misunderstood the question, or picked the wrong
join, filter, or aggregate. No amount of grammar constraint fixes these.

The distinction matters for the story a fine-tune tells. A model whose errors move from
"invalid" to "executed but wrong" has genuinely learned to write SQL and now has a harder,
more interesting problem left.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from text2sql_lora.eval.execute import ExecutionResult
from text2sql_lora.eval.metrics import canonicalize_rows


class FailureKind(StrEnum):
    """Mutually exclusive outcomes for a single prediction."""

    CORRECT = "correct"

    # --- did not execute ---
    INVALID_NO_SUCH_COLUMN = "invalid: no such column"
    INVALID_NO_SUCH_TABLE = "invalid: no such table"
    INVALID_AMBIGUOUS_COLUMN = "invalid: ambiguous column"
    INVALID_AGGREGATE_MISUSE = "invalid: aggregate misuse"
    INVALID_SYNTAX = "invalid: syntax error"
    INVALID_TIMEOUT = "invalid: timeout"
    INVALID_EMPTY = "invalid: empty output"
    INVALID_OTHER = "invalid: other"

    # --- executed, wrong answer ---
    WRONG_EMPTY = "wrong: returned no rows"
    WRONG_SHOULD_BE_EMPTY = "wrong: returned rows, gold returned none"
    WRONG_COLUMN_COUNT = "wrong: wrong number of columns"
    WRONG_ROW_COUNT = "wrong: wrong number of rows"
    WRONG_ORDER = "wrong: right rows, wrong order"
    WRONG_VALUES = "wrong: same shape, different values"

    @property
    def is_invalid(self) -> bool:
        return self.value.startswith("invalid")

    @property
    def is_semantic(self) -> bool:
        return self.value.startswith("wrong")


# SQLite's messages are stable enough to classify by substring. Ordered: the first match
# wins, so more specific patterns come first.
_ERROR_PATTERNS: tuple[tuple[re.Pattern, FailureKind], ...] = (
    (re.compile(r"no such column", re.I), FailureKind.INVALID_NO_SUCH_COLUMN),
    (re.compile(r"no such table", re.I), FailureKind.INVALID_NO_SUCH_TABLE),
    (re.compile(r"ambiguous column name", re.I), FailureKind.INVALID_AMBIGUOUS_COLUMN),
    (
        re.compile(r"misuse of aggregate|misuse of window", re.I),
        FailureKind.INVALID_AGGREGATE_MISUSE,
    ),
    (re.compile(r"no such function", re.I), FailureKind.INVALID_SYNTAX),
    (
        re.compile(r"syntax error|incomplete input|unrecognized token", re.I),
        FailureKind.INVALID_SYNTAX,
    ),
    (re.compile(r"^timeout$", re.I), FailureKind.INVALID_TIMEOUT),
    (re.compile(r"^empty query$", re.I), FailureKind.INVALID_EMPTY),
)


def classify_error_message(message: str) -> FailureKind:
    """Map a SQLite error string onto a failure kind."""
    if not message:
        return FailureKind.INVALID_OTHER
    for pattern, kind in _ERROR_PATTERNS:
        if pattern.search(message):
            return kind
    return FailureKind.INVALID_OTHER


def classify(
    gold_result: ExecutionResult,
    pred_result: ExecutionResult,
    execution_correct: bool,
    float_decimals: int = 4,
) -> FailureKind:
    """Assign one failure kind to a single prediction.

    `execution_correct` is passed in rather than recomputed so this always agrees with the
    headline metric -- a taxonomy that disagreed with the number it explains would be
    worse than no taxonomy.
    """
    if execution_correct:
        return FailureKind.CORRECT

    if not pred_result.ok:
        if pred_result.timed_out:
            return FailureKind.INVALID_TIMEOUT
        return classify_error_message(pred_result.error or "")

    # Executed, but wrong. Work out how.
    gold_rows = canonicalize_rows(gold_result.rows, float_decimals) or []
    pred_rows = canonicalize_rows(pred_result.rows, float_decimals) or []

    if not pred_rows and gold_rows:
        return FailureKind.WRONG_EMPTY
    if pred_rows and not gold_rows:
        return FailureKind.WRONG_SHOULD_BE_EMPTY

    gold_width = len(gold_rows[0]) if gold_rows else 0
    pred_width = len(pred_rows[0]) if pred_rows else 0
    if gold_width != pred_width:
        return FailureKind.WRONG_COLUMN_COUNT

    if len(gold_rows) != len(pred_rows):
        return FailureKind.WRONG_ROW_COUNT

    # Same shape and same row count. If the multisets match, only ordering differs --
    # which can only be a failure when the gold query specified an order.
    if sorted(gold_rows, key=repr) == sorted(pred_rows, key=repr):
        return FailureKind.WRONG_ORDER

    return FailureKind.WRONG_VALUES


@dataclass
class FailureProfile:
    """The distribution of failure kinds for one system."""

    system: str
    counts: Counter
    n: int

    def rate(self, kind: FailureKind) -> float:
        return self.counts.get(kind, 0) / self.n if self.n else 0.0

    @property
    def n_invalid(self) -> int:
        return sum(c for k, c in self.counts.items() if k.is_invalid)

    @property
    def n_semantic(self) -> int:
        return sum(c for k, c in self.counts.items() if k.is_semantic)

    @property
    def n_correct(self) -> int:
        return self.counts.get(FailureKind.CORRECT, 0)

    def semantic_share_of_errors(self) -> float:
        """Of the questions this system got wrong, what fraction at least *ran*?

        The number to watch across a fine-tune. Rising means errors are migrating from
        "cannot write valid SQL" to "wrote valid SQL, misread the question" -- real
        progress, even when overall accuracy moves less than you hoped.
        """
        errors = self.n_invalid + self.n_semantic
        return self.n_semantic / errors if errors else 0.0

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "n": self.n,
            "n_correct": self.n_correct,
            "n_invalid": self.n_invalid,
            "n_semantic_error": self.n_semantic,
            "invalid_rate": self.n_invalid / self.n if self.n else 0.0,
            "semantic_error_rate": self.n_semantic / self.n if self.n else 0.0,
            "semantic_share_of_errors": self.semantic_share_of_errors(),
            "by_kind": {
                kind.value: {"n": count, "rate": count / self.n if self.n else 0.0}
                for kind, count in sorted(
                    self.counts.items(), key=lambda kv: -kv[1]
                )
            },
        }


def build_profile(system: str, kinds: list[FailureKind]) -> FailureProfile:
    return FailureProfile(system=system, counts=Counter(kinds), n=len(kinds))
