"""Run SQL against a Spider database, safely and with a deadline.

Three things this has to get right, because each one can quietly corrupt the headline
number:

**Invalid SQL is data, not an exception.** A model that emits garbage should score zero on
that example and be counted in an invalid-SQL rate -- not blow up the eval loop. So every
outcome comes back as a `ExecutionResult` with either rows or an error, never a raised
exception.

**A timeout is a wrong answer, not a skipped one.** If a hung query were skipped, a model
could raise its score by emitting queries that never finish. SQLite has no query timeout,
so one is imposed via a progress handler that aborts from inside the VM.

**Read-only.** The databases are opened through a `file:...?mode=ro` URI. Evaluation must
not be able to mutate the corpus it is scoring against -- a `DROP TABLE` hallucinated on
example 40 would silently change the answers for every example after it.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

# How often SQLite checks in with the progress handler. Small enough that the deadline is
# honoured promptly, large enough not to dominate runtime.
_PROGRESS_INSTRUCTION_INTERVAL = 1000

# Rows beyond this are not read. Spider answers are small; a query returning millions of
# rows is a runaway cross join, and materializing it helps nobody.
MAX_ROWS = 10_000


@dataclass(frozen=True)
class ExecutionResult:
    """The outcome of running one query. Exactly one of `rows` / `error` is meaningful."""

    rows: list[tuple] | None
    error: str | None
    elapsed_seconds: float
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def execute_query(
    db_path: Path | str,
    query: str,
    timeout_seconds: float = 30.0,
) -> ExecutionResult:
    """Execute `query` read-only against `db_path` under a wall-clock deadline."""
    started = time.perf_counter()

    if not str(query).strip():
        return ExecutionResult(None, "empty query", time.perf_counter() - started)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, timeout=timeout_seconds
        )
        # Spider databases contain text in several encodings and a few with invalid
        # UTF-8. Latin-1 round-trips any byte sequence, so decoding never fails and the
        # comparison stays byte-faithful.
        connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")

        deadline = started + timeout_seconds
        timed_out = False

        def abort_if_past_deadline() -> int:
            nonlocal timed_out
            if time.perf_counter() > deadline:
                timed_out = True
                return 1  # non-zero aborts the running statement
            return 0

        connection.set_progress_handler(
            abort_if_past_deadline, _PROGRESS_INSTRUCTION_INTERVAL
        )

        cursor = connection.execute(query)
        rows = cursor.fetchmany(MAX_ROWS + 1)
        truncated = len(rows) > MAX_ROWS
        if truncated:
            rows = rows[:MAX_ROWS]

        elapsed = time.perf_counter() - started
        return ExecutionResult(
            rows=[tuple(row) for row in rows],
            error=None,
            elapsed_seconds=elapsed,
            truncated=truncated,
        )

    except sqlite3.Error as exc:
        elapsed = time.perf_counter() - started
        # An aborted statement surfaces as a generic OperationalError, so the progress
        # handler's flag is what distinguishes a timeout from a syntax error.
        hit_deadline = locals().get("timed_out", False) or elapsed >= timeout_seconds
        return ExecutionResult(
            rows=None,
            error="timeout" if hit_deadline else f"{type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed,
            timed_out=bool(hit_deadline),
        )
    except Exception as exc:
        return ExecutionResult(
            rows=None,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )
    finally:
        if connection is not None:
            connection.set_progress_handler(None, 0)
            connection.close()
