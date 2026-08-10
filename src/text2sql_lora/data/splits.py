"""Split Spider's training set by database, not by row.

Spider's whole premise is generalization to *unseen schemas*: its dev set uses databases
that appear nowhere in training. A monitoring split that shuffled rows would break that
premise locally -- the same database would sit on both sides, val loss would measure
schema memorization, and it would look far better than dev accuracy ever turns out to be.

So the internal train/val split is grouped by `db_id`. Whole databases go to one side or
the other, which makes the val split a miniature of the real dev condition.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def split_by_db(
    db_ids: Sequence[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return `(train_indices, val_indices)` with disjoint databases.

    Databases are shuffled and assigned to val until the target row count is reached, so
    the split lands near `val_fraction` by *rows* while staying clean by database. It will
    rarely be exact -- databases have different numbers of questions and are indivisible.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rows_per_db: dict[str, list[int]] = {}
    for index, db_id in enumerate(db_ids):
        rows_per_db.setdefault(db_id, []).append(index)

    if len(rows_per_db) < 2:
        raise ValueError("need at least 2 distinct databases to make a disjoint split")

    shuffled = sorted(rows_per_db)  # sort first so the shuffle is reproducible
    random.Random(seed).shuffle(shuffled)

    target = len(db_ids) * val_fraction
    val_indices: list[int] = []
    val_dbs: set[str] = set()

    for db_id in shuffled:
        if len(val_indices) >= target:
            break
        # Never take the last database -- train must not end up empty.
        if len(val_dbs) == len(rows_per_db) - 1:
            break
        val_dbs.add(db_id)
        val_indices.extend(rows_per_db[db_id])

    train_indices = [index for index, db_id in enumerate(db_ids) if db_id not in val_dbs]
    return sorted(train_indices), sorted(val_indices)
