"""The internal train/val split must be database-disjoint, or val loss is meaningless."""

from __future__ import annotations

import pytest

from text2sql_lora.data.splits import split_by_db

pytestmark = pytest.mark.unit


def make_db_ids(n_dbs: int, rows_per_db: int) -> list[str]:
    return [f"db_{i}" for i in range(n_dbs) for _ in range(rows_per_db)]


def test_no_database_appears_on_both_sides():
    db_ids = make_db_ids(n_dbs=20, rows_per_db=5)
    train, val = split_by_db(db_ids, val_fraction=0.2, seed=42)

    train_dbs = {db_ids[i] for i in train}
    val_dbs = {db_ids[i] for i in val}
    assert train_dbs & val_dbs == set()


def test_split_is_a_partition_losing_nothing():
    db_ids = make_db_ids(n_dbs=20, rows_per_db=5)
    train, val = split_by_db(db_ids, val_fraction=0.2, seed=42)

    assert sorted(train + val) == list(range(len(db_ids)))
    assert set(train) & set(val) == set()


def test_val_lands_near_the_requested_row_fraction():
    """Databases are indivisible, so exactness is impossible -- but it should be close."""
    db_ids = make_db_ids(n_dbs=100, rows_per_db=10)
    _, val = split_by_db(db_ids, val_fraction=0.1, seed=42)

    assert 0.05 <= len(val) / len(db_ids) <= 0.15


def test_split_is_reproducible_across_calls():
    db_ids = make_db_ids(n_dbs=20, rows_per_db=5)
    assert split_by_db(db_ids, 0.2, seed=42) == split_by_db(db_ids, 0.2, seed=42)


def test_different_seeds_give_different_splits():
    db_ids = make_db_ids(n_dbs=30, rows_per_db=5)
    assert split_by_db(db_ids, 0.2, seed=1) != split_by_db(db_ids, 0.2, seed=2)


def test_train_is_never_left_empty():
    """With few databases and a large val fraction, the guard must still leave training
    data behind."""
    db_ids = make_db_ids(n_dbs=2, rows_per_db=3)
    train, val = split_by_db(db_ids, val_fraction=0.9, seed=42)

    assert len(train) > 0
    assert len(val) > 0


def test_uneven_database_sizes_stay_intact():
    """A database with many rows must not be sliced in half to hit the target."""
    db_ids = ["big"] * 50 + ["small_a"] * 2 + ["small_b"] * 2
    train, val = split_by_db(db_ids, val_fraction=0.1, seed=42)

    for db in ("big", "small_a", "small_b"):
        in_train = any(db_ids[i] == db for i in train)
        in_val = any(db_ids[i] == db for i in val)
        assert not (in_train and in_val), f"{db} was split across train and val"


def test_a_single_database_cannot_be_split():
    with pytest.raises(ValueError):
        split_by_db(["only_db"] * 10, val_fraction=0.2, seed=42)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_fractions_are_rejected(fraction):
    with pytest.raises(ValueError):
        split_by_db(make_db_ids(5, 3), val_fraction=fraction, seed=42)
