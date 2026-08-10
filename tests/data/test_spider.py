"""Loading the real Spider corpus.

These are integration tests: they assert facts about the actual download, so they catch a
corrupted or half-finished fetch before it turns into a mysteriously bad metric. They skip
cleanly when the data is absent.
"""

from __future__ import annotations

import pytest

from text2sql_lora.config import DataConfig
from text2sql_lora.data.spider import (
    load_schemas,
    load_split,
    missing_databases,
    train_val_split,
)

from ..conftest import requires_spider

pytestmark = [pytest.mark.integration, requires_spider()]


@pytest.fixture(scope="module")
def schemas():
    return load_schemas()


@pytest.fixture(scope="module")
def dev(schemas):
    return load_split("validation", schemas=schemas)


@pytest.fixture(scope="module")
def train(schemas):
    return load_split("train", schemas=schemas)


def test_split_sizes_match_the_published_dataset(train, dev):
    assert len(train) == 7000
    assert len(dev) == 1034


def test_train_and_dev_databases_are_disjoint(train, dev):
    """This is the entire reason Spider was chosen over WikiSQL: the held-out number
    measures generalization to unseen schemas."""
    assert set(train.db_ids()) & set(dev.db_ids()) == set()


def test_every_referenced_database_has_a_sqlite_file(train, dev):
    """A missing .sqlite would silently score as wrong for every system rather than
    raising, so it is checked explicitly."""
    assert missing_databases(dev) == []
    assert missing_databases(train) == []


def test_examples_carry_rendered_schema_text(dev):
    example = dev.examples[0]
    assert example.question
    assert example.query
    assert "CREATE TABLE" in example.schema_text


def test_internal_val_split_is_database_disjoint(train):
    inner_train, inner_val = train_val_split(train, DataConfig())

    assert len(inner_train) + len(inner_val) == len(train)
    assert set(inner_train.db_ids()) & set(inner_val.db_ids()) == set()
    assert len(inner_val) > 0


def test_dev_uses_twenty_databases(dev):
    assert len(set(dev.db_ids())) == 20
