"""Load Spider question/SQL pairs and pair them with rendered schemas.

Spider arrives in two pieces from two places, and both are required:

* `xlangai/spider` on the Hub ships parquet with the question/SQL pairs and a `db_id` --
  and nothing else. There are no databases in it.
* The SQLite databases and `tables.json` come from the separate `spider_data.zip`.

Execution accuracy is impossible with only the first piece, which is why
`scripts/download_data.py` fetches both and why this module refuses to load if either
is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from text2sql_lora.config import DataConfig, Paths
from text2sql_lora.data.prompts import Example
from text2sql_lora.data.schema import DatabaseSchema, parse_tables_json, render_schema

SPLITS = ("train", "validation")


@dataclass(frozen=True)
class SpiderData:
    """A loaded split, with every example's schema already rendered."""

    split: str
    examples: list[Example]
    schemas: dict[str, DatabaseSchema]

    def __len__(self) -> int:
        return len(self.examples)

    def db_ids(self) -> list[str]:
        return [example.db_id for example in self.examples]

    def subset(self, indices: list[int]) -> SpiderData:
        return SpiderData(
            split=self.split,
            examples=[self.examples[i] for i in indices],
            schemas=self.schemas,
        )


def load_schemas(paths: Paths | None = None) -> dict[str, DatabaseSchema]:
    paths = paths or Paths()
    tables_path = paths.tables_json()
    if not tables_path.exists():
        raise FileNotFoundError(
            f"{tables_path} not found. Run `python scripts/download_data.py` first -- the "
            "Hub parquet does not include schemas or databases."
        )
    return parse_tables_json(tables_path)


def load_split(
    split: str,
    paths: Paths | None = None,
    schemas: dict[str, DatabaseSchema] | None = None,
) -> SpiderData:
    """Load one Spider split into `Example` objects with schema text attached."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    paths = paths or Paths()
    parquet_path = paths.parquet(split)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run `python scripts/download_data.py` first."
        )

    schemas = schemas if schemas is not None else load_schemas(paths)
    frame = pd.read_parquet(parquet_path, columns=["db_id", "question", "query"])

    # Cache rendered DDL per database rather than per row. Spider has ~1,000 questions
    # over ~20 dev databases, so this is a ~50x saving on string building.
    rendered: dict[str, str] = {}
    examples: list[Example] = []
    for row in frame.itertuples(index=False):
        db_id = str(row.db_id)
        if db_id not in rendered:
            if db_id not in schemas:
                raise KeyError(
                    f"database {db_id!r} appears in {split} but not in tables.json; the "
                    "parquet and spider_data.zip downloads are out of sync"
                )
            rendered[db_id] = render_schema(schemas[db_id])
        examples.append(
            Example(
                db_id=db_id,
                question=str(row.question),
                query=str(row.query),
                schema_text=rendered[db_id],
            )
        )

    return SpiderData(split=split, examples=examples, schemas=schemas)


def missing_databases(data: SpiderData, paths: Paths | None = None) -> list[str]:
    """Database ids referenced by the split whose `.sqlite` file is absent.

    Anything returned here cannot be executed, so it would silently become a wrong answer
    for every system. Callers should treat a non-empty result as a failed download rather
    than as a modelling problem.
    """
    paths = paths or Paths()
    seen: set[str] = set()
    missing: list[str] = []
    for db_id in data.db_ids():
        if db_id in seen:
            continue
        seen.add(db_id)
        if not paths.db_path(db_id).exists():
            missing.append(db_id)
    return sorted(missing)


def train_val_split(
    data: SpiderData,
    config: DataConfig | None = None,
) -> tuple[SpiderData, SpiderData]:
    """Carve a database-disjoint monitoring split out of a training split."""
    from text2sql_lora.data.splits import split_by_db

    config = config or DataConfig()
    train_indices, val_indices = split_by_db(
        data.db_ids(), val_fraction=config.val_fraction, seed=config.seed
    )
    return data.subset(train_indices), data.subset(val_indices)


def data_is_available(paths: Paths | None = None) -> bool:
    """Cheap check used by tests to skip integration cases when Spider is not downloaded."""
    paths = paths or Paths()
    return (
        paths.tables_json().exists()
        and paths.parquet("validation").exists()
        and paths.database_dir().is_dir()
    )


def resolve_db_path(db_id: str, paths: Paths | None = None) -> Path:
    paths = paths or Paths()
    path = paths.db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"no SQLite file for database {db_id!r} at {path}")
    return path
