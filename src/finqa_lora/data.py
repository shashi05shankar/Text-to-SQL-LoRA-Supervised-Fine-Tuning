"""Financial QA data: `virattt/financial-qa-10K` (question / answer / context from 10-K filings).

7,000 rows. Each row pairs a question with the filing passage that answers it and a
reference answer, so the task is context-grounded question answering.

Splitting is grouped by `context`: some passages appear in more than one row, and a
row-level shuffle would put the same passage on both sides of the split and let the model
score by recall rather than by reading. Whole contexts go to one side.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "virattt/financial-qa-10K"
FILENAME = "data/train-00000-of-00001.parquet"


@dataclass(frozen=True)
class QAExample:
    question: str
    context: str
    answer: str
    ticker: str = ""


def download(data_dir: Path) -> Path:
    """Fetch the parquet into `data_dir` (public, ungated)."""
    import shutil

    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "financial_qa_10K.parquet"
    if not target.exists():
        shutil.copyfile(hf_hub_download(REPO_ID, FILENAME, repo_type="dataset"), target)
    return target


def load_examples(parquet: Path) -> list[QAExample]:
    import pandas as pd

    frame = pd.read_parquet(parquet)
    missing = {"question", "answer", "context"} - set(frame.columns)
    if missing:
        raise ValueError(f"{parquet} lacks columns {sorted(missing)}")
    frame = frame.dropna(subset=["question", "answer", "context"])
    tickers = frame["ticker"] if "ticker" in frame.columns else [""] * len(frame)
    examples = [
        QAExample(str(q).strip(), str(c).strip(), str(a).strip(), str(t))
        for q, c, a, t in zip(
            frame["question"], frame["context"], frame["answer"], tickers, strict=True
        )
    ]
    return [e for e in examples if e.question and e.answer]


def split_by_context(
    contexts: list[str], val_fraction: float, test_fraction: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Return `(train, val, test)` row indices with no context shared across splits."""
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test fractions must be positive and sum to less than 1")

    rows_by_context: dict[str, list[int]] = {}
    for index, context in enumerate(contexts):
        rows_by_context.setdefault(context, []).append(index)

    keys = sorted(rows_by_context)  # sort first so the shuffle is reproducible
    random.Random(seed).shuffle(keys)

    n = len(contexts)
    test_target, val_target = n * test_fraction, n * val_fraction
    test: list[int] = []
    val: list[int] = []
    train: list[int] = []
    for key in keys:
        rows = rows_by_context[key]
        if len(test) < test_target:
            test.extend(rows)
        elif len(val) < val_target:
            val.extend(rows)
        else:
            train.extend(rows)
    return sorted(train), sorted(val), sorted(test)
