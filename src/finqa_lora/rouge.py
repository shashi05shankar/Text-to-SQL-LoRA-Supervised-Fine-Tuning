"""ROUGE-1 / ROUGE-2 / ROUGE-L (F-measure), dependency-free.

Tokenization: lowercase, runs of `[a-z0-9]`. No stemming. The reference `rouge-score`
package uses the same tokenizer and optionally applies a Porter stemmer, so numbers
reported from here are comparable with `rouge-score` run with `use_stemmer=False`.
`tests/finqa/test_rouge.py` cross-checks against that package when it is installed.
"""

from __future__ import annotations

import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if overlap == 0 or n_pred == 0 or n_ref == 0:
        return 0.0
    precision, recall = overlap / n_pred, overlap / n_ref
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(prediction: str, reference: str, n: int = 1) -> float:
    pred, ref = _ngrams(tokenize(prediction), n), _ngrams(tokenize(reference), n)
    overlap = sum((pred & ref).values())
    return _f1(overlap, sum(pred.values()), sum(ref.values()))


def _lcs(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            current.append(
                previous[j - 1] + 1 if token == other else max(previous[j], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = tokenize(prediction), tokenize(reference)
    return _f1(_lcs(pred, ref), len(pred), len(ref))


def score(prediction: str, reference: str) -> dict[str, float]:
    return {
        "rouge1": rouge_n(prediction, reference, 1),
        "rouge2": rouge_n(prediction, reference, 2),
        "rougeL": rouge_l(prediction, reference),
    }


def per_example(predictions: list[str], references: list[str]) -> list[dict[str, float]]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references differ in length")
    return [score(p, r) for p, r in zip(predictions, references, strict=True)]


def mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean per-example F-measure for each ROUGE variant."""
    if not rows:
        raise ValueError("no examples to score")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
