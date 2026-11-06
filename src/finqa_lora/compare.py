"""Statistics for the two questions the project asks.

1. Did fine-tuning help?  -> paired bootstrap on per-example ROUGE-1 (tuned minus base).
2. LoRA vs QLoRA: faster, and is quality comparable? -> relative training-time difference,
   plus a paired bootstrap on ROUGE-1 between the two tuned models.

Speed is the Trainer's `train_runtime` (the training loop only), NOT whole-run wall clock,
which also includes model loading and evaluation generation.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 42
) -> dict:
    """Bootstrap the mean of `b - a` over resampled examples (paired)."""
    if len(a) != len(b) or not a:
        raise ValueError("need equal-length, non-empty score lists")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    low, high = means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]
    return {
        "delta": sum(diffs) / n,
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": low > 0 or high < 0,
        "n_resamples": n_resamples,
    }


def relative_speedup(baseline_seconds: float, candidate_seconds: float) -> float:
    """Fraction of the baseline's training time saved by the candidate.

    Positive: candidate is faster (0.124 = 12.4% less time). Negative: candidate is slower
    -- QLoRA's dequantization overhead often makes it so.
    """
    if baseline_seconds <= 0:
        raise ValueError("baseline_seconds must be positive")
    return (baseline_seconds - candidate_seconds) / baseline_seconds
