"""Is the difference real, or is it noise?

A single accuracy number on 1,034 examples is a point estimate with meaningful
uncertainty around it: the standard error of a proportion near 0.4 at n=1034 is about
1.5 points. Quoting "41.0% vs 41.2%" as though that gap means something -- when it is a
tenth of a standard error -- is exactly the kind of claim this project exists to avoid
making.

Three tools, each answering a different question:

* **Wilson interval** -- how precisely is one system's accuracy known? Wilson rather than
  the textbook normal approximation because the latter misbehaves near 0 and 1 and can
  produce intervals extending past them.
* **McNemar's test** -- is system B better than system A? The systems are evaluated on the
  *same* questions, so the samples are paired and an unpaired two-proportion test would
  throw away that structure and be needlessly conservative. McNemar looks only at the
  questions where the two disagree, which is where all the information about the
  difference lives.
* **Bootstrap** -- a confidence interval on the *difference*, resampling questions with
  replacement. Makes no distributional assumption and covers the paired structure
  naturally by resampling question indices rather than systems.

No SciPy: the exact binomial and the chi-square survival function for 1 dof are both a
few lines, and one fewer dependency in the serving image is worth more than the import.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

Z_95 = 1.959963984540054


# --- one system -------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Never returns bounds outside [0, 1], and stays sensible when the count is 0 or n --
    both places the normal approximation fails outright.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} out of range for n={n}")

    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def standard_error(p: float, n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt(p * (1 - p) / n)


# --- comparing two systems --------------------------------------------------


def _chi2_sf_1dof(chi2: float) -> float:
    """Upper-tail probability of the chi-square distribution with 1 degree of freedom.

    For 1 dof this is exactly erfc(sqrt(chi2/2)), so no special-function library is
    needed.
    """
    if chi2 <= 0:
        return 1.0
    return math.erfc(math.sqrt(chi2 / 2.0))


def _binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test against p=0.5.

    Used when the number of disagreements is small enough that the chi-square
    approximation is untrustworthy. Two-sided by doubling the smaller tail, capped at 1.
    """
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of a paired comparison of two systems on identical questions."""

    b_gained: int  # A wrong, B right
    b_lost: int  # A right, B wrong
    n_agree_correct: int
    n_agree_wrong: int
    chi2: float
    p_value: float
    method: str

    @property
    def n_discordant(self) -> int:
        return self.b_gained + self.b_lost

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict:
        return {
            "gained": self.b_gained,
            "lost": self.b_lost,
            "net": self.b_gained - self.b_lost,
            "agree_correct": self.n_agree_correct,
            "agree_wrong": self.n_agree_wrong,
            "discordant": self.n_discordant,
            "chi2": self.chi2,
            "p_value": self.p_value,
            "method": self.method,
            "significant_at_05": self.significant,
        }


def mcnemar(
    baseline_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
    exact_threshold: int = 25,
) -> McNemarResult:
    """Test whether `candidate` beats `baseline` on the same set of questions.

    Only the disagreements carry information: a question both systems get right, or both
    get wrong, says nothing about which is better. So the test asks whether the split
    between "candidate fixed it" and "candidate broke it" is further from even than
    chance would give.

    Below `exact_threshold` disagreements the exact binomial test is used instead of the
    chi-square approximation, which is unreliable on small counts.
    """
    if len(baseline_correct) != len(candidate_correct):
        raise ValueError("the two systems must be scored on the same number of questions")
    if not baseline_correct:
        raise ValueError("no questions to compare")

    gained = lost = agree_correct = agree_wrong = 0
    for base, cand in zip(baseline_correct, candidate_correct, strict=True):
        if base and cand:
            agree_correct += 1
        elif not base and not cand:
            agree_wrong += 1
        elif cand:
            gained += 1
        else:
            lost += 1

    discordant = gained + lost
    if discordant == 0:
        # The systems are identical on every question; there is nothing to test.
        return McNemarResult(gained, lost, agree_correct, agree_wrong, 0.0, 1.0, "no-discordance")

    if discordant < exact_threshold:
        p_value = _binomial_two_sided_p(gained, discordant)
        chi2 = (abs(gained - lost) - 1) ** 2 / discordant if discordant > 1 else 0.0
        method = "exact binomial"
    else:
        # Edwards' continuity correction; standard for McNemar.
        chi2 = (abs(gained - lost) - 1) ** 2 / discordant
        p_value = _chi2_sf_1dof(chi2)
        method = "chi-square (continuity-corrected)"

    return McNemarResult(gained, lost, agree_correct, agree_wrong, chi2, p_value, method)


@dataclass(frozen=True)
class BootstrapResult:
    delta: float
    low: float
    high: float
    n_resamples: int
    excludes_zero: bool

    def as_dict(self) -> dict:
        return {
            "delta": self.delta,
            "ci_low": self.low,
            "ci_high": self.high,
            "n_resamples": self.n_resamples,
            "excludes_zero": self.excludes_zero,
        }


def bootstrap_delta(
    baseline_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap CI for the accuracy difference (candidate - baseline).

    Resamples **question indices**, taking both systems' outcomes for the drawn question
    together. That is what preserves the pairing: if the two systems were resampled
    independently the interval would be far too wide, because it would include variation
    that cancels out when the same questions are scored by both.
    """
    if len(baseline_correct) != len(candidate_correct):
        raise ValueError("the two systems must be scored on the same number of questions")
    n = len(baseline_correct)
    if n == 0:
        raise ValueError("no questions to compare")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    # Per-question difference: +1 candidate-only, -1 baseline-only, 0 agreement.
    differences = [
        int(cand) - int(base)
        for base, cand in zip(baseline_correct, candidate_correct, strict=True)
    ]
    observed = sum(differences) / n

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        total = 0
        for _ in range(n):
            total += differences[rng.randrange(n)]
        deltas.append(total / n)

    deltas.sort()
    alpha = (1 - confidence) / 2
    low = deltas[max(0, int(alpha * n_resamples) - 1)]
    high = deltas[min(n_resamples - 1, int((1 - alpha) * n_resamples))]
    return BootstrapResult(observed, low, high, n_resamples, low > 0 or high < 0)


# --- assembling a full comparison -------------------------------------------


@dataclass
class SystemComparison:
    """Everything needed to state whether one system genuinely beats another."""

    baseline: str
    candidate: str
    baseline_accuracy: float
    candidate_accuracy: float
    baseline_ci: tuple[float, float]
    candidate_ci: tuple[float, float]
    mcnemar: McNemarResult
    bootstrap: BootstrapResult = field(default=None)  # type: ignore[assignment]

    def as_dict(self) -> dict:
        payload = {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "baseline_accuracy": self.baseline_accuracy,
            "candidate_accuracy": self.candidate_accuracy,
            "baseline_ci": list(self.baseline_ci),
            "candidate_ci": list(self.candidate_ci),
            "delta": self.candidate_accuracy - self.baseline_accuracy,
            "mcnemar": self.mcnemar.as_dict(),
        }
        if self.bootstrap is not None:
            payload["bootstrap"] = self.bootstrap.as_dict()
        return payload


def compare_systems(
    baseline_name: str,
    baseline_correct: Sequence[bool],
    candidate_name: str,
    candidate_correct: Sequence[bool],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> SystemComparison:
    n = len(baseline_correct)
    base_hits = sum(baseline_correct)
    cand_hits = sum(candidate_correct)

    return SystemComparison(
        baseline=baseline_name,
        candidate=candidate_name,
        baseline_accuracy=base_hits / n,
        candidate_accuracy=cand_hits / n,
        baseline_ci=wilson_interval(base_hits, n),
        candidate_ci=wilson_interval(cand_hits, n),
        mcnemar=mcnemar(baseline_correct, candidate_correct),
        bootstrap=bootstrap_delta(
            baseline_correct, candidate_correct, n_resamples=n_resamples, seed=seed
        ),
    )
