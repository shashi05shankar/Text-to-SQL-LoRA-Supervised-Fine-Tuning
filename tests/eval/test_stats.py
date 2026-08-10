"""Significance machinery, against cases whose answers are known independently."""

from __future__ import annotations

import math

import pytest

from text2sql_lora.eval.stats import (
    bootstrap_delta,
    compare_systems,
    mcnemar,
    standard_error,
    wilson_interval,
)

pytestmark = pytest.mark.unit


# --- Wilson interval ---------------------------------------------------------


def test_wilson_brackets_the_point_estimate():
    low, high = wilson_interval(424, 1034)
    assert low < 424 / 1034 < high


def test_wilson_width_is_about_two_standard_errors():
    """Sanity: near p=0.4 with n~1000 the 95% interval should span roughly +/-3 points."""
    low, high = wilson_interval(410, 1000)
    assert 0.055 < high - low < 0.065


def test_wilson_never_leaves_the_unit_interval():
    """The normal approximation famously returns negative lower bounds here."""
    for successes, n in [(0, 50), (50, 50), (1, 1000), (999, 1000)]:
        low, high = wilson_interval(successes, n)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_narrows_as_n_grows():
    narrow = wilson_interval(400, 1000)
    wide = wilson_interval(40, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


@pytest.mark.parametrize(("successes", "n"), [(-1, 10), (11, 10), (5, 0)])
def test_wilson_rejects_impossible_inputs(successes, n):
    with pytest.raises(ValueError):
        wilson_interval(successes, n)


def test_standard_error_matches_the_closed_form():
    assert standard_error(0.41, 1034) == pytest.approx(math.sqrt(0.41 * 0.59 / 1034))


# --- McNemar -----------------------------------------------------------------


def make_outcomes(gained: int, lost: int, both_right: int, both_wrong: int):
    """Build two score vectors with an exactly specified agreement table."""
    baseline = [False] * gained + [True] * lost + [True] * both_right + [False] * both_wrong
    candidate = [True] * gained + [False] * lost + [True] * both_right + [False] * both_wrong
    return baseline, candidate


def test_mcnemar_counts_the_agreement_table():
    baseline, candidate = make_outcomes(gained=308, lost=55, both_right=369, both_wrong=302)
    result = mcnemar(baseline, candidate)

    assert result.b_gained == 308
    assert result.b_lost == 55
    assert result.n_agree_correct == 369
    assert result.n_agree_wrong == 302
    assert result.n_discordant == 363


def test_a_large_lopsided_difference_is_significant():
    baseline, candidate = make_outcomes(gained=308, lost=55, both_right=369, both_wrong=302)
    result = mcnemar(baseline, candidate)

    # (|308-55|-1)^2 / 363 = 252^2/363
    assert result.chi2 == pytest.approx(252**2 / 363)
    assert result.p_value < 1e-30
    assert result.significant


def test_an_even_split_is_not_significant():
    """134 fixed and 132 broken is a coin flip, however large the counts."""
    baseline, candidate = make_outcomes(gained=134, lost=132, both_right=290, both_wrong=478)
    result = mcnemar(baseline, candidate)

    assert result.p_value > 0.5
    assert not result.significant


def test_agreements_do_not_affect_the_verdict():
    """Only disagreements carry information about which system is better."""
    few = mcnemar(*make_outcomes(gained=30, lost=5, both_right=10, both_wrong=10))
    many = mcnemar(*make_outcomes(gained=30, lost=5, both_right=5000, both_wrong=5000))

    assert few.p_value == pytest.approx(many.p_value)


def test_small_discordance_uses_the_exact_test():
    result = mcnemar(*make_outcomes(gained=8, lost=1, both_right=100, both_wrong=100))
    assert result.method == "exact binomial"
    # 9 flips, 8 or more heads, two-sided: 2 * (9+1)/512
    assert result.p_value == pytest.approx(2 * (math.comb(9, 0) + math.comb(9, 1)) / 512)


def test_large_discordance_uses_the_chi_square_approximation():
    result = mcnemar(*make_outcomes(gained=200, lost=100, both_right=10, both_wrong=10))
    assert result.method.startswith("chi-square")


def test_identical_systems_are_not_a_difference():
    both = [True, False, True, False]
    result = mcnemar(both, both)

    assert result.n_discordant == 0
    assert result.p_value == 1.0
    assert not result.significant


def test_direction_matters():
    """Swapping the arguments swaps gained and lost."""
    baseline, candidate = make_outcomes(gained=50, lost=10, both_right=10, both_wrong=10)
    forward = mcnemar(baseline, candidate)
    backward = mcnemar(candidate, baseline)

    assert forward.b_gained == backward.b_lost == 50
    assert forward.p_value == pytest.approx(backward.p_value)  # two-sided


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        mcnemar([True, False], [True])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        mcnemar([], [])


# --- bootstrap ---------------------------------------------------------------


def test_bootstrap_interval_contains_the_observed_delta():
    baseline, candidate = make_outcomes(gained=308, lost=55, both_right=369, both_wrong=302)
    result = bootstrap_delta(baseline, candidate, n_resamples=2000, seed=1)

    assert result.low < result.delta < result.high
    assert result.delta == pytest.approx((308 - 55) / 1034, abs=1e-9)


def test_bootstrap_excludes_zero_for_a_real_effect():
    baseline, candidate = make_outcomes(gained=308, lost=55, both_right=369, both_wrong=302)
    result = bootstrap_delta(baseline, candidate, n_resamples=2000, seed=1)

    assert result.low > 0
    assert result.excludes_zero


def test_bootstrap_includes_zero_for_a_coin_flip():
    baseline, candidate = make_outcomes(gained=134, lost=132, both_right=290, both_wrong=478)
    result = bootstrap_delta(baseline, candidate, n_resamples=2000, seed=1)

    assert result.low < 0 < result.high
    assert not result.excludes_zero


def test_bootstrap_is_reproducible():
    baseline, candidate = make_outcomes(gained=50, lost=20, both_right=100, both_wrong=100)
    first = bootstrap_delta(baseline, candidate, n_resamples=500, seed=7)
    second = bootstrap_delta(baseline, candidate, n_resamples=500, seed=7)

    assert (first.low, first.high) == (second.low, second.high)


def test_bootstrap_respects_pairing():
    """Two systems that always agree have exactly zero difference, in every resample --
    which only holds if question indices are drawn jointly."""
    outcomes = [True, False, True, True, False] * 20
    result = bootstrap_delta(outcomes, outcomes, n_resamples=500, seed=3)

    assert result.delta == 0.0
    assert result.low == result.high == 0.0


# --- assembled comparison ----------------------------------------------------


def test_compare_systems_assembles_everything():
    baseline, candidate = make_outcomes(gained=308, lost=55, both_right=369, both_wrong=302)
    comparison = compare_systems("base", baseline, "tuned", candidate, n_resamples=500)

    payload = comparison.as_dict()
    assert payload["baseline"] == "base"
    assert payload["candidate"] == "tuned"
    assert payload["delta"] == pytest.approx(comparison.candidate_accuracy
                                             - comparison.baseline_accuracy)
    assert payload["mcnemar"]["significant_at_05"] is True
    assert payload["bootstrap"]["excludes_zero"] is True
    assert len(payload["baseline_ci"]) == 2
