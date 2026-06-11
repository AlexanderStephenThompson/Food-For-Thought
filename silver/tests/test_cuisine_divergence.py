"""Tests for silver.pipeline.cuisine_divergence.

All tests run on synthetic mini-indexes built in-memory from hand-made
Recipe objects -- no file I/O. Real-data validation lives in the CLI
(python -m silver.pipeline.cuisine_divergence) and is reported separately.
"""

import pytest

from silver.pipeline import cuisine_divergence
from silver.pipeline.cuisine_divergence import (
    VERDICT_MERGE,
    VERDICT_PRESERVE,
    VERDICT_REVIEW,
    classify_merge_verdict,
    compute_cuisine_distribution,
    evaluate_merge_candidate,
    jensen_shannon_divergence_bits,
    monte_carlo_null95,
)
from silver.pipeline.load_bronze_recipes import Recipe, TrainIndex, build_train_index
from silver.pipeline.merge_evidence import MergeEvidence

FAST_TRIALS = 200
JSD_TOLERANCE = 1e-9


def _build_synthetic_index() -> TrainIndex:
    """Mini train corpus: 8 recipes, 3 cuisines, overlapping ingredient strings."""
    recipes = [
        Recipe(id=1, cuisine="alpha", ingredients=("salt", "soy")),
        Recipe(id=2, cuisine="alpha", ingredients=("soy",)),
        Recipe(id=3, cuisine="alpha", ingredients=("soy", "pepper")),
        Recipe(id=4, cuisine="beta", ingredients=("salt",)),
        Recipe(id=5, cuisine="beta", ingredients=("salt", "pepper")),
        Recipe(id=6, cuisine="beta", ingredients=("soy",)),
        Recipe(id=7, cuisine="gamma", ingredients=("pepper",)),
        Recipe(id=8, cuisine="gamma", ingredients=("salt", "soy")),
    ]
    return build_train_index(recipes)


def test_jsd_zero_for_identical_distributions():
    distribution = (0.5, 0.3, 0.2)

    divergence = jensen_shannon_divergence_bits(distribution, distribution)

    assert divergence == pytest.approx(0.0, abs=JSD_TOLERANCE)


def test_jsd_bounded_by_one_bit():
    disjoint_left = (0.5, 0.5, 0.0, 0.0)
    disjoint_right = (0.0, 0.0, 0.25, 0.75)

    divergence = jensen_shannon_divergence_bits(disjoint_left, disjoint_right)

    assert divergence == pytest.approx(1.0, abs=JSD_TOLERANCE)
    assert divergence <= 1.0 + JSD_TOLERANCE


def test_jsd_symmetric():
    left = (0.7, 0.2, 0.1)
    right = (0.1, 0.3, 0.6)

    assert jensen_shannon_divergence_bits(left, right) == pytest.approx(
        jensen_shannon_divergence_bits(right, left), abs=JSD_TOLERANCE
    )


def test_jsd_handles_zero_entries_without_error():
    with_zero = (0.0, 0.5, 0.5)
    without_zero = (0.2, 0.4, 0.4)

    divergence = jensen_shannon_divergence_bits(with_zero, without_zero)

    assert 0.0 < divergence < 1.0


def test_jsd_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        jensen_shannon_divergence_bits((0.5, 0.5), (1.0,))


def test_jsd_raises_on_negative_probability():
    with pytest.raises(ValueError):
        jensen_shannon_divergence_bits((-0.1, 1.1), (0.5, 0.5))


def test_null95_shrinks_with_sample_size():
    base_distribution = (0.5, 0.3, 0.2)

    null_at_small_sample = monte_carlo_null95(
        base_distribution, sample_size=20, trials=FAST_TRIALS
    )
    null_at_large_sample = monte_carlo_null95(
        base_distribution, sample_size=400, trials=FAST_TRIALS
    )

    assert null_at_large_sample < null_at_small_sample


def test_null95_reproducible_with_fixed_seed():
    base_distribution = (0.6, 0.25, 0.15)

    first_value = monte_carlo_null95(base_distribution, sample_size=50, trials=FAST_TRIALS, seed=7)
    # Clear the memo cache so the second call recomputes instead of replaying it.
    cuisine_divergence._NULL95_CACHE.clear()
    second_value = monte_carlo_null95(base_distribution, sample_size=50, trials=FAST_TRIALS, seed=7)

    assert first_value == second_value


def test_null95_memoizes_repeat_calls():
    cuisine_divergence._NULL95_CACHE.clear()
    base_distribution = (0.4, 0.4, 0.2)

    monte_carlo_null95(base_distribution, sample_size=30, trials=FAST_TRIALS)
    monte_carlo_null95(base_distribution, sample_size=30, trials=FAST_TRIALS)

    assert len(cuisine_divergence._NULL95_CACHE) == 1


def test_null95_raises_on_zero_sample_size():
    with pytest.raises(ValueError):
        monte_carlo_null95((0.5, 0.5), sample_size=0)


def test_distribution_uses_recipe_level_union():
    index = _build_synthetic_index()

    # "salt" hits {1,4,5,8}, "soy" hits {1,2,3,6,8}; union has 7 recipes and
    # recipes 1 and 8 must be counted once each despite matching both strings.
    distribution = compute_cuisine_distribution({"salt", "soy"}, index)

    assert distribution == pytest.approx((3 / 7, 3 / 7, 1 / 7))


def test_distribution_is_all_zero_for_unknown_strings():
    index = _build_synthetic_index()

    distribution = compute_cuisine_distribution({"unicorn dust"}, index)

    assert distribution == (0.0, 0.0, 0.0)


def test_distribution_rejects_bare_string_argument():
    index = _build_synthetic_index()

    with pytest.raises(TypeError):
        compute_cuisine_distribution("salt", index)


def test_evaluate_merge_candidate_returns_consistent_evidence():
    index = _build_synthetic_index()

    evidence = evaluate_merge_candidate({"soy"}, {"salt"}, index, trials=FAST_TRIALS)

    assert isinstance(evidence, MergeEvidence)
    assert evidence.variant_count == 5
    assert evidence.base_count == 4
    expected_jsd = jensen_shannon_divergence_bits(
        compute_cuisine_distribution({"soy"}, index),
        compute_cuisine_distribution({"salt"}, index),
    )
    assert evidence.jsd_bits == pytest.approx(expected_jsd, abs=JSD_TOLERANCE)
    assert evidence.null95_bits > 0.0
    assert evidence.jsd_to_null_ratio == pytest.approx(
        evidence.jsd_bits / evidence.null95_bits
    )


def test_evaluate_merge_candidate_top_cuisines_sorted_with_lift():
    index = _build_synthetic_index()

    evidence = evaluate_merge_candidate({"soy"}, {"salt"}, index, trials=FAST_TRIALS)

    variant_shares = [entry.share for entry in evidence.variant_top_cuisines]
    assert variant_shares == sorted(variant_shares, reverse=True)
    top_entry = evidence.variant_top_cuisines[0]
    # "soy" is 3/5 alpha while alpha's prior is 3/8 -> lift = 0.6 / 0.375.
    assert top_entry.cuisine == "alpha"
    assert top_entry.share == pytest.approx(3 / 5)
    assert top_entry.lift == pytest.approx((3 / 5) / (3 / 8))


def test_evaluate_merge_candidate_zero_ratio_when_base_unknown():
    index = _build_synthetic_index()

    evidence = evaluate_merge_candidate({"soy"}, {"unicorn dust"}, index, trials=FAST_TRIALS)

    assert evidence.base_count == 0
    assert evidence.null95_bits == 0.0
    assert evidence.jsd_to_null_ratio == 0.0


def _make_evidence(variant_count: int, jsd_bits: float, null95_bits: float) -> MergeEvidence:
    return MergeEvidence(
        variant_count=variant_count,
        base_count=500,
        jsd_bits=jsd_bits,
        null95_bits=null95_bits,
        jsd_to_null_ratio=jsd_bits / null95_bits if null95_bits else 0.0,
        variant_top_cuisines=(),
        base_top_cuisines=(),
    )


def test_verdict_preserve_when_jsd_clears_floor_and_null():
    evidence = _make_evidence(variant_count=312, jsd_bits=0.1105, null95_bits=0.0174)

    assert classify_merge_verdict(evidence) == VERDICT_PRESERVE


def test_verdict_merge_when_jsd_below_floor_despite_support():
    evidence = _make_evidence(variant_count=425, jsd_bits=0.0181, null95_bits=0.0130)

    assert classify_merge_verdict(evidence) == VERDICT_MERGE


def test_verdict_merge_when_jsd_below_null_multiple():
    # Above the 0.07 floor but only 1.1x the null -> fails the 1.5x gate.
    evidence = _make_evidence(variant_count=40, jsd_bits=0.0900, null95_bits=0.0820)

    assert classify_merge_verdict(evidence) == VERDICT_MERGE


def test_verdict_review_when_support_below_minimum():
    evidence = _make_evidence(variant_count=17, jsd_bits=0.6500, null95_bits=0.1800)

    assert classify_merge_verdict(evidence) == VERDICT_REVIEW
