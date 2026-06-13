"""Tests for app_pipeline.score_blend — the pure scorer the JS must mirror.

Includes the cross-check against the numpy reference in
model_pipeline.calibrate_blend: same logits, same blends, within 1e-9.
"""

import numpy

from app_pipeline.score_blend import (
    build_feature_values,
    compute_logits,
    convert_logits_to_blend,
)
from model_pipeline.calibrate_blend import (
    compute_logits_from_parameters,
    convert_logits_to_blend as convert_logits_with_numpy,
)

FEATURE_IDS = ["basil", "dark_soy_sauce", "fish_sauce", "pasta", "rice", "soy_sauce"]
PARENT_INDICES = [None, 5, None, None, None, None]
COEFFICIENTS = [
    [2.1235, -1.0, -1.0, 2.0, -0.5, -0.6543],
    [-1.0, -1.0, -1.0, -1.0, 3.0, -0.5],
    [-2.0, 3.1111, 3.0, -2.0, -0.25, 1.0],
]
INTERCEPTS = [0.25, -0.1, -0.15]
PARENT_WEIGHT = 1.0


def test_build_feature_values_applies_parent_backoff():
    feature_values = build_feature_values(
        ["dark_soy_sauce", "fish_sauce"], FEATURE_IDS, PARENT_INDICES, PARENT_WEIGHT
    )

    assert feature_values == {1: 1.0, 2: 1.0, 5: PARENT_WEIGHT}


def test_build_feature_values_direct_presence_wins():
    feature_values = build_feature_values(
        ["dark_soy_sauce", "soy_sauce"], FEATURE_IDS, PARENT_INDICES, 0.5
    )

    assert feature_values[5] == 1.0


def test_build_feature_values_ignores_unknown_and_duplicate_ids():
    feature_values = build_feature_values(
        ["pasta", "pasta", "not_a_real_ingredient"],
        FEATURE_IDS,
        PARENT_INDICES,
        PARENT_WEIGHT,
    )

    assert feature_values == {3: 1.0}


def test_compute_logits_matches_manual_arithmetic():
    feature_values = {0: 1.0, 3: 1.0}

    logits = compute_logits(feature_values, COEFFICIENTS, INTERCEPTS)

    assert logits[0] == 0.25 + 2.1235 + 2.0
    assert logits[2] == -0.15 - 2.0 - 2.0


def test_blend_sums_to_one_and_orders_correctly():
    logits = compute_logits({1: 1.0, 2: 1.0, 5: 1.0}, COEFFICIENTS, INTERCEPTS)

    blend = convert_logits_to_blend(logits, temperature=1.05)

    assert abs(sum(blend) - 1.0) < 1e-12
    assert blend[2] == max(blend)


def test_pure_scorer_matches_numpy_reference():
    feature_values = {0: 1.0, 1: 1.0, 5: 0.3}
    feature_vector = numpy.zeros((1, len(FEATURE_IDS)))
    for index, value in feature_values.items():
        feature_vector[0, index] = value

    pure_logits = compute_logits(feature_values, COEFFICIENTS, INTERCEPTS)
    numpy_logits = compute_logits_from_parameters(
        feature_vector, COEFFICIENTS, INTERCEPTS
    )[0]
    pure_blend = convert_logits_to_blend(pure_logits, temperature=1.05)
    numpy_blend = convert_logits_with_numpy(
        numpy.array([pure_logits]), temperature=1.05
    )[0]

    for pure_value, numpy_value in zip(pure_logits, numpy_logits):
        assert abs(pure_value - numpy_value) < 1e-9
    for pure_value, numpy_value in zip(pure_blend, numpy_blend):
        assert abs(pure_value - numpy_value) < 1e-9
