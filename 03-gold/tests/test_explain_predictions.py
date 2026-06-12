"""Tests for model_pipeline.explain_predictions."""

from model_pipeline.explain_predictions import (
    compute_ingredient_contributions,
    summarize_blend_explanation,
)
from tests.model_payload_builders import (
    MODEL_BUILD_BLOCK,
    make_feature_space_payload,
)

CUISINE_IDS = ("italian", "mexican", "thai")


def _make_parameters_payload():
    """Hand-built parameters: thai loves features 1/2/5, italian loves 0/3."""
    return {
        "build": dict(MODEL_BUILD_BLOCK),
        "schema_version": 1,
        "cuisines": list(CUISINE_IDS),
        "intercepts": [0.0, 0.0, 0.0],
        "coefficients": [
            [2.0, -1.0, -1.0, 2.0, 0.0, -0.5],
            [-1.0, -1.0, -1.0, -1.0, 3.0, -0.5],
            [-2.0, 3.0, 3.0, -2.0, 0.0, 1.0],
        ],
    }


def test_contribution_equals_value_times_coefficient():
    thai_coefficients = _make_parameters_payload()["coefficients"][2]
    feature_values = {1: 1.0, 5: 0.3}

    contributions = compute_ingredient_contributions(
        feature_values, thai_coefficients
    )

    contribution_by_index = dict(contributions)
    assert contribution_by_index[1] == 3.0
    assert contribution_by_index[5] == 0.3 * 1.0


def test_contributions_are_sorted_descending():
    thai_coefficients = _make_parameters_payload()["coefficients"][2]
    feature_values = {0: 1.0, 1: 1.0, 5: 0.3}

    contributions = compute_ingredient_contributions(
        feature_values, thai_coefficients
    )

    values = [contribution for _, contribution in contributions]
    assert values == sorted(values, reverse=True)


def test_differentiators_rank_top_versus_runner_up():
    explanation = summarize_blend_explanation(
        feature_values={1: 1.0, 2: 1.0, 5: 0.3},
        parameters_payload=_make_parameters_payload(),
        feature_space_payload=make_feature_space_payload(),
        top_cuisine="thai",
        runner_up_cuisine="mexican",
        limit=2,
    )

    top_ids = [entry["ingredient_id"] for entry in explanation["top_contributions"]]
    assert top_ids[0] in ("dark_soy_sauce", "fish_sauce")
    first_differentiator = explanation["differentiators"][0]
    assert first_differentiator["ingredient_id"] in ("dark_soy_sauce", "fish_sauce")
    # thai coef 3.0 minus mexican coef -1.0 on a 1.0-valued feature.
    assert first_differentiator["advantage"] == 4.0
