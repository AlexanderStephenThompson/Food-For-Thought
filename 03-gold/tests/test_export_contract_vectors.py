"""Tests for app_pipeline.export_contract_vectors."""

from app_pipeline.export_contract_vectors import build_contract_vectors_asset
from app_pipeline.export_model import build_model_asset
from app_pipeline.score_blend import (
    build_feature_values,
    compute_logits,
    convert_logits_to_blend,
)
from tests.app_payload_builders import (
    APP_BUILD_BLOCK,
    make_calibration_payload,
    make_parameters_payload,
)
from tests.model_payload_builders import make_feature_space_payload

SYNTHETIC_RECIPES = (
    ("empty recipe falls back to base rates", True, ()),
    ("parent back-off fires", False, ("dark_soy_sauce", "fish_sauce")),
    ("variant and parent together", False, ("dark_soy_sauce", "soy_sauce")),
    ("unknown ids are ignored", False, ("pasta", "not_a_real_ingredient")),
)


def _build_assets():
    model_asset = build_model_asset(
        make_parameters_payload(),
        make_calibration_payload(),
        make_feature_space_payload(),
        APP_BUILD_BLOCK,
    )
    vectors_asset = build_contract_vectors_asset(
        model_asset, APP_BUILD_BLOCK, recipes=SYNTHETIC_RECIPES
    )
    return model_asset, vectors_asset


def test_contract_vectors_blends_sum_to_one():
    _, vectors_asset = _build_assets()

    for vector in vectors_asset["vectors"]:
        assert abs(sum(vector["expected_blend"]) - 1.0) < 0.002


def test_contract_vectors_recompute_through_the_pure_scorer():
    model_asset, vectors_asset = _build_assets()

    for vector in vectors_asset["vectors"]:
        feature_values = build_feature_values(
            vector["ingredient_ids"],
            model_asset["feature_ids"],
            model_asset["parent_indices"],
            model_asset["parent_weight"],
        )
        logits = compute_logits(
            feature_values, model_asset["coefficients"], model_asset["intercepts"]
        )
        blend = convert_logits_to_blend(logits, model_asset["temperature"])
        assert [round(value, 4) for value in blend] == vector["expected_blend"]


def test_backoff_vector_actually_exercises_backoff():
    model_asset, vectors_asset = _build_assets()

    backoff_vector = next(
        vector
        for vector in vectors_asset["vectors"]
        if vector["name"] == "parent back-off fires"
    )
    feature_values = build_feature_values(
        backoff_vector["ingredient_ids"],
        model_asset["feature_ids"],
        model_asset["parent_indices"],
        model_asset["parent_weight"],
    )
    soy_sauce_index = model_asset["feature_ids"].index("soy_sauce")
    assert feature_values[soy_sauce_index] == model_asset["parent_weight"]


def test_empty_vector_expects_the_calibrated_prior():
    model_asset, vectors_asset = _build_assets()

    empty_vector = next(
        vector for vector in vectors_asset["vectors"] if not vector["ingredient_ids"]
    )
    prior = convert_logits_to_blend(
        model_asset["intercepts"], model_asset["temperature"]
    )
    assert empty_vector["expected_blend"] == [round(value, 4) for value in prior]
    assert empty_vector["example"] is True


def test_contract_vectors_carry_explanations():
    _, vectors_asset = _build_assets()

    backoff_vector = next(
        vector
        for vector in vectors_asset["vectors"]
        if vector["name"] == "parent back-off fires"
    )
    assert backoff_vector["expected_top_cuisine"] == "thai"
    assert backoff_vector["expected_top_contributions"][0]["ingredient_id"] in (
        "dark_soy_sauce",
        "fish_sauce",
    )
    assert all(
        "advantage" in entry for entry in backoff_vector["expected_differentiators"]
    )
