"""Tests for app_pipeline.validate_app — corrupt-one-field gate checks."""

import copy

import pytest

from app_pipeline.export_contract_vectors import build_contract_vectors_asset
from app_pipeline.export_cuisines import build_cuisines_asset
from app_pipeline.export_ingredients import build_ingredients_asset
from app_pipeline.export_model import build_model_asset
from app_pipeline.export_model_card import build_model_card_asset
from app_pipeline.validate_app import ValidationError, validate_app_assets
from tests.app_payload_builders import (
    APP_BUILD_BLOCK,
    make_app_inputs,
)

SYNTHETIC_CUISINE_COUNT = 3
SYNTHETIC_FEATURE_COUNT = 6
SYNTHETIC_RECIPES = (
    ("empty", True, ()),
    ("thai-ish", False, ("dark_soy_sauce", "fish_sauce")),
)


def _build_valid_assets() -> dict:
    inputs = make_app_inputs()
    model_asset = build_model_asset(
        inputs["parameters"],
        inputs["calibration"],
        inputs["feature_space"],
        APP_BUILD_BLOCK,
    )
    return {
        "model": model_asset,
        "ingredients": build_ingredients_asset(
            inputs["ingredients"], APP_BUILD_BLOCK
        ),
        "cuisines": build_cuisines_asset(
            inputs["cuisines"], inputs["evaluation"], APP_BUILD_BLOCK
        ),
        "model_card": build_model_card_asset(
            inputs["evaluation"],
            inputs["calibration"],
            inputs["parameters"],
            APP_BUILD_BLOCK,
        ),
        "contract_vectors": build_contract_vectors_asset(
            model_asset, APP_BUILD_BLOCK, recipes=SYNTHETIC_RECIPES
        ),
    }


def _validate(assets: dict, expected_fingerprint: dict | None = None) -> None:
    validate_app_assets(
        assets["model"],
        assets["ingredients"],
        assets["cuisines"],
        assets["model_card"],
        assets["contract_vectors"],
        expected_fingerprint=expected_fingerprint or APP_BUILD_BLOCK,
        expected_cuisine_count=SYNTHETIC_CUISINE_COUNT,
        expected_feature_count=SYNTHETIC_FEATURE_COUNT,
    )


def test_valid_app_assets_pass():
    _validate(_build_valid_assets())


def test_stale_fingerprint_raises():
    assets = _build_valid_assets()
    stale = dict(APP_BUILD_BLOCK, parameters_sha256="0" * 64)

    with pytest.raises(ValidationError, match="fingerprint"):
        _validate(assets, expected_fingerprint=stale)


def test_wrong_feature_count_raises():
    assets = _build_valid_assets()
    corrupted = copy.deepcopy(assets["model"])
    corrupted["feature_ids"] = corrupted["feature_ids"][:-1]

    with pytest.raises(ValidationError, match="feature"):
        _validate({**assets, "model": corrupted})


def test_unrounded_coefficient_raises():
    assets = _build_valid_assets()
    corrupted = copy.deepcopy(assets["model"])
    corrupted["coefficients"][0][0] = 0.123456789

    with pytest.raises(ValidationError, match="coefficient"):
        _validate({**assets, "model": corrupted})


def test_unknown_distinctive_ingredient_raises():
    assets = _build_valid_assets()
    corrupted = copy.deepcopy(assets["cuisines"])
    corrupted["cuisines"][0]["distinctive"] = [
        {"id": "ghost_ingredient", "name": "ghost", "lift": 2.0, "coverage": 0.1}
    ]

    with pytest.raises(ValidationError, match="distinctive"):
        _validate({**assets, "cuisines": corrupted})


def test_contract_blend_not_summing_raises():
    assets = _build_valid_assets()
    corrupted = copy.deepcopy(assets["contract_vectors"])
    corrupted["vectors"][0]["expected_blend"] = [
        value * 0.5 for value in corrupted["vectors"][0]["expected_blend"]
    ]

    with pytest.raises(ValidationError, match="sum"):
        _validate({**assets, "contract_vectors": corrupted})


def test_position_off_unit_circle_raises():
    assets = _build_valid_assets()
    corrupted = copy.deepcopy(assets["cuisines"])
    corrupted["cuisines"][0]["position"] = {"x": 2.0, "y": 0.0}

    with pytest.raises(ValidationError, match="position"):
        _validate({**assets, "cuisines": corrupted})
