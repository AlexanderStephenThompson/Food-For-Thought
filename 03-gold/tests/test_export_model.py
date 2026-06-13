"""Tests for app_pipeline.export_model."""

from app_pipeline.export_model import COEFFICIENT_DECIMALS, build_model_asset
from tests.app_payload_builders import (
    APP_BUILD_BLOCK,
    make_calibration_payload,
    make_parameters_payload,
)
from tests.model_payload_builders import make_feature_space_payload


def _build_asset():
    return build_model_asset(
        make_parameters_payload(),
        make_calibration_payload(),
        make_feature_space_payload(),
        APP_BUILD_BLOCK,
    )


def test_model_asset_trims_coefficients_to_four_decimals():
    asset = _build_asset()

    assert asset["coefficients"][0][0] == round(2.123456, COEFFICIENT_DECIMALS)
    assert all(
        value == round(value, COEFFICIENT_DECIMALS)
        for cuisine_row in asset["coefficients"]
        for value in cuisine_row
    )


def test_model_asset_aligns_feature_arrays_with_indices():
    asset = _build_asset()

    assert asset["feature_ids"][1] == "dark_soy_sauce"
    assert asset["parent_indices"][1] == 5
    assert asset["parent_indices"][0] is None
    assert len(asset["feature_ids"]) == len(asset["parent_indices"]) == 6


def test_model_asset_carries_scoring_constants():
    asset = _build_asset()

    assert asset["temperature"] == 1.05
    assert asset["parent_weight"] == 1.0
    assert asset["cuisines"] == ["italian", "mexican", "thai"]
    assert asset["schema_version"] == 1
    assert asset["build"] == APP_BUILD_BLOCK
