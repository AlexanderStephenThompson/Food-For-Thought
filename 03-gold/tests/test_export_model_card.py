"""Tests for app_pipeline.export_model_card."""

from app_pipeline.export_model_card import build_model_card_asset
from tests.app_payload_builders import (
    APP_BUILD_BLOCK,
    make_calibration_payload,
    make_evaluation_payload,
    make_parameters_payload,
)


def _build_asset():
    return build_model_card_asset(
        make_evaluation_payload(),
        make_calibration_payload(),
        make_parameters_payload(),
        APP_BUILD_BLOCK,
    )


def test_model_card_extracts_calibration_with_reliability_bins():
    asset = _build_asset()

    assert asset["calibration"]["temperature"] == 1.05
    assert len(asset["calibration"]["reliability"]["after"]) == 10
    assert asset["calibration"]["ece_before"] == 0.05
    assert asset["calibration"]["ece_after"] == 0.02


def test_model_card_carries_per_cuisine_names_and_confusions():
    asset = _build_asset()

    per_cuisine_by_id = {entry["id"]: entry for entry in asset["per_cuisine"]}
    assert per_cuisine_by_id["thai"]["name"] == "Thai"
    assert per_cuisine_by_id["thai"]["recall"] == 0.8
    first_pair = asset["confusion_pairs"][0]
    assert first_pair["true_name"] == "Italian"
    assert first_pair["neighbor_similarity"] == 0.45


def test_model_card_records_training_facts():
    asset = _build_asset()

    assert asset["training"]["feature_count"] == 6
    assert asset["training"]["recipe_count"] == 12
    assert asset["training"]["coefficient_decimals"] == 4
    assert asset["mean"]["accuracy"] == 0.85
    assert asset["baseline_naive_bayes"]["mean_accuracy"] == 0.7
