"""Build the model-card asset: calibration honesty plus evaluation extracts."""

from __future__ import annotations

from app_pipeline.export_cuisines import format_display_name
from app_pipeline.export_model import COEFFICIENT_DECIMALS

MODEL_CARD_ASSET_FILENAME = "model-card.json"


def build_model_card_asset(
    evaluation_payload: dict,
    calibration_payload: dict,
    parameters_payload: dict,
    fingerprint: dict,
) -> dict:
    """Extract everything the model-card page states about the model.

    Args:
        evaluation_payload: Gold reports/evaluation.json content.
        calibration_payload: Gold model/calibration.json content.
        parameters_payload: Gold model/parameters.json content.
        fingerprint: App build block embedded in the asset.

    Returns:
        Asset with metrics, calibration (incl. reliability bins),
        per-cuisine recall, annotated confusion pairs, and training facts.
    """
    out_of_fold = calibration_payload["out_of_fold"]
    per_cuisine = [
        {
            "id": entry["cuisine"],
            "name": format_display_name(entry["cuisine"]),
            "recall": entry["recall"],
            "recipe_count": entry["recipe_count"],
        }
        for entry in evaluation_payload["per_cuisine"]
    ]
    confusion_pairs = [
        {
            **pair,
            "true_name": format_display_name(pair["true_cuisine"]),
            "predicted_name": format_display_name(pair["predicted_cuisine"]),
        }
        for pair in evaluation_payload["confusion_pairs"]
    ]
    return {
        "build": dict(fingerprint),
        "schema_version": 1,
        "mean": dict(evaluation_payload["mean"]),
        "folds": list(evaluation_payload["folds"]),
        "baseline_naive_bayes": dict(evaluation_payload["baseline_naive_bayes"]),
        "configuration": {
            "c_value": parameters_payload["configuration"]["c_value"],
            "parent_weight": parameters_payload["configuration"]["parent_weight"],
        },
        "grid_search": list(evaluation_payload["grid_search"]),
        "calibration": {
            "temperature": calibration_payload["temperature"],
            "log_loss_before": out_of_fold["log_loss_before"],
            "log_loss_after": out_of_fold["log_loss_after"],
            "ece_before": out_of_fold["ece_before"],
            "ece_after": out_of_fold["ece_after"],
            "reliability": dict(calibration_payload["reliability"]),
        },
        "per_cuisine": per_cuisine,
        "confusion_pairs": confusion_pairs,
        "training": {
            "recipe_count": sum(
                entry["recipe_count"] for entry in per_cuisine
            ),
            "feature_count": len(parameters_payload["coefficients"][0]),
            "coefficient_decimals": COEFFICIENT_DECIMALS,
        },
    }
