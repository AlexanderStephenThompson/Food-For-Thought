"""Build the browser's scoring asset from the gold model artifacts.

Coefficients are trimmed to 4 decimals: the blend displays at 4 decimals
and contributions at 3, so nothing visible changes, and combined with
compact serialization the asset roughly halves. Contract vectors are
generated from these trimmed values, so the browser reproduces them
exactly.
"""

from __future__ import annotations

COEFFICIENT_DECIMALS = 4
MODEL_ASSET_FILENAME = "model.json"


def build_model_asset(
    parameters_payload: dict,
    calibration_payload: dict,
    feature_space_payload: dict,
    fingerprint: dict,
) -> dict:
    """Assemble the complete client-side scoring contract.

    Args:
        parameters_payload: Gold model/parameters.json content.
        calibration_payload: Gold model/calibration.json content.
        feature_space_payload: Gold datasets/feature_space.json content.
        fingerprint: App build block embedded in the asset.

    Returns:
        Asset with cuisines, intercepts, 4-decimal coefficients,
        temperature, parent weight, and index-aligned feature arrays.
    """
    features = feature_space_payload["features"]
    return {
        "build": dict(fingerprint),
        "schema_version": 1,
        "cuisines": list(parameters_payload["cuisines"]),
        "intercepts": list(parameters_payload["intercepts"]),
        "coefficients": [
            [round(value, COEFFICIENT_DECIMALS) for value in cuisine_row]
            for cuisine_row in parameters_payload["coefficients"]
        ],
        "temperature": calibration_payload["temperature"],
        "parent_weight": parameters_payload["configuration"]["parent_weight"],
        "feature_ids": [feature["ingredient_id"] for feature in features],
        "parent_indices": [feature["parent_index"] for feature in features],
    }
