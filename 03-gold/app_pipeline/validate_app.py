"""Validation gates over the app data assets.

Pure validators over already-built asset dicts. Each gate failure raises
ValidationError naming the gate and the offending values; the umbrella
also proves all five assets share one build block that matches the
fingerprint freshly computed from the inputs on disk.
"""

from __future__ import annotations

import math

from app_pipeline.export_model import COEFFICIENT_DECIMALS

EXPECTED_CUISINE_COUNT = 20
EXPECTED_FEATURE_COUNT = 2_813
BLEND_SUM_TOLERANCE = 0.002
POSITION_RADIUS_TOLERANCE = 1e-3

REQUIRED_APP_FINGERPRINT_KEYS = frozenset(
    (
        "calibration_sha256",
        "cuisines_sha256",
        "evaluation_sha256",
        "feature_space_sha256",
        "ingredients_sha256",
        "parameters_sha256",
    )
)


class ValidationError(ValueError):
    """An app asset violates the pinned schema or a consistency gate."""


def _require_envelope(asset: dict, asset_name: str) -> None:
    """Gate: schema_version and a complete build block."""
    if asset.get("schema_version") != 1:
        raise ValidationError(
            f"{asset_name}: schema_version must be 1, "
            f"got {asset.get('schema_version')!r}"
        )
    build = asset.get("build")
    if not isinstance(build, dict) or set(build) != REQUIRED_APP_FINGERPRINT_KEYS:
        raise ValidationError(
            f"{asset_name}: build block must hold exactly "
            f"{sorted(REQUIRED_APP_FINGERPRINT_KEYS)}"
        )


def validate_model_asset(
    asset: dict, expected_cuisine_count: int, expected_feature_count: int
) -> None:
    """Gate the scoring asset: shapes, rounding, and constants."""
    _require_envelope(asset, "model")
    if len(asset["cuisines"]) != expected_cuisine_count:
        raise ValidationError(
            f"model: {len(asset['cuisines'])} cuisines, "
            f"expected {expected_cuisine_count}"
        )
    for array_name in ("feature_ids", "parent_indices"):
        if len(asset[array_name]) != expected_feature_count:
            raise ValidationError(
                f"model: {array_name} holds {len(asset[array_name])} feature "
                f"entries, expected {expected_feature_count}"
            )
    if len(asset["coefficients"]) != expected_cuisine_count:
        raise ValidationError(
            f"model: {len(asset['coefficients'])} coefficient rows for "
            f"{expected_cuisine_count} cuisines"
        )
    for cuisine, cuisine_row in zip(asset["cuisines"], asset["coefficients"]):
        if len(cuisine_row) != expected_feature_count:
            raise ValidationError(
                f"model: coefficient row for {cuisine} has {len(cuisine_row)} "
                f"feature columns, expected {expected_feature_count}"
            )
        for value in cuisine_row:
            if value != round(value, COEFFICIENT_DECIMALS):
                raise ValidationError(
                    f"model: coefficient {value!r} for {cuisine} is not "
                    f"rounded to {COEFFICIENT_DECIMALS} decimals"
                )
    if not asset["temperature"] > 0:
        raise ValidationError(
            f"model: temperature must be positive, got {asset['temperature']}"
        )
    if asset["parent_weight"] < 0:
        raise ValidationError(
            f"model: parent_weight must be non-negative, "
            f"got {asset['parent_weight']}"
        )


def validate_ingredients_asset(asset: dict, model_asset: dict) -> None:
    """Gate the ingredient asset: ordering and referential integrity."""
    _require_envelope(asset, "ingredients")
    ids = [entry["id"] for entry in asset["ingredients"]]
    if ids != sorted(ids):
        raise ValidationError("ingredients: entries must be sorted by id")
    if set(ids) != set(model_asset["feature_ids"]):
        raise ValidationError(
            "ingredients: ids do not match the model's feature ids"
        )
    known_ids = set(ids)
    for entry in asset["ingredients"]:
        if entry["parent_id"] is not None and entry["parent_id"] not in known_ids:
            raise ValidationError(
                f"ingredients: {entry['id']} has unknown parent "
                f"{entry['parent_id']!r}"
            )
        unknown_children = set(entry["children"]) - known_ids
        if unknown_children:
            raise ValidationError(
                f"ingredients: {entry['id']} has unknown children "
                f"{sorted(unknown_children)}"
            )


def validate_cuisines_asset(
    asset: dict, ingredients_asset: dict, expected_cuisine_count: int
) -> None:
    """Gate the atlas asset: coverage, positions, edges, references."""
    _require_envelope(asset, "cuisines")
    cuisines = asset["cuisines"]
    if len(cuisines) != expected_cuisine_count:
        raise ValidationError(
            f"cuisines: {len(cuisines)} entries, expected {expected_cuisine_count}"
        )
    known_ingredient_ids = {
        entry["id"] for entry in ingredients_asset["ingredients"]
    }
    known_cuisine_ids = {cuisine["id"] for cuisine in cuisines}
    for cuisine in cuisines:
        radius = math.hypot(cuisine["position"]["x"], cuisine["position"]["y"])
        if abs(radius - 1.0) > POSITION_RADIUS_TOLERANCE:
            raise ValidationError(
                f"cuisines: {cuisine['id']} position is off the unit circle "
                f"(radius {radius:.4f})"
            )
        if "recall" not in cuisine:
            raise ValidationError(f"cuisines: {cuisine['id']} is missing recall")
        unknown_distinctive = {
            entry["id"] for entry in cuisine["distinctive"]
        } - known_ingredient_ids
        if unknown_distinctive:
            raise ValidationError(
                f"cuisines: {cuisine['id']} distinctive ingredients unknown: "
                f"{sorted(unknown_distinctive)}"
            )
    for edge in asset["edges"]:
        if not {edge["a"], edge["b"]} <= known_cuisine_ids:
            raise ValidationError(f"cuisines: edge references unknown ids: {edge}")


def validate_model_card_asset(asset: dict, expected_cuisine_count: int) -> None:
    """Gate the model-card asset: per-cuisine coverage and bins."""
    _require_envelope(asset, "model_card")
    if len(asset["per_cuisine"]) != expected_cuisine_count:
        raise ValidationError(
            f"model_card: {len(asset['per_cuisine'])} per-cuisine entries, "
            f"expected {expected_cuisine_count}"
        )
    for phase in ("before", "after"):
        bins = asset["calibration"]["reliability"][phase]
        if len(bins) != 10:
            raise ValidationError(
                f"model_card: reliability {phase} holds {len(bins)} bins, "
                "expected 10"
            )


def validate_contract_vectors_asset(asset: dict, model_asset: dict) -> None:
    """Gate the contract vectors: blend shape, sums, and argmax."""
    _require_envelope(asset, "contract_vectors")
    cuisine_count = len(model_asset["cuisines"])
    cuisine_position = {
        cuisine: position
        for position, cuisine in enumerate(model_asset["cuisines"])
    }
    for vector in asset["vectors"]:
        blend = vector["expected_blend"]
        if len(blend) != cuisine_count:
            raise ValidationError(
                f"contract vector {vector['name']!r} blend has {len(blend)} "
                f"entries, expected {cuisine_count}"
            )
        blend_total = sum(blend)
        if abs(blend_total - 1.0) > BLEND_SUM_TOLERANCE:
            raise ValidationError(
                f"contract vector {vector['name']!r} blend sum {blend_total} "
                f"deviates from 1 beyond {BLEND_SUM_TOLERANCE}"
            )
        top_share = blend[cuisine_position[vector["expected_top_cuisine"]]]
        if top_share != max(blend):
            raise ValidationError(
                f"contract vector {vector['name']!r} top cuisine does not "
                "carry the maximum share"
            )


def _require_shared_fresh_build_blocks(
    asset_by_name: dict[str, dict], expected_fingerprint: dict
) -> None:
    """Gate: every asset carries the same, fresh build block."""
    blocks = [asset.get("build") for asset in asset_by_name.values()]
    if any(block != blocks[0] for block in blocks[1:]):
        raise ValidationError(
            f"app assets carry differing build blocks: {sorted(asset_by_name)}"
        )
    if blocks[0] != expected_fingerprint:
        raise ValidationError(
            "app build fingerprint is stale: assets were built from a "
            "different input state than the one on disk"
        )


def validate_app_assets(
    model_asset: dict,
    ingredients_asset: dict,
    cuisines_asset: dict,
    model_card_asset: dict,
    contract_vectors_asset: dict,
    *,
    expected_fingerprint: dict,
    expected_cuisine_count: int = EXPECTED_CUISINE_COUNT,
    expected_feature_count: int = EXPECTED_FEATURE_COUNT,
) -> None:
    """Run every app gate plus the cross-asset checks.

    Args:
        model_asset: The scoring asset.
        ingredients_asset: The search/explorer asset.
        cuisines_asset: The atlas asset.
        model_card_asset: The model-card asset.
        contract_vectors_asset: The scoring contract vectors.
        expected_fingerprint: Fingerprint freshly computed from the
            inputs on disk.
        expected_cuisine_count: Pinned cuisine count.
        expected_feature_count: Pinned feature-space width.

    Raises:
        ValidationError: If any gate fails.
    """
    _require_shared_fresh_build_blocks(
        {
            "model": model_asset,
            "ingredients": ingredients_asset,
            "cuisines": cuisines_asset,
            "model_card": model_card_asset,
            "contract_vectors": contract_vectors_asset,
        },
        expected_fingerprint,
    )
    validate_model_asset(
        model_asset, expected_cuisine_count, expected_feature_count
    )
    validate_ingredients_asset(ingredients_asset, model_asset)
    validate_cuisines_asset(
        cuisines_asset, ingredients_asset, expected_cuisine_count
    )
    validate_model_card_asset(model_card_asset, expected_cuisine_count)
    validate_contract_vectors_asset(contract_vectors_asset, model_asset)
