"""Validation gates over the gold artifacts.

Pure validators over already-loaded payload dicts — no file I/O. Each gate
failure raises ValidationError naming the gate and the offending values.
The umbrella validate_gold_artifacts additionally proves all artifacts
share one build block and that the block matches the silver inputs on disk
(freshness), then re-derives every feature row and fold property
independently of the builders.
"""

from __future__ import annotations

from gold_pipeline.assign_folds import FOLD_COUNT
from silver_pipeline.artifact_io import SCHEMA_VERSION

EXPECTED_FEATURE_COUNT = 2_813
EXPECTED_TRAIN_FEATURE_ROW_COUNT = 39_774
EXPECTED_TEST_FEATURE_ROW_COUNT = 9_944
EXPECTED_EMPTY_TEST_FEATURE_ROW_COUNT = 0
FOLD_BALANCE_MAXIMUM_SPREAD = 1

REQUIRED_FINGERPRINT_KEYS = frozenset(
    (
        "cuisines_sha256",
        "fold_count",
        "ingredients_sha256",
        "random_seed",
        "recipes_test_sha256",
        "recipes_train_sha256",
    )
)


class ValidationError(ValueError):
    """A gold artifact violates the pinned schema or a balance gate."""


def _require_envelope(payload: dict, artifact_name: str) -> None:
    """Gate: schema_version and a complete build block."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(
            f"{artifact_name}: schema_version must be {SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    build = payload.get("build")
    if not isinstance(build, dict) or set(build) != REQUIRED_FINGERPRINT_KEYS:
        raise ValidationError(
            f"{artifact_name}: build block must hold exactly "
            f"{sorted(REQUIRED_FINGERPRINT_KEYS)}"
        )


def _require_ascending_unique(values: list[int], context: str) -> None:
    """Gate: a strictly ascending, duplicate-free index list."""
    if any(
        later <= earlier for earlier, later in zip(values, values[1:])
    ):
        raise ValidationError(f"{context}: must be strictly ascending, got {values}")


def validate_feature_space_payload(
    payload: dict, ingredients_payload: dict, expected_feature_count: int
) -> None:
    """Gate the feature space: bijection over silver ids with parent links.

    Args:
        payload: Gold feature_space.json content.
        ingredients_payload: Parsed silver ingredients.json document.
        expected_feature_count: Pinned vocabulary size.

    Raises:
        ValidationError: On any schema, count, ordering, or parent-link
            violation.
    """
    _require_envelope(payload, "feature_space")
    features = payload["features"]
    if payload["feature_count"] != len(features):
        raise ValidationError(
            "feature_space: feature_count "
            f"{payload['feature_count']} != {len(features)} features"
        )
    if len(features) != expected_feature_count:
        raise ValidationError(
            f"feature_space: expected {expected_feature_count} features, "
            f"got {len(features)}"
        )

    silver_ids = {entry["id"] for entry in ingredients_payload["ingredients"]}
    feature_ids = [feature["ingredient_id"] for feature in features]
    if set(feature_ids) != silver_ids:
        raise ValidationError(
            "feature_space: ingredient ids do not match the silver vocabulary"
        )
    if feature_ids != sorted(feature_ids):
        raise ValidationError("feature_space: ingredient ids must be sorted")

    for position, feature in enumerate(features):
        if feature["index"] != position:
            raise ValidationError(
                f"feature_space: index at position {position} is "
                f"{feature['index']}, expected {position}"
            )

    parent_id_by_id = {
        entry["id"]: entry["parent_id"]
        for entry in ingredients_payload["ingredients"]
    }
    index_by_id = {
        feature["ingredient_id"]: feature["index"] for feature in features
    }
    for feature in features:
        parent_id = parent_id_by_id[feature["ingredient_id"]]
        expected_parent_index = (
            None if parent_id is None else index_by_id[parent_id]
        )
        if feature["parent_index"] != expected_parent_index:
            raise ValidationError(
                f"feature_space: {feature['ingredient_id']} has parent_index "
                f"{feature['parent_index']}, expected {expected_parent_index}"
            )


def _require_row_matches_recipe(
    row: dict,
    recipe: dict,
    index_by_id: dict[str, int],
    parent_index_by_index: dict[int, int | None],
) -> None:
    """Gate: a row's index lists re-derive exactly from its silver recipe."""
    expected_ingredient_indices = sorted(
        index_by_id[ingredient_id] for ingredient_id in recipe["ingredient_ids"]
    )
    if row["ingredient_indices"] != expected_ingredient_indices:
        raise ValidationError(
            f"features: recipe {row['recipe_id']} ingredient_indices "
            f"{row['ingredient_indices']} != derived "
            f"{expected_ingredient_indices}"
        )
    expected_parent_indices = sorted(
        {
            parent_index_by_index[index]
            for index in expected_ingredient_indices
            if parent_index_by_index[index] is not None
        }
    )
    if row["parent_indices"] != expected_parent_indices:
        raise ValidationError(
            f"features: recipe {row['recipe_id']} parent_indices "
            f"{row['parent_indices']} != derived {expected_parent_indices}"
        )


def validate_features_payload(
    payload: dict,
    feature_space_payload: dict,
    recipes_payload: dict,
    expected_row_count: int,
    requires_cuisine: bool,
    expected_empty_row_count: int,
) -> None:
    """Gate one feature split against its silver recipes and feature space.

    Args:
        payload: Gold features_train.json or features_test.json content.
        feature_space_payload: Gold feature_space.json content.
        recipes_payload: The matching silver recipes document.
        expected_row_count: Pinned recipe count for this split.
        requires_cuisine: True when rows must carry the recipe's cuisine
            (train), False when they must not (test).
        expected_empty_row_count: Pinned count of rows with no ingredient
            indices (0 for train; the measured count for test).

    Raises:
        ValidationError: On any schema, count, ordering, cuisine, index, or
            empty-row violation.
    """
    _require_envelope(payload, "features")
    rows = payload["rows"]
    if len(rows) != expected_row_count:
        raise ValidationError(
            f"features: expected {expected_row_count} rows, got {len(rows)}"
        )

    recipe_by_id = {
        recipe["id"]: recipe for recipe in recipes_payload["recipes"]
    }
    row_ids = [row["recipe_id"] for row in rows]
    _require_ascending_unique(row_ids, "features: recipe ids")
    if set(row_ids) != set(recipe_by_id):
        raise ValidationError(
            "features: row recipe ids do not match the silver split"
        )

    index_by_id = {
        feature["ingredient_id"]: feature["index"]
        for feature in feature_space_payload["features"]
    }
    parent_index_by_index = {
        feature["index"]: feature["parent_index"]
        for feature in feature_space_payload["features"]
    }
    valid_indices = set(parent_index_by_index)
    empty_row_count = 0
    for row in rows:
        if not set(row["ingredient_indices"]) <= valid_indices:
            raise ValidationError(
                f"features: recipe {row['recipe_id']} ingredient_indices "
                "contain out-of-range values"
            )
        _require_ascending_unique(
            row["ingredient_indices"],
            f"features: recipe {row['recipe_id']} ingredient_indices",
        )
        _require_row_matches_recipe(
            row, recipe_by_id[row["recipe_id"]], index_by_id, parent_index_by_index
        )
        if requires_cuisine and row.get("cuisine") != recipe_by_id[
            row["recipe_id"]
        ].get("cuisine"):
            raise ValidationError(
                f"features: recipe {row['recipe_id']} cuisine mismatch"
            )
        if not requires_cuisine and "cuisine" in row:
            raise ValidationError(
                f"features: test recipe {row['recipe_id']} must not carry cuisine"
            )
        if not row["ingredient_indices"]:
            empty_row_count += 1

    if empty_row_count != expected_empty_row_count:
        raise ValidationError(
            f"features: expected {expected_empty_row_count} empty rows, "
            f"got {empty_row_count}"
        )


def validate_folds_payload(
    payload: dict,
    recipes_train_payload: dict,
    expected_fold_count: int = FOLD_COUNT,
    maximum_spread: int = FOLD_BALANCE_MAXIMUM_SPREAD,
) -> None:
    """Gate the fold assignment: complete, in range, and balanced.

    Args:
        payload: Gold folds.json content.
        recipes_train_payload: Parsed silver recipes_train.json document.
        expected_fold_count: Pinned number of folds.
        maximum_spread: Largest allowed difference between any cuisine's
            biggest and smallest fold.

    Raises:
        ValidationError: On any schema, coverage, range, ordering, or
            balance violation.
    """
    _require_envelope(payload, "folds")
    if payload["fold_count"] != expected_fold_count:
        raise ValidationError(
            f"folds: fold_count must be {expected_fold_count}, "
            f"got {payload['fold_count']}"
        )

    assignments = payload["assignments"]
    assigned_ids = [entry["recipe_id"] for entry in assignments]
    _require_ascending_unique(assigned_ids, "folds: assignment recipe ids")
    train_ids = {recipe["id"] for recipe in recipes_train_payload["recipes"]}
    if set(assigned_ids) != train_ids:
        raise ValidationError(
            "folds: assignment recipe ids do not cover the train split "
            f"exactly ({len(assigned_ids)} assignments, {len(train_ids)} recipes)"
        )
    out_of_range = [
        entry
        for entry in assignments
        if not 0 <= entry["fold"] < expected_fold_count
    ]
    if out_of_range:
        raise ValidationError(
            f"folds: assignment folds out of range: {out_of_range[:5]}"
        )

    cuisine_by_recipe_id = {
        recipe["id"]: recipe["cuisine"]
        for recipe in recipes_train_payload["recipes"]
    }
    fold_counts_by_cuisine: dict[str, list[int]] = {}
    for entry in assignments:
        cuisine = cuisine_by_recipe_id[entry["recipe_id"]]
        fold_counts_by_cuisine.setdefault(
            cuisine, [0] * expected_fold_count
        )[entry["fold"]] += 1
    for cuisine, fold_counts in sorted(fold_counts_by_cuisine.items()):
        spread = max(fold_counts) - min(fold_counts)
        if spread > maximum_spread:
            raise ValidationError(
                f"folds: cuisine {cuisine} fold spread {spread} exceeds "
                f"{maximum_spread} (balance violated: {fold_counts})"
            )


def validate_fold_balance_payload(
    payload: dict, folds_payload: dict, recipes_train_payload: dict
) -> None:
    """Gate the fold-balance report against an independent recount.

    Args:
        payload: Gold fold_balance.json content.
        folds_payload: Gold folds.json content.
        recipes_train_payload: Parsed silver recipes_train.json document.

    Raises:
        ValidationError: When any reported count disagrees with the counts
            recomputed from the folds artifact.
    """
    _require_envelope(payload, "fold_balance")
    fold_count = folds_payload["fold_count"]
    if payload["fold_count"] != fold_count:
        raise ValidationError(
            f"fold_balance: fold_count {payload['fold_count']} != "
            f"folds artifact's {fold_count}"
        )

    cuisine_by_recipe_id = {
        recipe["id"]: recipe["cuisine"]
        for recipe in recipes_train_payload["recipes"]
    }
    recounted_fold_sizes = [0] * fold_count
    recounted_by_cuisine: dict[str, list[int]] = {}
    for entry in folds_payload["assignments"]:
        cuisine = cuisine_by_recipe_id[entry["recipe_id"]]
        recounted_fold_sizes[entry["fold"]] += 1
        recounted_by_cuisine.setdefault(cuisine, [0] * fold_count)[
            entry["fold"]
        ] += 1

    if payload["fold_sizes"] != recounted_fold_sizes:
        raise ValidationError(
            f"fold_balance: fold_sizes {payload['fold_sizes']} != recounted "
            f"{recounted_fold_sizes}"
        )
    reported_by_cuisine = {
        entry["cuisine"]: entry for entry in payload["by_cuisine"]
    }
    if set(reported_by_cuisine) != set(recounted_by_cuisine):
        raise ValidationError(
            "fold_balance: reported cuisines do not match the train split"
        )
    for cuisine, fold_counts in recounted_by_cuisine.items():
        reported = reported_by_cuisine[cuisine]
        if reported["fold_counts"] != fold_counts:
            raise ValidationError(
                f"fold_balance: {cuisine} fold_counts "
                f"{reported['fold_counts']} != recounted {fold_counts}"
            )
        if reported["recipe_count"] != sum(fold_counts):
            raise ValidationError(
                f"fold_balance: {cuisine} recipe_count "
                f"{reported['recipe_count']} != {sum(fold_counts)}"
            )


def _require_shared_fresh_build_blocks(
    payload_by_name: dict[str, dict], expected_fingerprint: dict
) -> None:
    """Gate: every artifact carries the same, fresh build block."""
    blocks = {name: payload.get("build") for name, payload in payload_by_name.items()}
    distinct_blocks = [
        block for position, block in enumerate(blocks.values())
        if block not in list(blocks.values())[:position]
    ]
    if len(distinct_blocks) > 1:
        raise ValidationError(
            f"gold artifacts carry differing build blocks: {sorted(blocks)}"
        )
    shared_block = next(iter(blocks.values()))
    if shared_block != expected_fingerprint:
        raise ValidationError(
            "gold build fingerprint is stale: artifacts were built from a "
            "different silver state than the one on disk"
        )


def validate_gold_artifacts(
    feature_space_payload: dict,
    features_train_payload: dict,
    features_test_payload: dict,
    folds_payload: dict,
    fold_balance_payload: dict,
    ingredients_payload: dict,
    recipes_train_payload: dict,
    recipes_test_payload: dict,
    *,
    expected_fingerprint: dict,
    expected_feature_count: int = EXPECTED_FEATURE_COUNT,
    expected_train_row_count: int = EXPECTED_TRAIN_FEATURE_ROW_COUNT,
    expected_test_row_count: int = EXPECTED_TEST_FEATURE_ROW_COUNT,
    expected_empty_test_row_count: int = EXPECTED_EMPTY_TEST_FEATURE_ROW_COUNT,
) -> None:
    """Run every gold gate plus the cross-artifact checks.

    Args:
        feature_space_payload: Gold feature_space.json content.
        features_train_payload: Gold features_train.json content.
        features_test_payload: Gold features_test.json content.
        folds_payload: Gold folds.json content.
        fold_balance_payload: Gold fold_balance.json content.
        ingredients_payload: Parsed silver ingredients.json document.
        recipes_train_payload: Parsed silver recipes_train.json document.
        recipes_test_payload: Parsed silver recipes_test.json document.
        expected_fingerprint: Fingerprint freshly computed from the silver
            files on disk; every artifact's build block must equal it.
        expected_feature_count: Pinned vocabulary size.
        expected_train_row_count: Pinned train recipe count.
        expected_test_row_count: Pinned test recipe count.
        expected_empty_test_row_count: Pinned count of empty test rows.

    Raises:
        ValidationError: If any gate fails.
    """
    _require_shared_fresh_build_blocks(
        {
            "feature_space": feature_space_payload,
            "features_train": features_train_payload,
            "features_test": features_test_payload,
            "folds": folds_payload,
            "fold_balance": fold_balance_payload,
        },
        expected_fingerprint,
    )
    validate_feature_space_payload(
        feature_space_payload, ingredients_payload, expected_feature_count
    )
    validate_features_payload(
        features_train_payload,
        feature_space_payload,
        recipes_train_payload,
        expected_row_count=expected_train_row_count,
        requires_cuisine=True,
        expected_empty_row_count=0,
    )
    validate_features_payload(
        features_test_payload,
        feature_space_payload,
        recipes_test_payload,
        expected_row_count=expected_test_row_count,
        requires_cuisine=False,
        expected_empty_row_count=expected_empty_test_row_count,
    )
    validate_folds_payload(folds_payload, recipes_train_payload)
    validate_fold_balance_payload(
        fold_balance_payload, folds_payload, recipes_train_payload
    )
