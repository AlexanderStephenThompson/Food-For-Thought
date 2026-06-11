"""Build per-recipe gold feature rows with parent back-off indices.

Each row carries two index lists over the feature space: the recipe's own
ingredient indices, and the deduplicated indices of those ingredients'
parents. Keeping them separate leaves the back-off weighting decision to
the model phase instead of baking it into the data.
"""

from __future__ import annotations

from silver_pipeline.artifact_io import SCHEMA_VERSION


def _derive_row(
    recipe: dict,
    index_by_id: dict[str, int],
    parent_index_by_index: dict[int, int | None],
    includes_cuisine: bool,
) -> dict:
    """Build one feature row from a silver recipe record."""
    ingredient_indices = sorted(
        index_by_id[ingredient_id] for ingredient_id in recipe["ingredient_ids"]
    )
    parent_indices = sorted(
        {
            parent_index_by_index[index]
            for index in ingredient_indices
            if parent_index_by_index[index] is not None
        }
    )
    row = {
        "ingredient_indices": ingredient_indices,
        "parent_indices": parent_indices,
        "recipe_id": recipe["id"],
    }
    if includes_cuisine:
        row["cuisine"] = recipe["cuisine"]
    return row


def build_features_payload(
    recipes_payload: dict,
    feature_space_payload: dict,
    fingerprint: dict,
    *,
    includes_cuisine: bool,
) -> dict:
    """Build the gold feature rows for one recipe split.

    Args:
        recipes_payload: Parsed silver recipes_train.json or
            recipes_test.json document.
        feature_space_payload: Gold feature space from
            build_feature_space_payload.
        fingerprint: Gold build block embedded in the artifact.
        includes_cuisine: True for the labeled train split (rows carry
            the recipe's cuisine); False for the unlabeled test split.

    Returns:
        Payload with rows sorted by recipe id, ready for
        write_artifact_json.

    Raises:
        KeyError: If a recipe references an ingredient id missing from
            the feature space.
    """
    index_by_id = {
        feature["ingredient_id"]: feature["index"]
        for feature in feature_space_payload["features"]
    }
    parent_index_by_index = {
        feature["index"]: feature["parent_index"]
        for feature in feature_space_payload["features"]
    }
    rows = [
        _derive_row(recipe, index_by_id, parent_index_by_index, includes_cuisine)
        for recipe in sorted(
            recipes_payload["recipes"], key=lambda recipe: recipe["id"]
        )
    ]
    return {
        "build": dict(fingerprint),
        "rows": rows,
        "schema_version": SCHEMA_VERSION,
    }
