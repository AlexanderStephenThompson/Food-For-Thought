"""Build the gold feature space: a bijection between ingredient ids and indices.

Indices are assigned in sorted-ingredient-id order so the space is
deterministic. Each feature carries its parent's index (or null for root
ingredients), centralizing the parent back-off map so feature rows and
validators derive it from one place.
"""

from __future__ import annotations

from silver_pipeline.artifact_io import SCHEMA_VERSION


def map_ingredient_ids_to_indices(ingredients_payload: dict) -> dict[str, int]:
    """Map each silver ingredient id to its feature index.

    Args:
        ingredients_payload: Parsed silver ingredients.json document.

    Returns:
        Mapping of ingredient id -> index, assigned in sorted-id order
        from 0 to feature_count - 1.
    """
    sorted_ids = sorted(entry["id"] for entry in ingredients_payload["ingredients"])
    return {
        ingredient_id: index for index, ingredient_id in enumerate(sorted_ids)
    }


def build_feature_space_payload(
    ingredients_payload: dict, fingerprint: dict
) -> dict:
    """Build the gold feature-space artifact from the silver vocabulary.

    Args:
        ingredients_payload: Parsed silver ingredients.json document.
        fingerprint: Gold build block embedded in the artifact.

    Returns:
        Payload with feature_count and one feature record per ingredient
        ({index, ingredient_id, parent_index}), ready for
        write_artifact_json.
    """
    index_by_id = map_ingredient_ids_to_indices(ingredients_payload)
    parent_id_by_id = {
        entry["id"]: entry["parent_id"]
        for entry in ingredients_payload["ingredients"]
    }
    features = []
    for ingredient_id in sorted(index_by_id):
        parent_id = parent_id_by_id[ingredient_id]
        parent_index = None if parent_id is None else index_by_id[parent_id]
        features.append(
            {
                "index": index_by_id[ingredient_id],
                "ingredient_id": ingredient_id,
                "parent_index": parent_index,
            }
        )
    return {
        "build": dict(fingerprint),
        "feature_count": len(features),
        "features": features,
        "schema_version": SCHEMA_VERSION,
    }
