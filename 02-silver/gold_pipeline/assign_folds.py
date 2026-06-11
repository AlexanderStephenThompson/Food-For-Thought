"""Assign every train recipe to one of five stratified cross-validation folds.

Stratification is per cuisine with an independent keyed RNG per cuisine
(random.Random seeded with "<seed>:<cuisine>"), so one cuisine's membership
change can never reshuffle another cuisine's folds. Round-robin assignment
over each cuisine's shuffled order makes per-cuisine fold balance exact by
construction (spread of at most one recipe between folds).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from silver_pipeline.artifact_io import BUILD_RANDOM_SEED, SCHEMA_VERSION

FOLD_COUNT = 5


def assign_folds_for_cuisine(
    recipe_ids: Iterable[int], cuisine_name: str, seed: int = BUILD_RANDOM_SEED
) -> dict[int, int]:
    """Assign one cuisine's recipes to folds, shuffled then round-robin.

    The RNG is keyed by both seed and cuisine name so each cuisine's
    shuffle is independent: adding or removing recipes in one cuisine
    leaves every other cuisine's assignment untouched.

    Args:
        recipe_ids: Train recipe ids belonging to this cuisine.
        cuisine_name: Cuisine label; part of the RNG key.
        seed: Base random seed shared by the whole build.

    Returns:
        Mapping of recipe id -> fold number in [0, FOLD_COUNT).
    """
    shuffled_ids = sorted(recipe_ids)
    generator = random.Random(f"{seed}:{cuisine_name}")
    generator.shuffle(shuffled_ids)
    return {
        recipe_id: position % FOLD_COUNT
        for position, recipe_id in enumerate(shuffled_ids)
    }


def build_folds_payload(recipes_train_payload: dict, fingerprint: dict) -> dict:
    """Build the gold folds artifact over every labeled train recipe.

    Args:
        recipes_train_payload: Parsed silver recipes_train.json document.
        fingerprint: Gold build block embedded in the artifact.

    Returns:
        Payload with fold_count and assignments sorted by recipe id, ready
        for write_artifact_json.
    """
    recipe_ids_by_cuisine: dict[str, list[int]] = defaultdict(list)
    for recipe in recipes_train_payload["recipes"]:
        recipe_ids_by_cuisine[recipe["cuisine"]].append(recipe["id"])

    fold_by_recipe_id: dict[int, int] = {}
    for cuisine_name in sorted(recipe_ids_by_cuisine):
        fold_by_recipe_id.update(
            assign_folds_for_cuisine(
                recipe_ids_by_cuisine[cuisine_name], cuisine_name
            )
        )

    assignments = [
        {"fold": fold_by_recipe_id[recipe_id], "recipe_id": recipe_id}
        for recipe_id in sorted(fold_by_recipe_id)
    ]
    return {
        "assignments": assignments,
        "build": dict(fingerprint),
        "fold_count": FOLD_COUNT,
        "schema_version": SCHEMA_VERSION,
    }
