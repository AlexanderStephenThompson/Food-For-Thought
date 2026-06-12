"""Shared builders for the synthetic gold payloads the model tests use.

One tiny, learnable corpus drives every model suite: six features with one
parent link (dark_soy_sauce -> soy_sauce), three cleanly separable cuisines
(italian = basil+pasta, mexican = rice, thai = dark_soy_sauce+fish_sauce),
twelve train rows, three test rows including one empty, and a two-fold
assignment — small enough that scikit-learn fits take milliseconds.
"""

from collections.abc import Sequence

from silver_pipeline.artifact_io import SCHEMA_VERSION

MODEL_BUILD_BLOCK = {
    "cuisines_sha256": "a" * 64,
    "feature_space_sha256": "b" * 64,
    "features_test_sha256": "c" * 64,
    "features_train_sha256": "d" * 64,
    "folds_sha256": "e" * 64,
    "random_seed": 42,
    "sklearn_version": "1.9.0",
}

SYNTHETIC_FOLD_COUNT = 2

_FEATURES = (
    ("basil", None),
    ("dark_soy_sauce", 5),
    ("fish_sauce", None),
    ("pasta", None),
    ("rice", None),
    ("soy_sauce", None),
)

ITALIAN_INDICES = [0, 3]
MEXICAN_INDICES = [4]
THAI_INDICES = [1, 2]
THAI_PARENT_INDICES = [5]


def make_feature_space_payload() -> dict:
    """Build a gold-shaped feature space: 6 features, 1 parent link."""
    features = [
        {"index": index, "ingredient_id": ingredient_id, "parent_index": parent}
        for index, (ingredient_id, parent) in enumerate(_FEATURES)
    ]
    return {
        "build": dict(MODEL_BUILD_BLOCK),
        "feature_count": len(features),
        "features": features,
        "schema_version": SCHEMA_VERSION,
    }


def make_feature_row(
    recipe_id: int,
    ingredient_indices: Sequence[int],
    parent_indices: Sequence[int] = (),
    cuisine: str | None = None,
) -> dict:
    """Build one gold-shaped feature row; omit cuisine for test rows."""
    row = {
        "ingredient_indices": list(ingredient_indices),
        "parent_indices": list(parent_indices),
        "recipe_id": recipe_id,
    }
    if cuisine is not None:
        row["cuisine"] = cuisine
    return row


def make_features_payload(rows: Sequence[dict]) -> dict:
    """Wrap feature rows in the gold payload envelope, sorted by recipe id."""
    return {
        "build": dict(MODEL_BUILD_BLOCK),
        "rows": sorted(rows, key=lambda row: row["recipe_id"]),
        "schema_version": SCHEMA_VERSION,
    }


def make_train_feature_rows() -> list[dict]:
    """Default labeled corpus: italian x4, mexican x4, thai x4 (ids 1-12)."""
    italian = [
        make_feature_row(recipe_id, ITALIAN_INDICES, cuisine="italian")
        for recipe_id in range(1, 5)
    ]
    mexican = [
        make_feature_row(recipe_id, MEXICAN_INDICES, cuisine="mexican")
        for recipe_id in range(5, 9)
    ]
    thai = [
        make_feature_row(
            recipe_id, THAI_INDICES, THAI_PARENT_INDICES, cuisine="thai"
        )
        for recipe_id in range(9, 13)
    ]
    return italian + mexican + thai


def make_test_feature_rows() -> list[dict]:
    """Default unlabeled corpus: italian-ish, thai-ish, and one empty row."""
    return [
        make_feature_row(100, ITALIAN_INDICES),
        make_feature_row(101, THAI_INDICES, THAI_PARENT_INDICES),
        make_feature_row(102, []),
    ]


def make_folds_payload(rows: Sequence[dict] | None = None) -> dict:
    """Build a two-fold assignment alternating within each cuisine."""
    if rows is None:
        rows = make_train_feature_rows()
    position_within_cuisine: dict[str, int] = {}
    assignments = []
    for row in sorted(rows, key=lambda row: row["recipe_id"]):
        cuisine = row["cuisine"]
        position = position_within_cuisine.get(cuisine, 0)
        position_within_cuisine[cuisine] = position + 1
        assignments.append(
            {"fold": position % SYNTHETIC_FOLD_COUNT, "recipe_id": row["recipe_id"]}
        )
    return {
        "assignments": assignments,
        "build": dict(MODEL_BUILD_BLOCK),
        "fold_count": SYNTHETIC_FOLD_COUNT,
        "schema_version": SCHEMA_VERSION,
    }


def make_cuisines_payload() -> dict:
    """Build a silver-shaped cuisines payload with neighbor annotations."""
    cuisines = [
        {
            "distinctive_ingredients": [],
            "family": "mediterranean",
            "id": "italian",
            "neighbors": [{"id": "mexican", "similarity": 0.45}],
            "recipe_count": 4,
        },
        {
            "distinctive_ingredients": [],
            "family": "latin_american",
            "id": "mexican",
            "neighbors": [{"id": "italian", "similarity": 0.45}],
            "recipe_count": 4,
        },
        {
            "distinctive_ingredients": [],
            "family": "southeast_asian",
            "id": "thai",
            "neighbors": [{"id": "mexican", "similarity": 0.3}],
            "recipe_count": 4,
        },
    ]
    return {
        "build": dict(MODEL_BUILD_BLOCK),
        "cuisines": cuisines,
        "schema_version": SCHEMA_VERSION,
    }
