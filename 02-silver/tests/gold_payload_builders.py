"""Shared builders for the synthetic silver payloads the gold tests use.

Every gold suite composes its inputs from this one tiny corpus: three
cuisines with deliberately uneven recipe counts and eight ingredients of
which two are parented variants, so feature-space, feature-row, fold, and
gate behavior are all exercisable without touching real silver data.
"""

from collections.abc import Sequence

from silver_pipeline.artifact_io import SCHEMA_VERSION

SILVER_BUILD_BLOCK = {
    "lexicon_fingerprint": "b" * 64,
    "random_seed": 42,
    "train_sha256": "a" * 64,
}

GOLD_BUILD_BLOCK = {
    "cuisines_sha256": "c" * 64,
    "fold_count": 5,
    "ingredients_sha256": "d" * 64,
    "random_seed": 42,
    "recipes_test_sha256": "e" * 64,
    "recipes_train_sha256": "f" * 64,
}

PARENTED_INGREDIENT_IDS = ("dark_soy_sauce", "thai_basil")


def make_ingredient_entry(
    ingredient_id: str, parent_id: str | None = None, mention_count: int = 5
) -> dict:
    """Build one silver-shaped ingredient record."""
    return {
        "aliases": [
            {
                "alias": ingredient_id.replace("_", " "),
                "rule": None,
                "source": "canonical_surface_form",
                "train_frequency": mention_count,
            }
        ],
        "category": None,
        "id": ingredient_id,
        "name": ingredient_id.replace("_", " "),
        "parent_id": parent_id,
        "preserve_evidence": None,
        "train_mention_count": mention_count,
    }


def make_ingredients_payload(entries: Sequence[dict] | None = None) -> dict:
    """Build a silver-shaped ingredients payload (default: 8-entry corpus).

    Default corpus, sorted by id with two parented variants:
    basil, dark_soy_sauce (-> soy_sauce), fish_sauce, olive_oil, pasta,
    rice, soy_sauce, thai_basil (-> basil).
    """
    if entries is None:
        entries = [
            make_ingredient_entry("basil"),
            make_ingredient_entry("dark_soy_sauce", parent_id="soy_sauce"),
            make_ingredient_entry("fish_sauce"),
            make_ingredient_entry("olive_oil"),
            make_ingredient_entry("pasta"),
            make_ingredient_entry("rice"),
            make_ingredient_entry("soy_sauce"),
            make_ingredient_entry("thai_basil", parent_id="basil"),
        ]
    return {
        "build": dict(SILVER_BUILD_BLOCK),
        "ingredients": list(entries),
        "schema_version": SCHEMA_VERSION,
    }


def make_recipe_record(
    recipe_id: int, ingredient_ids: Sequence[str], cuisine: str | None = None
) -> dict:
    """Build one silver-shaped recipe record; omit cuisine for test recipes."""
    record = {
        "id": recipe_id,
        "ingredient_ids": list(ingredient_ids),
        "raw_ingredient_count": len(ingredient_ids),
        "unresolved_ingredients": [],
    }
    if cuisine is not None:
        record["cuisine"] = cuisine
    return record


def make_recipes_payload(records: Sequence[dict]) -> dict:
    """Wrap recipe records in the silver payload envelope."""
    return {
        "build": dict(SILVER_BUILD_BLOCK),
        "recipes": sorted(records, key=lambda record: record["id"]),
        "schema_version": SCHEMA_VERSION,
    }


def make_train_records() -> list[dict]:
    """Default labeled corpus: italian x7, mexican x4, thai x6 (ids 1-17)."""
    italian = [
        make_recipe_record(recipe_id, ["basil", "olive_oil", "pasta"], "italian")
        for recipe_id in range(1, 8)
    ]
    mexican = [
        make_recipe_record(recipe_id, ["olive_oil", "rice"], "mexican")
        for recipe_id in range(8, 12)
    ]
    thai = [
        make_recipe_record(
            recipe_id, ["dark_soy_sauce", "fish_sauce", "rice", "thai_basil"], "thai"
        )
        for recipe_id in range(12, 18)
    ]
    return italian + mexican + thai


def make_test_records() -> list[dict]:
    """Default unlabeled corpus: two normal recipes plus one empty one."""
    return [
        make_recipe_record(100, ["pasta", "soy_sauce"]),
        make_recipe_record(101, ["dark_soy_sauce"]),
        make_recipe_record(102, []),
    ]


def make_cuisines_payload() -> dict:
    """Build a silver-shaped cuisines payload matching the train corpus."""
    cuisines = [
        {
            "distinctive_ingredients": [],
            "family": "mediterranean",
            "id": "italian",
            "neighbors": [],
            "recipe_count": 7,
        },
        {
            "distinctive_ingredients": [],
            "family": "latin_american",
            "id": "mexican",
            "neighbors": [],
            "recipe_count": 4,
        },
        {
            "distinctive_ingredients": [],
            "family": "southeast_asian",
            "id": "thai",
            "neighbors": [],
            "recipe_count": 6,
        },
    ]
    return {
        "build": dict(SILVER_BUILD_BLOCK),
        "cuisines": cuisines,
        "schema_version": SCHEMA_VERSION,
    }
