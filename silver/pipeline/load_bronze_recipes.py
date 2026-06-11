"""Load the bronze Kaggle What's Cooking JSON files into validated Recipe records.

Provides frozen Recipe records, fail-fast schema validation against the known
Kaggle dataset shape (39,774 labeled train recipes across exactly 20 cuisines;
9,944 unlabeled test recipes), and a TrainIndex that maps raw ingredient
strings to the recipes containing them for downstream divergence testing.

No file I/O happens at import time: callers invoke load_train_recipes /
load_test_recipes explicitly and pass the resulting structures onward.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from silver.pipeline import locations

BRONZE_TRAIN_PATH = locations.BRONZE_KAGGLE_DIRECTORY / "train.json"
BRONZE_TEST_PATH = locations.BRONZE_KAGGLE_DIRECTORY / "test.json"

EXPECTED_TRAIN_RECIPE_COUNT = 39_774
EXPECTED_TEST_RECIPE_COUNT = 9_944
EXPECTED_CUISINE_COUNT = 20

BRONZE_FILE_ENCODING = "utf-8"


class SchemaValidationError(ValueError):
    """Raised when a bronze recipe file violates the expected Kaggle schema.

    Covers malformed JSON, records with missing or mistyped fields, and
    dataset-level violations (wrong recipe count, wrong cuisine count).
    """


@dataclass(frozen=True)
class Recipe:
    """One bronze recipe exactly as read from the Kaggle JSON.

    Attributes:
        id: Kaggle recipe identifier.
        cuisine: Cuisine label for train recipes; None for test recipes.
        ingredients: Raw ingredient strings, untouched (no normalization).
    """

    id: int
    cuisine: str | None
    ingredients: tuple[str, ...]


@dataclass(frozen=True)
class TrainIndex:
    """Lookup structures over the labeled train recipes.

    Attributes:
        recipe_count: Total number of indexed recipes.
        cuisine_names: Sorted tuple of distinct cuisine labels.
        cuisine_recipe_counts: Recipes per cuisine, keyed by cuisine name.
        string_to_recipe_ids: Raw ingredient string -> ids of recipes
            containing it (whole-string keys, never substrings).
        recipe_id_to_cuisine: Recipe id -> its cuisine label.
    """

    recipe_count: int
    cuisine_names: tuple[str, ...]
    cuisine_recipe_counts: dict[str, int]
    string_to_recipe_ids: dict[str, frozenset[int]]
    recipe_id_to_cuisine: dict[int, str]


def load_train_recipes(path: Path = BRONZE_TRAIN_PATH) -> list[Recipe]:
    """Load and validate the labeled Kaggle train recipes.

    Args:
        path: JSON file to read; defaults to bronze/kaggle/train.json.

    Returns:
        All train recipes in file order, each with a cuisine label.

    Raises:
        SchemaValidationError: If the JSON is malformed, any record violates
            the {id, cuisine, ingredients} schema, the recipe count is not
            39,774, or the distinct cuisine count is not exactly 20.
        FileNotFoundError: If the file does not exist.
    """
    records = _read_json_records(path)
    recipes = [
        _parse_record(record, position, is_cuisine_required=True)
        for position, record in enumerate(records)
    ]
    if len(recipes) != EXPECTED_TRAIN_RECIPE_COUNT:
        raise SchemaValidationError(
            f"{path}: expected {EXPECTED_TRAIN_RECIPE_COUNT} train recipes, "
            f"found {len(recipes)}"
        )
    distinct_cuisines = {recipe.cuisine for recipe in recipes}
    if len(distinct_cuisines) != EXPECTED_CUISINE_COUNT:
        raise SchemaValidationError(
            f"{path}: expected {EXPECTED_CUISINE_COUNT} distinct cuisines, "
            f"found {len(distinct_cuisines)}"
        )
    return recipes


def load_test_recipes(path: Path = BRONZE_TEST_PATH) -> list[Recipe]:
    """Load and validate the unlabeled Kaggle test recipes.

    Args:
        path: JSON file to read; defaults to bronze/kaggle/test.json.

    Returns:
        All test recipes in file order, each with cuisine set to None.

    Raises:
        SchemaValidationError: If the JSON is malformed, any record violates
            the {id, ingredients} schema, or the count is not 9,944.
        FileNotFoundError: If the file does not exist.
    """
    records = _read_json_records(path)
    recipes = [
        _parse_record(record, position, is_cuisine_required=False)
        for position, record in enumerate(records)
    ]
    if len(recipes) != EXPECTED_TEST_RECIPE_COUNT:
        raise SchemaValidationError(
            f"{path}: expected {EXPECTED_TEST_RECIPE_COUNT} test recipes, "
            f"found {len(recipes)}"
        )
    return recipes


def build_train_index(recipes: Sequence[Recipe]) -> TrainIndex:
    """Build lookup structures over labeled recipes for divergence testing.

    Args:
        recipes: Labeled recipes; every cuisine must be a string, ids unique.

    Returns:
        A frozen TrainIndex with sorted cuisine names and whole-string
        ingredient -> recipe-id mappings.

    Raises:
        SchemaValidationError: If recipes is empty, any recipe lacks a
            cuisine label, or two recipes share an id.
    """
    if not recipes:
        raise SchemaValidationError("cannot build a train index from zero recipes")
    cuisine_recipe_counts: dict[str, int] = {}
    recipe_id_to_cuisine: dict[int, str] = {}
    for recipe in recipes:
        if recipe.cuisine is None:
            raise SchemaValidationError(
                f"recipe {recipe.id} has no cuisine label; only labeled "
                "train recipes can be indexed"
            )
        recipe_id_to_cuisine[recipe.id] = recipe.cuisine
        cuisine_recipe_counts[recipe.cuisine] = (
            cuisine_recipe_counts.get(recipe.cuisine, 0) + 1
        )
    if len(recipe_id_to_cuisine) != len(recipes):
        raise SchemaValidationError("duplicate recipe ids in train recipes")
    cuisine_names = tuple(sorted(cuisine_recipe_counts))
    return TrainIndex(
        recipe_count=len(recipes),
        cuisine_names=cuisine_names,
        cuisine_recipe_counts={name: cuisine_recipe_counts[name] for name in cuisine_names},
        string_to_recipe_ids=_collect_string_to_recipe_ids(recipes),
        recipe_id_to_cuisine=recipe_id_to_cuisine,
    )


def _read_json_records(path: Path) -> list[object]:
    """Read a JSON file and require a top-level list of records."""
    try:
        with path.open(encoding=BRONZE_FILE_ENCODING) as bronze_file:
            payload = json.load(bronze_file)
    except json.JSONDecodeError as error:
        raise SchemaValidationError(f"{path}: invalid JSON ({error})") from error
    if not isinstance(payload, list):
        raise SchemaValidationError(
            f"{path}: expected a top-level JSON list, got {type(payload).__name__}"
        )
    return payload


def _parse_record(record: object, position: int, is_cuisine_required: bool) -> Recipe:
    """Validate one bronze record and convert it to a frozen Recipe."""
    if not isinstance(record, dict):
        raise SchemaValidationError(
            f"record {position}: expected an object, got {type(record).__name__}"
        )
    recipe_id = record.get("id")
    if isinstance(recipe_id, bool) or not isinstance(recipe_id, int):
        raise SchemaValidationError(f"record {position}: 'id' must be an integer")
    ingredients = record.get("ingredients")
    if not isinstance(ingredients, list) or not ingredients:
        raise SchemaValidationError(
            f"record {position}: 'ingredients' must be a non-empty list"
        )
    if any(not isinstance(ingredient, str) for ingredient in ingredients):
        raise SchemaValidationError(
            f"record {position}: every ingredient must be a string"
        )
    cuisine = None
    if is_cuisine_required:
        cuisine = record.get("cuisine")
        if not isinstance(cuisine, str) or not cuisine:
            raise SchemaValidationError(
                f"record {position}: 'cuisine' must be a non-empty string"
            )
    return Recipe(id=recipe_id, cuisine=cuisine, ingredients=tuple(ingredients))


def _collect_string_to_recipe_ids(
    recipes: Sequence[Recipe],
) -> dict[str, frozenset[int]]:
    """Map each raw ingredient string to the ids of recipes containing it."""
    mutable_ids: dict[str, set[int]] = {}
    for recipe in recipes:
        for ingredient in recipe.ingredients:
            mutable_ids.setdefault(ingredient, set()).add(recipe.id)
    # Sorted keys keep index iteration order deterministic downstream.
    return {
        ingredient: frozenset(mutable_ids[ingredient])
        for ingredient in sorted(mutable_ids)
    }
