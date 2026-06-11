"""Tests for pipeline.load_bronze_recipes.

Schema/shape tests run against tiny JSON fixtures under
tests/fixtures/cuisine_divergence/; count-validation tests run against the
real Kaggle files (read-only).
"""

from pathlib import Path

import pytest

from pipeline.load_bronze_recipes import (
    BRONZE_TEST_PATH,
    BRONZE_TRAIN_PATH,
    Recipe,
    SchemaValidationError,
    build_train_index,
    load_test_recipes,
    load_train_recipes,
)

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures" / "cuisine_divergence"
BAD_RECORD_FIXTURE = FIXTURES_DIRECTORY / "train_with_bad_record.json"
WRONG_COUNT_FIXTURE = FIXTURES_DIRECTORY / "train_with_wrong_count.json"
INVALID_JSON_FIXTURE = FIXTURES_DIRECTORY / "train_invalid_json.json"

EXPECTED_TRAIN_COUNT = 39_774
EXPECTED_TEST_COUNT = 9_944
EXPECTED_CUISINE_COUNT = 20


@pytest.fixture(scope="module")
def real_train_recipes() -> list[Recipe]:
    """Load the real train file once for all real-data tests in this module."""
    return load_train_recipes()


def _make_synthetic_recipes() -> list[Recipe]:
    return [
        Recipe(id=1, cuisine="alpha", ingredients=("salt", "soy")),
        Recipe(id=2, cuisine="alpha", ingredients=("soy",)),
        Recipe(id=3, cuisine="beta", ingredients=("salt",)),
        Recipe(id=4, cuisine="gamma", ingredients=("pepper", "salt")),
    ]


def test_load_train_recipes_returns_expected_count(real_train_recipes):
    assert len(real_train_recipes) == EXPECTED_TRAIN_COUNT


def test_load_train_recipes_yields_frozen_recipes_with_cuisine(real_train_recipes):
    first_recipe = real_train_recipes[0]

    assert isinstance(first_recipe, Recipe)
    assert isinstance(first_recipe.id, int)
    assert isinstance(first_recipe.cuisine, str)
    assert isinstance(first_recipe.ingredients, tuple)
    with pytest.raises(AttributeError):
        first_recipe.cuisine = "other"


def test_load_test_recipes_returns_expected_count_with_no_cuisine():
    test_recipes = load_test_recipes()

    assert len(test_recipes) == EXPECTED_TEST_COUNT
    assert all(recipe.cuisine is None for recipe in test_recipes[:50])


def test_load_train_recipes_raises_on_bad_record_schema():
    with pytest.raises(SchemaValidationError):
        load_train_recipes(path=BAD_RECORD_FIXTURE)


def test_load_train_recipes_raises_on_wrong_recipe_count():
    with pytest.raises(SchemaValidationError):
        load_train_recipes(path=WRONG_COUNT_FIXTURE)


def test_load_train_recipes_raises_on_invalid_json():
    with pytest.raises(SchemaValidationError):
        load_train_recipes(path=INVALID_JSON_FIXTURE)


def test_paths_point_at_bronze_kaggle_files():
    assert BRONZE_TRAIN_PATH.name == "train.json"
    assert BRONZE_TEST_PATH.name == "test.json"
    assert BRONZE_TRAIN_PATH.parent.name == "kaggle"


def test_build_train_index_counts_recipes_and_cuisines():
    index = build_train_index(_make_synthetic_recipes())

    assert index.recipe_count == 4
    assert index.cuisine_names == ("alpha", "beta", "gamma")
    assert index.cuisine_recipe_counts == {"alpha": 2, "beta": 1, "gamma": 1}


def test_build_train_index_maps_raw_strings_to_recipe_id_sets():
    index = build_train_index(_make_synthetic_recipes())

    assert index.string_to_recipe_ids["salt"] == frozenset({1, 3, 4})
    assert index.string_to_recipe_ids["soy"] == frozenset({1, 2})
    assert index.recipe_id_to_cuisine[4] == "gamma"


def test_build_train_index_rejects_recipes_without_cuisine():
    unlabeled_recipe = Recipe(id=9, cuisine=None, ingredients=("salt",))

    with pytest.raises(SchemaValidationError):
        build_train_index([unlabeled_recipe])


def test_build_train_index_on_real_data_has_twenty_cuisines(real_train_recipes):
    index = build_train_index(real_train_recipes)

    assert index.recipe_count == EXPECTED_TRAIN_COUNT
    assert len(index.cuisine_names) == EXPECTED_CUISINE_COUNT
    assert sum(index.cuisine_recipe_counts.values()) == EXPECTED_TRAIN_COUNT
