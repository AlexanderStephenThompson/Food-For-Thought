"""Tests for gold_pipeline.build_features.

Feature-space indices for the default corpus (sorted ingredient ids):
basil=0, dark_soy_sauce=1, fish_sauce=2, olive_oil=3, pasta=4, rice=5,
soy_sauce=6, thai_basil=7. Parents: dark_soy_sauce->soy_sauce(6),
thai_basil->basil(0).
"""

from gold_pipeline.build_feature_space import build_feature_space_payload
from gold_pipeline.build_features import build_features_payload
from tests.gold_payload_builders import (
    GOLD_BUILD_BLOCK,
    make_ingredient_entry,
    make_ingredients_payload,
    make_recipe_record,
    make_recipes_payload,
    make_test_records,
    make_train_records,
)


def _default_feature_space():
    return build_feature_space_payload(make_ingredients_payload(), GOLD_BUILD_BLOCK)


def test_feature_row_carries_sorted_unique_ingredient_indices():
    recipes = make_recipes_payload(
        [make_recipe_record(1, ["pasta", "basil", "olive_oil"], "italian")]
    )

    payload = build_features_payload(
        recipes, _default_feature_space(), GOLD_BUILD_BLOCK, includes_cuisine=True
    )

    assert payload["rows"][0]["ingredient_indices"] == [0, 3, 4]


def test_feature_row_collects_parent_indices_of_variants():
    recipes = make_recipes_payload(
        [
            make_recipe_record(
                12, ["dark_soy_sauce", "fish_sauce", "rice", "thai_basil"], "thai"
            )
        ]
    )

    payload = build_features_payload(
        recipes, _default_feature_space(), GOLD_BUILD_BLOCK, includes_cuisine=True
    )

    assert payload["rows"][0]["ingredient_indices"] == [1, 2, 5, 7]
    assert payload["rows"][0]["parent_indices"] == [0, 6]


def test_feature_row_deduplicates_shared_parent_indices():
    ingredients = make_ingredients_payload(
        [
            make_ingredient_entry("dark_soy_sauce", parent_id="soy_sauce"),
            make_ingredient_entry("light_soy_sauce", parent_id="soy_sauce"),
            make_ingredient_entry("soy_sauce"),
        ]
    )
    feature_space = build_feature_space_payload(ingredients, GOLD_BUILD_BLOCK)
    recipes = make_recipes_payload(
        [make_recipe_record(1, ["dark_soy_sauce", "light_soy_sauce"], "chinese")]
    )

    payload = build_features_payload(
        recipes, feature_space, GOLD_BUILD_BLOCK, includes_cuisine=True
    )

    soy_sauce_index = 2
    assert payload["rows"][0]["parent_indices"] == [soy_sauce_index]


def test_feature_row_without_parented_ingredients_has_empty_parent_indices():
    recipes = make_recipes_payload(
        [make_recipe_record(1, ["basil", "olive_oil", "pasta"], "italian")]
    )

    payload = build_features_payload(
        recipes, _default_feature_space(), GOLD_BUILD_BLOCK, includes_cuisine=True
    )

    assert payload["rows"][0]["parent_indices"] == []


def test_train_rows_carry_cuisine_and_test_rows_do_not():
    train_payload = build_features_payload(
        make_recipes_payload(make_train_records()),
        _default_feature_space(),
        GOLD_BUILD_BLOCK,
        includes_cuisine=True,
    )
    test_payload = build_features_payload(
        make_recipes_payload(make_test_records()),
        _default_feature_space(),
        GOLD_BUILD_BLOCK,
        includes_cuisine=False,
    )

    assert all("cuisine" in row for row in train_payload["rows"])
    assert train_payload["rows"][0]["cuisine"] == "italian"
    assert all("cuisine" not in row for row in test_payload["rows"])


def test_rows_are_sorted_by_recipe_id():
    shuffled_records = [
        make_recipe_record(9, ["rice"], "mexican"),
        make_recipe_record(2, ["pasta"], "italian"),
        make_recipe_record(15, ["fish_sauce"], "thai"),
    ]

    payload = build_features_payload(
        make_recipes_payload(shuffled_records),
        _default_feature_space(),
        GOLD_BUILD_BLOCK,
        includes_cuisine=True,
    )

    assert [row["recipe_id"] for row in payload["rows"]] == [2, 9, 15]


def test_empty_test_recipe_produces_empty_index_lists():
    payload = build_features_payload(
        make_recipes_payload(make_test_records()),
        _default_feature_space(),
        GOLD_BUILD_BLOCK,
        includes_cuisine=False,
    )

    empty_row = next(row for row in payload["rows"] if row["recipe_id"] == 102)
    assert empty_row["ingredient_indices"] == []
    assert empty_row["parent_indices"] == []


def test_features_payload_embeds_schema_version_and_build_block():
    payload = build_features_payload(
        make_recipes_payload(make_train_records()),
        _default_feature_space(),
        GOLD_BUILD_BLOCK,
        includes_cuisine=True,
    )

    assert payload["schema_version"] == 1
    assert payload["build"] == GOLD_BUILD_BLOCK
    assert len(payload["rows"]) == 17
