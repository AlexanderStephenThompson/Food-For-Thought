"""Tests for gold_pipeline.build_feature_space.

All tests run on the synthetic 8-ingredient corpus from
gold_payload_builders (two parented variants: dark_soy_sauce -> soy_sauce,
thai_basil -> basil).
"""

from gold_pipeline.build_feature_space import (
    build_feature_space_payload,
    map_ingredient_ids_to_indices,
)
from tests.gold_payload_builders import (
    GOLD_BUILD_BLOCK,
    make_ingredients_payload,
)


def test_feature_space_assigns_indices_in_sorted_ingredient_id_order():
    payload = build_feature_space_payload(make_ingredients_payload(), GOLD_BUILD_BLOCK)

    ingredient_ids = [feature["ingredient_id"] for feature in payload["features"]]
    indices = [feature["index"] for feature in payload["features"]]

    assert ingredient_ids == sorted(ingredient_ids)
    assert indices == list(range(len(ingredient_ids)))


def test_map_ingredient_ids_to_indices_matches_feature_space_order():
    ingredients_payload = make_ingredients_payload()

    index_by_id = map_ingredient_ids_to_indices(ingredients_payload)

    assert index_by_id["basil"] == 0
    assert index_by_id["thai_basil"] == 7
    assert sorted(index_by_id.values()) == list(range(8))


def test_feature_space_maps_parent_index_for_parented_ingredients():
    payload = build_feature_space_payload(make_ingredients_payload(), GOLD_BUILD_BLOCK)

    feature_by_id = {
        feature["ingredient_id"]: feature for feature in payload["features"]
    }

    soy_sauce_index = feature_by_id["soy_sauce"]["index"]
    basil_index = feature_by_id["basil"]["index"]
    assert feature_by_id["dark_soy_sauce"]["parent_index"] == soy_sauce_index
    assert feature_by_id["thai_basil"]["parent_index"] == basil_index


def test_feature_space_leaves_parent_index_null_for_root_ingredients():
    payload = build_feature_space_payload(make_ingredients_payload(), GOLD_BUILD_BLOCK)

    root_parent_indices = {
        feature["ingredient_id"]: feature["parent_index"]
        for feature in payload["features"]
        if feature["ingredient_id"] in ("basil", "fish_sauce", "soy_sauce")
    }

    assert root_parent_indices == {
        "basil": None,
        "fish_sauce": None,
        "soy_sauce": None,
    }


def test_feature_space_embeds_schema_version_and_build_block():
    payload = build_feature_space_payload(make_ingredients_payload(), GOLD_BUILD_BLOCK)

    assert payload["schema_version"] == 1
    assert payload["build"] == GOLD_BUILD_BLOCK
    assert payload["feature_count"] == 8
    assert payload["feature_count"] == len(payload["features"])
