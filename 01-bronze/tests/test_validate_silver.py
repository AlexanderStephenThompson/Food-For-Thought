"""Tests for silver_pipeline.validate_silver.

Each validator gate gets one test that corrupts an otherwise valid tiny
payload and asserts ValidationError names the gate; one end-to-end test
confirms fully valid artifacts pass with expected counts patched down to
fixture size.
"""

import copy

import pytest

import silver_pipeline.validate_silver as validate_silver
from silver_pipeline.artifact_io import SCHEMA_VERSION
from silver_pipeline.validate_silver import (
    EXPECTED_CUISINE_NAMES,
    EXPECTED_TEST_RECIPE_COUNT,
    EXPECTED_TRAIN_RECIPE_COUNT,
    ValidationError,
    validate_ingredients_payload,
    validate_recipes_payload,
    validate_resolution_statistics,
    validate_silver_artifacts,
)

BUILD_FINGERPRINT = {
    "train_sha256": "a" * 64,
    "lexicon_fingerprint": "b" * 64,
    "random_seed": 42,
}

FIXTURE_TRAIN_RECIPE_COUNT = 2
FIXTURE_TEST_RECIPE_COUNT = 1


def _build_ingredients_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "build": dict(BUILD_FINGERPRINT),
        "ingredients": [
            {
                "id": "feta_cheese",
                "name": "feta cheese",
                "category": None,
                "parent_id": None,
                "train_mention_count": 1,
                "preserve_evidence": None,
                "aliases": [
                    {
                        "alias": "feta cheese",
                        "source": "canonical_surface_form",
                        "rule": None,
                        "train_frequency": 1,
                    }
                ],
            },
            {
                "id": "green_onion",
                "name": "green onion",
                "category": None,
                "parent_id": None,
                "train_mention_count": 1,
                "preserve_evidence": None,
                "aliases": [
                    {
                        "alias": "green onion",
                        "source": "canonical_surface_form",
                        "rule": None,
                        "train_frequency": 1,
                    },
                    {
                        "alias": "scallions",
                        "source": "manual_alias",
                        "rule": None,
                        "train_frequency": 1,
                    },
                ],
            },
            {
                "id": "low_sodium_soy_sauce",
                "name": "low sodium soy sauce",
                "category": None,
                "parent_id": "soy_sauce",
                "train_mention_count": 1,
                "preserve_evidence": {
                    "layer": "statistical_gate",
                    "jsd_bits": 0.41,
                    "null95_bits": 0.2,
                    "variant_count": 12,
                },
                "aliases": [
                    {
                        "alias": "low sodium soy sauce",
                        "source": "canonical_surface_form",
                        "rule": None,
                        "train_frequency": 1,
                    }
                ],
            },
            {
                "id": "soy_sauce",
                "name": "soy sauce",
                "category": None,
                "parent_id": None,
                "train_mention_count": 2,
                "preserve_evidence": None,
                "aliases": [
                    {
                        "alias": "soy sauce",
                        "source": "canonical_surface_form",
                        "rule": None,
                        "train_frequency": 2,
                    }
                ],
            },
        ],
    }


def _build_train_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "build": dict(BUILD_FINGERPRINT),
        "recipes": [
            {
                "id": 1,
                "cuisine": "greek",
                "ingredient_ids": ["feta_cheese", "soy_sauce"],
                "unresolved_ingredients": [],
                "raw_ingredient_count": 2,
            },
            {
                "id": 2,
                "cuisine": "thai",
                "ingredient_ids": ["soy_sauce", "green_onion"],
                "unresolved_ingredients": ["mystery goo"],
                "raw_ingredient_count": 3,
            },
        ],
    }


def _build_test_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "build": dict(BUILD_FINGERPRINT),
        "recipes": [
            {
                "id": 7,
                "ingredient_ids": ["low_sodium_soy_sauce"],
                "unresolved_ingredients": [],
                "raw_ingredient_count": 1,
            }
        ],
    }


def _build_statistics() -> dict:
    return {
        "train": {
            "mentions_total": 1000,
            "by_method": {
                "exact_alias": 900,
                "cleaned_match": 50,
                "modifier_stripped_match": 30,
                "brand_resolved_match": 15,
                "token_drop_match": 2,
                "unresolved": 3,
            },
            "top_unresolved": [{"string": "mystery goo", "count": 3}],
        },
        "test": {
            "mentions_total": 200,
            "by_method": {
                "exact_alias": 190,
                "cleaned_match": 5,
                "modifier_stripped_match": 2,
                "brand_resolved_match": 0,
                "token_drop_match": 1,
                "unresolved": 2,
            },
            "top_unresolved": [{"string": "mystery goo", "count": 2}],
        },
    }


def _patch_expected_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_silver, "EXPECTED_TRAIN_RECIPE_COUNT", FIXTURE_TRAIN_RECIPE_COUNT
    )
    monkeypatch.setattr(
        validate_silver, "EXPECTED_TEST_RECIPE_COUNT", FIXTURE_TEST_RECIPE_COUNT
    )


def test_expected_constants_match_pinned_values():
    assert EXPECTED_TRAIN_RECIPE_COUNT == 39774
    assert EXPECTED_TEST_RECIPE_COUNT == 9944
    assert len(EXPECTED_CUISINE_NAMES) == 20
    assert "southern_us" in EXPECTED_CUISINE_NAMES


def test_valid_ingredients_payload_passes():
    validate_ingredients_payload(_build_ingredients_payload())


def test_bad_slug_raises():
    payload = _build_ingredients_payload()
    payload["ingredients"][0]["id"] = "Feta-Cheese"

    with pytest.raises(ValidationError, match="Feta-Cheese"):
        validate_ingredients_payload(payload)


def test_duplicate_ingredient_id_raises():
    payload = _build_ingredients_payload()
    payload["ingredients"][1] = copy.deepcopy(payload["ingredients"][0])

    with pytest.raises(ValidationError, match="sorted|duplicate"):
        validate_ingredients_payload(payload)


def test_unsorted_ingredients_raise():
    payload = _build_ingredients_payload()
    payload["ingredients"].reverse()

    with pytest.raises(ValidationError, match="sorted"):
        validate_ingredients_payload(payload)


def test_alias_mapped_to_two_ingredients_raises():
    payload = _build_ingredients_payload()
    payload["ingredients"][0]["aliases"][0]["alias"] = "soy sauce"

    with pytest.raises(ValidationError, match="alias"):
        validate_ingredients_payload(payload)


def test_unknown_alias_source_raises():
    payload = _build_ingredients_payload()
    payload["ingredients"][0]["aliases"][0]["source"] = "telepathy"

    with pytest.raises(ValidationError, match="telepathy"):
        validate_ingredients_payload(payload)


def test_dangling_parent_raises():
    payload = _build_ingredients_payload()
    payload["ingredients"][2]["parent_id"] = "phantom_sauce"

    with pytest.raises(ValidationError, match="phantom_sauce"):
        validate_ingredients_payload(payload)


def test_parent_cycle_raises():
    payload = _build_ingredients_payload()
    payload["ingredients"][2]["parent_id"] = "soy_sauce"
    payload["ingredients"][3]["parent_id"] = "low_sodium_soy_sauce"

    with pytest.raises(ValidationError, match="cycle"):
        validate_ingredients_payload(payload)


def test_parent_depth_three_raises():
    payload = _build_ingredients_payload()
    # Chain feta_cheese -> green_onion -> soy_sauce is depth 3.
    payload["ingredients"][0]["parent_id"] = "green_onion"
    payload["ingredients"][1]["parent_id"] = "soy_sauce"

    with pytest.raises(ValidationError, match="depth"):
        validate_ingredients_payload(payload)


def test_valid_recipes_payload_passes():
    validate_recipes_payload(
        _build_train_payload(),
        _build_ingredients_payload(),
        FIXTURE_TRAIN_RECIPE_COUNT,
        True,
    )


def test_wrong_recipe_count_raises():
    with pytest.raises(ValidationError, match="count"):
        validate_recipes_payload(
            _build_train_payload(), _build_ingredients_payload(), 3, True
        )


def test_unknown_cuisine_raises():
    payload = _build_train_payload()
    payload["recipes"][0]["cuisine"] = "klingon"

    with pytest.raises(ValidationError, match="klingon"):
        validate_recipes_payload(
            payload, _build_ingredients_payload(), FIXTURE_TRAIN_RECIPE_COUNT, True
        )


def test_cuisine_present_in_test_split_raises():
    payload = _build_test_payload()
    payload["recipes"][0]["cuisine"] = "greek"

    with pytest.raises(ValidationError, match="cuisine"):
        validate_recipes_payload(
            payload, _build_ingredients_payload(), FIXTURE_TEST_RECIPE_COUNT, False
        )


def test_dangling_ingredient_id_raises():
    payload = _build_train_payload()
    payload["recipes"][0]["ingredient_ids"].append("phantom_sauce")

    with pytest.raises(ValidationError, match="phantom_sauce"):
        validate_recipes_payload(
            payload, _build_ingredients_payload(), FIXTURE_TRAIN_RECIPE_COUNT, True
        )


def test_duplicate_ingredient_ids_in_recipe_raise():
    payload = _build_train_payload()
    payload["recipes"][0]["ingredient_ids"] = ["soy_sauce", "soy_sauce"]

    with pytest.raises(ValidationError, match="duplicate"):
        validate_recipes_payload(
            payload, _build_ingredients_payload(), FIXTURE_TRAIN_RECIPE_COUNT, True
        )


def test_train_recipe_with_no_resolved_ids_raises():
    payload = _build_train_payload()
    payload["recipes"][0]["ingredient_ids"] = []

    with pytest.raises(ValidationError, match="no resolved"):
        validate_recipes_payload(
            payload, _build_ingredients_payload(), FIXTURE_TRAIN_RECIPE_COUNT, True
        )


def test_empty_unresolved_string_raises():
    payload = _build_train_payload()
    payload["recipes"][1]["unresolved_ingredients"] = [""]

    with pytest.raises(ValidationError, match="unresolved"):
        validate_recipes_payload(
            payload, _build_ingredients_payload(), FIXTURE_TRAIN_RECIPE_COUNT, True
        )


def test_recipes_out_of_order_raise():
    payload = _build_train_payload()
    payload["recipes"].reverse()

    with pytest.raises(ValidationError, match="sorted"):
        validate_recipes_payload(
            payload, _build_ingredients_payload(), FIXTURE_TRAIN_RECIPE_COUNT, True
        )


def test_valid_statistics_pass():
    validate_resolution_statistics(_build_statistics())


def test_alias_tier_coverage_below_threshold_raises():
    statistics = _build_statistics()
    statistics["train"]["by_method"]["exact_alias"] = 980
    statistics["train"]["by_method"]["cleaned_match"] = 0
    statistics["train"]["by_method"]["modifier_stripped_match"] = 0
    statistics["train"]["by_method"]["brand_resolved_match"] = 0
    statistics["train"]["by_method"]["token_drop_match"] = 17

    with pytest.raises(ValidationError, match="alias tier"):
        validate_resolution_statistics(statistics)


def test_train_full_chain_coverage_below_threshold_raises():
    # Alias tier 0.989 passes its 0.988 gate, but the full chain at 0.989
    # sits below the 0.990 train gate.
    statistics = _build_statistics()
    statistics["train"]["by_method"]["exact_alias"] = 988
    statistics["train"]["by_method"]["cleaned_match"] = 1
    statistics["train"]["by_method"]["modifier_stripped_match"] = 0
    statistics["train"]["by_method"]["brand_resolved_match"] = 0
    statistics["train"]["by_method"]["token_drop_match"] = 0
    statistics["train"]["by_method"]["unresolved"] = 11

    with pytest.raises(ValidationError, match="full chain"):
        validate_resolution_statistics(statistics)


def test_test_full_chain_coverage_below_threshold_raises():
    statistics = _build_statistics()
    statistics["test"]["by_method"]["modifier_stripped_match"] = 0
    statistics["test"]["by_method"]["unresolved"] = 4

    with pytest.raises(ValidationError, match="full chain"):
        validate_resolution_statistics(statistics)


def test_by_method_sum_mismatch_raises():
    statistics = _build_statistics()
    statistics["train"]["mentions_total"] = 999

    with pytest.raises(ValidationError, match="mentions_total"):
        validate_resolution_statistics(statistics)


def test_valid_artifacts_pass(monkeypatch):
    _patch_expected_counts(monkeypatch)

    validate_silver_artifacts(
        _build_ingredients_payload(),
        _build_train_payload(),
        _build_test_payload(),
        _build_statistics(),
    )


def test_fingerprint_mismatch_raises(monkeypatch):
    _patch_expected_counts(monkeypatch)
    test_payload = _build_test_payload()
    test_payload["build"]["train_sha256"] = "c" * 64

    with pytest.raises(ValidationError, match="fingerprint"):
        validate_silver_artifacts(
            _build_ingredients_payload(),
            _build_train_payload(),
            test_payload,
            _build_statistics(),
        )
