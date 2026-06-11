"""Tests for gold_pipeline.validate_gold.

Builds a fully valid synthetic gold artifact set through the real builder
modules, then corrupts one field at a time and asserts the matching gate
fires (mirroring the test_validate_silver corruption style).
"""

import copy

import pytest

from gold_pipeline.assign_folds import build_folds_payload
from gold_pipeline.build_feature_space import build_feature_space_payload
from gold_pipeline.build_features import build_features_payload
from gold_pipeline.build_fold_balance_report import build_fold_balance_payload
from gold_pipeline.validate_gold import (
    ValidationError,
    validate_feature_space_payload,
    validate_features_payload,
    validate_fold_balance_payload,
    validate_folds_payload,
    validate_gold_artifacts,
)
from tests.gold_payload_builders import (
    GOLD_BUILD_BLOCK,
    make_ingredients_payload,
    make_recipes_payload,
    make_test_records,
    make_train_records,
)

SYNTHETIC_FEATURE_COUNT = 8
SYNTHETIC_TRAIN_ROW_COUNT = 17
SYNTHETIC_TEST_ROW_COUNT = 3
SYNTHETIC_EMPTY_TEST_ROW_COUNT = 1


def _build_valid_artifacts() -> dict:
    """Build a coherent synthetic gold artifact set plus its silver inputs."""
    ingredients = make_ingredients_payload()
    recipes_train = make_recipes_payload(make_train_records())
    recipes_test = make_recipes_payload(make_test_records())
    feature_space = build_feature_space_payload(ingredients, GOLD_BUILD_BLOCK)
    return {
        "ingredients": ingredients,
        "recipes_train": recipes_train,
        "recipes_test": recipes_test,
        "feature_space": feature_space,
        "features_train": build_features_payload(
            recipes_train, feature_space, GOLD_BUILD_BLOCK, includes_cuisine=True
        ),
        "features_test": build_features_payload(
            recipes_test, feature_space, GOLD_BUILD_BLOCK, includes_cuisine=False
        ),
        "folds": build_folds_payload(recipes_train, GOLD_BUILD_BLOCK),
    }


def _add_fold_balance(artifacts: dict) -> dict:
    artifacts["fold_balance"] = build_fold_balance_payload(
        artifacts["folds"], artifacts["recipes_train"], GOLD_BUILD_BLOCK
    )
    return artifacts


def _validate(artifacts: dict) -> None:
    validate_gold_artifacts(
        artifacts["feature_space"],
        artifacts["features_train"],
        artifacts["features_test"],
        artifacts["folds"],
        artifacts["fold_balance"],
        artifacts["ingredients"],
        artifacts["recipes_train"],
        artifacts["recipes_test"],
        expected_fingerprint=GOLD_BUILD_BLOCK,
        expected_feature_count=SYNTHETIC_FEATURE_COUNT,
        expected_train_row_count=SYNTHETIC_TRAIN_ROW_COUNT,
        expected_test_row_count=SYNTHETIC_TEST_ROW_COUNT,
        expected_empty_test_row_count=SYNTHETIC_EMPTY_TEST_ROW_COUNT,
    )


def test_valid_gold_artifacts_pass():
    artifacts = _add_fold_balance(_build_valid_artifacts())

    _validate(artifacts)


def test_schema_version_mismatch_raises():
    artifacts = _add_fold_balance(_build_valid_artifacts())
    artifacts["feature_space"] = copy.deepcopy(artifacts["feature_space"])
    artifacts["feature_space"]["schema_version"] = 99

    with pytest.raises(ValidationError, match="schema_version"):
        _validate(artifacts)


def test_build_fingerprint_disagreement_between_artifacts_raises():
    artifacts = _add_fold_balance(_build_valid_artifacts())
    artifacts["folds"] = copy.deepcopy(artifacts["folds"])
    artifacts["folds"]["build"]["random_seed"] = 7

    with pytest.raises(ValidationError, match="build"):
        _validate(artifacts)


def test_stale_fingerprint_against_expected_raises():
    artifacts = _add_fold_balance(_build_valid_artifacts())
    stale_fingerprint = dict(GOLD_BUILD_BLOCK, ingredients_sha256="0" * 64)

    with pytest.raises(ValidationError, match="fingerprint"):
        validate_gold_artifacts(
            artifacts["feature_space"],
            artifacts["features_train"],
            artifacts["features_test"],
            artifacts["folds"],
            artifacts["fold_balance"],
            artifacts["ingredients"],
            artifacts["recipes_train"],
            artifacts["recipes_test"],
            expected_fingerprint=stale_fingerprint,
            expected_feature_count=SYNTHETIC_FEATURE_COUNT,
            expected_train_row_count=SYNTHETIC_TRAIN_ROW_COUNT,
            expected_test_row_count=SYNTHETIC_TEST_ROW_COUNT,
            expected_empty_test_row_count=SYNTHETIC_EMPTY_TEST_ROW_COUNT,
        )


def test_nonsequential_feature_index_raises():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["feature_space"])
    corrupted["features"][0]["index"] = 5

    with pytest.raises(ValidationError, match="index"):
        validate_feature_space_payload(
            corrupted, artifacts["ingredients"], SYNTHETIC_FEATURE_COUNT
        )


def test_parent_index_on_parentless_ingredient_raises():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["feature_space"])
    fish_sauce_feature = next(
        feature
        for feature in corrupted["features"]
        if feature["ingredient_id"] == "fish_sauce"
    )
    fish_sauce_feature["parent_index"] = 0

    with pytest.raises(ValidationError, match="parent"):
        validate_feature_space_payload(
            corrupted, artifacts["ingredients"], SYNTHETIC_FEATURE_COUNT
        )


def test_out_of_range_ingredient_index_raises():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["features_train"])
    corrupted["rows"][0]["ingredient_indices"] = [0, 99]

    with pytest.raises(ValidationError, match="ingredient_indices"):
        validate_features_payload(
            corrupted,
            artifacts["feature_space"],
            artifacts["recipes_train"],
            expected_row_count=SYNTHETIC_TRAIN_ROW_COUNT,
            requires_cuisine=True,
            expected_empty_row_count=0,
        )


def test_wrong_parent_indices_raise():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["features_train"])
    thai_row = next(row for row in corrupted["rows"] if row["recipe_id"] == 12)
    thai_row["parent_indices"] = []

    with pytest.raises(ValidationError, match="parent_indices"):
        validate_features_payload(
            corrupted,
            artifacts["feature_space"],
            artifacts["recipes_train"],
            expected_row_count=SYNTHETIC_TRAIN_ROW_COUNT,
            requires_cuisine=True,
            expected_empty_row_count=0,
        )


def test_missing_fold_assignment_raises():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["folds"])
    corrupted["assignments"] = corrupted["assignments"][:-1]

    with pytest.raises(ValidationError, match="assignment"):
        validate_folds_payload(corrupted, artifacts["recipes_train"])


def test_cuisine_fold_imbalance_raises():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["folds"])
    mexican_entries = [
        entry for entry in corrupted["assignments"] if 8 <= entry["recipe_id"] < 12
    ]
    crowded_fold = mexican_entries[0]["fold"]
    for entry in mexican_entries:
        entry["fold"] = crowded_fold

    with pytest.raises(ValidationError, match="spread|balance"):
        validate_folds_payload(corrupted, artifacts["recipes_train"])


def test_empty_test_row_count_above_pinned_raises():
    artifacts = _build_valid_artifacts()

    with pytest.raises(ValidationError, match="empty"):
        validate_features_payload(
            artifacts["features_test"],
            artifacts["feature_space"],
            artifacts["recipes_test"],
            expected_row_count=SYNTHETIC_TEST_ROW_COUNT,
            requires_cuisine=False,
            expected_empty_row_count=0,
        )


def test_fold_balance_report_mismatch_raises():
    artifacts = _add_fold_balance(_build_valid_artifacts())
    corrupted = copy.deepcopy(artifacts["fold_balance"])
    corrupted["fold_sizes"][0] += 1

    with pytest.raises(ValidationError, match="fold_sizes"):
        validate_fold_balance_payload(
            corrupted, artifacts["folds"], artifacts["recipes_train"]
        )
