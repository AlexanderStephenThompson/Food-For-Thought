"""Tests for gold_pipeline.assign_folds.

The assignment must be deterministic, exactly balanced within each cuisine
(spread <= 1 by round-robin construction), shuffled within a cuisine, and
isolated between cuisines (per-cuisine keyed seeds).
"""

from collections import Counter

from gold_pipeline.assign_folds import (
    FOLD_COUNT,
    assign_folds_for_cuisine,
    build_folds_payload,
)
from tests.gold_payload_builders import (
    GOLD_BUILD_BLOCK,
    make_recipe_record,
    make_recipes_payload,
    make_train_records,
)


def test_assign_folds_is_deterministic_across_calls():
    recipe_ids = list(range(100, 123))

    first = assign_folds_for_cuisine(recipe_ids, "thai")
    second = assign_folds_for_cuisine(recipe_ids, "thai")

    assert first == second


def test_assign_folds_balances_each_cuisine_within_one_recipe():
    recipe_ids = list(range(23))

    fold_by_recipe_id = assign_folds_for_cuisine(recipe_ids, "italian")

    fold_counts = Counter(fold_by_recipe_id.values())
    assert sum(fold_counts.values()) == 23
    assert max(fold_counts.values()) - min(fold_counts.values()) <= 1


def test_assign_folds_shuffles_within_cuisine():
    recipe_ids = list(range(50))

    fold_by_recipe_id = assign_folds_for_cuisine(recipe_ids, "mexican")

    unshuffled_round_robin = {
        recipe_id: position % FOLD_COUNT
        for position, recipe_id in enumerate(recipe_ids)
    }
    assert fold_by_recipe_id != unshuffled_round_robin


def test_every_train_recipe_lands_in_exactly_one_fold():
    payload = build_folds_payload(
        make_recipes_payload(make_train_records()), GOLD_BUILD_BLOCK
    )

    assigned_ids = [entry["recipe_id"] for entry in payload["assignments"]]
    assert assigned_ids == list(range(1, 18))
    assert all(
        0 <= entry["fold"] < FOLD_COUNT for entry in payload["assignments"]
    )
    assert payload["fold_count"] == FOLD_COUNT
    assert payload["schema_version"] == 1
    assert payload["build"] == GOLD_BUILD_BLOCK


def test_changing_one_cuisine_leaves_other_cuisine_folds_unchanged():
    base_records = make_train_records()
    extra_mexican = [
        make_recipe_record(recipe_id, ["rice"], "mexican")
        for recipe_id in range(200, 210)
    ]

    base_payload = build_folds_payload(
        make_recipes_payload(base_records), GOLD_BUILD_BLOCK
    )
    grown_payload = build_folds_payload(
        make_recipes_payload(base_records + extra_mexican), GOLD_BUILD_BLOCK
    )

    thai_ids = {
        record["id"] for record in base_records if record["cuisine"] == "thai"
    }
    base_thai_folds = {
        entry["recipe_id"]: entry["fold"]
        for entry in base_payload["assignments"]
        if entry["recipe_id"] in thai_ids
    }
    grown_thai_folds = {
        entry["recipe_id"]: entry["fold"]
        for entry in grown_payload["assignments"]
        if entry["recipe_id"] in thai_ids
    }
    assert base_thai_folds == grown_thai_folds
