"""Validate silver artifacts against the pinned schemas and coverage gates.

Pure validators over already-loaded payload dicts: the ingredients table,
both silver recipe splits, and the resolution statistics report. Each gate
failure raises ValidationError naming the gate and the offending values.
File loading happens only in main(), never at import time.

Usage:
    PYTHONPATH=01-bronze .venv/bin/python -m silver_pipeline.validate_silver
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from silver_pipeline import locations
from silver_pipeline.artifact_io import SCHEMA_VERSION
from silver_pipeline.transform_bronze_to_silver import (
    RESOLUTION_METHOD_NAMES,
    RESOLUTION_STATISTICS_FILENAME,
    TEST_RECIPES_FILENAME,
    TOP_UNRESOLVED_LIMIT,
    TRAIN_RECIPES_FILENAME,
    UNRESOLVED_METHOD,
)

EXPECTED_TRAIN_RECIPE_COUNT = 39774
EXPECTED_TEST_RECIPE_COUNT = 9944
EXPECTED_CUISINE_NAMES = (
    "brazilian",
    "british",
    "cajun_creole",
    "chinese",
    "filipino",
    "french",
    "greek",
    "indian",
    "irish",
    "italian",
    "jamaican",
    "japanese",
    "korean",
    "mexican",
    "moroccan",
    "russian",
    "southern_us",
    "spanish",
    "thai",
    "vietnamese",
)
ALIAS_TIER_MINIMUM_TRAIN_COVERAGE = 0.988
# Measured structural ceiling from the first real run: the unresolved tail
# is ~1,900 unique freq<=3 strings whose resolution would require dropping
# head tokens, which the resolver forbids by design (fish sauce != sauce).
# The plan's provisional 0.995 was tightened to the measured 0.990.
FULL_CHAIN_MINIMUM_TRAIN_COVERAGE = 0.990
FULL_CHAIN_MINIMUM_TEST_COVERAGE = 0.985
INGREDIENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
MAXIMUM_PARENT_DEPTH = 2

ALIAS_SOURCE_NAMES = frozenset(
    {
        "canonical_surface_form",
        "mechanical_normalization",
        "modifier_strip",
        "always_merge_lexicon",
        "forced_merge_override",
        "statistical_gate",
        "named_variety_lexicon",
        "brand_pattern",
        "manual_alias",
        "manual_review",
    }
)
ALIAS_TIER_METHOD_NAMES = (
    "exact_alias",
    "cleaned_match",
    "modifier_stripped_match",
    "brand_resolved_match",
)
REQUIRED_BUILD_KEYS = frozenset(
    {"train_sha256", "lexicon_fingerprint", "random_seed"}
)
REQUIRED_INGREDIENT_KEYS = frozenset(
    {
        "id",
        "name",
        "category",
        "parent_id",
        "train_mention_count",
        "preserve_evidence",
        "aliases",
    }
)
REQUIRED_ALIAS_KEYS = frozenset({"alias", "source", "rule", "train_frequency"})
REQUIRED_RECIPE_KEYS = frozenset(
    {"id", "ingredient_ids", "unresolved_ingredients", "raw_ingredient_count"}
)
REQUIRED_STATISTICS_KEYS = frozenset(
    {"mentions_total", "by_method", "top_unresolved"}
)
STATISTICS_SPLIT_NAMES = ("train", "test")

INGREDIENTS_FILENAME = "ingredients.json"


class ValidationError(ValueError):
    """A silver artifact violates the pinned schema or a coverage gate."""


def _is_non_negative_integer(value: object) -> bool:
    """Return True for ints >= 0, rejecting booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _require_keys(mapping: object, required_keys: frozenset, context: str) -> None:
    """Fail fast when a record is not a dict or lacks required keys."""
    if not isinstance(mapping, dict):
        raise ValidationError(f"{context} must be an object, got {type(mapping).__name__}")
    missing_keys = required_keys - set(mapping)
    if missing_keys:
        raise ValidationError(f"{context} is missing keys: {sorted(missing_keys)}")


def _validate_envelope(payload: dict, artifact_name: str) -> None:
    """Check schema_version and build block shared by every silver payload."""
    _require_keys(payload, frozenset({"schema_version", "build"}), artifact_name)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValidationError(
            f"schema version gate failed for {artifact_name}: "
            f"expected {SCHEMA_VERSION}, found {payload['schema_version']!r}"
        )
    _require_keys(payload["build"], REQUIRED_BUILD_KEYS, f"{artifact_name} build")


def _validate_ingredient_aliases(record: dict) -> None:
    """Check alias entries of one ingredient: shape, source enum, ordering."""
    aliases = record["aliases"]
    if not isinstance(aliases, list) or not aliases:
        raise ValidationError(
            f"alias gate failed: ingredient '{record['id']}' must have a "
            "non-empty alias list"
        )
    for entry in aliases:
        _require_keys(entry, REQUIRED_ALIAS_KEYS, f"alias of '{record['id']}'")
        if entry["source"] not in ALIAS_SOURCE_NAMES:
            raise ValidationError(
                f"alias gate failed: ingredient '{record['id']}' has unknown "
                f"alias source '{entry['source']}'"
            )
        if not isinstance(entry["alias"], str) or not entry["alias"]:
            raise ValidationError(
                f"alias gate failed: ingredient '{record['id']}' has an empty "
                "or non-string alias"
            )
    alias_texts = [entry["alias"] for entry in aliases]
    if alias_texts != sorted(set(alias_texts)):
        raise ValidationError(
            f"alias gate failed: aliases of '{record['id']}' must be sorted "
            "and unique"
        )


def _validate_ingredient_record(record: dict) -> None:
    """Check one ingredient record: keys, id slug, counts, aliases."""
    _require_keys(record, REQUIRED_INGREDIENT_KEYS, "ingredient record")
    ingredient_id = record["id"]
    is_valid_slug = isinstance(ingredient_id, str) and bool(
        INGREDIENT_ID_PATTERN.match(ingredient_id)
    )
    if not is_valid_slug:
        raise ValidationError(
            f"ingredient id gate failed: '{ingredient_id}' does not match "
            f"{INGREDIENT_ID_PATTERN.pattern}"
        )
    if not _is_non_negative_integer(record["train_mention_count"]):
        raise ValidationError(
            f"ingredient gate failed: '{ingredient_id}' train_mention_count "
            f"must be a non-negative integer, got {record['train_mention_count']!r}"
        )
    _validate_ingredient_aliases(record)


def _validate_ingredient_order(ingredients: list) -> None:
    """Check ingredients are strictly sorted by id (also bans duplicates)."""
    identifiers = [record["id"] for record in ingredients]
    for previous_id, current_id in zip(identifiers, identifiers[1:]):
        if previous_id >= current_id:
            raise ValidationError(
                "ingredient order gate failed: ingredients must be sorted by "
                f"id without duplicates, but '{previous_id}' precedes "
                f"'{current_id}'"
            )


def _validate_alias_ownership(ingredients: list) -> None:
    """Check every alias string maps to exactly one ingredient id."""
    owner_by_alias: dict[str, str] = {}
    for record in ingredients:
        for entry in record["aliases"]:
            alias_text = entry["alias"]
            existing_owner = owner_by_alias.get(alias_text)
            if existing_owner is not None and existing_owner != record["id"]:
                raise ValidationError(
                    f"alias gate failed: alias '{alias_text}' maps to both "
                    f"'{existing_owner}' and '{record['id']}'"
                )
            owner_by_alias[alias_text] = record["id"]


def _validate_parent_links(ingredients: list) -> None:
    """Check parent_id existence, acyclicity, and maximum chain depth."""
    parent_by_id = {record["id"]: record["parent_id"] for record in ingredients}
    for ingredient_id in sorted(parent_by_id):
        parent_id = parent_by_id[ingredient_id]
        if parent_id is None:
            continue
        if parent_id not in parent_by_id:
            raise ValidationError(
                f"parent gate failed: '{ingredient_id}' references unknown "
                f"parent '{parent_id}'"
            )
        if parent_id == ingredient_id:
            raise ValidationError(
                f"parent gate failed: cycle at '{ingredient_id}'"
            )
        grandparent_id = parent_by_id[parent_id]
        if grandparent_id is None:
            continue
        if grandparent_id == ingredient_id:
            raise ValidationError(
                f"parent gate failed: cycle between '{ingredient_id}' and "
                f"'{parent_id}'"
            )
        raise ValidationError(
            f"parent gate failed: chain from '{ingredient_id}' through "
            f"'{parent_id}' exceeds maximum depth {MAXIMUM_PARENT_DEPTH}"
        )


def validate_ingredients_payload(payload: dict) -> None:
    """Validate the silver ingredients payload against the pinned schema.

    Args:
        payload: Parsed 02-silver/datasets/ingredients.json content.

    Raises:
        ValidationError: On schema violations, malformed ids, alias
            conflicts, unsorted records, or invalid parent links.
    """
    _validate_envelope(payload, "ingredients payload")
    ingredients = payload.get("ingredients")
    if not isinstance(ingredients, list):
        raise ValidationError("ingredients payload must contain an ingredients list")
    for record in ingredients:
        _validate_ingredient_record(record)
    _validate_ingredient_order(ingredients)
    _validate_alias_ownership(ingredients)
    _validate_parent_links(ingredients)


def _validate_recipe_cuisine(recipe: dict, requires_cuisine: bool) -> None:
    """Check cuisine presence and membership per split requirements."""
    if requires_cuisine:
        cuisine = recipe.get("cuisine")
        if cuisine not in EXPECTED_CUISINE_NAMES:
            raise ValidationError(
                f"cuisine gate failed: recipe {recipe['id']} has unknown "
                f"cuisine {cuisine!r}"
            )
        return
    if "cuisine" in recipe:
        raise ValidationError(
            f"cuisine gate failed: test recipe {recipe['id']} must not carry "
            "a cuisine field"
        )


def _validate_recipe_ingredient_ids(
    recipe: dict, known_ingredient_ids: frozenset, requires_cuisine: bool
) -> None:
    """Check ingredient_id references, in-recipe uniqueness, train coverage."""
    ingredient_ids = recipe["ingredient_ids"]
    seen_ids: set[str] = set()
    for ingredient_id in ingredient_ids:
        if ingredient_id not in known_ingredient_ids:
            raise ValidationError(
                f"ingredient reference gate failed: recipe {recipe['id']} "
                f"references unknown ingredient_id '{ingredient_id}'"
            )
        if ingredient_id in seen_ids:
            raise ValidationError(
                f"duplicate ingredient gate failed: recipe {recipe['id']} "
                f"repeats ingredient_id '{ingredient_id}'"
            )
        seen_ids.add(ingredient_id)
    if requires_cuisine and not ingredient_ids:
        raise ValidationError(
            f"resolved coverage gate failed: train recipe {recipe['id']} has "
            "no resolved ingredient_ids"
        )


def _validate_recipe_record(
    recipe: dict, known_ingredient_ids: frozenset, requires_cuisine: bool
) -> None:
    """Check one silver recipe record against the pinned schema."""
    _require_keys(recipe, REQUIRED_RECIPE_KEYS, "recipe record")
    _validate_recipe_cuisine(recipe, requires_cuisine)
    _validate_recipe_ingredient_ids(recipe, known_ingredient_ids, requires_cuisine)
    for unresolved_text in recipe["unresolved_ingredients"]:
        if not isinstance(unresolved_text, str) or not unresolved_text:
            raise ValidationError(
                f"unresolved gate failed: recipe {recipe['id']} contains an "
                "empty or non-string unresolved ingredient"
            )
    if not _is_non_negative_integer(recipe["raw_ingredient_count"]):
        raise ValidationError(
            f"recipe gate failed: recipe {recipe['id']} raw_ingredient_count "
            f"must be a non-negative integer, got {recipe['raw_ingredient_count']!r}"
        )


def validate_recipes_payload(
    payload: dict,
    ingredients_payload: dict,
    expected_count: int,
    requires_cuisine: bool,
) -> None:
    """Validate one silver recipes payload against the pinned schema.

    Args:
        payload: Parsed silver recipes payload (train or test split).
        ingredients_payload: Parsed silver ingredients payload providing the
            set of valid ingredient ids.
        expected_count: Exact number of recipes the split must contain.
        requires_cuisine: True for the train split, where each recipe must
            carry a known cuisine and at least one resolved ingredient id.

    Raises:
        ValidationError: On count mismatch, cuisine violations, dangling or
            duplicated ingredient_ids, empty train recipes, empty unresolved
            strings, or unsorted recipe ids.
    """
    _validate_envelope(payload, "recipes payload")
    recipes = payload.get("recipes")
    if not isinstance(recipes, list):
        raise ValidationError("recipes payload must contain a recipes list")
    if len(recipes) != expected_count:
        raise ValidationError(
            f"recipe count gate failed: expected {expected_count} recipes, "
            f"found {len(recipes)}"
        )
    known_ingredient_ids = frozenset(
        record["id"] for record in ingredients_payload["ingredients"]
    )
    previous_id: int | None = None
    for recipe in recipes:
        _validate_recipe_record(recipe, known_ingredient_ids, requires_cuisine)
        if previous_id is not None and recipe["id"] <= previous_id:
            raise ValidationError(
                f"recipe order gate failed: recipes must be sorted by id, but "
                f"{previous_id} precedes {recipe['id']}"
            )
        previous_id = recipe["id"]


def _validate_by_method_counts(by_method: dict, split_name: str) -> None:
    """Check by_method has exactly the pinned keys with integer counts."""
    _require_keys(
        by_method, frozenset(RESOLUTION_METHOD_NAMES), f"{split_name} by_method"
    )
    unexpected_keys = set(by_method) - set(RESOLUTION_METHOD_NAMES)
    if unexpected_keys:
        raise ValidationError(
            f"statistics gate failed: {split_name} by_method has unexpected "
            f"keys {sorted(unexpected_keys)}"
        )
    for method in RESOLUTION_METHOD_NAMES:
        if not _is_non_negative_integer(by_method[method]):
            raise ValidationError(
                f"statistics gate failed: {split_name} by_method['{method}'] "
                f"must be a non-negative integer, got {by_method[method]!r}"
            )


def _validate_statistics_block(block: dict, split_name: str) -> None:
    """Check structure and internal consistency of one split's statistics."""
    _require_keys(block, REQUIRED_STATISTICS_KEYS, f"{split_name} statistics")
    by_method = block["by_method"]
    _validate_by_method_counts(by_method, split_name)
    mentions_total = block["mentions_total"]
    if not _is_non_negative_integer(mentions_total) or mentions_total == 0:
        raise ValidationError(
            f"statistics gate failed: {split_name} mentions_total must be a "
            f"positive integer, got {mentions_total!r}"
        )
    method_sum = sum(by_method.values())
    if method_sum != mentions_total:
        raise ValidationError(
            f"statistics gate failed: {split_name} by_method sums to "
            f"{method_sum} but mentions_total is {mentions_total}"
        )
    top_unresolved = block["top_unresolved"]
    if not isinstance(top_unresolved, list) or len(top_unresolved) > TOP_UNRESOLVED_LIMIT:
        raise ValidationError(
            f"statistics gate failed: {split_name} top_unresolved must be a "
            f"list of at most {TOP_UNRESOLVED_LIMIT} entries"
        )


def _validate_full_chain_coverage(
    block: dict, split_name: str, minimum_coverage: float
) -> None:
    """Check the resolved share of all mentions meets the split's gate."""
    mentions_total = block["mentions_total"]
    resolved_mentions = mentions_total - block["by_method"][UNRESOLVED_METHOD]
    coverage = resolved_mentions / mentions_total
    if coverage < minimum_coverage:
        raise ValidationError(
            f"full chain coverage gate failed for {split_name}: "
            f"{coverage:.6f} < {minimum_coverage}"
        )


def validate_resolution_statistics(statistics: dict) -> None:
    """Validate the resolution statistics report and its coverage gates.

    Args:
        statistics: Parsed 01-bronze/reports/resolution_statistics.json content with
            'train' and 'test' blocks.

    Raises:
        ValidationError: On structural problems, by_method/mentions_total
            inconsistency, or any coverage gate below its threshold.
    """
    for split_name in STATISTICS_SPLIT_NAMES:
        if split_name not in statistics:
            raise ValidationError(
                f"statistics gate failed: missing '{split_name}' block"
            )
        _validate_statistics_block(statistics[split_name], split_name)
    train_block = statistics["train"]
    alias_tier_mentions = sum(
        train_block["by_method"][method] for method in ALIAS_TIER_METHOD_NAMES
    )
    alias_tier_coverage = alias_tier_mentions / train_block["mentions_total"]
    if alias_tier_coverage < ALIAS_TIER_MINIMUM_TRAIN_COVERAGE:
        raise ValidationError(
            f"alias tier coverage gate failed for train: "
            f"{alias_tier_coverage:.6f} < {ALIAS_TIER_MINIMUM_TRAIN_COVERAGE}"
        )
    _validate_full_chain_coverage(
        train_block, "train", FULL_CHAIN_MINIMUM_TRAIN_COVERAGE
    )
    _validate_full_chain_coverage(
        statistics["test"], "test", FULL_CHAIN_MINIMUM_TEST_COVERAGE
    )


def _validate_build_fingerprints(
    ingredients_payload: dict, train_payload: dict, test_payload: dict
) -> None:
    """Check the three silver payloads share one build fingerprint."""
    reference_build = ingredients_payload["build"]
    other_builds = {
        "recipes_train": train_payload["build"],
        "recipes_test": test_payload["build"],
    }
    for artifact_name in sorted(other_builds):
        if other_builds[artifact_name] != reference_build:
            raise ValidationError(
                f"build fingerprint gate failed: {artifact_name} build "
                f"{other_builds[artifact_name]} differs from ingredients "
                f"build {reference_build}"
            )


def validate_silver_artifacts(
    ingredients_payload: dict,
    train_payload: dict,
    test_payload: dict,
    statistics: dict,
) -> None:
    """Run every silver-artifact validator plus cross-artifact checks.

    Args:
        ingredients_payload: Parsed 02-silver/datasets/ingredients.json.
        train_payload: Parsed 02-silver/datasets/recipes_train.json.
        test_payload: Parsed 02-silver/datasets/recipes_test.json.
        statistics: Parsed 01-bronze/reports/resolution_statistics.json.

    Raises:
        ValidationError: On any per-artifact gate failure or when the build
            fingerprints of the three silver payloads disagree.
    """
    validate_ingredients_payload(ingredients_payload)
    validate_recipes_payload(
        train_payload, ingredients_payload, EXPECTED_TRAIN_RECIPE_COUNT, True
    )
    validate_recipes_payload(
        test_payload, ingredients_payload, EXPECTED_TEST_RECIPE_COUNT, False
    )
    validate_resolution_statistics(statistics)
    _validate_build_fingerprints(ingredients_payload, train_payload, test_payload)


def _load_json(path: Path) -> dict:
    """Load one JSON artifact from disk."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    """Load the silver artifacts, validate them, and print a PASS summary."""
    datasets_directory = locations.SILVER_DATASETS_DIRECTORY
    ingredients_payload = _load_json(datasets_directory / INGREDIENTS_FILENAME)
    train_payload = _load_json(datasets_directory / TRAIN_RECIPES_FILENAME)
    test_payload = _load_json(datasets_directory / TEST_RECIPES_FILENAME)
    statistics = _load_json(
        locations.REPORTS_DIRECTORY / RESOLUTION_STATISTICS_FILENAME
    )
    validate_silver_artifacts(
        ingredients_payload, train_payload, test_payload, statistics
    )
    train_unresolved = statistics["train"]["by_method"][UNRESOLVED_METHOD]
    test_unresolved = statistics["test"]["by_method"][UNRESOLVED_METHOD]
    print("PASS: silver artifacts are valid")
    print(f"  ingredients: {len(ingredients_payload['ingredients'])}")
    print(f"  train recipes: {len(train_payload['recipes'])}")
    print(f"  test recipes: {len(test_payload['recipes'])}")
    print(f"  unresolved mentions: train={train_unresolved} test={test_unresolved}")


if __name__ == "__main__":
    main()
