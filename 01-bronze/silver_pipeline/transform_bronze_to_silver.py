"""Stage bronze recipes into the pinned silver payloads via an injected resolver.

stage_recipes maps every raw ingredient string of every recipe through a
duck-typed resolver (any object with resolve(raw_text) returning an object
with ingredient_id, method, dropped_tokens) and produces both the silver
recipes payload and the per-split resolution statistics block. The concrete
IngredientResolver is constructed elsewhere and injected by the orchestrator;
this module never imports it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Protocol, Sequence

from silver_pipeline.artifact_io import SCHEMA_VERSION, write_artifact_json
from silver_pipeline.load_bronze_recipes import Recipe

TOP_UNRESOLVED_LIMIT = 50
UNRESOLVED_METHOD = "unresolved"
RESOLUTION_METHOD_NAMES = (
    "exact_alias",
    "cleaned_match",
    "modifier_stripped_match",
    "brand_resolved_match",
    "token_drop_match",
    UNRESOLVED_METHOD,
)
REQUIRED_FINGERPRINT_KEYS = frozenset(
    {"train_sha256", "lexicon_fingerprint", "random_seed"}
)
REQUIRED_STATISTICS_SPLITS = ("train", "test")
TRAIN_RECIPES_FILENAME = "recipes_train.json"
TEST_RECIPES_FILENAME = "recipes_test.json"
RESOLUTION_STATISTICS_FILENAME = "resolution_statistics.json"


class ResolutionLike(Protocol):
    """Structural contract for one resolver outcome."""

    ingredient_id: str | None
    method: str
    dropped_tokens: tuple[str, ...]


class ResolverLike(Protocol):
    """Structural contract for the injected ingredient resolver."""

    def resolve(self, raw_text: str) -> ResolutionLike:
        """Resolve one raw ingredient string to a resolution outcome."""
        ...


def _validate_fingerprint(fingerprint: dict) -> None:
    """Fail fast when the build fingerprint block is malformed."""
    missing_keys = REQUIRED_FINGERPRINT_KEYS - set(fingerprint)
    if missing_keys:
        raise ValueError(
            "build fingerprint is missing required keys: "
            f"{sorted(missing_keys)}"
        )


def _validate_resolution(raw_text: str, resolution: ResolutionLike) -> None:
    """Fail fast when the resolver returns an out-of-contract outcome."""
    if resolution.method not in RESOLUTION_METHOD_NAMES:
        raise ValueError(
            f"unknown resolution method '{resolution.method}' for '{raw_text}'"
        )
    is_resolved = resolution.method != UNRESOLVED_METHOD
    if is_resolved and resolution.ingredient_id is None:
        raise ValueError(
            f"method '{resolution.method}' for '{raw_text}' requires an "
            "ingredient_id but got None"
        )
    if not is_resolved and resolution.ingredient_id is not None:
        raise ValueError(
            f"unresolved outcome for '{raw_text}' must not carry an "
            f"ingredient_id, got '{resolution.ingredient_id}'"
        )


def _stage_single_recipe(
    recipe: Recipe, resolver: ResolverLike, unresolved_counter: Counter
) -> tuple[dict, Counter]:
    """Resolve one recipe's mentions into a silver record and method counts."""
    ingredient_ids: list[str] = []
    unresolved_ingredients: list[str] = []
    method_counts: Counter = Counter()
    for raw_text in recipe.ingredients:
        resolution = resolver.resolve(raw_text)
        _validate_resolution(raw_text, resolution)
        method_counts[resolution.method] += 1
        if resolution.method == UNRESOLVED_METHOD:
            unresolved_ingredients.append(raw_text)
            unresolved_counter[raw_text] += 1
        elif resolution.ingredient_id not in ingredient_ids:
            ingredient_ids.append(resolution.ingredient_id)
    record: dict = {"id": recipe.id}
    if recipe.cuisine is not None:
        record["cuisine"] = recipe.cuisine
    record["ingredient_ids"] = ingredient_ids
    record["unresolved_ingredients"] = unresolved_ingredients
    record["raw_ingredient_count"] = len(recipe.ingredients)
    return record, method_counts


def _build_top_unresolved(unresolved_counter: Counter) -> list[dict]:
    """Rank unresolved strings by count descending, then string ascending."""
    ranked = sorted(unresolved_counter.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"string": text, "count": count}
        for text, count in ranked[:TOP_UNRESOLVED_LIMIT]
    ]


def stage_recipes(
    recipes: Sequence[Recipe], resolver: ResolverLike, fingerprint: dict
) -> tuple[dict, dict]:
    """Stage one split of recipes and tally its resolution statistics.

    Args:
        recipes: Bronze recipes for one split (train or test).
        resolver: Injected object whose resolve(raw_text) returns an outcome
            with ingredient_id, method, and dropped_tokens attributes.
        fingerprint: Build block with train_sha256, lexicon_fingerprint,
            and random_seed.

    Returns:
        Tuple of (silver recipes payload per the pinned schema, statistics
        block with mentions_total, zero-filled by_method counts, and
        top_unresolved capped at TOP_UNRESOLVED_LIMIT).

    Raises:
        ValueError: If the fingerprint is missing required keys, the resolver
            reports an unknown method, or a resolved outcome lacks an
            ingredient_id (or an unresolved one carries one).
    """
    _validate_fingerprint(fingerprint)
    unresolved_counter: Counter = Counter()
    total_method_counts: Counter = Counter()
    silver_records: list[dict] = []
    for recipe in recipes:
        record, method_counts = _stage_single_recipe(
            recipe, resolver, unresolved_counter
        )
        silver_records.append(record)
        total_method_counts.update(method_counts)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "build": dict(fingerprint),
        "recipes": sorted(silver_records, key=lambda record: record["id"]),
    }
    statistics = {
        "mentions_total": sum(total_method_counts.values()),
        "by_method": {
            method: total_method_counts.get(method, 0)
            for method in RESOLUTION_METHOD_NAMES
        },
        "top_unresolved": _build_top_unresolved(unresolved_counter),
    }
    return payload, statistics


def write_silver_recipes(
    train_payload: dict,
    test_payload: dict,
    statistics_by_split: dict,
    output_directory: Path,
    reports_directory: Path,
) -> None:
    """Persist both silver recipe splits and the combined statistics report.

    Args:
        train_payload: Silver train recipes payload from stage_recipes.
        test_payload: Silver test recipes payload from stage_recipes.
        statistics_by_split: Mapping with 'train' and 'test' statistics blocks.
        output_directory: Directory receiving the silver recipe files.
        reports_directory: Directory receiving the resolution statistics.

    Raises:
        ValueError: If statistics_by_split lacks a required split key.
    """
    missing_splits = [
        split for split in REQUIRED_STATISTICS_SPLITS
        if split not in statistics_by_split
    ]
    if missing_splits:
        raise ValueError(
            f"statistics_by_split is missing splits: {missing_splits}"
        )
    write_artifact_json(train_payload, output_directory / TRAIN_RECIPES_FILENAME)
    write_artifact_json(test_payload, output_directory / TEST_RECIPES_FILENAME)
    statistics_report = {
        split: statistics_by_split[split] for split in REQUIRED_STATISTICS_SPLITS
    }
    write_artifact_json(
        statistics_report, reports_directory / RESOLUTION_STATISTICS_FILENAME
    )
