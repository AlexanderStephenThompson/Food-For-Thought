"""Regenerate the cuisine taxonomy deterministically from staged payloads.

Builds staged/cuisines.json content from the staged train recipes payload:
per-cuisine TF-IDF ingredient vectors, cosine-similarity neighbors, and
lift-ranked distinctive ingredients, with curated family assignments loaded
from lexicons/cuisine_families.json.

No file I/O happens at import time: callers invoke load_cuisine_families
explicitly and pass the resulting mapping onward.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from pipeline.staged_io import SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CUISINE_FAMILIES_PATH = PROJECT_ROOT / "lexicons" / "cuisine_families.json"

FAMILIES_KEY = "families"
PRIVATE_KEY_PREFIX = "_"
LEXICON_FILE_ENCODING = "utf-8"

NEIGHBOR_COUNT = 4
DISTINCTIVE_INGREDIENT_COUNT = 15
MINIMUM_INGREDIENT_SUPPORT = 10
MINIMUM_CUISINE_COVERAGE = 0.02
ROUNDED_DECIMAL_PLACES = 4


@dataclass(frozen=True)
class _CorpusStatistics:
    """Aggregate recipe counts derived from one staged recipes payload.

    Attributes:
        total_recipe_count: Number of recipes in the payload.
        ingredient_document_frequency: Ingredient id -> number of recipes
            (across all cuisines) containing it.
        cuisine_recipe_counts: Cuisine id -> number of recipes.
        cuisine_ingredient_counts: Cuisine id -> ingredient id -> number of
            the cuisine's recipes containing that ingredient.
    """

    total_recipe_count: int
    ingredient_document_frequency: dict[str, int]
    cuisine_recipe_counts: dict[str, int]
    cuisine_ingredient_counts: dict[str, dict[str, int]]


def load_cuisine_families(
    path: Path = DEFAULT_CUISINE_FAMILIES_PATH,
) -> dict[str, str]:
    """Load the curated cuisine -> family mapping from a lexicon file.

    Args:
        path: Path to a cuisine families JSON file whose mapping lives under
            its "families" key.

    Returns:
        Mapping of cuisine id to family id, sorted by cuisine id, with
        documentation keys (those starting with an underscore) removed.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If the file has no "families" key.
    """
    with open(path, encoding=LEXICON_FILE_ENCODING) as handle:
        document = json.load(handle)
    return {
        cuisine: family
        for cuisine, family in sorted(document[FAMILIES_KEY].items())
        if not cuisine.startswith(PRIVATE_KEY_PREFIX)
    }


def build_cuisines_payload(
    recipes_train_payload: dict, families: dict[str, str], fingerprint: dict
) -> dict:
    """Build the staged cuisines payload from staged train recipes.

    Args:
        recipes_train_payload: Staged recipes_train.json payload (each recipe
            carries "cuisine" and "ingredient_ids").
        families: Cuisine id -> family id mapping from the curated lexicon.
        fingerprint: Build block from compute_build_fingerprint.

    Returns:
        Payload with schema_version, the build block, and cuisines sorted by
        id, each carrying family, recipe_count, top-4 cosine neighbors, and
        top-15 lift-ranked distinctive ingredients.

    Raises:
        ValueError: If any observed cuisine lacks a family assignment, or any
            family-mapped cuisine has no recipes.
    """
    statistics = _summarize_corpus(recipes_train_payload)
    _validate_family_assignments(statistics.cuisine_recipe_counts, families)
    vectors = _build_tfidf_vectors(statistics)
    cuisines = [
        {
            "id": cuisine,
            "family": families[cuisine],
            "recipe_count": statistics.cuisine_recipe_counts[cuisine],
            "neighbors": _top_neighbors(cuisine, vectors),
            "distinctive_ingredients": _distinctive_ingredients(cuisine, statistics),
        }
        for cuisine in sorted(statistics.cuisine_recipe_counts)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "build": fingerprint,
        "cuisines": cuisines,
    }


def _summarize_corpus(recipes_train_payload: dict) -> _CorpusStatistics:
    """Count recipes and per-recipe-deduplicated ingredient occurrences."""
    document_frequency: Counter[str] = Counter()
    cuisine_recipe_counts: Counter[str] = Counter()
    cuisine_ingredient_counts: dict[str, Counter[str]] = defaultdict(Counter)
    recipes = recipes_train_payload["recipes"]
    for recipe in recipes:
        cuisine = recipe["cuisine"]
        unique_ingredient_ids = set(recipe["ingredient_ids"])
        cuisine_recipe_counts[cuisine] += 1
        document_frequency.update(unique_ingredient_ids)
        cuisine_ingredient_counts[cuisine].update(unique_ingredient_ids)
    return _CorpusStatistics(
        total_recipe_count=len(recipes),
        ingredient_document_frequency=dict(document_frequency),
        cuisine_recipe_counts=dict(cuisine_recipe_counts),
        cuisine_ingredient_counts={
            cuisine: dict(counts)
            for cuisine, counts in cuisine_ingredient_counts.items()
        },
    )


def _validate_family_assignments(
    cuisine_recipe_counts: dict[str, int], families: dict[str, str]
) -> None:
    """Fail fast when cuisines and family assignments do not match exactly."""
    observed_cuisines = set(cuisine_recipe_counts)
    assigned_cuisines = set(families)
    missing_families = sorted(observed_cuisines - assigned_cuisines)
    if missing_families:
        raise ValueError(
            f"Cuisines missing a family assignment: {', '.join(missing_families)}"
        )
    unused_families = sorted(assigned_cuisines - observed_cuisines)
    if unused_families:
        raise ValueError(
            f"Family-mapped cuisines with no recipes: {', '.join(unused_families)}"
        )


def _build_tfidf_vectors(
    statistics: _CorpusStatistics,
) -> dict[str, dict[str, float]]:
    """Build per-cuisine TF-IDF vectors over ingredient ids."""
    vectors: dict[str, dict[str, float]] = {}
    total_recipe_count = statistics.total_recipe_count
    for cuisine, ingredient_counts in sorted(
        statistics.cuisine_ingredient_counts.items()
    ):
        cuisine_recipe_count = statistics.cuisine_recipe_counts[cuisine]
        vector: dict[str, float] = {}
        for ingredient_id, recipe_count in sorted(ingredient_counts.items()):
            term_frequency = recipe_count / cuisine_recipe_count
            inverse_document_frequency = math.log(
                total_recipe_count
                / statistics.ingredient_document_frequency[ingredient_id]
            )
            vector[ingredient_id] = term_frequency * inverse_document_frequency
        vectors[cuisine] = vector
    return vectors


def _cosine_similarity(
    left_vector: dict[str, float], right_vector: dict[str, float]
) -> float:
    """Cosine similarity of two sparse vectors; 0.0 when either norm is zero."""
    shared_ingredient_ids = sorted(set(left_vector) & set(right_vector))
    dot_product = sum(
        left_vector[ingredient_id] * right_vector[ingredient_id]
        for ingredient_id in shared_ingredient_ids
    )
    left_norm = math.sqrt(sum(value * value for value in left_vector.values()))
    right_norm = math.sqrt(sum(value * value for value in right_vector.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _top_neighbors(
    cuisine: str, vectors: dict[str, dict[str, float]]
) -> list[dict]:
    """Top NEIGHBOR_COUNT other cuisines by cosine similarity, ties by id."""
    similarities = [
        (_cosine_similarity(vectors[cuisine], vectors[other_cuisine]), other_cuisine)
        for other_cuisine in sorted(vectors)
        if other_cuisine != cuisine
    ]
    similarities.sort(key=lambda pair: (-pair[0], pair[1]))
    return [
        {
            "id": other_cuisine,
            "similarity": round(similarity, ROUNDED_DECIMAL_PLACES),
        }
        for similarity, other_cuisine in similarities[:NEIGHBOR_COUNT]
    ]


def _distinctive_ingredients(
    cuisine: str, statistics: _CorpusStatistics
) -> list[dict]:
    """Top DISTINCTIVE_INGREDIENT_COUNT eligible ingredients by lift, ties by id."""
    cuisine_recipe_count = statistics.cuisine_recipe_counts[cuisine]
    total_recipe_count = statistics.total_recipe_count
    candidates = []
    for ingredient_id, support in sorted(
        statistics.cuisine_ingredient_counts[cuisine].items()
    ):
        cuisine_coverage = support / cuisine_recipe_count
        if support < MINIMUM_INGREDIENT_SUPPORT:
            continue
        if cuisine_coverage < MINIMUM_CUISINE_COVERAGE:
            continue
        overall_share = (
            statistics.ingredient_document_frequency[ingredient_id]
            / total_recipe_count
        )
        lift = cuisine_coverage / overall_share
        candidates.append((lift, ingredient_id, cuisine_coverage))
    candidates.sort(key=lambda entry: (-entry[0], entry[1]))
    return [
        {
            "id": ingredient_id,
            "lift": round(lift, ROUNDED_DECIMAL_PLACES),
            "cuisine_coverage": round(cuisine_coverage, ROUNDED_DECIMAL_PLACES),
        }
        for lift, ingredient_id, cuisine_coverage in candidates[
            :DISTINCTIVE_INGREDIENT_COUNT
        ]
    ]
