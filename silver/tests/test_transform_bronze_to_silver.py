"""Tests for silver.pipeline.transform_bronze_to_silver.

stage_recipes turns bronze Recipe records plus an injected resolver into the
pinned silver recipes payload and a per-split resolution statistics block;
write_silver_recipes persists both splits and the combined statistics report.

The resolver is a duck-typed stub here: the concrete IngredientResolver is
built elsewhere and injected by the orchestrator, never imported.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from silver.pipeline.artifact_io import SCHEMA_VERSION
from silver.pipeline.load_bronze_recipes import Recipe
from silver.pipeline.transform_bronze_to_silver import (
    TOP_UNRESOLVED_LIMIT,
    stage_recipes,
    write_silver_recipes,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "transform_bronze_to_silver"
SAMPLE_RECIPES_PATH = FIXTURE_DIRECTORY / "sample_recipes.json"

FINGERPRINT = {
    "train_sha256": "a" * 64,
    "lexicon_fingerprint": "b" * 64,
    "random_seed": 42,
}

UNRESOLVED_METHOD = "unresolved"


@dataclass(frozen=True)
class StubResolution:
    """Duck-typed stand-in for resolve_ingredient.ResolutionResult."""

    ingredient_id: str | None
    method: str
    dropped_tokens: tuple[str, ...] = ()


class StubResolver:
    """Resolver double returning canned outcomes; unresolved by default."""

    def __init__(self, outcomes: dict[str, StubResolution] | None = None) -> None:
        self._outcomes = outcomes or {}

    def resolve(self, raw_text: str) -> StubResolution:
        default = StubResolution(ingredient_id=None, method=UNRESOLVED_METHOD)
        return self._outcomes.get(raw_text, default)


SAMPLE_OUTCOMES = {
    "soy sauce": StubResolution("soy_sauce", "exact_alias"),
    "low sodium soy sauce": StubResolution("soy_sauce", "modifier_stripped_match"),
    "scallions": StubResolution("green_onion", "cleaned_match"),
    "feta cheese crumbles": StubResolution("feta_cheese", "token_drop_match", ("crumbles",)),
    "romaine lettuce": StubResolution("romaine_lettuce", "exact_alias"),
    "Old El Paso taco shells": StubResolution("taco_shell", "brand_resolved_match"),
}


def _load_sample_recipes() -> tuple[list[Recipe], list[Recipe]]:
    with SAMPLE_RECIPES_PATH.open(encoding="utf-8") as handle:
        sample = json.load(handle)
    train_recipes = [
        Recipe(id=record["id"], cuisine=record["cuisine"], ingredients=tuple(record["ingredients"]))
        for record in sample["train"]
    ]
    test_recipes = [
        Recipe(id=record["id"], cuisine=None, ingredients=tuple(record["ingredients"]))
        for record in sample["test"]
    ]
    return train_recipes, test_recipes


def test_stage_recipes_deduplicates_preserving_order():
    recipe = Recipe(
        id=1,
        cuisine="chinese",
        ingredients=("low sodium soy sauce", "scallions", "soy sauce"),
    )

    payload, _ = stage_recipes([recipe], StubResolver(SAMPLE_OUTCOMES), FINGERPRINT)

    assert payload["recipes"][0]["ingredient_ids"] == ["soy_sauce", "green_onion"]


def test_unresolved_strings_recorded_verbatim():
    recipe = Recipe(
        id=1,
        cuisine="thai",
        ingredients=("Mystery Brand™ Goo", "soy sauce", "Mystery Brand™ Goo"),
    )

    payload, _ = stage_recipes([recipe], StubResolver(SAMPLE_OUTCOMES), FINGERPRINT)

    silver_recipe = payload["recipes"][0]
    assert silver_recipe["unresolved_ingredients"] == [
        "Mystery Brand™ Goo",
        "Mystery Brand™ Goo",
    ]
    assert silver_recipe["ingredient_ids"] == ["soy_sauce"]


def test_cuisine_omitted_for_test_split():
    recipe = Recipe(id=1, cuisine=None, ingredients=("soy sauce",))

    payload, _ = stage_recipes([recipe], StubResolver(SAMPLE_OUTCOMES), FINGERPRINT)

    assert "cuisine" not in payload["recipes"][0]


def test_cuisine_present_for_train_split():
    recipe = Recipe(id=1, cuisine="greek", ingredients=("soy sauce",))

    payload, _ = stage_recipes([recipe], StubResolver(SAMPLE_OUTCOMES), FINGERPRINT)

    assert payload["recipes"][0]["cuisine"] == "greek"


def test_raw_ingredient_count_counts_every_mention():
    recipe = Recipe(
        id=1, cuisine="greek", ingredients=("soy sauce", "soy sauce", "scallions")
    )

    payload, _ = stage_recipes([recipe], StubResolver(SAMPLE_OUTCOMES), FINGERPRINT)

    assert payload["recipes"][0]["raw_ingredient_count"] == 3


def test_recipes_sorted_by_id():
    recipes = [
        Recipe(id=30, cuisine="greek", ingredients=("soy sauce",)),
        Recipe(id=10, cuisine="thai", ingredients=("soy sauce",)),
        Recipe(id=20, cuisine="french", ingredients=("soy sauce",)),
    ]

    payload, _ = stage_recipes(recipes, StubResolver(SAMPLE_OUTCOMES), FINGERPRINT)

    assert [recipe["id"] for recipe in payload["recipes"]] == [10, 20, 30]


def test_payload_envelope_matches_pinned_schema():
    recipe = Recipe(id=1, cuisine="greek", ingredients=("soy sauce",))

    payload, _ = stage_recipes([recipe], StubResolver(SAMPLE_OUTCOMES), FINGERPRINT)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["build"] == FINGERPRINT
    assert payload["build"] is not FINGERPRINT


def test_statistics_method_counts_zero_filled():
    recipe = Recipe(id=1, cuisine="greek", ingredients=("soy sauce",))

    _, statistics = stage_recipes([recipe], StubResolver(SAMPLE_OUTCOMES), FINGERPRINT)

    assert statistics["by_method"] == {
        "exact_alias": 1,
        "cleaned_match": 0,
        "modifier_stripped_match": 0,
        "brand_resolved_match": 0,
        "token_drop_match": 0,
        "unresolved": 0,
    }


def test_statistics_mentions_total_counts_every_raw_mention():
    recipes = [
        Recipe(id=1, cuisine="greek", ingredients=("soy sauce", "scallions", "goo")),
        Recipe(id=2, cuisine="thai", ingredients=("soy sauce", "soy sauce")),
    ]

    _, statistics = stage_recipes(recipes, StubResolver(SAMPLE_OUTCOMES), FINGERPRINT)

    assert statistics["mentions_total"] == 5


def test_top_unresolved_sorted_and_capped():
    frequent_strings = [f"mystery {index:02d}" for index in range(5)]
    rare_strings = [f"mystery {index:02d}" for index in range(5, 55)]
    recipe = Recipe(
        id=1, cuisine="greek", ingredients=tuple(frequent_strings * 3 + rare_strings)
    )

    _, statistics = stage_recipes([recipe], StubResolver(), FINGERPRINT)

    top_unresolved = statistics["top_unresolved"]
    assert len(top_unresolved) == TOP_UNRESOLVED_LIMIT
    assert top_unresolved[:5] == [
        {"string": text, "count": 3} for text in frequent_strings
    ]
    assert top_unresolved[5:] == [
        {"string": text, "count": 1} for text in rare_strings[:45]
    ]


def test_token_drop_match_resolves_and_counts():
    recipe = Recipe(id=1, cuisine="greek", ingredients=("feta cheese crumbles",))

    payload, statistics = stage_recipes(
        [recipe], StubResolver(SAMPLE_OUTCOMES), FINGERPRINT
    )

    assert payload["recipes"][0]["ingredient_ids"] == ["feta_cheese"]
    assert statistics["by_method"]["token_drop_match"] == 1


def test_unknown_method_raises_value_error():
    resolver = StubResolver({"weird": StubResolution("thing", "telepathy")})
    recipe = Recipe(id=1, cuisine="greek", ingredients=("weird",))

    with pytest.raises(ValueError, match="telepathy"):
        stage_recipes([recipe], resolver, FINGERPRINT)


def test_resolved_method_without_ingredient_id_raises():
    resolver = StubResolver({"ghost": StubResolution(None, "exact_alias")})
    recipe = Recipe(id=1, cuisine="greek", ingredients=("ghost",))

    with pytest.raises(ValueError, match="ingredient_id"):
        stage_recipes([recipe], resolver, FINGERPRINT)


def test_missing_fingerprint_field_raises():
    recipe = Recipe(id=1, cuisine="greek", ingredients=("soy sauce",))

    with pytest.raises(ValueError, match="fingerprint"):
        stage_recipes([recipe], StubResolver(SAMPLE_OUTCOMES), {"random_seed": 42})


def test_write_silver_recipes_writes_three_files(tmp_path):
    train_recipes, test_recipes = _load_sample_recipes()
    resolver = StubResolver(SAMPLE_OUTCOMES)
    train_payload, train_statistics = stage_recipes(train_recipes, resolver, FINGERPRINT)
    test_payload, test_statistics = stage_recipes(test_recipes, resolver, FINGERPRINT)
    silver_directory = tmp_path / "silver"
    reports_directory = tmp_path / "reports"
    silver_directory.mkdir()
    reports_directory.mkdir()

    write_silver_recipes(
        train_payload,
        test_payload,
        {"train": train_statistics, "test": test_statistics},
        silver_directory,
        reports_directory,
    )

    written_train = json.loads(
        (silver_directory / "recipes_train.json").read_text(encoding="utf-8")
    )
    written_test = json.loads(
        (silver_directory / "recipes_test.json").read_text(encoding="utf-8")
    )
    written_statistics = json.loads(
        (reports_directory / "resolution_statistics.json").read_text(encoding="utf-8")
    )
    assert written_train == train_payload
    assert written_test == test_payload
    assert written_statistics == {"train": train_statistics, "test": test_statistics}


def test_write_silver_recipes_missing_split_raises(tmp_path):
    train_recipes, test_recipes = _load_sample_recipes()
    resolver = StubResolver(SAMPLE_OUTCOMES)
    train_payload, train_statistics = stage_recipes(train_recipes, resolver, FINGERPRINT)
    test_payload, _ = stage_recipes(test_recipes, resolver, FINGERPRINT)

    with pytest.raises(ValueError, match="test"):
        write_silver_recipes(
            train_payload,
            test_payload,
            {"train": train_statistics},
            tmp_path,
            tmp_path,
        )
