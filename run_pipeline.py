"""Rebuild every silver artifact from bronze data and curated lexicons.

Usage:
    .venv/bin/python run_pipeline.py                    # full rebuild
    .venv/bin/python run_pipeline.py --check-idempotent # rebuild in memory,
                                                        # verify disk matches

The pipeline is deterministic end to end: rerunning it against unchanged
bronze data and lexicons produces byte-identical silver artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline.build_coverage_report import (
    build_coverage_payload,
    write_coverage_reports,
)
from pipeline.build_cuisines import build_cuisines_payload, load_cuisine_families
from pipeline.build_vocabulary import (
    build_vocabulary_from_index,
    load_pipeline_lexicons,
    write_review_queue_to_path,
)
from pipeline.compile_alias_table import (
    CompileStatistics,
    compile_ingredients_payload,
    load_merge_decisions,
    validate_compiled_payload,
)
from pipeline.artifact_io import (
    ARTIFACT_JSON_INDENT,
    compute_build_fingerprint,
    write_artifact_json,
)
from pipeline.load_bronze_recipes import (
    BRONZE_TRAIN_PATH,
    build_train_index,
    load_test_recipes,
    load_train_recipes,
)
from pipeline.resolve_ingredient import IngredientResolver
from pipeline.transform_bronze_to_silver import stage_recipes, write_silver_recipes
from pipeline.validate_silver import validate_silver_artifacts

PROJECT_ROOT = Path(__file__).resolve().parent
LEXICONS_DIRECTORY = PROJECT_ROOT / "lexicons"
SILVER_DIRECTORY = PROJECT_ROOT / "silver"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
MERGE_DECISIONS_PATH = LEXICONS_DIRECTORY / "merge_decisions.jsonl"
CUISINE_FAMILIES_PATH = LEXICONS_DIRECTORY / "cuisine_families.json"
REVIEW_QUEUE_PATH = REPORTS_DIRECTORY / "merge_review_queue.jsonl"

INGREDIENTS_PATH = SILVER_DIRECTORY / "ingredients.json"
RECIPES_TRAIN_PATH = SILVER_DIRECTORY / "recipes_train.json"
RECIPES_TEST_PATH = SILVER_DIRECTORY / "recipes_test.json"
CUISINES_PATH = SILVER_DIRECTORY / "cuisines.json"
RESOLUTION_STATISTICS_PATH = REPORTS_DIRECTORY / "resolution_statistics.json"
COVERAGE_PATH = REPORTS_DIRECTORY / "coverage.json"


@dataclass
class SilverArtifacts:
    """Every payload one full pipeline run produces."""

    review_queue_entries: list[dict]
    ingredients: dict
    recipes_train: dict
    recipes_test: dict
    resolution_statistics: dict
    cuisines: dict
    coverage: dict
    compile_statistics: CompileStatistics


def build_silver_artifacts() -> SilverArtifacts:
    """Run the full bronze-to-silver build in memory.

    Returns:
        SilverArtifacts with every payload, already validated by the
        compile gates and the silver-artifact validators.

    Raises:
        ValidationError: If any silver-artifact gate fails.
        ValueError: If merge decisions are missing, malformed, or stale.
    """
    lexicons = load_pipeline_lexicons(LEXICONS_DIRECTORY)
    fingerprint = compute_build_fingerprint(BRONZE_TRAIN_PATH, LEXICONS_DIRECTORY)
    train_recipes = load_train_recipes()
    test_recipes = load_test_recipes()
    index = build_train_index(train_recipes)
    print(f"loaded {len(train_recipes)} train / {len(test_recipes)} test recipes")

    vocabulary_build = build_vocabulary_from_index(index, lexicons)
    review_queue_entries = list(vocabulary_build.review_entries)
    print(
        f"vocabulary build: {len(vocabulary_build.groups)} groups, "
        f"{len(review_queue_entries)} review entries"
    )

    decisions = load_merge_decisions(MERGE_DECISIONS_PATH)
    ingredients = compile_ingredients_payload(
        vocabulary_build, decisions, index, fingerprint
    )
    compile_statistics = validate_compiled_payload(ingredients, index)
    print(
        f"compiled {compile_statistics.ingredient_count} ingredients, "
        f"{compile_statistics.alias_count} aliases, "
        f"coverage {compile_statistics.coverage_ratio:.4f}"
    )

    resolver = IngredientResolver.from_payload(ingredients, lexicons)
    recipes_train, train_statistics = stage_recipes(
        train_recipes, resolver, fingerprint
    )
    recipes_test, test_statistics = stage_recipes(test_recipes, resolver, fingerprint)
    resolution_statistics = {"test": test_statistics, "train": train_statistics}
    for split_name, split_statistics in sorted(resolution_statistics.items()):
        unresolved = split_statistics["by_method"]["unresolved"]
        total = split_statistics["mentions_total"]
        print(f"silver {split_name}: {total} mentions, {unresolved} unresolved")

    families = load_cuisine_families(CUISINE_FAMILIES_PATH)
    cuisines = build_cuisines_payload(recipes_train, families, fingerprint)
    validate_silver_artifacts(
        ingredients, recipes_train, recipes_test, resolution_statistics
    )
    print("validation gates: PASS")

    coverage = build_coverage_payload(resolution_statistics, ingredients, fingerprint)
    return SilverArtifacts(
        review_queue_entries=review_queue_entries,
        ingredients=ingredients,
        recipes_train=recipes_train,
        recipes_test=recipes_test,
        resolution_statistics=resolution_statistics,
        cuisines=cuisines,
        coverage=coverage,
        compile_statistics=compile_statistics,
    )


def write_silver_artifacts(artifacts: SilverArtifacts) -> None:
    """Persist every artifact atomically to silver/ and reports/."""
    write_review_queue_to_path(artifacts.review_queue_entries, REVIEW_QUEUE_PATH)
    write_artifact_json(artifacts.ingredients, INGREDIENTS_PATH)
    write_silver_recipes(
        artifacts.recipes_train,
        artifacts.recipes_test,
        artifacts.resolution_statistics,
        SILVER_DIRECTORY,
        REPORTS_DIRECTORY,
    )
    write_artifact_json(artifacts.cuisines, CUISINES_PATH)
    write_coverage_reports(artifacts.coverage, REPORTS_DIRECTORY)
    print(f"wrote silver artifacts to {SILVER_DIRECTORY} and {REPORTS_DIRECTORY}")


def _serialize_artifact(payload: dict) -> str:
    return (
        json.dumps(
            payload, ensure_ascii=False, indent=ARTIFACT_JSON_INDENT, sort_keys=True
        )
        + "\n"
    )


def verify_rebuild_matches_disk(artifacts: SilverArtifacts) -> list[str]:
    """Compare freshly built payloads against the files on disk.

    Args:
        artifacts: Payloads from build_silver_artifacts.

    Returns:
        Names of artifact files whose on-disk bytes differ from the rebuild
        (empty when the pipeline is idempotent).
    """
    expected_by_path = {
        INGREDIENTS_PATH: _serialize_artifact(artifacts.ingredients),
        RECIPES_TRAIN_PATH: _serialize_artifact(artifacts.recipes_train),
        RECIPES_TEST_PATH: _serialize_artifact(artifacts.recipes_test),
        CUISINES_PATH: _serialize_artifact(artifacts.cuisines),
        RESOLUTION_STATISTICS_PATH: _serialize_artifact(
            artifacts.resolution_statistics
        ),
        COVERAGE_PATH: _serialize_artifact(artifacts.coverage),
    }
    mismatches = []
    for path, expected_content in expected_by_path.items():
        if not path.is_file():
            mismatches.append(f"{path.name} (missing)")
            continue
        if path.read_text(encoding="utf-8") != expected_content:
            mismatches.append(path.name)
    return mismatches


def main() -> int:
    """Entry point: rebuild silver artifacts or verify idempotency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-idempotent",
        action="store_true",
        help="rebuild in memory and verify the silver files on disk match",
    )
    arguments = parser.parse_args()

    artifacts = build_silver_artifacts()
    if not arguments.check_idempotent:
        write_silver_artifacts(artifacts)
        return 0

    mismatches = verify_rebuild_matches_disk(artifacts)
    if mismatches:
        print(f"IDEMPOTENCY FAILURE: {', '.join(sorted(mismatches))}")
        return 1
    print("idempotency check: PASS (rebuild matches disk byte-for-byte)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
