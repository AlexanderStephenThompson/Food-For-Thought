"""Rebuild every silver artifact from bronze data and curated lexicons.

Usage:
    .venv/bin/python 01-bronze/build.py                    # full rebuild
    .venv/bin/python 01-bronze/build.py --check-idempotent # rebuild in memory,
                                                           # verify disk matches

The pipeline is deterministic end to end: rerunning it against unchanged
bronze data and lexicons produces byte-identical silver artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from silver_pipeline import locations
from silver_pipeline.artifact_io import (
    compute_build_fingerprint,
    find_artifact_mismatches,
    serialize_artifact_json,
    write_artifact_json,
)
from silver_pipeline.build_coverage_report import (
    build_coverage_payload,
    write_coverage_reports,
)
from silver_pipeline.build_cuisines import build_cuisines_payload, load_cuisine_families
from silver_pipeline.build_vocabulary import (
    build_vocabulary_from_index,
    load_pipeline_lexicons,
    write_review_queue_to_path,
)
from silver_pipeline.compile_alias_table import (
    CompileStatistics,
    compile_ingredients_payload,
    load_merge_decisions,
    validate_compiled_payload,
)
from silver_pipeline.load_bronze_recipes import (
    BRONZE_TRAIN_PATH,
    build_train_index,
    load_test_recipes,
    load_train_recipes,
)
from silver_pipeline.resolve_ingredient import IngredientResolver
from silver_pipeline.transform_bronze_to_silver import stage_recipes, write_silver_recipes
from silver_pipeline.validate_silver import validate_silver_artifacts

MERGE_DECISIONS_PATH = locations.LEXICONS_DIRECTORY / "merge_decisions.jsonl"
CUISINE_FAMILIES_PATH = locations.LEXICONS_DIRECTORY / "cuisine_families.json"
REVIEW_QUEUE_PATH = locations.REPORTS_DIRECTORY / "merge_review_queue.jsonl"

INGREDIENTS_PATH = locations.SILVER_DATASETS_DIRECTORY / "ingredients.json"
RECIPES_TRAIN_PATH = locations.SILVER_DATASETS_DIRECTORY / "recipes_train.json"
RECIPES_TEST_PATH = locations.SILVER_DATASETS_DIRECTORY / "recipes_test.json"
CUISINES_PATH = locations.SILVER_DATASETS_DIRECTORY / "cuisines.json"
RESOLUTION_STATISTICS_PATH = (
    locations.REPORTS_DIRECTORY / "resolution_statistics.json"
)
COVERAGE_PATH = locations.REPORTS_DIRECTORY / "coverage.json"

PROGRESS_LOG_FORMAT = "%(message)s"

logger = logging.getLogger(__name__)


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
    lexicons = load_pipeline_lexicons(locations.LEXICONS_DIRECTORY)
    fingerprint = compute_build_fingerprint(
        BRONZE_TRAIN_PATH, locations.LEXICONS_DIRECTORY
    )
    train_recipes = load_train_recipes()
    test_recipes = load_test_recipes()
    index = build_train_index(train_recipes)
    logger.info(
        "loaded %d train / %d test recipes", len(train_recipes), len(test_recipes)
    )

    vocabulary_build = build_vocabulary_from_index(index, lexicons)
    review_queue_entries = list(vocabulary_build.review_entries)
    logger.info(
        "vocabulary build: %d groups, %d review entries",
        len(vocabulary_build.groups),
        len(review_queue_entries),
    )

    decisions = load_merge_decisions(MERGE_DECISIONS_PATH)
    ingredients = compile_ingredients_payload(
        vocabulary_build, decisions, index, fingerprint
    )
    compile_statistics = validate_compiled_payload(ingredients, index)
    logger.info(
        "compiled %d ingredients, %d aliases, coverage %.4f",
        compile_statistics.ingredient_count,
        compile_statistics.alias_count,
        compile_statistics.coverage_ratio,
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
        logger.info(
            "silver %s: %d mentions, %d unresolved", split_name, total, unresolved
        )

    families = load_cuisine_families(CUISINE_FAMILIES_PATH)
    cuisines = build_cuisines_payload(recipes_train, families, fingerprint)
    validate_silver_artifacts(
        ingredients, recipes_train, recipes_test, resolution_statistics
    )
    logger.info("validation gates: PASS")

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
    """Persist every artifact atomically to 02-silver/datasets/ and 01-bronze/reports/."""
    write_review_queue_to_path(artifacts.review_queue_entries, REVIEW_QUEUE_PATH)
    write_artifact_json(artifacts.ingredients, INGREDIENTS_PATH)
    write_silver_recipes(
        artifacts.recipes_train,
        artifacts.recipes_test,
        artifacts.resolution_statistics,
        output_directory=locations.SILVER_DATASETS_DIRECTORY,
        reports_directory=locations.REPORTS_DIRECTORY,
    )
    write_artifact_json(artifacts.cuisines, CUISINES_PATH)
    write_coverage_reports(artifacts.coverage, locations.REPORTS_DIRECTORY)
    logger.info(
        "wrote silver artifacts to %s and %s",
        locations.SILVER_DATASETS_DIRECTORY,
        locations.REPORTS_DIRECTORY,
    )


def verify_rebuild_matches_disk(artifacts: SilverArtifacts) -> list[str]:
    """Compare freshly built payloads against the silver files on disk.

    Args:
        artifacts: Payloads from build_silver_artifacts.

    Returns:
        Names of artifact files whose on-disk bytes differ from the rebuild
        (empty when the pipeline is idempotent).
    """
    return find_artifact_mismatches(
        {
            INGREDIENTS_PATH: serialize_artifact_json(artifacts.ingredients),
            RECIPES_TRAIN_PATH: serialize_artifact_json(artifacts.recipes_train),
            RECIPES_TEST_PATH: serialize_artifact_json(artifacts.recipes_test),
            CUISINES_PATH: serialize_artifact_json(artifacts.cuisines),
            RESOLUTION_STATISTICS_PATH: serialize_artifact_json(
                artifacts.resolution_statistics
            ),
            COVERAGE_PATH: serialize_artifact_json(artifacts.coverage),
        }
    )


def main() -> int:
    """Entry point: rebuild silver artifacts or verify idempotency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-idempotent",
        action="store_true",
        help="rebuild in memory and verify the silver files on disk match",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=PROGRESS_LOG_FORMAT)

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
