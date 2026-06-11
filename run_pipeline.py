"""Rebuild every staged artifact from raw data and curated lexicons.

Usage:
    .venv/bin/python run_pipeline.py                    # full rebuild
    .venv/bin/python run_pipeline.py --check-idempotent # rebuild in memory,
                                                        # verify disk matches

The pipeline is deterministic end to end: rerunning it against unchanged
raw data and lexicons produces byte-identical staged artifacts.
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
from pipeline.load_raw_recipes import (
    RAW_TRAIN_PATH,
    build_train_index,
    load_test_recipes,
    load_train_recipes,
)
from pipeline.resolve_ingredient import IngredientResolver
from pipeline.staged_io import (
    STAGED_JSON_INDENT,
    compute_build_fingerprint,
    write_staged_json,
)
from pipeline.transform_raw_to_staged import stage_recipes, write_staged_recipes
from pipeline.validate_staged import validate_staged_artifacts

PROJECT_ROOT = Path(__file__).resolve().parent
LEXICONS_DIRECTORY = PROJECT_ROOT / "lexicons"
STAGED_DIRECTORY = PROJECT_ROOT / "staged"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
MERGE_DECISIONS_PATH = LEXICONS_DIRECTORY / "merge_decisions.jsonl"
CUISINE_FAMILIES_PATH = LEXICONS_DIRECTORY / "cuisine_families.json"
REVIEW_QUEUE_PATH = REPORTS_DIRECTORY / "merge_review_queue.jsonl"

INGREDIENTS_PATH = STAGED_DIRECTORY / "ingredients.json"
RECIPES_TRAIN_PATH = STAGED_DIRECTORY / "recipes_train.json"
RECIPES_TEST_PATH = STAGED_DIRECTORY / "recipes_test.json"
CUISINES_PATH = STAGED_DIRECTORY / "cuisines.json"
RESOLUTION_STATISTICS_PATH = REPORTS_DIRECTORY / "resolution_statistics.json"
COVERAGE_PATH = REPORTS_DIRECTORY / "coverage.json"


@dataclass
class StagedArtifacts:
    """Every payload one full pipeline run produces."""

    review_queue_entries: list[dict]
    ingredients: dict
    recipes_train: dict
    recipes_test: dict
    resolution_statistics: dict
    cuisines: dict
    coverage: dict
    compile_statistics: CompileStatistics


def build_staged_artifacts() -> StagedArtifacts:
    """Run the full raw-to-staged build in memory.

    Returns:
        StagedArtifacts with every payload, already validated by the
        compile gates and the staged-artifact validators.

    Raises:
        ValidationError: If any staged-artifact gate fails.
        ValueError: If merge decisions are missing, malformed, or stale.
    """
    lexicons = load_pipeline_lexicons(LEXICONS_DIRECTORY)
    fingerprint = compute_build_fingerprint(RAW_TRAIN_PATH, LEXICONS_DIRECTORY)
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
        print(f"staged {split_name}: {total} mentions, {unresolved} unresolved")

    families = load_cuisine_families(CUISINE_FAMILIES_PATH)
    cuisines = build_cuisines_payload(recipes_train, families, fingerprint)
    validate_staged_artifacts(
        ingredients, recipes_train, recipes_test, resolution_statistics
    )
    print("validation gates: PASS")

    coverage = build_coverage_payload(resolution_statistics, ingredients, fingerprint)
    return StagedArtifacts(
        review_queue_entries=review_queue_entries,
        ingredients=ingredients,
        recipes_train=recipes_train,
        recipes_test=recipes_test,
        resolution_statistics=resolution_statistics,
        cuisines=cuisines,
        coverage=coverage,
        compile_statistics=compile_statistics,
    )


def write_staged_artifacts(artifacts: StagedArtifacts) -> None:
    """Persist every artifact atomically to staged/ and reports/."""
    write_review_queue_to_path(artifacts.review_queue_entries, REVIEW_QUEUE_PATH)
    write_staged_json(artifacts.ingredients, INGREDIENTS_PATH)
    write_staged_recipes(
        artifacts.recipes_train,
        artifacts.recipes_test,
        artifacts.resolution_statistics,
        STAGED_DIRECTORY,
        REPORTS_DIRECTORY,
    )
    write_staged_json(artifacts.cuisines, CUISINES_PATH)
    write_coverage_reports(artifacts.coverage, REPORTS_DIRECTORY)
    print(f"wrote staged artifacts to {STAGED_DIRECTORY} and {REPORTS_DIRECTORY}")


def _serialize_staged(payload: dict) -> str:
    return (
        json.dumps(
            payload, ensure_ascii=False, indent=STAGED_JSON_INDENT, sort_keys=True
        )
        + "\n"
    )


def verify_rebuild_matches_disk(artifacts: StagedArtifacts) -> list[str]:
    """Compare freshly built payloads against the files on disk.

    Args:
        artifacts: Payloads from build_staged_artifacts.

    Returns:
        Names of staged files whose on-disk bytes differ from the rebuild
        (empty when the pipeline is idempotent).
    """
    expected_by_path = {
        INGREDIENTS_PATH: _serialize_staged(artifacts.ingredients),
        RECIPES_TRAIN_PATH: _serialize_staged(artifacts.recipes_train),
        RECIPES_TEST_PATH: _serialize_staged(artifacts.recipes_test),
        CUISINES_PATH: _serialize_staged(artifacts.cuisines),
        RESOLUTION_STATISTICS_PATH: _serialize_staged(
            artifacts.resolution_statistics
        ),
        COVERAGE_PATH: _serialize_staged(artifacts.coverage),
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
    """Entry point: rebuild staged artifacts or verify idempotency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-idempotent",
        action="store_true",
        help="rebuild in memory and verify the staged files on disk match",
    )
    arguments = parser.parse_args()

    artifacts = build_staged_artifacts()
    if not arguments.check_idempotent:
        write_staged_artifacts(artifacts)
        return 0

    mismatches = verify_rebuild_matches_disk(artifacts)
    if mismatches:
        print(f"IDEMPOTENCY FAILURE: {', '.join(sorted(mismatches))}")
        return 1
    print("idempotency check: PASS (rebuild matches disk byte-for-byte)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
