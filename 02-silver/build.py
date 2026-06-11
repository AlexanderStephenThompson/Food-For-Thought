"""Rebuild every gold artifact from the silver datasets.

Usage:
    .venv/bin/python 02-silver/build.py                    # full rebuild
    .venv/bin/python 02-silver/build.py --check-idempotent # rebuild in memory,
                                                           # verify disk matches

The build is deterministic end to end: rerunning it against unchanged
silver datasets produces byte-identical gold artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# gold_pipeline imports silver_pipeline's shared infrastructure (artifact
# I/O, locations), so the bronze tier root must be importable before the
# gold imports below. Python already puts this script's own directory
# (02-silver) on sys.path, which makes gold_pipeline importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01-bronze"))

from gold_pipeline import locations
from gold_pipeline.assign_folds import build_folds_payload
from gold_pipeline.build_feature_space import build_feature_space_payload
from gold_pipeline.build_features import build_features_payload
from gold_pipeline.build_fold_balance_report import (
    FOLD_BALANCE_JSON_FILENAME,
    build_fold_balance_payload,
    write_fold_balance_reports,
)
from gold_pipeline.load_silver_datasets import (
    compute_gold_build_fingerprint,
    load_silver_inputs,
)
from gold_pipeline.validate_gold import validate_gold_artifacts
from silver_pipeline.artifact_io import (
    find_artifact_mismatches,
    serialize_artifact_json,
    write_artifact_json,
)

FEATURE_SPACE_PATH = locations.GOLD_DATASETS_DIRECTORY / "feature_space.json"
FEATURES_TRAIN_PATH = locations.GOLD_DATASETS_DIRECTORY / "features_train.json"
FEATURES_TEST_PATH = locations.GOLD_DATASETS_DIRECTORY / "features_test.json"
FOLDS_PATH = locations.GOLD_DATASETS_DIRECTORY / "folds.json"
FOLD_BALANCE_PATH = (
    locations.SILVER_REPORTS_DIRECTORY / FOLD_BALANCE_JSON_FILENAME
)

PROGRESS_LOG_FORMAT = "%(message)s"

logger = logging.getLogger(__name__)


@dataclass
class GoldArtifacts:
    """Every payload one full gold build produces."""

    feature_space: dict
    features_train: dict
    features_test: dict
    folds: dict
    fold_balance: dict


def build_gold_artifacts() -> GoldArtifacts:
    """Run the full silver-to-gold build in memory.

    Returns:
        GoldArtifacts with every payload, already validated by the gold
        gates against the silver inputs.

    Raises:
        ValidationError: If any gold gate fails.
        FileNotFoundError: If a silver dataset file is missing.
        ValueError: If a silver dataset holds invalid JSON.
    """
    fingerprint = compute_gold_build_fingerprint()
    inputs = load_silver_inputs()
    logger.info(
        "loaded silver inputs: %d ingredients, %d train / %d test recipes",
        len(inputs.ingredients["ingredients"]),
        len(inputs.recipes_train["recipes"]),
        len(inputs.recipes_test["recipes"]),
    )

    feature_space = build_feature_space_payload(inputs.ingredients, fingerprint)
    features_train = build_features_payload(
        inputs.recipes_train, feature_space, fingerprint, includes_cuisine=True
    )
    features_test = build_features_payload(
        inputs.recipes_test, feature_space, fingerprint, includes_cuisine=False
    )
    logger.info(
        "feature space: %d features; rows: %d train / %d test",
        feature_space["feature_count"],
        len(features_train["rows"]),
        len(features_test["rows"]),
    )

    folds = build_folds_payload(inputs.recipes_train, fingerprint)
    fold_balance = build_fold_balance_payload(
        folds, inputs.recipes_train, fingerprint
    )
    logger.info(
        "folds: sizes %s",
        ", ".join(str(size) for size in fold_balance["fold_sizes"]),
    )

    validate_gold_artifacts(
        feature_space,
        features_train,
        features_test,
        folds,
        fold_balance,
        inputs.ingredients,
        inputs.recipes_train,
        inputs.recipes_test,
        expected_fingerprint=fingerprint,
    )
    logger.info("validation gates: PASS")

    return GoldArtifacts(
        feature_space=feature_space,
        features_train=features_train,
        features_test=features_test,
        folds=folds,
        fold_balance=fold_balance,
    )


def write_gold_artifacts(artifacts: GoldArtifacts) -> None:
    """Persist every artifact atomically to 03-gold/datasets/ and 02-silver/reports/."""
    locations.GOLD_DATASETS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    locations.SILVER_REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_artifact_json(artifacts.feature_space, FEATURE_SPACE_PATH)
    write_artifact_json(artifacts.features_train, FEATURES_TRAIN_PATH)
    write_artifact_json(artifacts.features_test, FEATURES_TEST_PATH)
    write_artifact_json(artifacts.folds, FOLDS_PATH)
    write_fold_balance_reports(
        artifacts.fold_balance, locations.SILVER_REPORTS_DIRECTORY
    )
    logger.info(
        "wrote gold artifacts to %s and %s",
        locations.GOLD_DATASETS_DIRECTORY,
        locations.SILVER_REPORTS_DIRECTORY,
    )


def verify_rebuild_matches_disk(artifacts: GoldArtifacts) -> list[str]:
    """Compare freshly built payloads against the gold files on disk.

    Args:
        artifacts: Payloads from build_gold_artifacts.

    Returns:
        Names of artifact files whose on-disk bytes differ from the rebuild
        (empty when the build is idempotent).
    """
    return find_artifact_mismatches(
        {
            FEATURE_SPACE_PATH: serialize_artifact_json(artifacts.feature_space),
            FEATURES_TRAIN_PATH: serialize_artifact_json(artifacts.features_train),
            FEATURES_TEST_PATH: serialize_artifact_json(artifacts.features_test),
            FOLDS_PATH: serialize_artifact_json(artifacts.folds),
            FOLD_BALANCE_PATH: serialize_artifact_json(artifacts.fold_balance),
        }
    )


def main() -> int:
    """Entry point: rebuild gold artifacts or verify idempotency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-idempotent",
        action="store_true",
        help="rebuild in memory and verify the gold files on disk match",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=PROGRESS_LOG_FORMAT)

    artifacts = build_gold_artifacts()
    if not arguments.check_idempotent:
        write_gold_artifacts(artifacts)
        return 0

    mismatches = verify_rebuild_matches_disk(artifacts)
    if mismatches:
        print(f"IDEMPOTENCY FAILURE: {', '.join(sorted(mismatches))}")
        return 1
    print("idempotency check: PASS (rebuild matches disk byte-for-byte)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
