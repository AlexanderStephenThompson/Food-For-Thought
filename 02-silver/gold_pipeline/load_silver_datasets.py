"""Load the four silver datasets and fingerprint them for the gold build.

The fingerprint hashes every silver input file plus the build's random seed
and fold count, so each gold artifact records exactly which silver state
produced it and a stale gold build is detectable. No file I/O happens at
import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gold_pipeline import locations
from gold_pipeline.assign_folds import FOLD_COUNT
from silver_pipeline.artifact_io import (
    ARTIFACT_TEXT_ENCODING,
    BUILD_RANDOM_SEED,
    sha256_of_file,
)

INGREDIENTS_FILENAME = "ingredients.json"
RECIPES_TRAIN_FILENAME = "recipes_train.json"
RECIPES_TEST_FILENAME = "recipes_test.json"
CUISINES_FILENAME = "cuisines.json"


@dataclass(frozen=True)
class SilverInputs:
    """The four parsed silver payloads the gold build consumes."""

    ingredients: dict
    recipes_train: dict
    recipes_test: dict
    cuisines: dict


def _read_json_document(path: Path) -> dict:
    """Parse one silver JSON file, failing fast with the path in context."""
    try:
        return json.loads(path.read_text(encoding=ARTIFACT_TEXT_ENCODING))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON ({error})") from error


def load_silver_inputs(
    datasets_directory: Path = locations.SILVER_DATASETS_DIRECTORY,
) -> SilverInputs:
    """Load all four silver datasets from one directory.

    Args:
        datasets_directory: Directory holding the silver dataset files.

    Returns:
        SilverInputs with every payload parsed.

    Raises:
        FileNotFoundError: If any silver dataset file is missing.
        ValueError: If any file holds invalid JSON.
    """
    return SilverInputs(
        ingredients=_read_json_document(datasets_directory / INGREDIENTS_FILENAME),
        recipes_train=_read_json_document(
            datasets_directory / RECIPES_TRAIN_FILENAME
        ),
        recipes_test=_read_json_document(datasets_directory / RECIPES_TEST_FILENAME),
        cuisines=_read_json_document(datasets_directory / CUISINES_FILENAME),
    )


def compute_gold_build_fingerprint(
    datasets_directory: Path = locations.SILVER_DATASETS_DIRECTORY,
) -> dict:
    """Fingerprint the silver inputs that determine every gold artifact.

    Args:
        datasets_directory: Directory holding the silver dataset files.

    Returns:
        Build block with the sha256 of each silver file, the build's
        random seed, and the fold count.

    Raises:
        FileNotFoundError: If any silver dataset file is missing.
    """
    return {
        "cuisines_sha256": sha256_of_file(datasets_directory / CUISINES_FILENAME),
        "fold_count": FOLD_COUNT,
        "ingredients_sha256": sha256_of_file(
            datasets_directory / INGREDIENTS_FILENAME
        ),
        "random_seed": BUILD_RANDOM_SEED,
        "recipes_test_sha256": sha256_of_file(
            datasets_directory / RECIPES_TEST_FILENAME
        ),
        "recipes_train_sha256": sha256_of_file(
            datasets_directory / RECIPES_TRAIN_FILENAME
        ),
    }
