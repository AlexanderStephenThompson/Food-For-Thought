"""Load the gold datasets and fingerprint them for the model build.

The fingerprint hashes every input file (the four gold datasets plus the
silver cuisine taxonomy) and records the build seed and the installed
scikit-learn version, so each model artifact declares exactly which data
and environment produced it. Byte-identity of model artifacts is therefore
environment-conditional: a rebuild matches disk only under the same
scikit-learn version, and the validation gates say so explicitly. No file
I/O happens at import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import sklearn

from model_pipeline import locations
from silver_pipeline.artifact_io import (
    ARTIFACT_TEXT_ENCODING,
    BUILD_RANDOM_SEED,
    sha256_of_file,
)

FEATURE_SPACE_FILENAME = "feature_space.json"
FEATURES_TRAIN_FILENAME = "features_train.json"
FEATURES_TEST_FILENAME = "features_test.json"
FOLDS_FILENAME = "folds.json"
CUISINES_FILENAME = "cuisines.json"


@dataclass(frozen=True)
class GoldModelInputs:
    """The five parsed payloads the model build consumes."""

    feature_space: dict
    features_train: dict
    features_test: dict
    folds: dict
    cuisines: dict


def _read_json_document(path: Path) -> dict:
    """Parse one input JSON file, failing fast with the path in context."""
    try:
        return json.loads(path.read_text(encoding=ARTIFACT_TEXT_ENCODING))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON ({error})") from error


def load_gold_model_inputs(
    gold_datasets_directory: Path = locations.GOLD_DATASETS_DIRECTORY,
    silver_datasets_directory: Path = locations.SILVER_DATASETS_DIRECTORY,
) -> GoldModelInputs:
    """Load the four gold datasets plus the silver cuisine taxonomy.

    Args:
        gold_datasets_directory: Directory holding the gold dataset files.
        silver_datasets_directory: Directory holding cuisines.json.

    Returns:
        GoldModelInputs with every payload parsed.

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: If any file holds invalid JSON.
    """
    return GoldModelInputs(
        feature_space=_read_json_document(
            gold_datasets_directory / FEATURE_SPACE_FILENAME
        ),
        features_train=_read_json_document(
            gold_datasets_directory / FEATURES_TRAIN_FILENAME
        ),
        features_test=_read_json_document(
            gold_datasets_directory / FEATURES_TEST_FILENAME
        ),
        folds=_read_json_document(gold_datasets_directory / FOLDS_FILENAME),
        cuisines=_read_json_document(
            silver_datasets_directory / CUISINES_FILENAME
        ),
    )


def compute_model_build_fingerprint(
    gold_datasets_directory: Path = locations.GOLD_DATASETS_DIRECTORY,
    silver_datasets_directory: Path = locations.SILVER_DATASETS_DIRECTORY,
) -> dict:
    """Fingerprint the inputs and environment that determine the model.

    Args:
        gold_datasets_directory: Directory holding the gold dataset files.
        silver_datasets_directory: Directory holding cuisines.json.

    Returns:
        Build block with the sha256 of each input file, the build's random
        seed, and the installed scikit-learn version.

    Raises:
        FileNotFoundError: If any input file is missing.
    """
    return {
        "cuisines_sha256": sha256_of_file(
            silver_datasets_directory / CUISINES_FILENAME
        ),
        "feature_space_sha256": sha256_of_file(
            gold_datasets_directory / FEATURE_SPACE_FILENAME
        ),
        "features_test_sha256": sha256_of_file(
            gold_datasets_directory / FEATURES_TEST_FILENAME
        ),
        "features_train_sha256": sha256_of_file(
            gold_datasets_directory / FEATURES_TRAIN_FILENAME
        ),
        "folds_sha256": sha256_of_file(gold_datasets_directory / FOLDS_FILENAME),
        "random_seed": BUILD_RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
    }
