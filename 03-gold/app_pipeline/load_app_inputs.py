"""Load and fingerprint the six inputs the app export consumes.

Inputs: the gold model artifacts (parameters, calibration), the gold
feature space, the evaluation report, and the silver ingredient/cuisine
taxonomies. No file I/O happens at import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app_pipeline import locations
from silver_pipeline.artifact_io import ARTIFACT_TEXT_ENCODING, sha256_of_file

PARAMETERS_RELATIVE_PATH = Path("model/parameters.json")
CALIBRATION_RELATIVE_PATH = Path("model/calibration.json")
FEATURE_SPACE_RELATIVE_PATH = Path("datasets/feature_space.json")
EVALUATION_RELATIVE_PATH = Path("reports/evaluation.json")
INGREDIENTS_FILENAME = "ingredients.json"
CUISINES_FILENAME = "cuisines.json"


@dataclass(frozen=True)
class AppExportInputs:
    """The six parsed payloads the app export consumes."""

    parameters: dict
    calibration: dict
    feature_space: dict
    evaluation: dict
    ingredients: dict
    cuisines: dict


def _read_json_document(path: Path) -> dict:
    """Parse one input file, failing fast with the path in context."""
    try:
        return json.loads(path.read_text(encoding=ARTIFACT_TEXT_ENCODING))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON ({error})") from error


def load_app_export_inputs(
    gold_root: Path = locations.GOLD_ROOT,
    silver_datasets_directory: Path = locations.SILVER_DATASETS_DIRECTORY,
) -> AppExportInputs:
    """Load every exporter input from the gold and silver tiers.

    Args:
        gold_root: Gold tier root holding model/, datasets/, reports/.
        silver_datasets_directory: Directory holding the silver taxonomies.

    Returns:
        AppExportInputs with every payload parsed.

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: If any file holds invalid JSON.
    """
    return AppExportInputs(
        parameters=_read_json_document(gold_root / PARAMETERS_RELATIVE_PATH),
        calibration=_read_json_document(gold_root / CALIBRATION_RELATIVE_PATH),
        feature_space=_read_json_document(gold_root / FEATURE_SPACE_RELATIVE_PATH),
        evaluation=_read_json_document(gold_root / EVALUATION_RELATIVE_PATH),
        ingredients=_read_json_document(
            silver_datasets_directory / INGREDIENTS_FILENAME
        ),
        cuisines=_read_json_document(silver_datasets_directory / CUISINES_FILENAME),
    )


def compute_app_build_fingerprint(
    gold_root: Path = locations.GOLD_ROOT,
    silver_datasets_directory: Path = locations.SILVER_DATASETS_DIRECTORY,
) -> dict:
    """Fingerprint the six inputs that determine every app asset.

    The app build is a pure transform — no seed, no library version — so
    the fingerprint is exactly the input hashes.

    Args:
        gold_root: Gold tier root holding model/, datasets/, reports/.
        silver_datasets_directory: Directory holding the silver taxonomies.

    Returns:
        Build block with one sha256 per input file.

    Raises:
        FileNotFoundError: If any input file is missing.
    """
    return {
        "calibration_sha256": sha256_of_file(
            gold_root / CALIBRATION_RELATIVE_PATH
        ),
        "cuisines_sha256": sha256_of_file(
            silver_datasets_directory / CUISINES_FILENAME
        ),
        "evaluation_sha256": sha256_of_file(gold_root / EVALUATION_RELATIVE_PATH),
        "feature_space_sha256": sha256_of_file(
            gold_root / FEATURE_SPACE_RELATIVE_PATH
        ),
        "ingredients_sha256": sha256_of_file(
            silver_datasets_directory / INGREDIENTS_FILENAME
        ),
        "parameters_sha256": sha256_of_file(gold_root / PARAMETERS_RELATIVE_PATH),
    }
