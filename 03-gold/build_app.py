"""Rebuild every app data asset from the gold artifacts and silver taxonomy.

Usage:
    .venv/bin/python 03-gold/build_app.py                    # full rebuild (seconds)
    .venv/bin/python 03-gold/build_app.py --check-idempotent # rebuild in memory,
                                                             # verify disk matches

Pure standard-library transforms of fingerprinted inputs: byte-identical
rebuilds on every environment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# app_pipeline reads gold data via gold_pipeline.locations, reuses
# model_pipeline's explanation math, and shares silver_pipeline's artifact
# I/O — so both earlier tier roots must be importable before the imports
# below. Python already puts this script's directory (03-gold) on sys.path.
TIER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TIER_ROOT.parent / "02-silver"))
sys.path.insert(0, str(TIER_ROOT.parent / "01-bronze"))

from app_pipeline import locations
from app_pipeline.asset_io import serialize_asset_json
from app_pipeline.export_contract_vectors import (
    CONTRACT_VECTORS_ASSET_FILENAME,
    build_contract_vectors_asset,
)
from app_pipeline.export_cuisines import (
    CUISINES_ASSET_FILENAME,
    build_cuisines_asset,
)
from app_pipeline.export_ingredients import (
    INGREDIENTS_ASSET_FILENAME,
    build_ingredients_asset,
)
from app_pipeline.export_model import MODEL_ASSET_FILENAME, build_model_asset
from app_pipeline.export_model_card import (
    MODEL_CARD_ASSET_FILENAME,
    build_model_card_asset,
)
from app_pipeline.load_app_inputs import (
    compute_app_build_fingerprint,
    load_app_export_inputs,
)
from app_pipeline.validate_app import validate_app_assets
from silver_pipeline.artifact_io import (
    find_artifact_mismatches,
    write_text_atomically,
)

MODEL_ASSET_PATH = locations.APP_DATA_DIRECTORY / MODEL_ASSET_FILENAME
INGREDIENTS_ASSET_PATH = locations.APP_DATA_DIRECTORY / INGREDIENTS_ASSET_FILENAME
CUISINES_ASSET_PATH = locations.APP_DATA_DIRECTORY / CUISINES_ASSET_FILENAME
MODEL_CARD_ASSET_PATH = locations.APP_DATA_DIRECTORY / MODEL_CARD_ASSET_FILENAME
CONTRACT_VECTORS_ASSET_PATH = (
    locations.APP_DATA_DIRECTORY / CONTRACT_VECTORS_ASSET_FILENAME
)

PROGRESS_LOG_FORMAT = "%(message)s"

logger = logging.getLogger(__name__)


@dataclass
class AppAssets:
    """Every asset one full app export produces."""

    model: dict
    ingredients: dict
    cuisines: dict
    model_card: dict
    contract_vectors: dict


def build_app_assets() -> AppAssets:
    """Run the full app export in memory.

    Returns:
        AppAssets with every asset, already validated against the inputs
        on disk.

    Raises:
        ValidationError: If any app gate fails.
        ValueError: If a contract vector references an unknown ingredient.
        FileNotFoundError: If an input file is missing.
    """
    fingerprint = compute_app_build_fingerprint()
    inputs = load_app_export_inputs()
    logger.info(
        "loaded app inputs: %d features, %d ingredients, %d cuisines",
        inputs.feature_space["feature_count"],
        len(inputs.ingredients["ingredients"]),
        len(inputs.cuisines["cuisines"]),
    )

    model_asset = build_model_asset(
        inputs.parameters, inputs.calibration, inputs.feature_space, fingerprint
    )
    ingredients_asset = build_ingredients_asset(inputs.ingredients, fingerprint)
    cuisines_asset = build_cuisines_asset(
        inputs.cuisines, inputs.evaluation, fingerprint
    )
    model_card_asset = build_model_card_asset(
        inputs.evaluation, inputs.calibration, inputs.parameters, fingerprint
    )
    contract_vectors_asset = build_contract_vectors_asset(model_asset, fingerprint)
    logger.info(
        "built assets: %d contract vectors, %d similarity edges",
        len(contract_vectors_asset["vectors"]),
        len(cuisines_asset["edges"]),
    )

    validate_app_assets(
        model_asset,
        ingredients_asset,
        cuisines_asset,
        model_card_asset,
        contract_vectors_asset,
        expected_fingerprint=fingerprint,
    )
    logger.info("validation gates: PASS")

    return AppAssets(
        model=model_asset,
        ingredients=ingredients_asset,
        cuisines=cuisines_asset,
        model_card=model_card_asset,
        contract_vectors=contract_vectors_asset,
    )


def _serialized_assets_by_path(assets: AppAssets) -> dict[Path, str]:
    """Pair every asset's destination path with its canonical bytes."""
    return {
        MODEL_ASSET_PATH: serialize_asset_json(assets.model),
        INGREDIENTS_ASSET_PATH: serialize_asset_json(assets.ingredients),
        CUISINES_ASSET_PATH: serialize_asset_json(assets.cuisines),
        MODEL_CARD_ASSET_PATH: serialize_asset_json(assets.model_card),
        CONTRACT_VECTORS_ASSET_PATH: serialize_asset_json(assets.contract_vectors),
    }


def write_app_assets(assets: AppAssets) -> None:
    """Persist every asset atomically to 04-app/data/."""
    locations.APP_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path, content in _serialized_assets_by_path(assets).items():
        write_text_atomically(content, path)
        logger.info("wrote %s (%d bytes)", path.name, len(content.encode("utf-8")))


def verify_rebuild_matches_disk(assets: AppAssets) -> list[str]:
    """Compare freshly built assets against the files on disk.

    Args:
        assets: Assets from build_app_assets.

    Returns:
        Names of asset files whose on-disk bytes differ from the rebuild
        (empty when the export is idempotent).
    """
    return find_artifact_mismatches(_serialized_assets_by_path(assets))


def main() -> int:
    """Entry point: rebuild app assets or verify idempotency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-idempotent",
        action="store_true",
        help="rebuild in memory and verify the app assets on disk match",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=PROGRESS_LOG_FORMAT)

    assets = build_app_assets()
    if not arguments.check_idempotent:
        write_app_assets(assets)
        return 0

    mismatches = verify_rebuild_matches_disk(assets)
    if mismatches:
        print(f"IDEMPOTENCY FAILURE: {', '.join(sorted(mismatches))}")
        return 1
    print("idempotency check: PASS (rebuild matches disk byte-for-byte)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
