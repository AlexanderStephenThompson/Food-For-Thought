"""Single source of truth for the project's filesystem anchors.

Every path the pipeline reads from or writes to is derived from the
constants below. Modules import these anchors instead of recomputing
``Path(__file__)`` parent chains, so a future move of any tier only
requires updating this one file.

Constants:
    PROJECT_ROOT: Repository root containing the medallion tiers.
    BRONZE_ROOT: The bronze tier — raw Kaggle data plus the pipeline
        code, lexicons, tests, and reports that build the silver tier.
    BRONZE_DATA_DIRECTORY: Raw Kaggle competition JSON (train/test).
    LEXICONS_DIRECTORY: Curated lexicons that drive the vocabulary build.
    REPORTS_DIRECTORY: Build reports and the merge review queue.
    SILVER_ROOT: The silver tier — canonical datasets only.
    SILVER_DATASETS_DIRECTORY: Canonical silver entities (ingredients,
        recipes, cuisines).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_ROOT = PROJECT_ROOT / "01-bronze"
BRONZE_DATA_DIRECTORY = BRONZE_ROOT / "data"
LEXICONS_DIRECTORY = BRONZE_ROOT / "lexicons"
REPORTS_DIRECTORY = BRONZE_ROOT / "reports"
SILVER_ROOT = PROJECT_ROOT / "02-silver"
SILVER_DATASETS_DIRECTORY = SILVER_ROOT / "datasets"
