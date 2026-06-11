"""Single source of truth for the project's filesystem anchors.

Every path the pipeline reads from or writes to is derived from the
constants below. Modules import these anchors instead of recomputing
``Path(__file__)`` parent chains, so a future move of any tier only
requires updating this one file.

Constants:
    PROJECT_ROOT: Repository root containing the medallion tiers.
    BRONZE_KAGGLE_DIRECTORY: Raw Kaggle competition JSON (train/test).
    SILVER_ROOT: The silver tier — pipeline code, lexicons, datasets,
        tests, and reports.
    SILVER_DATASETS_DIRECTORY: Canonical silver entities (ingredients,
        recipes, cuisines).
    LEXICONS_DIRECTORY: Curated lexicons that drive the vocabulary build.
    REPORTS_DIRECTORY: Build reports and the merge review queue.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_KAGGLE_DIRECTORY = PROJECT_ROOT / "bronze" / "kaggle"
SILVER_ROOT = PROJECT_ROOT / "silver"
SILVER_DATASETS_DIRECTORY = SILVER_ROOT / "datasets"
LEXICONS_DIRECTORY = SILVER_ROOT / "lexicons"
REPORTS_DIRECTORY = SILVER_ROOT / "reports"
