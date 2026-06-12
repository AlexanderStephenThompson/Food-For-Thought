"""Single source of truth for the model build's filesystem anchors.

Extends gold_pipeline.locations with the directories the model build
writes to; the re-exports keep model modules on one import path for every
anchor they need.

Constants:
    PROJECT_ROOT: Repository root containing the medallion tiers (re-export).
    SILVER_DATASETS_DIRECTORY: Canonical silver datasets (re-export; the
        predict CLI loads the ingredient vocabulary from here).
    LEXICONS_DIRECTORY: Curated lexicons (re-export; the predict CLI's
        resolver loads them).
    GOLD_ROOT: The gold tier (re-export).
    GOLD_DATASETS_DIRECTORY: Gold data artifacts the model reads (re-export).
    GOLD_MODEL_DIRECTORY: Trained parameters, calibration, and test blends.
    GOLD_REPORTS_DIRECTORY: The evaluation report.
    GOLD_SUBMISSION_DIRECTORY: The Kaggle submission file.
"""

from __future__ import annotations

from gold_pipeline.locations import (
    GOLD_DATASETS_DIRECTORY,
    GOLD_ROOT,
    PROJECT_ROOT,
    SILVER_DATASETS_DIRECTORY,
)
from silver_pipeline.locations import LEXICONS_DIRECTORY

GOLD_MODEL_DIRECTORY = GOLD_ROOT / "model"
GOLD_REPORTS_DIRECTORY = GOLD_ROOT / "reports"
GOLD_SUBMISSION_DIRECTORY = GOLD_ROOT / "submission"
