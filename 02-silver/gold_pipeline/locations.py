"""Single source of truth for the gold build's filesystem anchors.

Extends silver_pipeline.locations with the anchors the silver-to-gold build
reads from and writes to; the re-exports keep gold modules on one import
path for every anchor they need.

Constants:
    PROJECT_ROOT: Repository root containing the medallion tiers (re-export).
    SILVER_DATASETS_DIRECTORY: Canonical silver inputs (re-export).
    SILVER_REPORTS_DIRECTORY: Reports the gold build writes (fold balance).
    GOLD_ROOT: The gold tier — model-ready datasets only.
    GOLD_DATASETS_DIRECTORY: Gold artifacts (feature space, features, folds).
"""

from __future__ import annotations

from silver_pipeline.locations import (
    PROJECT_ROOT,
    SILVER_DATASETS_DIRECTORY,
    SILVER_ROOT,
)

SILVER_REPORTS_DIRECTORY = SILVER_ROOT / "reports"
GOLD_ROOT = PROJECT_ROOT / "03-gold"
GOLD_DATASETS_DIRECTORY = GOLD_ROOT / "datasets"
