"""Single source of truth for the app build's filesystem anchors.

Constants:
    PROJECT_ROOT: Repository root containing the medallion tiers (re-export).
    GOLD_ROOT: The gold tier holding the exporter's inputs (re-export).
    SILVER_DATASETS_DIRECTORY: Silver taxonomy inputs (re-export).
    APP_ROOT: The app tier — the static site.
    APP_DATA_DIRECTORY: Generated data assets the pages fetch.
"""

from __future__ import annotations

from gold_pipeline.locations import GOLD_ROOT, PROJECT_ROOT, SILVER_DATASETS_DIRECTORY

APP_ROOT = PROJECT_ROOT / "04-app"
APP_DATA_DIRECTORY = APP_ROOT / "data"
