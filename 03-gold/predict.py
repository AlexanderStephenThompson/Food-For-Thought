"""Launcher for the cuisine blend prediction CLI.

Usage:
    .venv/bin/python 03-gold/predict.py \\
        --ingredients "fish sauce, coconut milk, thai basil"

See model_pipeline.predict_blend for the CLI body.
"""

from __future__ import annotations

import sys
from pathlib import Path

TIER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TIER_ROOT.parent / "02-silver"))
sys.path.insert(0, str(TIER_ROOT.parent / "01-bronze"))

from model_pipeline.predict_blend import main

if __name__ == "__main__":
    sys.exit(main())
