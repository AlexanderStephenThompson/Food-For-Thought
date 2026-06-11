"""Make gold_pipeline and silver_pipeline importable for the gold test suite.

gold_pipeline imports silver_pipeline's shared infrastructure (artifact_io,
locations), so both tier roots go on sys.path — the silver insert keeps
`pytest 02-silver/tests` working standalone, not just the full-repo run.
"""
import sys
from pathlib import Path

TIER_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(TIER_ROOT))
sys.path.insert(0, str(TIER_ROOT.parent / "01-bronze"))
