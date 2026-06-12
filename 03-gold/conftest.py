"""Make model_pipeline, gold_pipeline, and silver_pipeline importable.

model_pipeline reads gold datasets and reuses infrastructure from both
earlier tiers (gold_pipeline.locations, silver_pipeline.artifact_io, the
ingredient resolver), so all three tier roots go on sys.path. The parent
inserts keep `pytest 03-gold/tests` working standalone, not just the
full-repo run.
"""
import sys
from pathlib import Path

TIER_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(TIER_ROOT))
sys.path.insert(0, str(TIER_ROOT.parent / "02-silver"))
sys.path.insert(0, str(TIER_ROOT.parent / "01-bronze"))
