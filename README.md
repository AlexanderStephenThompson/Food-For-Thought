# What's Cooking — Cuisine Blend Classifier

Predicts the cuisine of a dish from its ingredient list — as a **blend**, not a single label
("58% thai · 24% vietnamese · 11% chinese"), with per-prediction ingredient explanations.

Data: Kaggle "What's Cooking" playground competition — 39,774 training recipes,
9,944 test recipes, 20 cuisines.

## Layout

| Path | What it is |
|------|------------|
| `raw/kaggle/` | Immutable source data, exactly as downloaded |
| `lexicons/` | Curated, version-controlled mapping data: modifier lists, brand patterns, merge/preserve rules, resolved merge decisions |
| `pipeline/` | Pure-stdlib transform code (raw → staged) |
| `staged/` | Generated canonical entities: ingredient vocabulary, ID-resolved recipes, cuisine taxonomy |
| `reports/` | Generated review queue and coverage reports |
| `tests/` | Pytest suite (TDD) |

## Core rule

Raw ingredient strings ("Kikkoman Less Sodium Soy Sauce") resolve to canonical ingredient IDs
(`soy_sauce`) via an evidence-driven alias table. Variants are merged **unless the variant
carries cuisine signal** — "dark soy sauce" stays separate from "soy sauce" (74% Chinese vs 41%),
"low sodium soy sauce" does not. Every merge decision records its provenance and statistical
evidence (Jensen-Shannon divergence vs a Monte Carlo null).

## Usage

```bash
# one-time setup
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install pytest

# rebuild staged data from raw + lexicons
.venv/bin/python run_pipeline.py

# verify a rebuild produces byte-identical output
.venv/bin/python run_pipeline.py --check-idempotent

# run tests
.venv/bin/python -m pytest
```
