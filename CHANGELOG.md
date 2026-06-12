# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.5.0] — 2026-06-12 — Region Feel / Model: calibrated cuisine blend classifier

### Added
- Multinomial logistic regression over the gold features, tuned on the
  5 folds (C × parent-back-off-weight grid, 60 fits) by pooled
  out-of-fold log loss — the blend-honesty metric
- Temperature-scaled calibration fit on pooled out-of-fold logits
  (fixed-iteration golden-section search), with ECE and reliability
  tables before/after in `03-gold/model/calibration.json`
- Model artifacts derived entirely from 6-decimal rounded parameters:
  `parameters.json`, `blends_test.json` (per-recipe calibrated 20-way
  blend), `reports/evaluation.json` + `.md` (grid, per-fold metrics,
  per-cuisine recall, confusion pairs annotated with taxonomy neighbor
  similarities, MultinomialNB baseline), `submission/submission.csv`
- `03-gold/predict.py` CLI: resolves raw ingredient strings through the
  silver alias chain and prints the calibrated blend plus the model's
  own per-ingredient explanations
- `03-gold/model_pipeline/` package with 41 tests (suite now 434);
  12-gate `validate_model` including a scikit-learn version freshness
  check
- scikit-learn dependency at the model tier only — every data tier
  stays pure standard library; model byte-identity is per-environment
  and recorded in each artifact's build block

### Changed
- `manage.sh` rebuild/verify chain all three builders (the model grid
  adds minutes); `test` runs all three suites

---

## [0.4.0] — 2026-06-11 — Region Feel / Gold: feature space, features, stratified folds

### Added
- Gold tier datasets, regenerable from silver + code: `feature_space.json`
  (sorted-id ↔ index bijection with per-feature parent indices),
  `features_train.json` / `features_test.json` (per-recipe
  `ingredient_indices` plus deduplicated `parent_indices`), and
  `folds.json` (stratified 5-fold CV, independently seeded per cuisine,
  fold spread ≤ 1 within every cuisine)
- `02-silver/gold_pipeline/` package — load_silver_datasets,
  build_feature_space, build_features, assign_folds,
  build_fold_balance_report, validate_gold — built test-first (35 tests;
  suite now 393)
- `02-silver/build.py` orchestrator with `--check-idempotent`; fold-balance
  report (JSON + Markdown) in `02-silver/reports/`
- Gold build fingerprint: sha256 of all four silver inputs plus the build
  seed and fold count, embedded in every gold artifact and checked fresh
  against disk by the validation gates

### Changed
- `01-bronze/pipeline/` renamed to `01-bronze/silver_pipeline/`
  (convention: `<tier>_pipeline` is the package that builds that tier),
  clearing the package-name collision with the new gold pipeline
- `find_artifact_mismatches` moved into `silver_pipeline.artifact_io`,
  shared by both tier builders
- `manage.sh` rebuild/verify now chain both tiers; `test` runs both suites

---

## [0.3.0] — 2026-06-11 — Region Feel / Quality: single-owner contracts & naming

### Added
- `pipeline.artifact_io.serialize_artifact_json` and `write_text_atomically`:
  single owners of the canonical artifact byte format and the atomic write
  path (previously three copies of the temp-file rename dance and two copies
  of the serializer existed across modules)
- Test suites for `pipeline.artifact_io` and the rebuild verifier in
  `build.py` — 358 tests total, up from 342
- `tests/recipe_builders.py`: shared synthetic-recipe builders used by the
  vocabulary and alias-table suites (previously duplicated in both)
- Tier README for `02-silver/`
- `pyproject.toml` with pytest configuration so bare `pytest` works from the
  repo root

### Changed
- Renamed public functions for verb-first clarity: `group_frequency` →
  `count_group_recipes`, `representative_cleaned` →
  `select_representative_cleaned`, `monte_carlo_null95` →
  `estimate_null95_bits`
- `build.py` progress output goes through `logging` instead of `print`
  inside library functions; CLI output is unchanged
- The decision-id format (`__vs__`) is owned by
  `build_vocabulary.make_decision_id` and imported by the alias compiler;
  the review-queue writer no longer hardcodes it inline
- `follow_redirects` is the single shared redirect-chaser (two identical
  private copies removed)

### Fixed
- `apply_pair_outcomes` return type annotation declared a 3-tuple while the
  function returns 4 values
- Atomic writes now remove their temporary file when the write fails partway
- Coverage Markdown percentage precision is a named constant
  (`MARKDOWN_PERCENTAGE_DECIMAL_PLACES`) instead of a hardcoded format

---

## [0.2.0] — 2026-06-11 — Region Feel / Structure: Numbered medallion tiers

### Changed
- Tiers renamed to `01-bronze/`, `02-silver/`, `03-gold/`; bronze owns the
  pipeline, lexicons, tests, and reports that build silver; Kaggle source
  files moved to `01-bronze/data/`
- `manage.sh`, `locations.py`, and all tier READMEs updated to the new
  layout; organizing rule restated: each tier owns its data plus the code
  that transforms it into the next tier

---

## [0.1.0] — 2026-06-11 — Region Feel / Data Foundation: Silver datasets

### Added
- Bronze→silver pipeline: text normalization, singularization, modifier
  stripping, brand resolution, four-pass vocabulary build, statistical merge
  gate (Jensen-Shannon divergence vs a Monte Carlo null), alias compilation,
  recipe staging, validation gates, and coverage reporting
- Four canonical silver datasets: ingredients (2,813), train recipes
  (39,774), test recipes (9,944), and the 20-cuisine taxonomy
- 94 borderline merge decisions resolved via a judge/skeptic/tiebreaker
  review panel, each individually reversible
- Deterministic, idempotent rebuilds with SHA256 input fingerprints embedded
  in every artifact
