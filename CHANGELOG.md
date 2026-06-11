# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

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
