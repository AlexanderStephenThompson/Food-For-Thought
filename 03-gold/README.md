# Gold tier — model-ready data and the blend model

The data half is produced by `02-silver/gold_pipeline/`; the model half by
this tier's own `model_pipeline/` (the last tier transforms its own data
into model artifacts). Everything is regenerable from silver + code via
`./manage.sh rebuild`.

## Data artifacts (`datasets/`)

| File | What it is |
|------|------------|
| `datasets/feature_space.json` | Ingredient id ↔ feature index bijection (sorted ids, 2,813 features); each feature carries its parent's index (or null) for back-off |
| `datasets/features_train.json` | 39,774 labeled rows: `ingredient_indices` plus deduplicated `parent_indices` per recipe |
| `datasets/features_test.json` | 9,944 unlabeled rows, same form minus cuisine |
| `datasets/folds.json` | Stratified 5-fold cross-validation assignment, independently seeded per cuisine; fold spread ≤ 1 within every cuisine |

## Model artifacts

| File | What it is |
|------|------------|
| `model/parameters.json` | Multinomial logistic regression coefficients (20 × 2,813) and intercepts, rounded to 6 decimals — every downstream artifact derives from these rounded values |
| `model/calibration.json` | Blend temperature (fit on pooled out-of-fold logits) with ECE and reliability tables before/after |
| `model/blends_test.json` | Per-test-recipe calibrated 20-cuisine blend (4 decimals) + top cuisine |
| `reports/evaluation.json` + `.md` | Grid search, per-fold and mean metrics, per-cuisine recall, confusion pairs annotated with the taxonomy's neighbor similarities, Naive Bayes baseline comparison |
| `submission/submission.csv` | Kaggle submission: argmax over the blend, `id,cuisine` format |

## How the model treats parent back-off (max semantics)

A feature cell is 1.0 when the recipe contains the ingredient directly,
`parent_weight` when the index is only a parent of one of its ingredients —
never their sum. Direct evidence wins; back-off only fills absence. The
weight itself is a tuned hyperparameter (`parent_weight` in
`model/parameters.json`), selected on the folds by pooled out-of-fold log
loss.

## Determinism is environment-conditional here

The data tiers are byte-identical everywhere; the model tier is
byte-identical *per environment*. Every model artifact embeds the
scikit-learn version in its build block, and the validation gates compare
it against the installed version — `--check-idempotent` passes under the
environment that produced the artifacts, and a version change fails loudly
instead of masquerading as a determinism bug.

## Try it

```bash
.venv/bin/python 03-gold/predict.py --ingredients "fish sauce, coconut milk, thai basil, lime juice, rice noodles"
```

Prints the resolution of each raw string through the silver alias chain,
the calibrated blend, and the model's own per-ingredient explanation
(contribution = feature value × coefficient).

## App data export (`app_pipeline/`)

A second builder, `build_app.py`, derives the static web app's data assets
(`04-app/data/`) from the model artifacts and silver taxonomy:
`model.json` (4-decimal coefficients), `ingredients.json`, `cuisines.json`,
`model-card.json`, and `contract-vectors.json`. Unlike the model build,
this export is pure standard-library — no scikit-learn, no randomness — so
its assets are byte-identical on **every** environment, not just the one
that trained the model. The contract vectors pin the browser's JavaScript
scorer to the Python model's exact output. Run it with
`.venv/bin/python 03-gold/build_app.py` (seconds, not minutes).
