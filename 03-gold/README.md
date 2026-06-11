# Gold tier — model-ready data

Produced by `02-silver/gold_pipeline/` and regenerable from silver + code,
the same way silver is regenerable from bronze + lexicons. Every artifact
embeds a fingerprint of the four silver inputs plus the build's random seed
and fold count, so a gold build from stale silver is detectable.

| File | What it is |
|------|------------|
| `datasets/feature_space.json` | Ingredient id ↔ feature index bijection (sorted ids, 2,813 features); each feature carries its parent's index (or null) for back-off |
| `datasets/features_train.json` | 39,774 labeled rows: `ingredient_indices` plus deduplicated `parent_indices` per recipe |
| `datasets/features_test.json` | 9,944 unlabeled rows, same form minus cuisine |
| `datasets/folds.json` | Stratified 5-fold cross-validation assignment, independently seeded per cuisine; fold spread ≤ 1 within every cuisine |

Feature rows keep direct ingredient evidence and parent back-off in
separate index lists, so the model phase chooses how to weight back-off
instead of having that decision baked into the data.

## Still to come (the model phase)

- the blend model itself ("58% thai · 24% vietnamese"), trained over the folds
- calibration artifacts for the blend output
- the Kaggle submission file (argmax over the blend)
