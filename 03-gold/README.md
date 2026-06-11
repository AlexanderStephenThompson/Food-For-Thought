# Gold tier — model-ready data

This tier is intentionally empty until the model phase. When it lands, gold
holds `datasets/` only — the silver→gold pipeline and its tests will live in
`02-silver/`, beside the datasets they read, the same way `01-bronze/` holds
the bronze→silver build.

Planned contents:

- feature matrices built from `02-silver/datasets/recipes_train.json`
  (canonical ingredient IDs, with `parent_id` back-off from
  `02-silver/datasets/ingredients.json`)
- stratified train/validation splits (the 20 cuisines are imbalanced:
  italian 7,838 recipes, brazilian 467)
- calibration artifacts for the blend output ("58% thai · 24% vietnamese")
- the Kaggle submission file (argmax over the blend)

Everything here must be regenerable from silver + code, the same way silver
is regenerable from bronze + lexicons.
