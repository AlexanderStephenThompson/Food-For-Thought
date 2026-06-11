# Gold tier — model-ready data

This tier is intentionally empty until the model phase. It will hold the
artifacts the classifier trains and serves from:

- feature matrices built from `silver/recipes_train.json` (canonical
  ingredient IDs, with `parent_id` back-off from `silver/ingredients.json`)
- stratified train/validation splits (the 20 cuisines are imbalanced:
  italian 7,838 recipes, brazilian 467)
- calibration artifacts for the blend output ("58% thai · 24% vietnamese")
- the Kaggle submission file (argmax over the blend)

Everything here must be regenerable from silver + code, the same way
silver is regenerable from bronze + lexicons.
