# Gold tier — model-ready data

This tier is intentionally empty until the model phase. When it lands, gold
mirrors silver's internal shape:

```
gold/
├── datasets/    # feature matrices, stratified splits, calibration artifacts,
│                # the Kaggle submission file (argmax over the blend)
├── pipeline/    # silver -> gold transforms (feature build, training, calibration)
└── tests/       # the gold pipeline's suite
```

Planned contents:

- feature matrices built from `silver/datasets/recipes_train.json` (canonical
  ingredient IDs, with `parent_id` back-off from `silver/datasets/ingredients.json`)
- stratified train/validation splits (the 20 cuisines are imbalanced:
  italian 7,838 recipes, brazilian 467)
- calibration artifacts for the blend output ("58% thai · 24% vietnamese")

Everything here must be regenerable from silver + code, the same way silver
is regenerable from bronze + lexicons.
