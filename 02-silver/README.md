# Silver tier — canonical datasets

The product of the bronze→silver pipeline in `01-bronze/`. Every file here
is regenerable from bronze data + lexicons via `./manage.sh rebuild` and is
never edited by hand.

| File | What it is |
|------|------------|
| `datasets/ingredients.json` | Canonical vocabulary: 2,813 ingredients; every raw string mapped with provenance and statistical evidence |
| `datasets/recipes_train.json` | 39,774 labeled recipes, ingredients as canonical IDs |
| `datasets/recipes_test.json` | 9,944 unlabeled recipes, same form |
| `datasets/cuisines.json` | 20-cuisine taxonomy: families, neighbors, distinctive ingredients |

Each artifact embeds a build fingerprint (train-file hash, lexicon hash,
random seed), so stale data is detectable. `./manage.sh verify` proves a
rebuild matches these files byte-for-byte.

When the model phase lands, the silver→gold pipeline and its tests will live
in this tier, beside the datasets they read — see `../03-gold/README.md`.
