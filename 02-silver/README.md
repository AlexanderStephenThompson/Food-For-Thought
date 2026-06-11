# Silver tier — canonical datasets and the gold build

`datasets/` holds the product of the bronze→silver pipeline in `01-bronze/`.
Every file there is regenerable from bronze data + lexicons via
`./manage.sh rebuild` and is never edited by hand.

| File | What it is |
|------|------------|
| `datasets/ingredients.json` | Canonical vocabulary: 2,813 ingredients; every raw string mapped with provenance and statistical evidence |
| `datasets/recipes_train.json` | 39,774 labeled recipes, ingredients as canonical IDs |
| `datasets/recipes_test.json` | 9,944 unlabeled recipes, same form |
| `datasets/cuisines.json` | 20-cuisine taxonomy: families, neighbors, distinctive ingredients |

Each artifact embeds a build fingerprint (train-file hash, lexicon hash,
random seed), so stale data is detectable. `./manage.sh verify` proves a
rebuild matches these files byte-for-byte.

The rest of this tier is everything that turns silver into gold:

- `gold_pipeline/` — the silver→gold transform modules (see the gold module
  map in the root README)
- `reports/` — the fold-balance report the gold build writes
- `tests/` — the gold pipeline's suite
- `build.py` — the gold rebuild entry point

Run everything through the task runner at the repo root: `./manage.sh
<test|rebuild|verify|all>`.
