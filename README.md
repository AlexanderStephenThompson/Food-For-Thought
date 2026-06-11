# What's Cooking — Cuisine Blend Classifier

Predicts the cuisine of a dish from its ingredient list — as a **blend**, not a single label
("58% thai · 24% vietnamese · 11% chinese"), with per-prediction ingredient explanations.

Data: Kaggle "What's Cooking" playground competition — 39,774 training recipes,
9,944 test recipes, 20 cuisines.

## Layout — medallion architecture

**One rule organizes everything: each tier owns its data plus the code, rules,
tests, and reports that transform it into the next tier.**

| Tier | Produced by | Contents |
|------|-------------|----------|
| `01-bronze/` | Downloading from Kaggle | `data/` (immutable source files) · `pipeline/` (bronze→silver transforms) · `lexicons/` (curated rules & merge decisions) · `reports/` (coverage, review queue) · `tests/` · `build.py` |
| `02-silver/` | `01-bronze/pipeline/` | `datasets/` — the four canonical entities |
| `03-gold/` | The model phase (upcoming) | `datasets/` only — see `03-gold/README.md` |

## Silver datasets (the current product)

| File | What it is |
|------|------------|
| `02-silver/datasets/ingredients.json` | Canonical vocabulary: 2,813 ingredients, every raw string mapped with provenance and statistical evidence |
| `02-silver/datasets/recipes_train.json` | 39,774 labeled recipes, ingredients as canonical IDs |
| `02-silver/datasets/recipes_test.json` | 9,944 unlabeled recipes, same form |
| `02-silver/datasets/cuisines.json` | 20-cuisine taxonomy: families, neighbors, distinctive ingredients |

## Pipeline module map (`01-bronze/pipeline/`)

In data-flow order:

| Stage | Modules |
|-------|---------|
| Load bronze | `load_bronze_recipes` |
| String cleaning | `normalize_text` → `singularize` → `strip_modifiers` → `resolve_brands` |
| Evidence & merge gate | `cuisine_divergence`, `merge_evidence`, `merge_gate` |
| Vocabulary build | `build_vocabulary` → review queue → `compile_alias_table` |
| Silver staging | `resolve_ingredient` (runtime fallback chain), `transform_bronze_to_silver`, `build_cuisines` |
| Quality gates | `validate_silver`, `build_coverage_report` |
| Shared infrastructure | `artifact_io` (serialization, atomic writes, fingerprints), `locations` (every filesystem anchor) |

## Core rule of the vocabulary

Raw ingredient strings ("Kikkoman Less Sodium Soy Sauce") resolve to canonical ingredient IDs
(`soy_sauce`) via an evidence-driven alias table. Variants are merged **unless the variant
carries cuisine signal** — "dark soy sauce" stays separate from "soy sauce" (74% Chinese vs 41%),
"low sodium soy sauce" does not. Every merge decision records its provenance and statistical
evidence (Jensen-Shannon divergence vs a Monte Carlo null); borderline calls were resolved by
a judge/skeptic/tiebreaker review panel and live in `01-bronze/lexicons/merge_decisions.jsonl`,
each one individually reversible.

## Usage

```bash
# one-time setup
python3 -m venv --without-pip .venv
.venv/bin/python get-pip.py          # system python has no pip/ensurepip
.venv/bin/python -m pip install pytest

./manage.sh test      # run the suite (358 tests)
./manage.sh rebuild   # rebuild silver from bronze + lexicons (~35s)
./manage.sh verify    # prove a rebuild matches disk byte-for-byte
./manage.sh all       # the full confidence pass
```

To interrogate a merge decision (or evaluate a new variant when the vocabulary grows):

```bash
PYTHONPATH=01-bronze .venv/bin/python -m pipeline.cuisine_divergence --variant "dark soy sauce" --base "soy sauce"
```

## Known limitations (candidates for future refinement)

- **Parent links are back-off hints, not an ontology.** Head-token pairing
  gives some preserved variants a semantically loose parent (e.g.
  `green_pepper` — a vegetable — parents to `pepper`, whose recipes skew
  toward the spice). The variants themselves stay correctly separate.
- **Alias provenance records the last hop only.** A string that merged via
  modifier stripping and whose group later merged again shows the final
  merge's source/rule.
- **Preserve evidence is measured against the immediate base** while
  `parent_id` points at the chain root (schema requires root parents).
- **Vocabulary fragmentation tail**: a few near-duplicate canonicals remain
  (`lime_leaves`/`kaffir_lime_leaves`, `lemongrass`/`lemon_grass`,
  `harissa`/`harissa_paste`, `yoghurt`/`yogurt`). Fix by adding
  `01-bronze/lexicons/manual_aliases.json` entries — but check the
  cuisine-divergence CLI first; British spellings can carry real cuisine signal.
- **~100 canonical names retain prep tokens** (e.g. `fresh_lemon_juice`)
  where the prep form is the dominant surface and no bare-form group
  existed to anchor the name.
