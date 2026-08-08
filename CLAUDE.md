# CLAUDE.md

<!-- LINEAGE: layer=3 brand=crop-and-chop hub=../_Assets/crop-and-chop-AI-Hub -->

Working conventions for this repository. `README.md` describes what the project
is and how to run it — read it first. This file covers the rules that are easy
to break and expensive to unbreak.

## The one rule the layout enforces

**Each tier owns its data plus the code, rules, tests, and reports that turn it
into the next tier.** `01-bronze/silver_pipeline/` produces silver;
`02-silver/gold_pipeline/` produces gold; `03-gold/model_pipeline/` and
`app_pipeline/` produce the model and the app's assets.

The consequence people get wrong: **code lives with the tier it reads from, not
the tier it writes to.** The model reads gold datasets, so `model_pipeline/`
sits in `03-gold/`, not in a top-level `src/`. Before adding a module, ask which
tier's data it consumes — that answers where it goes.

## Dependency boundary

Every data tier is **pure stdlib**. scikit-learn is permitted in the model tier
only (`03-gold/model_pipeline/`), and the web app in `04-app/` has zero
dependencies at all.

This is not a preference. It is what lets the data tiers rebuild identically
anywhere, and it is why `04-app/`'s assets are byte-identical across
environments while the model build only guarantees per-environment
reproducibility. Adding an import to a data tier silently gives that guarantee
away.

## Determinism is a tested property

`./manage.sh verify` rebuilds everything and proves it matches disk
byte-for-byte. Treat a `verify` failure as a real defect, not as noise from a
seed or a timestamp — the pipeline is written to have neither.

Each tier gates its inputs with a fingerprint, and the model's fingerprint
includes the sklearn version. If you upgrade sklearn, expect the model
fingerprint to change and the artifacts with it; that is the mechanism working.

## The scorer contract

`03-gold/app_pipeline/score_blend.py` is a no-numpy scorer that mirrors the
browser's JavaScript **op-for-op**, and `export_contract_vectors` exists to pin
the two together. Change one side without the other and the app will disagree
with the CLI in ways no unit test catches on its own. Change both, then run the
contract vectors.

## Vocabulary changes

Merges are evidence-driven and individually reversible. Before adding anything
to `01-bronze/lexicons/manual_aliases.json`, run the divergence CLI:

```bash
PYTHONPATH=01-bronze .venv/bin/python -m silver_pipeline.cuisine_divergence \
  --variant "dark soy sauce" --base "soy sauce"
```

A variant that carries cuisine signal must stay separate — "dark soy sauce" is
74% Chinese against the base's 41%, so merging it would destroy a real feature.
British spellings can carry signal too; check before assuming a duplicate. Every
decision lands in `01-bronze/lexicons/merge_decisions.jsonl` with its
provenance, so a merge can always be undone.

The known-fragmentation list at the end of `README.md` is a working queue, not
a defect list — each entry has been looked at and left deliberately.

## Environment

System Python has no `pip` or `ensurepip`, so the venv is built in two steps:

```bash
python3 -m venv --without-pip .venv
.venv/bin/python get-pip.py
```

The app's JavaScript suite needs Node >= 22.7 and installs nothing.

## Before calling a change done

```bash
./manage.sh all       # tests, rebuild, and verify in one pass
```

`./manage.sh test` alone runs 471 Python tests and 31 JavaScript tests but does
not prove the artifacts on disk still match what the code produces. `all` does.

## Publishing

This repository is public (`AlexanderStephenThompson/Food-For-Thought`) and
carries a `LICENSE`. `CHANGELOG.md` follows Keep a Changelog with semantic
versioning — update it in the same change that ships the behaviour, not after.
