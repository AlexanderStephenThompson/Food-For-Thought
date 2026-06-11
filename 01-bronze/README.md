# Bronze tier — immutable source data and the silver build

`data/` holds the "What's Cooking" competition files exactly as downloaded
(train.json, test.json, sample_submission.csv, plus the original archives as
provenance). Nothing in `data/` is ever edited or written by the pipeline —
silver is rebuilt *from* it, never the other way around.

The rest of this tier is everything that turns bronze into silver:

- `pipeline/` — the bronze→silver transform modules (see the module map in
  the root README)
- `lexicons/` — curated rules and reviewed merge decisions that drive the
  vocabulary build
- `reports/` — build reports and the merge review queue
- `tests/` — the pipeline's suite
- `build.py` — the rebuild entry point

Run everything through the task runner at the repo root: `./manage.sh
<test|rebuild|verify|all>`.
