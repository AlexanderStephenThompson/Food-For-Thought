# Bronze tier — immutable source data

`kaggle/` holds the "What's Cooking" competition files exactly as downloaded
(train.json, test.json, sample_submission.csv, plus the original archives as
provenance). Nothing in this tier is ever edited or written by the pipeline —
silver is rebuilt *from* it, never the other way around.

Bronze has no code by design: this tier is produced by downloading, not by
the pipeline. The code that consumes it lives in `silver/pipeline/`.
