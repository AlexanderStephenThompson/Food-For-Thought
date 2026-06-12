"""Model pipeline for the What's Cooking cuisine blend classifier.

Trains a multinomial logistic regression over the gold feature rows,
calibrates the blend output with temperature scaling on pooled
out-of-fold logits, and produces the model artifacts: rounded parameters,
calibration, per-recipe test blends, the evaluation report, and the
Kaggle submission. scikit-learn enters the project here and only here —
every earlier tier stays pure standard library.
"""
