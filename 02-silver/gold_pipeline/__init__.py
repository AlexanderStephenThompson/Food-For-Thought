"""Silver-to-gold data pipeline for the What's Cooking classifier.

Pure standard-library code. Transforms the canonical silver datasets into
model-ready gold artifacts: the ingredient feature space, per-recipe feature
rows with parent back-off indices, and a seeded stratified five-fold
cross-validation assignment. Deterministic end to end: every artifact embeds
a fingerprint of the four silver inputs, and rebuilds are byte-identical.
"""
