"""App-asset pipeline for the What's Cooking blend explorer.

Derives the static web app's data assets (04-app/data/) from the gold
model artifacts and silver taxonomy. Pure standard-library transforms of
fingerprinted inputs — no randomness, no scikit-learn — so rebuilds are
byte-identical on every environment, a stronger guarantee than the model
build's per-environment one.
"""
