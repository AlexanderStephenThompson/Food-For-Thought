"""Raw-to-staged data pipeline for the What's Cooking cuisine blend classifier.

Pure standard-library code. Transforms raw Kaggle recipe JSON into three
canonical staged entities: the ingredient vocabulary (ingredients.json),
ID-resolved recipes (recipes_train.json / recipes_test.json), and the
cuisine taxonomy (cuisines.json).
"""
