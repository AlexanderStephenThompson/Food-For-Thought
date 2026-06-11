"""Bronze-to-silver data pipeline for the What's Cooking cuisine blend classifier.

Pure standard-library code. Transforms bronze Kaggle recipe JSON into three
canonical silver entities: the ingredient vocabulary (ingredients.json),
ID-resolved recipes (recipes_train.json / recipes_test.json), and the
cuisine taxonomy (cuisines.json).
"""
