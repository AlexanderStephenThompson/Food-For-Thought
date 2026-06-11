"""Shared builders for the synthetic recipes and train indexes tests use.

Test modules compose tiny corpora from (id, cuisine, ingredients) rows;
these builders turn those rows into real Recipe records and TrainIndex
structures so every suite constructs them the same way.
"""

from collections.abc import Iterable, Sequence

from pipeline.load_bronze_recipes import Recipe, TrainIndex, build_train_index

RecipeRow = tuple[int, str, Sequence[str]]


def make_index(recipe_rows: Iterable[RecipeRow]) -> TrainIndex:
    """Build a TrainIndex from (id, cuisine, ingredients) rows."""
    recipes = [
        Recipe(id=row[0], cuisine=row[1], ingredients=tuple(row[2]))
        for row in recipe_rows
    ]
    return build_train_index(recipes)


def repeat_recipes(
    start_id: int, cuisine: str, ingredients: Sequence[str], count: int
) -> list[RecipeRow]:
    """Generate count identical single-ingredient-list recipe rows."""
    return [(start_id + offset, cuisine, ingredients) for offset in range(count)]
