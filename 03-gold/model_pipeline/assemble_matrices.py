"""Assemble sparse design matrices and label vectors from gold feature rows.

The parent back-off rule is max semantics: a feature cell is 1.0 when the
recipe contains the ingredient directly, parent_weight when the index is
only a parent of one of its ingredients, and never their sum — direct
evidence wins, back-off only fills absence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy
from scipy.sparse import csr_matrix

DIRECT_PRESENCE_VALUE = 1.0


def assemble_design_matrix(
    rows: Sequence[dict], feature_count: int, parent_weight: float
) -> csr_matrix:
    """Build the sparse feature matrix for one split of gold rows.

    Args:
        rows: Gold feature rows ({ingredient_indices, parent_indices, ...}).
        feature_count: Width of the feature space.
        parent_weight: Value given to parent back-off features; 0.0
            disables back-off entirely.

    Returns:
        CSR matrix of shape (len(rows), feature_count) with float values.
    """
    row_positions: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for position, row in enumerate(rows):
        direct_indices = set(row["ingredient_indices"])
        for index in row["ingredient_indices"]:
            row_positions.append(position)
            column_indices.append(index)
            values.append(DIRECT_PRESENCE_VALUE)
        if parent_weight <= 0.0:
            continue
        for parent_index in row["parent_indices"]:
            if parent_index in direct_indices:
                continue
            row_positions.append(position)
            column_indices.append(parent_index)
            values.append(parent_weight)
    return csr_matrix(
        (values, (row_positions, column_indices)),
        shape=(len(rows), feature_count),
    )


def encode_cuisine_labels(
    rows: Sequence[dict], cuisine_ids: Sequence[str]
) -> numpy.ndarray:
    """Encode each row's cuisine as its index into the sorted cuisine list.

    Args:
        rows: Labeled gold feature rows (each carrying a cuisine).
        cuisine_ids: Sorted cuisine identifiers defining the label order.

    Returns:
        Integer label array aligned with the rows.

    Raises:
        KeyError: If a row carries a cuisine outside cuisine_ids.
    """
    index_by_cuisine = {
        cuisine: position for position, cuisine in enumerate(cuisine_ids)
    }
    return numpy.array(
        [index_by_cuisine[row["cuisine"]] for row in rows], dtype=numpy.int64
    )


def split_row_positions_by_fold(
    rows: Sequence[dict], folds_payload: dict
) -> dict[int, list[int]]:
    """Group row positions (not recipe ids) by their assigned fold.

    Args:
        rows: Gold feature rows in matrix order.
        folds_payload: Gold folds artifact mapping recipe ids to folds.

    Returns:
        Mapping of fold number -> row positions, covering every row once.

    Raises:
        KeyError: If a row's recipe id has no fold assignment.
    """
    fold_by_recipe_id = {
        assignment["recipe_id"]: assignment["fold"]
        for assignment in folds_payload["assignments"]
    }
    positions_by_fold: dict[int, list[int]] = {}
    for position, row in enumerate(rows):
        fold = fold_by_recipe_id[row["recipe_id"]]
        positions_by_fold.setdefault(fold, []).append(position)
    return positions_by_fold
