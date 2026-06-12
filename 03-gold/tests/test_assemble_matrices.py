"""Tests for model_pipeline.assemble_matrices.

Synthetic corpus indices: basil=0, dark_soy_sauce=1 (parent soy_sauce=5),
fish_sauce=2, pasta=3, rice=4, soy_sauce=5.
"""

from model_pipeline.assemble_matrices import (
    assemble_design_matrix,
    encode_cuisine_labels,
    split_row_positions_by_fold,
)
from tests.model_payload_builders import (
    make_feature_row,
    make_folds_payload,
    make_train_feature_rows,
)

SYNTHETIC_FEATURE_COUNT = 6
CUISINE_IDS = ("italian", "mexican", "thai")
PARENT_WEIGHT = 0.3


def test_design_matrix_sets_one_for_direct_ingredients():
    rows = [make_feature_row(1, [0, 3], cuisine="italian")]

    matrix = assemble_design_matrix(rows, SYNTHETIC_FEATURE_COUNT, PARENT_WEIGHT)

    dense = matrix.toarray()
    assert dense[0, 0] == 1.0
    assert dense[0, 3] == 1.0
    assert dense[0, 4] == 0.0


def test_design_matrix_sets_parent_weight_for_absent_parents():
    rows = [make_feature_row(9, [1, 2], [5], cuisine="thai")]

    matrix = assemble_design_matrix(rows, SYNTHETIC_FEATURE_COUNT, PARENT_WEIGHT)

    dense = matrix.toarray()
    assert dense[0, 1] == 1.0
    assert dense[0, 5] == PARENT_WEIGHT


def test_direct_presence_wins_over_parent_weight():
    recipe_with_variant_and_parent = make_feature_row(
        9, [1, 5], [5], cuisine="thai"
    )

    matrix = assemble_design_matrix(
        [recipe_with_variant_and_parent], SYNTHETIC_FEATURE_COUNT, PARENT_WEIGHT
    )

    assert matrix.toarray()[0, 5] == 1.0


def test_zero_parent_weight_adds_no_parent_entries():
    rows = [make_feature_row(9, [1, 2], [5], cuisine="thai")]

    matrix = assemble_design_matrix(rows, SYNTHETIC_FEATURE_COUNT, 0.0)

    dense = matrix.toarray()
    assert dense[0, 5] == 0.0
    assert matrix.nnz == 2


def test_encode_cuisine_labels_maps_sorted_cuisine_order():
    rows = [
        make_feature_row(1, [0], cuisine="thai"),
        make_feature_row(2, [0], cuisine="italian"),
    ]

    labels = encode_cuisine_labels(rows, CUISINE_IDS)

    assert labels.tolist() == [2, 0]


def test_split_row_positions_by_fold_covers_every_row_once():
    rows = make_train_feature_rows()
    folds_payload = make_folds_payload(rows)

    positions_by_fold = split_row_positions_by_fold(rows, folds_payload)

    all_positions = sorted(
        position
        for positions in positions_by_fold.values()
        for position in positions
    )
    assert all_positions == list(range(len(rows)))
    assert set(positions_by_fold) == {0, 1}
