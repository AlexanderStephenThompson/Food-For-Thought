"""Tests for model_pipeline.build_submission."""

from model_pipeline.build_submission import (
    SUBMISSION_HEADER,
    build_submission_rows,
    render_submission_csv,
)
from tests.model_payload_builders import MODEL_BUILD_BLOCK


def _make_blends_payload():
    return {
        "build": dict(MODEL_BUILD_BLOCK),
        "schema_version": 1,
        "cuisines": ["italian", "mexican", "thai"],
        "rows": [
            {"blend": [0.7, 0.2, 0.1], "recipe_id": 100, "top_cuisine": "italian"},
            {"blend": [0.1, 0.2, 0.7], "recipe_id": 101, "top_cuisine": "thai"},
        ],
    }


def test_submission_rows_use_blend_top_cuisine_in_order():
    rows = build_submission_rows(_make_blends_payload())

    assert rows == [(100, "italian"), (101, "thai")]


def test_submission_csv_renders_header_and_rows():
    csv_text = render_submission_csv([(100, "italian"), (101, "thai")])

    assert csv_text == f"{SUBMISSION_HEADER}\n100,italian\n101,thai\n"
