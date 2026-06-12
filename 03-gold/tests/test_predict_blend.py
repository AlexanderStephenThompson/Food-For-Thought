"""Tests for model_pipeline.predict_blend — the CLI's pure building blocks."""

from silver_pipeline.resolve_ingredient import ResolutionResult

from model_pipeline.predict_blend import (
    build_feature_values,
    format_blend_lines,
    format_resolution_table,
    split_raw_ingredient_arguments,
)
from tests.model_payload_builders import make_feature_space_payload

PARENT_WEIGHT = 0.3


def test_split_raw_ingredient_arguments_trims_and_drops_empties():
    raw_arguments = split_raw_ingredient_arguments(
        " fish sauce , coconut milk ,, rice "
    )

    assert raw_arguments == ["fish sauce", "coconut milk", "rice"]


def test_build_feature_values_applies_parent_weight_to_absent_parents():
    feature_values = build_feature_values(
        ["dark_soy_sauce", "fish_sauce"],
        make_feature_space_payload(),
        PARENT_WEIGHT,
    )

    assert feature_values[1] == 1.0
    assert feature_values[2] == 1.0
    assert feature_values[5] == PARENT_WEIGHT


def test_build_feature_values_direct_presence_wins():
    feature_values = build_feature_values(
        ["dark_soy_sauce", "soy_sauce"],
        make_feature_space_payload(),
        PARENT_WEIGHT,
    )

    assert feature_values[5] == 1.0


def test_format_blend_lines_render_percentages():
    blend_by_cuisine = {"thai": 0.583, "vietnamese": 0.241, "chinese": 0.110}

    lines = format_blend_lines(blend_by_cuisine, top_count=2)

    assert lines == ["58.3%  thai", "24.1%  vietnamese"]


def test_format_resolution_table_marks_unresolved():
    resolutions = [
        ("fish sauce", ResolutionResult(ingredient_id="fish_sauce", method="exact_alias")),
        ("mystery goo", ResolutionResult(ingredient_id=None, method="unresolved")),
    ]

    lines = format_resolution_table(resolutions)

    joined = "\n".join(lines)
    assert "fish_sauce" in joined
    assert "UNRESOLVED" in joined
