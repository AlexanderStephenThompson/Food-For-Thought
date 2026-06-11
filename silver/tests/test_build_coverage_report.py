"""Tests for silver.pipeline.build_coverage_report.

All tests run on small in-test statistics and ingredients payloads; file
writes go to pytest tmp_path only.
"""

import json

import pytest

from silver.pipeline.build_coverage_report import (
    build_coverage_payload,
    render_coverage_markdown,
    write_coverage_reports,
)

FINGERPRINT = {
    "train_sha256": "a" * 64,
    "lexicon_fingerprint": "b" * 64,
    "random_seed": 42,
}

ALL_ALIAS_SOURCES = (
    "always_merge_lexicon",
    "brand_pattern",
    "canonical_surface_form",
    "forced_merge_override",
    "manual_alias",
    "manual_review",
    "mechanical_normalization",
    "modifier_strip",
    "named_variety_lexicon",
    "statistical_gate",
)
ALL_HISTOGRAM_BUCKETS = ("1", "2-5", "6-10", "11-20", "21+")
TOP_UNRESOLVED_LIMIT = 50


def _make_statistics() -> dict:
    """Resolution statistics whose shares are exact at four decimal places."""
    return {
        "train": {
            "mentions_total": 200,
            "by_method": {
                "exact_alias": 120,
                "cleaned_match": 40,
                "modifier_stripped_match": 20,
                "brand_resolved_match": 10,
                "token_drop_match": 7,
                "unresolved": 3,
            },
            "top_unresolved": [
                {"string": "mystery paste", "count": 2},
                {"string": "shared oddity", "count": 1},
            ],
        },
        "test": {
            "mentions_total": 50,
            "by_method": {
                "exact_alias": 30,
                "cleaned_match": 10,
                "modifier_stripped_match": 5,
                "brand_resolved_match": 2,
                "token_drop_match": 1,
                "unresolved": 2,
            },
            "top_unresolved": [
                {"string": "shared oddity", "count": 2},
            ],
        },
    }


def _make_ingredient(
    ingredient_id: str, alias_sources: list[str], parent_id: str | None = None
) -> dict:
    """Build one schema-shaped ingredient with one alias per listed source."""
    aliases = [
        {
            "alias": f"{ingredient_id} alias {index}",
            "source": source,
            "rule": None,
            "train_frequency": index + 1,
        }
        for index, source in enumerate(alias_sources)
    ]
    return {
        "id": ingredient_id,
        "name": ingredient_id.replace("_", " "),
        "category": None,
        "parent_id": parent_id,
        "train_mention_count": len(aliases),
        "preserve_evidence": None,
        "aliases": aliases,
    }


def _make_ingredients_payload() -> dict:
    """Six ingredients whose alias counts land in every histogram bucket."""
    ingredients = [
        _make_ingredient("a_one", ["canonical_surface_form"]),
        _make_ingredient("b_one", ["canonical_surface_form"], parent_id="a_one"),
        _make_ingredient(
            "c_three",
            ["canonical_surface_form"] + ["mechanical_normalization"] * 2,
        ),
        _make_ingredient(
            "d_seven", ["canonical_surface_form"] + ["modifier_strip"] * 6
        ),
        _make_ingredient(
            "e_fifteen", ["canonical_surface_form"] + ["statistical_gate"] * 14
        ),
        _make_ingredient(
            "f_twenty_five", ["canonical_surface_form"] + ["manual_alias"] * 24
        ),
    ]
    return {"schema_version": 1, "build": FINGERPRINT, "ingredients": ingredients}


def _build_payload() -> dict:
    """Coverage payload from the shared fixtures."""
    return build_coverage_payload(
        _make_statistics(), _make_ingredients_payload(), FINGERPRINT
    )


def _reverse_key_order(value):
    """Rebuild nested dicts with reversed key insertion order (lists untouched)."""
    if isinstance(value, dict):
        return {key: _reverse_key_order(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_key_order(item) for item in value]
    return value


def test_percentages_sum_to_one_per_split():
    payload = _build_payload()

    for split in ("train", "test"):
        summary = payload["resolution"][split]
        total_percentage = sum(
            entry["percentage"] for entry in summary["by_method"].values()
        )
        assert total_percentage == pytest.approx(1.0, abs=1e-9)


def test_method_counts_and_coverage_per_split():
    payload = _build_payload()

    train = payload["resolution"]["train"]
    assert train["mentions_total"] == 200
    assert train["by_method"]["exact_alias"] == {"count": 120, "percentage": 0.6}
    assert train["by_method"]["unresolved"] == {"count": 3, "percentage": 0.015}
    assert train["alias_tier_coverage"] == 0.6
    assert train["full_chain_coverage"] == 0.985
    test_split = payload["resolution"]["test"]
    assert test_split["alias_tier_coverage"] == 0.6
    assert test_split["full_chain_coverage"] == 0.96


def test_histogram_buckets():
    payload = _build_payload()

    histogram = payload["vocabulary"]["merge_group_size_histogram"]
    assert histogram == {"1": 2, "2-5": 1, "6-10": 1, "11-20": 1, "21+": 1}
    assert tuple(histogram) == ALL_HISTOGRAM_BUCKETS


def test_aliases_by_source_counts():
    payload = _build_payload()

    assert payload["vocabulary"]["aliases_by_source"] == {
        "always_merge_lexicon": 0,
        "brand_pattern": 0,
        "canonical_surface_form": 6,
        "forced_merge_override": 0,
        "manual_alias": 24,
        "manual_review": 0,
        "mechanical_normalization": 2,
        "modifier_strip": 6,
        "named_variety_lexicon": 0,
        "statistical_gate": 14,
    }


def test_vocabulary_counts():
    payload = _build_payload()

    vocabulary = payload["vocabulary"]
    assert vocabulary["ingredient_count"] == 6
    assert vocabulary["alias_count"] == 52
    assert vocabulary["preserved_variant_count"] == 1


def test_unknown_alias_source_raises():
    ingredients_payload = _make_ingredients_payload()
    ingredients_payload["ingredients"][0]["aliases"][0]["source"] = "not_a_source"

    with pytest.raises(ValueError, match="not_a_source"):
        build_coverage_payload(_make_statistics(), ingredients_payload, FINGERPRINT)


def test_top_unresolved_merged_and_sorted():
    payload = _build_payload()

    assert payload["top_unresolved"] == [
        {"string": "shared oddity", "count": 3},
        {"string": "mystery paste", "count": 2},
    ]


def test_top_unresolved_capped_at_fifty():
    statistics = _make_statistics()
    statistics["train"]["top_unresolved"] = [
        {"string": f"string {index:03d}", "count": 60 - index} for index in range(60)
    ]
    statistics["test"]["top_unresolved"] = []

    payload = build_coverage_payload(
        statistics, _make_ingredients_payload(), FINGERPRINT
    )

    assert len(payload["top_unresolved"]) == TOP_UNRESOLVED_LIMIT
    assert payload["top_unresolved"][0] == {"string": "string 000", "count": 60}


def test_markdown_contains_method_table_and_unresolved():
    markdown = render_coverage_markdown(_build_payload())

    assert "| Method | Count | % |" in markdown
    assert "| exact_alias | 120 | 60.00% |" in markdown
    assert "| unresolved | 3 | 1.50% |" in markdown
    assert "shared oddity" in markdown
    assert "mystery paste" in markdown


def test_markdown_deterministic():
    payload = _build_payload()

    first_render = render_coverage_markdown(payload)
    second_render = render_coverage_markdown(payload)
    reordered_render = render_coverage_markdown(_reverse_key_order(payload))

    assert first_render == second_render
    assert first_render == reordered_render
    assert first_render.endswith("\n")


def test_write_coverage_reports_creates_files(tmp_path):
    payload = _build_payload()

    write_coverage_reports(payload, tmp_path)

    json_path = tmp_path / "coverage.json"
    markdown_path = tmp_path / "coverage_report.md"
    written_json = json_path.read_text(encoding="utf-8")
    assert json.loads(written_json) == payload
    assert written_json.endswith("\n")
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert markdown_text == render_coverage_markdown(payload)
    assert markdown_text.endswith("\n")
