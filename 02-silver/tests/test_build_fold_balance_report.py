"""Tests for gold_pipeline.build_fold_balance_report."""

from gold_pipeline.assign_folds import FOLD_COUNT, build_folds_payload
from gold_pipeline.build_fold_balance_report import (
    build_fold_balance_payload,
    write_fold_balance_reports,
)
from tests.gold_payload_builders import (
    GOLD_BUILD_BLOCK,
    make_recipes_payload,
    make_train_records,
)


def _build_corpus_payloads():
    recipes_payload = make_recipes_payload(make_train_records())
    folds_payload = build_folds_payload(recipes_payload, GOLD_BUILD_BLOCK)
    return recipes_payload, folds_payload


def test_fold_balance_counts_recipes_per_cuisine_per_fold():
    recipes_payload, folds_payload = _build_corpus_payloads()

    payload = build_fold_balance_payload(
        folds_payload, recipes_payload, GOLD_BUILD_BLOCK
    )

    assert sum(payload["fold_sizes"]) == 17
    assert len(payload["fold_sizes"]) == FOLD_COUNT
    counts_by_cuisine = {
        entry["cuisine"]: entry for entry in payload["by_cuisine"]
    }
    assert sum(counts_by_cuisine["italian"]["fold_counts"]) == 7
    assert sum(counts_by_cuisine["mexican"]["fold_counts"]) == 4
    assert sum(counts_by_cuisine["thai"]["fold_counts"]) == 6
    assert counts_by_cuisine["italian"]["recipe_count"] == 7
    assert payload["schema_version"] == 1
    assert payload["build"] == GOLD_BUILD_BLOCK


def test_fold_balance_markdown_lists_every_cuisine_and_fold_sizes(tmp_path):
    recipes_payload, folds_payload = _build_corpus_payloads()
    payload = build_fold_balance_payload(
        folds_payload, recipes_payload, GOLD_BUILD_BLOCK
    )

    write_fold_balance_reports(payload, tmp_path)

    markdown = (tmp_path / "fold_balance.md").read_text(encoding="utf-8")
    assert "italian" in markdown
    assert "mexican" in markdown
    assert "thai" in markdown
    assert (tmp_path / "fold_balance.json").is_file()
