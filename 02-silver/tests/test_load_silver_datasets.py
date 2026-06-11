"""Tests for gold_pipeline.load_silver_datasets.

All tests write tiny silver-shaped JSON files into tmp_path; nothing reads
the real silver datasets.
"""

import hashlib
import json

from gold_pipeline.load_silver_datasets import (
    compute_gold_build_fingerprint,
    load_silver_inputs,
)

SILVER_FILENAMES = (
    "cuisines.json",
    "ingredients.json",
    "recipes_test.json",
    "recipes_train.json",
)


def _write_silver_files(tmp_path):
    """Write four distinct, valid JSON files named like the silver datasets."""
    contents_by_filename = {
        filename: json.dumps({"marker": filename}) + "\n"
        for filename in SILVER_FILENAMES
    }
    for filename, content in contents_by_filename.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")
    return contents_by_filename


def test_load_silver_inputs_returns_all_four_payloads(tmp_path):
    _write_silver_files(tmp_path)

    inputs = load_silver_inputs(tmp_path)

    assert inputs.ingredients == {"marker": "ingredients.json"}
    assert inputs.recipes_train == {"marker": "recipes_train.json"}
    assert inputs.recipes_test == {"marker": "recipes_test.json"}
    assert inputs.cuisines == {"marker": "cuisines.json"}


def test_compute_gold_build_fingerprint_hashes_each_silver_file(tmp_path):
    contents_by_filename = _write_silver_files(tmp_path)

    fingerprint = compute_gold_build_fingerprint(tmp_path)

    expected_hash_by_key = {
        "ingredients_sha256": "ingredients.json",
        "recipes_train_sha256": "recipes_train.json",
        "recipes_test_sha256": "recipes_test.json",
        "cuisines_sha256": "cuisines.json",
    }
    for fingerprint_key, filename in expected_hash_by_key.items():
        expected_digest = hashlib.sha256(
            contents_by_filename[filename].encode("utf-8")
        ).hexdigest()
        assert fingerprint[fingerprint_key] == expected_digest
    assert fingerprint["random_seed"] == 42
    assert fingerprint["fold_count"] == 5


def test_compute_gold_build_fingerprint_changes_when_any_input_changes(tmp_path):
    _write_silver_files(tmp_path)
    before = compute_gold_build_fingerprint(tmp_path)

    (tmp_path / "cuisines.json").write_text('{"changed": true}\n', encoding="utf-8")
    after = compute_gold_build_fingerprint(tmp_path)

    assert before["cuisines_sha256"] != after["cuisines_sha256"]
    assert before["ingredients_sha256"] == after["ingredients_sha256"]
    assert before["recipes_train_sha256"] == after["recipes_train_sha256"]
    assert before["recipes_test_sha256"] == after["recipes_test_sha256"]
