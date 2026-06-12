"""Tests for model_pipeline.load_gold_datasets.

All tests write tiny marker JSON files into tmp_path directories; nothing
reads the real gold or silver datasets.
"""

import hashlib
import json

import sklearn

from model_pipeline.load_gold_datasets import (
    compute_model_build_fingerprint,
    load_gold_model_inputs,
)

GOLD_FILENAMES = (
    "feature_space.json",
    "features_test.json",
    "features_train.json",
    "folds.json",
)
SILVER_CUISINES_FILENAME = "cuisines.json"


def _write_input_files(tmp_path):
    """Write marker JSON files into gold/ and silver/ subdirectories."""
    gold_directory = tmp_path / "gold"
    silver_directory = tmp_path / "silver"
    gold_directory.mkdir()
    silver_directory.mkdir()
    contents_by_filename = {}
    for filename in GOLD_FILENAMES:
        content = json.dumps({"marker": filename}) + "\n"
        (gold_directory / filename).write_text(content, encoding="utf-8")
        contents_by_filename[filename] = content
    cuisines_content = json.dumps({"marker": SILVER_CUISINES_FILENAME}) + "\n"
    (silver_directory / SILVER_CUISINES_FILENAME).write_text(
        cuisines_content, encoding="utf-8"
    )
    contents_by_filename[SILVER_CUISINES_FILENAME] = cuisines_content
    return gold_directory, silver_directory, contents_by_filename


def test_load_gold_model_inputs_returns_all_five_payloads(tmp_path):
    gold_directory, silver_directory, _ = _write_input_files(tmp_path)

    inputs = load_gold_model_inputs(gold_directory, silver_directory)

    assert inputs.feature_space == {"marker": "feature_space.json"}
    assert inputs.features_train == {"marker": "features_train.json"}
    assert inputs.features_test == {"marker": "features_test.json"}
    assert inputs.folds == {"marker": "folds.json"}
    assert inputs.cuisines == {"marker": "cuisines.json"}


def test_compute_model_build_fingerprint_hashes_each_input(tmp_path):
    gold_directory, silver_directory, contents = _write_input_files(tmp_path)

    fingerprint = compute_model_build_fingerprint(gold_directory, silver_directory)

    expected_hash_by_key = {
        "feature_space_sha256": "feature_space.json",
        "features_train_sha256": "features_train.json",
        "features_test_sha256": "features_test.json",
        "folds_sha256": "folds.json",
        "cuisines_sha256": "cuisines.json",
    }
    for fingerprint_key, filename in expected_hash_by_key.items():
        expected_digest = hashlib.sha256(
            contents[filename].encode("utf-8")
        ).hexdigest()
        assert fingerprint[fingerprint_key] == expected_digest
    assert fingerprint["random_seed"] == 42
    assert fingerprint["sklearn_version"] == sklearn.__version__


def test_compute_model_build_fingerprint_changes_when_gold_input_changes(tmp_path):
    gold_directory, silver_directory, _ = _write_input_files(tmp_path)
    before = compute_model_build_fingerprint(gold_directory, silver_directory)

    (gold_directory / "folds.json").write_text('{"changed": true}\n', encoding="utf-8")
    after = compute_model_build_fingerprint(gold_directory, silver_directory)

    assert before["folds_sha256"] != after["folds_sha256"]
    assert before["feature_space_sha256"] == after["feature_space_sha256"]
