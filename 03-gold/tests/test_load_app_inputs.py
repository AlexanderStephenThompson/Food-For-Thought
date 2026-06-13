"""Tests for app_pipeline.load_app_inputs."""

import hashlib
import json

from app_pipeline.load_app_inputs import (
    compute_app_build_fingerprint,
    load_app_export_inputs,
)

GOLD_RELATIVE_PATHS = (
    "model/parameters.json",
    "model/calibration.json",
    "datasets/feature_space.json",
    "reports/evaluation.json",
)
SILVER_FILENAMES = ("ingredients.json", "cuisines.json")


def _write_input_files(tmp_path):
    gold_root = tmp_path / "gold"
    silver_directory = tmp_path / "silver"
    contents_by_name = {}
    for relative_path in GOLD_RELATIVE_PATHS:
        path = gold_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps({"marker": relative_path}) + "\n"
        path.write_text(content, encoding="utf-8")
        contents_by_name[relative_path] = content
    silver_directory.mkdir()
    for filename in SILVER_FILENAMES:
        content = json.dumps({"marker": filename}) + "\n"
        (silver_directory / filename).write_text(content, encoding="utf-8")
        contents_by_name[filename] = content
    return gold_root, silver_directory, contents_by_name


def test_load_app_export_inputs_returns_all_six_payloads(tmp_path):
    gold_root, silver_directory, _ = _write_input_files(tmp_path)

    inputs = load_app_export_inputs(gold_root, silver_directory)

    assert inputs.parameters == {"marker": "model/parameters.json"}
    assert inputs.calibration == {"marker": "model/calibration.json"}
    assert inputs.feature_space == {"marker": "datasets/feature_space.json"}
    assert inputs.evaluation == {"marker": "reports/evaluation.json"}
    assert inputs.ingredients == {"marker": "ingredients.json"}
    assert inputs.cuisines == {"marker": "cuisines.json"}


def test_compute_app_build_fingerprint_hashes_each_input(tmp_path):
    gold_root, silver_directory, contents = _write_input_files(tmp_path)

    fingerprint = compute_app_build_fingerprint(gold_root, silver_directory)

    expected_by_key = {
        "parameters_sha256": "model/parameters.json",
        "calibration_sha256": "model/calibration.json",
        "feature_space_sha256": "datasets/feature_space.json",
        "evaluation_sha256": "reports/evaluation.json",
        "ingredients_sha256": "ingredients.json",
        "cuisines_sha256": "cuisines.json",
    }
    assert set(fingerprint) == set(expected_by_key)
    for fingerprint_key, name in expected_by_key.items():
        expected_digest = hashlib.sha256(contents[name].encode("utf-8")).hexdigest()
        assert fingerprint[fingerprint_key] == expected_digest


def test_compute_app_build_fingerprint_changes_when_input_changes(tmp_path):
    gold_root, silver_directory, _ = _write_input_files(tmp_path)
    before = compute_app_build_fingerprint(gold_root, silver_directory)

    (silver_directory / "ingredients.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )
    after = compute_app_build_fingerprint(gold_root, silver_directory)

    assert before["ingredients_sha256"] != after["ingredients_sha256"]
    assert before["parameters_sha256"] == after["parameters_sha256"]
