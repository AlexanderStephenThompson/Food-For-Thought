"""Tests for silver_pipeline.artifact_io — serialization, atomic writes, fingerprints.

All tests run against temporary files; nothing touches real bronze data or
silver artifacts. The serialization tests pin the exact byte format because
the idempotency guarantee depends on it.
"""

import hashlib
import json

import pytest

from silver_pipeline.artifact_io import (
    TEMPORARY_FILE_SUFFIX,
    compute_build_fingerprint,
    find_artifact_mismatches,
    serialize_artifact_json,
    sha256_of_file,
    write_artifact_json,
    write_text_atomically,
)

SAMPLE_FILE_BYTES = b"silver pipeline fingerprint sample\n"


def test_sha256_of_file_matches_hashlib_digest(tmp_path):
    sample_path = tmp_path / "sample.bin"
    sample_path.write_bytes(SAMPLE_FILE_BYTES)

    digest = sha256_of_file(sample_path)

    assert digest == hashlib.sha256(SAMPLE_FILE_BYTES).hexdigest()


def test_sha256_of_file_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_of_file(tmp_path / "does_not_exist.bin")


def _write_fingerprint_inputs(tmp_path):
    """Create a train file plus a two-file lexicon directory."""
    train_path = tmp_path / "train.json"
    train_path.write_text('[{"id": 1}]', encoding="utf-8")
    lexicons_directory = tmp_path / "lexicons"
    lexicons_directory.mkdir()
    (lexicons_directory / "aliases.json").write_text("{}", encoding="utf-8")
    (lexicons_directory / "decisions.jsonl").write_text("", encoding="utf-8")
    return train_path, lexicons_directory


def test_compute_build_fingerprint_is_deterministic(tmp_path):
    train_path, lexicons_directory = _write_fingerprint_inputs(tmp_path)

    first = compute_build_fingerprint(train_path, lexicons_directory)
    second = compute_build_fingerprint(train_path, lexicons_directory)

    assert first == second


def test_compute_build_fingerprint_changes_when_lexicon_content_changes(tmp_path):
    train_path, lexicons_directory = _write_fingerprint_inputs(tmp_path)
    before = compute_build_fingerprint(train_path, lexicons_directory)

    (lexicons_directory / "aliases.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    after = compute_build_fingerprint(train_path, lexicons_directory)

    assert before["lexicon_fingerprint"] != after["lexicon_fingerprint"]
    assert before["train_sha256"] == after["train_sha256"]


def test_compute_build_fingerprint_includes_jsonl_files(tmp_path):
    train_path, lexicons_directory = _write_fingerprint_inputs(tmp_path)
    before = compute_build_fingerprint(train_path, lexicons_directory)

    (lexicons_directory / "decisions.jsonl").write_text(
        '{"decision_id": "x"}\n', encoding="utf-8"
    )
    after = compute_build_fingerprint(train_path, lexicons_directory)

    assert before["lexicon_fingerprint"] != after["lexicon_fingerprint"]


def test_serialize_artifact_json_sorts_keys_and_ends_with_newline():
    payload = {"beta": 2, "alpha": 1}

    content = serialize_artifact_json(payload)

    assert content == '{\n  "alpha": 1,\n  "beta": 2\n}\n'


def test_serialize_artifact_json_preserves_non_ascii_text():
    payload = {"name": "jalapeño"}

    content = serialize_artifact_json(payload)

    assert "jalapeño" in content


def test_write_text_atomically_writes_exact_content(tmp_path):
    target_path = tmp_path / "report.md"

    write_text_atomically("# Coverage\n", target_path)

    assert target_path.read_text(encoding="utf-8") == "# Coverage\n"


def test_write_text_atomically_replaces_existing_file(tmp_path):
    target_path = tmp_path / "report.md"
    target_path.write_text("stale content\n", encoding="utf-8")

    write_text_atomically("fresh content\n", target_path)

    assert target_path.read_text(encoding="utf-8") == "fresh content\n"


def test_write_text_atomically_leaves_no_temporary_files(tmp_path):
    target_path = tmp_path / "report.md"

    write_text_atomically("content\n", target_path)

    leftover_names = [
        entry.name
        for entry in tmp_path.iterdir()
        if entry.name.endswith(TEMPORARY_FILE_SUFFIX)
    ]
    assert leftover_names == []


def test_write_artifact_json_round_trips_payload(tmp_path):
    target_path = tmp_path / "artifact.json"
    payload = {"schema_version": 1, "entries": [{"id": "salt"}]}

    write_artifact_json(payload, target_path)

    assert json.loads(target_path.read_text(encoding="utf-8")) == payload


def test_write_artifact_json_bytes_match_serialize_artifact_json(tmp_path):
    target_path = tmp_path / "artifact.json"
    payload = {"zulu": 1, "alpha": {"nested": True}}

    write_artifact_json(payload, target_path)

    assert target_path.read_text(encoding="utf-8") == serialize_artifact_json(payload)


def test_find_artifact_mismatches_empty_when_disk_matches(tmp_path):
    artifact_path = tmp_path / "cuisines.json"
    artifact_path.write_text('{"a": 1}\n', encoding="utf-8")

    mismatches = find_artifact_mismatches({artifact_path: '{"a": 1}\n'})

    assert mismatches == []


def test_find_artifact_mismatches_reports_missing_file(tmp_path):
    missing_path = tmp_path / "ingredients.json"

    mismatches = find_artifact_mismatches({missing_path: "{}\n"})

    assert mismatches == ["ingredients.json (missing)"]


def test_find_artifact_mismatches_reports_content_drift(tmp_path):
    artifact_path = tmp_path / "coverage.json"
    artifact_path.write_text('{"stale": true}\n', encoding="utf-8")

    mismatches = find_artifact_mismatches({artifact_path: '{"fresh": true}\n'})

    assert mismatches == ["coverage.json"]


def test_find_artifact_mismatches_checks_every_expected_file(tmp_path):
    matching_path = tmp_path / "matching.json"
    matching_path.write_text("{}\n", encoding="utf-8")
    drifted_path = tmp_path / "drifted.json"
    drifted_path.write_text("old\n", encoding="utf-8")

    mismatches = find_artifact_mismatches(
        {matching_path: "{}\n", drifted_path: "new\n"}
    )

    assert mismatches == ["drifted.json"]
