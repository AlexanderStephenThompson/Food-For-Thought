"""Tests for build — the silver rebuild orchestrator's verification core.

The full bronze-to-silver build takes ~35 seconds, so end-to-end coverage
lives in `./manage.sh verify` (rebuild + byte-for-byte disk comparison).
These tests cover the comparison logic itself against temporary files.
"""

from build import find_artifact_mismatches


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
