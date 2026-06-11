"""Build the fold-balance report for one gold build.

Condenses the folds artifact into per-fold and per-cuisine counts, written
as both a JSON artifact and readable Markdown (mirroring the silver
coverage report). Pure functions over already-loaded payloads.
"""

from __future__ import annotations

from pathlib import Path

from silver_pipeline.artifact_io import (
    SCHEMA_VERSION,
    write_artifact_json,
    write_text_atomically,
)

FOLD_BALANCE_JSON_FILENAME = "fold_balance.json"
FOLD_BALANCE_MARKDOWN_FILENAME = "fold_balance.md"
LINE_SEPARATOR = "\n"


def build_fold_balance_payload(
    folds_payload: dict, recipes_train_payload: dict, fingerprint: dict
) -> dict:
    """Count assigned recipes per fold, overall and per cuisine.

    Args:
        folds_payload: Gold folds artifact from build_folds_payload.
        recipes_train_payload: Parsed silver recipes_train.json document
            (source of each recipe's cuisine).
        fingerprint: Gold build block embedded in the artifact.

    Returns:
        Payload with fold_sizes and per-cuisine fold_counts, ready for
        write_artifact_json.
    """
    fold_count = folds_payload["fold_count"]
    cuisine_by_recipe_id = {
        recipe["id"]: recipe["cuisine"]
        for recipe in recipes_train_payload["recipes"]
    }
    fold_sizes = [0] * fold_count
    fold_counts_by_cuisine: dict[str, list[int]] = {}
    for assignment in folds_payload["assignments"]:
        fold = assignment["fold"]
        cuisine = cuisine_by_recipe_id[assignment["recipe_id"]]
        fold_sizes[fold] += 1
        fold_counts_by_cuisine.setdefault(cuisine, [0] * fold_count)[fold] += 1

    by_cuisine = [
        {
            "cuisine": cuisine,
            "fold_counts": fold_counts_by_cuisine[cuisine],
            "recipe_count": sum(fold_counts_by_cuisine[cuisine]),
        }
        for cuisine in sorted(fold_counts_by_cuisine)
    ]
    return {
        "build": dict(fingerprint),
        "by_cuisine": by_cuisine,
        "fold_count": fold_count,
        "fold_sizes": fold_sizes,
        "schema_version": SCHEMA_VERSION,
    }


def _render_markdown_report(payload: dict) -> str:
    """Render the fold-balance payload as a readable Markdown report."""
    fold_headers = " | ".join(
        f"fold {fold}" for fold in range(payload["fold_count"])
    )
    lines = [
        "# Fold balance",
        "",
        f"- Folds: {payload['fold_count']}",
        f"- Recipes: {sum(payload['fold_sizes'])}",
        f"- Fold sizes: {', '.join(str(size) for size in payload['fold_sizes'])}",
        "",
        "## Per cuisine",
        "",
        f"| Cuisine | Recipes | {fold_headers} |",
        "| --- | ---: | " + " | ".join("---:" for _ in range(payload["fold_count"])) + " |",
    ]
    for entry in payload["by_cuisine"]:
        fold_cells = " | ".join(str(count) for count in entry["fold_counts"])
        lines.append(
            f"| {entry['cuisine']} | {entry['recipe_count']} | {fold_cells} |"
        )
    lines.append("")
    return LINE_SEPARATOR.join(lines)


def write_fold_balance_reports(payload: dict, reports_directory: Path) -> None:
    """Atomically write the fold-balance JSON artifact and Markdown report.

    Args:
        payload: Fold-balance payload from build_fold_balance_payload.
        reports_directory: Destination directory; must exist.
    """
    write_artifact_json(payload, reports_directory / FOLD_BALANCE_JSON_FILENAME)
    write_text_atomically(
        _render_markdown_report(payload),
        reports_directory / FOLD_BALANCE_MARKDOWN_FILENAME,
    )
