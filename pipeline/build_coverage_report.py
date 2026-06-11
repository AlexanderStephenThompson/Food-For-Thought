"""Build the human-facing coverage summary for one pipeline run.

Condenses reports/resolution_statistics.json and silver/ingredients.json
into a single coverage payload, renders it as readable Markdown, and writes
both artifacts atomically. Pure functions over already-loaded payloads:
no file I/O happens at import time.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pipeline.artifact_io import write_artifact_json

RESOLUTION_SPLITS = ("train", "test")
RESOLUTION_METHODS = (
    "exact_alias",
    "cleaned_match",
    "modifier_stripped_match",
    "brand_resolved_match",
    "token_drop_match",
    "unresolved",
)
# The alias tier is a direct raw-string hit on the precompiled alias table;
# every other method needs the runtime normalization chain.
ALIAS_TIER_METHODS = ("exact_alias",)
UNRESOLVED_METHOD = "unresolved"

ALIAS_SOURCE_VALUES = (
    "canonical_surface_form",
    "mechanical_normalization",
    "modifier_strip",
    "always_merge_lexicon",
    "forced_merge_override",
    "statistical_gate",
    "named_variety_lexicon",
    "brand_pattern",
    "manual_alias",
    "manual_review",
)

HISTOGRAM_BUCKET_UPPER_BOUNDS = ((1, "1"), (5, "2-5"), (10, "6-10"), (20, "11-20"))
HISTOGRAM_OVERFLOW_LABEL = "21+"
HISTOGRAM_BUCKET_LABELS = tuple(
    label for _, label in HISTOGRAM_BUCKET_UPPER_BOUNDS
) + (HISTOGRAM_OVERFLOW_LABEL,)

TOP_UNRESOLVED_LIMIT = 50
MARKDOWN_UNRESOLVED_LIMIT = 20
PERCENTAGE_DECIMAL_PLACES = 4
SHORT_HASH_LENGTH = 12

COVERAGE_JSON_FILENAME = "coverage.json"
COVERAGE_MARKDOWN_FILENAME = "coverage_report.md"
REPORT_FILE_ENCODING = "utf-8"
LINE_SEPARATOR = "\n"


def build_coverage_payload(
    resolution_statistics: dict, ingredients_payload: dict, fingerprint: dict
) -> dict:
    """Condense resolution statistics and the vocabulary into one summary.

    Args:
        resolution_statistics: reports/resolution_statistics.json payload with
            "train" and "test" splits.
        ingredients_payload: silver/ingredients.json payload.
        fingerprint: Build block from compute_build_fingerprint.

    Returns:
        Dict with "build", per-split "resolution" summaries (counts,
        percentages, alias-tier and full-chain coverage), "vocabulary"
        statistics, and a merged "top_unresolved" list (max 50 entries).

    Raises:
        KeyError: If a split or required field is missing.
        ValueError: If an alias carries an unknown source value.
    """
    return {
        "build": fingerprint,
        "resolution": {
            split: _summarize_split(resolution_statistics[split])
            for split in RESOLUTION_SPLITS
        },
        "vocabulary": _summarize_vocabulary(ingredients_payload["ingredients"]),
        "top_unresolved": _merge_top_unresolved(resolution_statistics),
    }


def render_coverage_markdown(coverage_payload: dict) -> str:
    """Render the coverage payload as deterministic, readable Markdown.

    Args:
        coverage_payload: Output of build_coverage_payload.

    Returns:
        Markdown document (title, build fingerprint, per-split resolution
        tables, vocabulary statistics, histogram, top-20 unresolved list)
        ending with a trailing newline.
    """
    lines = ["# Coverage Report", ""]
    lines.extend(_render_build_section(coverage_payload["build"]))
    for split in RESOLUTION_SPLITS:
        lines.extend(
            _render_resolution_section(split, coverage_payload["resolution"][split])
        )
    lines.extend(_render_vocabulary_section(coverage_payload["vocabulary"]))
    lines.extend(_render_unresolved_section(coverage_payload["top_unresolved"]))
    return LINE_SEPARATOR.join(lines) + LINE_SEPARATOR


def write_coverage_reports(coverage_payload: dict, reports_directory: Path) -> None:
    """Write coverage.json and coverage_report.md atomically.

    Args:
        coverage_payload: Output of build_coverage_payload.
        reports_directory: Existing directory to write both reports into.
    """
    write_artifact_json(coverage_payload, reports_directory / COVERAGE_JSON_FILENAME)
    markdown = render_coverage_markdown(coverage_payload)
    _write_text_atomically(markdown, reports_directory / COVERAGE_MARKDOWN_FILENAME)


def _summarize_split(split_statistics: dict) -> dict:
    """Per-method counts and shares plus tier coverage for one split."""
    mentions_total = split_statistics["mentions_total"]
    method_counts = split_statistics["by_method"]
    by_method = {
        method: {
            "count": method_counts.get(method, 0),
            "percentage": _share(method_counts.get(method, 0), mentions_total),
        }
        for method in RESOLUTION_METHODS
    }
    alias_tier_count = sum(
        method_counts.get(method, 0) for method in ALIAS_TIER_METHODS
    )
    resolved_count = mentions_total - method_counts.get(UNRESOLVED_METHOD, 0)
    return {
        "mentions_total": mentions_total,
        "by_method": by_method,
        "alias_tier_coverage": _share(alias_tier_count, mentions_total),
        "full_chain_coverage": _share(resolved_count, mentions_total),
    }


def _share(count: int, total: int) -> float:
    """Fraction of total, rounded; 0.0 when the total is zero."""
    if total == 0:
        return 0.0
    return round(count / total, PERCENTAGE_DECIMAL_PLACES)


def _summarize_vocabulary(ingredients: list[dict]) -> dict:
    """Alias, source, parent, and group-size statistics over the vocabulary."""
    aliases_by_source = {source: 0 for source in ALIAS_SOURCE_VALUES}
    histogram = {label: 0 for label in HISTOGRAM_BUCKET_LABELS}
    alias_count = 0
    preserved_variant_count = 0
    for ingredient in ingredients:
        aliases = ingredient["aliases"]
        alias_count += len(aliases)
        if ingredient["parent_id"] is not None:
            preserved_variant_count += 1
        histogram[_histogram_bucket_label(len(aliases))] += 1
        for alias in aliases:
            source = alias["source"]
            if source not in aliases_by_source:
                raise ValueError(f"Unknown alias source: {source}")
            aliases_by_source[source] += 1
    return {
        "ingredient_count": len(ingredients),
        "alias_count": alias_count,
        "preserved_variant_count": preserved_variant_count,
        "aliases_by_source": aliases_by_source,
        "merge_group_size_histogram": histogram,
    }


def _histogram_bucket_label(alias_count: int) -> str:
    """Histogram bucket label for an ingredient's alias count."""
    for upper_bound, label in HISTOGRAM_BUCKET_UPPER_BOUNDS:
        if alias_count <= upper_bound:
            return label
    return HISTOGRAM_OVERFLOW_LABEL


def _merge_top_unresolved(resolution_statistics: dict) -> list[dict]:
    """Merge per-split unresolved strings; count desc, then string asc."""
    merged_counts: dict[str, int] = {}
    for split in RESOLUTION_SPLITS:
        for entry in resolution_statistics[split].get("top_unresolved", []):
            unresolved_string = entry["string"]
            merged_counts[unresolved_string] = (
                merged_counts.get(unresolved_string, 0) + entry["count"]
            )
    ranked = sorted(merged_counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"string": unresolved_string, "count": count}
        for unresolved_string, count in ranked[:TOP_UNRESOLVED_LIMIT]
    ]


def _render_build_section(build: dict) -> list[str]:
    """Markdown lines for the build fingerprint (short hashes)."""
    return [
        "## Build",
        "",
        f"- Train sha256: `{build['train_sha256'][:SHORT_HASH_LENGTH]}`",
        f"- Lexicon fingerprint: `{build['lexicon_fingerprint'][:SHORT_HASH_LENGTH]}`",
        f"- Random seed: {build['random_seed']}",
        "",
    ]


def _render_resolution_section(split: str, split_summary: dict) -> list[str]:
    """Markdown lines for one split's resolution table."""
    lines = [
        f"## Resolution ({split})",
        "",
        f"- Mentions total: {split_summary['mentions_total']}",
        f"- Alias tier coverage: {_format_percentage(split_summary['alias_tier_coverage'])}",
        f"- Full chain coverage: {_format_percentage(split_summary['full_chain_coverage'])}",
        "",
        "| Method | Count | % |",
        "| --- | ---: | ---: |",
    ]
    for method in RESOLUTION_METHODS:
        entry = split_summary["by_method"][method]
        percentage = _format_percentage(entry["percentage"])
        lines.append(f"| {method} | {entry['count']} | {percentage} |")
    lines.append("")
    return lines


def _format_percentage(fraction: float) -> str:
    """Render a 0..1 fraction as a fixed two-decimal percentage string."""
    return f"{fraction * 100:.2f}%"


def _render_vocabulary_section(vocabulary: dict) -> list[str]:
    """Markdown lines for vocabulary statistics and both tables."""
    lines = [
        "## Vocabulary",
        "",
        f"- Ingredients: {vocabulary['ingredient_count']}",
        f"- Aliases: {vocabulary['alias_count']}",
        f"- Preserved variants: {vocabulary['preserved_variant_count']}",
        "",
        "### Aliases by source",
        "",
        "| Source | Count |",
        "| --- | ---: |",
    ]
    for source in ALIAS_SOURCE_VALUES:
        lines.append(f"| {source} | {vocabulary['aliases_by_source'][source]} |")
    lines.extend(
        [
            "",
            "### Merge group size histogram",
            "",
            "| Aliases per ingredient | Ingredients |",
            "| --- | ---: |",
        ]
    )
    for label in HISTOGRAM_BUCKET_LABELS:
        lines.append(f"| {label} | {vocabulary['merge_group_size_histogram'][label]} |")
    lines.append("")
    return lines


def _render_unresolved_section(top_unresolved: list[dict]) -> list[str]:
    """Markdown lines for the top unresolved strings (capped for readability)."""
    lines = [f"## Top unresolved (first {MARKDOWN_UNRESOLVED_LIMIT})", ""]
    if not top_unresolved:
        lines.extend(["No unresolved ingredient strings.", ""])
        return lines
    for entry in top_unresolved[:MARKDOWN_UNRESOLVED_LIMIT]:
        lines.append(f"- {entry['string']} ({entry['count']})")
    lines.append("")
    return lines


def _write_text_atomically(content: str, path: Path) -> None:
    """Write text via a sibling temp file and os.replace; ensure newline end."""
    if not content.endswith(LINE_SEPARATOR):
        content += LINE_SEPARATOR
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=path.name, suffix=".tmp"
    )
    with os.fdopen(descriptor, "w", encoding=REPORT_FILE_ENCODING) as handle:
        handle.write(content)
    os.replace(temporary_path, path)
