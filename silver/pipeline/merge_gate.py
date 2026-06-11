"""Layered merge gate deciding MERGE / PRESERVE / REVIEW for ingredient variants.

Implements the validated three-plus-one layer rule from the signal-variants
analysis of train.json:

    L0  do_not_merge exceptions        -> PRESERVE (exact cleaned strings)
    L1  always-merge marketing lexicon -> MERGE (beats ALL statistics; e.g.
        'lower sodium soy sauce' diverges at JSD=0.1449 through a
        recipe-source artifact yet must merge)
    L2  statistical gate (n >= 20)     -> PRESERVE iff JSD >= 0.07 bits AND
        JSD >= 1.5x the Monte Carlo null95; borderline values route to REVIEW
    L3  small-sample layer (n < 20)    -> named-variety lexicon preserves
        (usukuchi, kecap manis, dende, ...), concentrated samples go to
        REVIEW, everything else default-merges

Lexicons are loaded explicitly via load_gate_lexicons() — no import-time
file I/O — and passed into decide_merge().
"""

import enum
import json
import re
from dataclasses import dataclass
from pathlib import Path

from silver.pipeline.merge_evidence import MergeEvidence

# Statistical gate thresholds validated against ~160 (variant, base) pairs.
MIN_SUPPORT = 20
JSD_FLOOR_BITS = 0.07
NULL_MULTIPLIER = 1.5
# Flip-sensitive zones: moving the floor by +-0.01 flips tamari (0.0771),
# smoked paprika (0.0777), reduced sodium soy (0.0639) — route to REVIEW.
BORDERLINE_JSD_BAND = (0.06, 0.08)
BORDERLINE_RATIO_BAND = (1.3, 1.7)
# Sub-threshold variants concentrated in one cuisine deserve a human look.
SMALL_SAMPLE_REVIEW_SHARE = 0.80
SMALL_SAMPLE_REVIEW_MINIMUM = 5

LAYER_DO_NOT_MERGE_EXCEPTION = "do_not_merge_exception"
LAYER_ALWAYS_MERGE_LEXICON = "always_merge_lexicon"
LAYER_FORCED_MERGE_OVERRIDE = "forced_merge_override"
LAYER_STATISTICAL_GATE = "statistical_gate"
LAYER_NAMED_VARIETY_LEXICON = "named_variety_lexicon"
LAYER_SMALL_SAMPLE_DEFAULT = "small_sample_default"
LAYER_SMALL_SAMPLE_REVIEW = "small_sample_review"

ALWAYS_MERGE_FILE_NAME = "always_merge_patterns.json"
FORCED_MERGE_FILE_NAME = "forced_merge_overrides.json"
NAMED_VARIETIES_FILE_NAME = "named_varieties.json"
DO_NOT_MERGE_FILE_NAME = "do_not_merge.json"


class GateAction(enum.Enum):
    """Outcome of a merge-gate decision for one (variant, base) pair."""

    MERGE = "merge"
    PRESERVE = "preserve"
    REVIEW = "review"


@dataclass(frozen=True)
class GateLexicons:
    """Frozen bundle of the four curated gate lexicons.

    Attributes:
        always_merge_patterns: Compiled marketing-phrase regexes (layer L1).
        forced_merge_overrides: Exact cleaned variant strings force-merged at L1.
        named_varieties: Variety phrases preserving sub-threshold variants at L3.
        do_not_merge: Exact cleaned variant strings hard-preserved at L0.
    """

    always_merge_patterns: tuple[re.Pattern[str], ...]
    forced_merge_overrides: frozenset[str]
    named_varieties: frozenset[str]
    do_not_merge: frozenset[str]


@dataclass(frozen=True)
class GateDecision:
    """One merge-gate verdict with its provenance.

    Attributes:
        action: MERGE, PRESERVE, or REVIEW.
        layer: Which gate layer produced the verdict (one of the LAYER_*
            constants in this module).
        evidence: The statistical evidence consulted, or None when the
            decision was lexicon-only or no evidence was available.
        reason: Human-readable one-liner explaining the verdict.
    """

    action: GateAction
    layer: str
    evidence: MergeEvidence | None
    reason: str


def load_gate_lexicons(lexicons_directory: Path) -> GateLexicons:
    """Load and validate the four gate lexicon files from one directory.

    Args:
        lexicons_directory: Directory containing always_merge_patterns.json,
            forced_merge_overrides.json, named_varieties.json, and
            do_not_merge.json.

    Returns:
        A frozen GateLexicons with compiled patterns and frozensets.

    Raises:
        FileNotFoundError: If the directory or any lexicon file is missing.
        ValueError: If a file has the wrong shape, an empty entry, or an
            invalid regex source.
    """
    if not lexicons_directory.is_dir():
        raise FileNotFoundError(f"Lexicon directory not found: {lexicons_directory}")
    pattern_entries = _read_entry_list(
        lexicons_directory / ALWAYS_MERGE_FILE_NAME, "patterns"
    )
    override_entries = _read_entry_list(
        lexicons_directory / FORCED_MERGE_FILE_NAME, "overrides"
    )
    variety_entries = _read_entry_list(
        lexicons_directory / NAMED_VARIETIES_FILE_NAME, "varieties"
    )
    exception_entries = _read_entry_list(
        lexicons_directory / DO_NOT_MERGE_FILE_NAME, "exceptions"
    )
    return GateLexicons(
        always_merge_patterns=_compile_patterns(pattern_entries),
        forced_merge_overrides=_collect_field(override_entries, "variant"),
        named_varieties=_collect_field(variety_entries, "phrase"),
        do_not_merge=_collect_field(exception_entries, "variant"),
    )


def decide_merge(
    variant_cleaned: str,
    base_cleaned: str,
    evidence: MergeEvidence | None,
    lexicons: GateLexicons,
) -> GateDecision:
    """Decide whether a cleaned variant string merges into its base.

    Applies the gate layers strictly in order: do-not-merge exceptions (L0),
    always-merge patterns and forced overrides (L1), the statistical gate
    when evidence has at least MIN_SUPPORT recipes (L2), and the
    small-sample layer otherwise (L3).

    Args:
        variant_cleaned: Cleaned (lowercased, normalized) variant string.
        base_cleaned: Cleaned base string the variant would merge into.
        evidence: Divergence evidence for the pair, or None when the variant
            has no train support (e.g. test-only strings).
        lexicons: Loaded gate lexicons from load_gate_lexicons().

    Returns:
        A GateDecision naming the action, the deciding layer, and a reason.

    Raises:
        ValueError: If either string is empty or blank.
        TypeError: If evidence is neither None nor a MergeEvidence.

    Examples:
        >>> decide_merge("usukuchi soy sauce", "soy sauce", None, lexicons).action
        <GateAction.PRESERVE: 'preserve'>
    """
    _validate_decision_inputs(variant_cleaned, base_cleaned, evidence)
    lexicon_decision = _decide_lexicon_layers(variant_cleaned, base_cleaned, evidence, lexicons)
    if lexicon_decision is not None:
        return lexicon_decision
    if evidence is not None and evidence.variant_count >= MIN_SUPPORT:
        return _decide_statistical_gate(variant_cleaned, base_cleaned, evidence)
    return _decide_small_sample(variant_cleaned, base_cleaned, evidence, lexicons)


def _validate_decision_inputs(
    variant_cleaned: str,
    base_cleaned: str,
    evidence: MergeEvidence | None,
) -> None:
    """Fail fast on blank strings or a wrongly typed evidence object."""
    if not isinstance(variant_cleaned, str) or not variant_cleaned.strip():
        raise ValueError("variant_cleaned must be a non-blank string")
    if not isinstance(base_cleaned, str) or not base_cleaned.strip():
        raise ValueError("base_cleaned must be a non-blank string")
    if evidence is not None and not isinstance(evidence, MergeEvidence):
        raise TypeError(
            f"evidence must be MergeEvidence or None, got {type(evidence).__name__}"
        )


def _decision(
    action: GateAction,
    layer: str,
    evidence: MergeEvidence | None,
    reason: str,
) -> GateDecision:
    """Build a GateDecision; keeps the layer deciders short and uniform."""
    return GateDecision(action=action, layer=layer, evidence=evidence, reason=reason)


def _decide_lexicon_layers(
    variant_cleaned: str,
    base_cleaned: str,
    evidence: MergeEvidence | None,
    lexicons: GateLexicons,
) -> GateDecision | None:
    """Apply L0 (do-not-merge) then L1 (always-merge / forced override)."""
    if variant_cleaned in lexicons.do_not_merge:
        reason = f"'{variant_cleaned}' is a curated do-not-merge exception"
        return _decision(
            GateAction.PRESERVE, LAYER_DO_NOT_MERGE_EXCEPTION, evidence, reason
        )
    matched_pattern = _find_always_merge_pattern(variant_cleaned, lexicons)
    if matched_pattern is not None:
        reason = (
            f"'{variant_cleaned}' matches always-merge marketing pattern "
            f"{matched_pattern.pattern!r}; merges into '{base_cleaned}'"
        )
        return _decision(
            GateAction.MERGE, LAYER_ALWAYS_MERGE_LEXICON, evidence, reason
        )
    if variant_cleaned in lexicons.forced_merge_overrides:
        reason = (
            f"'{variant_cleaned}' is a curated forced-merge override "
            f"into '{base_cleaned}'"
        )
        return _decision(
            GateAction.MERGE, LAYER_FORCED_MERGE_OVERRIDE, evidence, reason
        )
    return None


def _decide_statistical_gate(
    variant_cleaned: str,
    base_cleaned: str,
    evidence: MergeEvidence,
) -> GateDecision:
    """Apply L2: borderline -> REVIEW, then the JSD-floor-and-null test."""
    jsd_bits = evidence.jsd_bits
    ratio = evidence.jsd_to_null_ratio
    statistics_note = f"jsd={jsd_bits:.4f} bits, {ratio:.2f}x null95"
    # Borderline wins over a clean verdict: these zones flip with +-0.01
    # threshold moves, so they go to human review instead.
    if _is_borderline(jsd_bits, ratio):
        reason = (
            f"borderline statistics for '{variant_cleaned}' vs "
            f"'{base_cleaned}': {statistics_note}"
        )
        return _decision(GateAction.REVIEW, LAYER_STATISTICAL_GATE, evidence, reason)
    is_clean_preserve = (
        jsd_bits >= JSD_FLOOR_BITS
        and jsd_bits >= NULL_MULTIPLIER * evidence.null95_bits
    )
    if is_clean_preserve:
        reason = (
            f"'{variant_cleaned}' diverges from '{base_cleaned}': {statistics_note}"
        )
        return _decision(GateAction.PRESERVE, LAYER_STATISTICAL_GATE, evidence, reason)
    reason = (
        f"'{variant_cleaned}' carries no cuisine signal vs "
        f"'{base_cleaned}': {statistics_note}"
    )
    return _decision(GateAction.MERGE, LAYER_STATISTICAL_GATE, evidence, reason)


def _is_borderline(jsd_bits: float, ratio: float) -> bool:
    """True when either statistic sits inside its flip-sensitive band."""
    is_borderline_jsd = BORDERLINE_JSD_BAND[0] <= jsd_bits <= BORDERLINE_JSD_BAND[1]
    is_borderline_ratio = BORDERLINE_RATIO_BAND[0] <= ratio <= BORDERLINE_RATIO_BAND[1]
    return is_borderline_jsd or is_borderline_ratio


def _decide_small_sample(
    variant_cleaned: str,
    base_cleaned: str,
    evidence: MergeEvidence | None,
    lexicons: GateLexicons,
) -> GateDecision:
    """Apply L3: named varieties, then concentrated-sample review, then merge."""
    matched_variety = _find_named_variety(variant_cleaned, lexicons.named_varieties)
    if matched_variety is not None:
        reason = (
            f"'{variant_cleaned}' names the variety '{matched_variety}'; "
            "preserved despite sub-threshold support"
        )
        return _decision(
            GateAction.PRESERVE, LAYER_NAMED_VARIETY_LEXICON, evidence, reason
        )
    if _is_concentrated_small_sample(evidence):
        top_cuisine = evidence.variant_top_cuisines[0]
        reason = (
            f"'{variant_cleaned}' is small (n={evidence.variant_count}) but "
            f"{top_cuisine.share:.0%} {top_cuisine.cuisine}; needs review"
        )
        return _decision(GateAction.REVIEW, LAYER_SMALL_SAMPLE_REVIEW, evidence, reason)
    support_note = (
        f"n={evidence.variant_count}" if evidence is not None else "no train evidence"
    )
    reason = (
        f"'{variant_cleaned}' ({support_note}) is below support threshold "
        f"{MIN_SUPPORT} with no variety phrase; default-merge into '{base_cleaned}'"
    )
    return _decision(GateAction.MERGE, LAYER_SMALL_SAMPLE_DEFAULT, evidence, reason)


def _is_concentrated_small_sample(evidence: MergeEvidence | None) -> bool:
    """True when a sub-threshold variant is concentrated enough for review."""
    if evidence is None or not evidence.variant_top_cuisines:
        return False
    has_minimum_support = evidence.variant_count >= SMALL_SAMPLE_REVIEW_MINIMUM
    is_concentrated = (
        evidence.variant_top_cuisines[0].share >= SMALL_SAMPLE_REVIEW_SHARE
    )
    return has_minimum_support and is_concentrated


def _find_always_merge_pattern(
    variant_cleaned: str,
    lexicons: GateLexicons,
) -> re.Pattern[str] | None:
    """Return the first always-merge pattern matching the variant, if any."""
    for pattern in lexicons.always_merge_patterns:
        if pattern.search(variant_cleaned):
            return pattern
    return None


def _find_named_variety(
    variant_cleaned: str,
    named_varieties: frozenset[str],
) -> str | None:
    """Return the variety phrase contained in the variant, if any.

    Matching is whole-word: 'tamarind paste' must not match the variety
    'tamari' (naive substring rules corrupt the tamarind/ancho/soy families).
    Iteration is sorted for deterministic tie-breaking.
    """
    for phrase in sorted(named_varieties):
        boundary_pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        if re.search(boundary_pattern, variant_cleaned):
            return phrase
    return None


def _read_entry_list(file_path: Path, list_key: str) -> tuple[dict, ...]:
    """Read one lexicon file and return its validated entry list.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the payload is not an object with a non-empty list of
            objects under list_key.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Lexicon file not found: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    entries = payload.get(list_key) if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"{file_path.name}: expected a non-empty list under {list_key!r}"
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{file_path.name}: every entry must be an object")
    return tuple(entries)


def _collect_field(entries: tuple[dict, ...], field_name: str) -> frozenset[str]:
    """Extract one required non-blank string field from every entry."""
    values = []
    for entry in entries:
        value = entry.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"lexicon entry missing non-blank string field {field_name!r}: {entry}"
            )
        values.append(value)
    return frozenset(values)


def _compile_patterns(entries: tuple[dict, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile the 'pattern' field of every entry, failing on bad regexes."""
    compiled = []
    for entry in entries:
        source = entry.get("pattern")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"pattern entry missing non-blank 'pattern': {entry}")
        try:
            compiled.append(re.compile(source))
        except re.error as regex_error:
            raise ValueError(f"invalid regex source {source!r}") from regex_error
    return tuple(compiled)
