"""Build the canonical ingredient vocabulary from raw train strings.

Orchestrates the four-pass vocabulary build:

1. Mechanical grouping — every raw string is cleaned and grouped by its
   singularized lookup key; manual aliases join their prescribed targets.
2. Strip + brand reduction — groups whose modifier-stripped or
   brand-resolved form lands on another group's key merge into it.
3. Candidate pair generation — alias-scope variant groups whose key is
   (whitelisted modifier tokens) + (another group's FULL key) pair with
   that base. Nouns are never modifiers, so 'fish sauce' finds no parent
   in 'sauce'. Word-order permutations are handled upstream: the strip
   pass drops safe modifiers order-independently, and the remaining
   known reorder artifacts live in lexicons/manual_aliases.json.
4. Gate application — each pair runs through the three-layer merge gate;
   decisive outcomes apply immediately, borderline ones serialize to the
   review queue for human/LLM judgment.

The build is deterministic: fixed Monte Carlo seeds, sorted iteration,
and rounded evidence floats make regeneration byte-identical.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pipeline.cuisine_divergence import evaluate_merge_candidate
from pipeline.load_raw_recipes import (
    RAW_TRAIN_PATH,
    TrainIndex,
    build_train_index,
    load_train_recipes,
)
from pipeline.merge_evidence import MergeEvidence
from pipeline.merge_gate import (
    JSD_FLOOR_BITS,
    NULL_MULTIPLIER,
    GateAction,
    GateLexicons,
    decide_merge,
    load_gate_lexicons,
)
from pipeline.normalize_text import clean_ingredient_text
from pipeline.resolve_brands import (
    BrandLexicon,
    load_brand_lexicon,
    resolve_brand_to_generic,
)
from pipeline.singularize import (
    SingularizeExceptions,
    load_singularize_exceptions,
    make_lookup_key,
)
from pipeline.strip_modifiers import (
    ModifierLexicon,
    load_modifier_lexicon,
    strip_safe_modifiers,
)

ALIAS_SCOPE_MINIMUM_FREQUENCY = 4
REDUCTION_PASS_LIMIT = 3
EXAMPLE_RECIPE_ID_LIMIT = 3
EVIDENCE_DECIMAL_PLACES = 4

MANUAL_ALIASES_FILENAME = "manual_aliases.json"
VARIANT_MODIFIER_TOKENS_FILENAME = "variant_modifier_tokens.json"
SINGULARIZE_EXCEPTIONS_FILENAME = "singularize_exceptions.json"
MODIFIER_STRIP_FILENAME = "modifier_strip.json"
BRAND_PATTERNS_FILENAME = "brand_patterns.json"

DEFAULT_LEXICONS_DIRECTORY = Path(__file__).resolve().parent.parent / "lexicons"
DEFAULT_QUEUE_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "merge_review_queue.jsonl"
)

LAYER_TO_ALIAS_SOURCE = {
    "always_merge_lexicon": "always_merge_lexicon",
    "forced_merge_override": "forced_merge_override",
    "statistical_gate": "statistical_gate",
    "small_sample_default": "statistical_gate",
}


@dataclass
class GroupMember:
    """One raw string's place inside a vocabulary group.

    Attributes:
        cleaned: The cleaned form of the raw string (or of its manual-alias
            target when the raw string is a known artifact).
        source: Provenance of the raw -> group mapping, e.g.
            'mechanical_normalization', 'manual_alias', 'modifier_strip',
            'brand_pattern', 'always_merge_lexicon', 'statistical_gate'.
        rule: Optional detail for the source (gate reason, alias target).
    """

    cleaned: str
    source: str
    rule: str | None = None


@dataclass
class VocabularyGroup:
    """A set of raw strings sharing one canonical ingredient concept.

    Attributes:
        key: Singularized lookup key identifying the group.
        members: Raw string -> GroupMember provenance record.
        canonical_override: Display name override for synthetic groups
            created from brand-resolution targets that had no natural
            group of their own.
    """

    key: str
    members: dict[str, GroupMember] = field(default_factory=dict)
    canonical_override: str | None = None


@dataclass(frozen=True)
class PairCandidate:
    """One (variant, base) merge candidate identified by group keys."""

    variant_key: str
    base_key: str


@dataclass(frozen=True)
class PreservedVariant:
    """A gate PRESERVE outcome linking a variant to its base with evidence."""

    base_key: str
    layer: str
    reason: str
    evidence: MergeEvidence | None


@dataclass(frozen=True)
class ReviewCandidate:
    """A gate REVIEW outcome carrying everything the queue entry needs."""

    variant_key: str
    base_key: str
    variant_cleaned: str
    base_cleaned: str
    variant_raw_strings: tuple[str, ...]
    evidence: MergeEvidence
    gate_layer: str
    gate_reason: str


@dataclass(frozen=True)
class PipelineLexicons:
    """Every curated lexicon the vocabulary build consumes, preloaded."""

    singularize_exceptions: SingularizeExceptions
    modifier_lexicon: ModifierLexicon
    brand_lexicon: BrandLexicon
    gate_lexicons: GateLexicons
    manual_aliases: dict[str, str]
    variant_modifier_tokens: frozenset[str]


@dataclass
class VocabularyBuild:
    """Result of one vocabulary build over the train index."""

    groups: dict[str, VocabularyGroup]
    preserved: dict[str, PreservedVariant]
    alias_scope_keys: frozenset[str]
    review_candidates: tuple[ReviewCandidate, ...]
    review_entries: list[dict]

    @property
    def parent_keys(self) -> dict[str, str]:
        """Variant key -> base key for every gate-preserved variant."""
        return {
            variant: preserved.base_key
            for variant, preserved in self.preserved.items()
        }


def load_manual_aliases(path: Path) -> dict[str, str]:
    """Load the raw-string -> target-string manual alias map.

    Args:
        path: Path to manual_aliases.json.

    Returns:
        Mapping of exact raw ingredient strings to their replacement text.

    Raises:
        ValueError: If any entry is not a string-to-string pair.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    aliases = {
        key: value for key, value in data.items() if not key.startswith("_")
    }
    for key, value in aliases.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Manual alias for {key!r} must be a non-empty string")
    return aliases


def load_variant_modifier_tokens(path: Path) -> frozenset[str]:
    """Load the candidate-modifier token whitelist.

    Args:
        path: Path to variant_modifier_tokens.json.

    Returns:
        Frozen set of tokens allowed to prefix a base in pair generation.

    Raises:
        ValueError: If the 'tokens' list is missing or holds non-strings.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    tokens = data.get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(token, str) for token in tokens
    ):
        raise ValueError(f"{path} must hold a 'tokens' list of strings")
    return frozenset(tokens)


def load_pipeline_lexicons(lexicons_directory: Path) -> PipelineLexicons:
    """Load every lexicon the vocabulary build needs from one directory.

    Args:
        lexicons_directory: Directory holding the curated lexicon files.

    Returns:
        A frozen PipelineLexicons bundle.
    """
    return PipelineLexicons(
        singularize_exceptions=load_singularize_exceptions(
            lexicons_directory / SINGULARIZE_EXCEPTIONS_FILENAME
        ),
        modifier_lexicon=load_modifier_lexicon(
            lexicons_directory / MODIFIER_STRIP_FILENAME
        ),
        brand_lexicon=load_brand_lexicon(
            lexicons_directory / BRAND_PATTERNS_FILENAME
        ),
        gate_lexicons=load_gate_lexicons(lexicons_directory),
        manual_aliases=load_manual_aliases(
            lexicons_directory / MANUAL_ALIASES_FILENAME
        ),
        variant_modifier_tokens=load_variant_modifier_tokens(
            lexicons_directory / VARIANT_MODIFIER_TOKENS_FILENAME
        ),
    )


def group_strings_mechanically(
    raw_strings: Iterable[str], lexicons: PipelineLexicons
) -> dict[str, VocabularyGroup]:
    """Group raw strings by cleaned, singularized lookup key (pass 1).

    Manual-alias artifacts are redirected to their prescribed target text
    before keying, so 'dri leav rosemari' lands in the group its repaired
    form belongs to.

    Args:
        raw_strings: Unique raw ingredient strings.
        lexicons: Loaded lexicon bundle.

    Returns:
        Mapping of lookup key -> VocabularyGroup.
    """
    groups: dict[str, VocabularyGroup] = {}
    for raw in raw_strings:
        alias_target = lexicons.manual_aliases.get(raw)
        if alias_target is None:
            cleaned = clean_ingredient_text(raw)
            member = GroupMember(cleaned=cleaned, source="mechanical_normalization")
        else:
            cleaned = clean_ingredient_text(alias_target)
            member = GroupMember(
                cleaned=cleaned, source="manual_alias", rule=alias_target
            )
        key = make_lookup_key(cleaned, lexicons.singularize_exceptions)
        groups.setdefault(key, VocabularyGroup(key=key)).members[raw] = member
    return groups


def group_frequency(group: VocabularyGroup, index: TrainIndex) -> int:
    """Count distinct train recipes containing any member of the group."""
    recipe_ids: set[int] = set()
    for raw in group.members:
        recipe_ids.update(index.string_to_recipe_ids.get(raw, frozenset()))
    return len(recipe_ids)


def representative_cleaned(group: VocabularyGroup, index: TrainIndex) -> str:
    """Return the cleaned form of the group's highest-frequency member.

    Synthetic groups created from brand targets return their override name.
    Frequency ties break toward the lexicographically smallest raw string.
    """
    if group.canonical_override is not None:
        return group.canonical_override
    best_raw = min(
        group.members,
        key=lambda raw: (-len(index.string_to_recipe_ids.get(raw, frozenset())), raw),
    )
    return group.members[best_raw].cleaned


def _matches_always_merge(cleaned: str, gate_lexicons: GateLexicons) -> bool:
    return any(
        pattern.search(cleaned) for pattern in gate_lexicons.always_merge_patterns
    )


def _reduction_move(
    group: VocabularyGroup, lexicons: PipelineLexicons, index: TrainIndex
) -> tuple[str, str, bool] | None:
    """Find where a group reduces to: (target_key, source, may_create)."""
    cleaned = representative_cleaned(group, index)
    brand_generic = resolve_brand_to_generic(cleaned, lexicons.brand_lexicon)
    if brand_generic is not None:
        target_cleaned = clean_ingredient_text(brand_generic)
        target_key = make_lookup_key(target_cleaned, lexicons.singularize_exceptions)
        if target_key != group.key:
            return target_key, "brand_pattern", True
    stripped = strip_safe_modifiers(cleaned, lexicons.modifier_lexicon)
    stripped_key = make_lookup_key(stripped, lexicons.singularize_exceptions)
    if stripped_key != group.key:
        source = (
            "always_merge_lexicon"
            if _matches_always_merge(cleaned, lexicons.gate_lexicons)
            else "modifier_strip"
        )
        return stripped_key, source, False
    return None


def _apply_move(
    groups: dict[str, VocabularyGroup],
    move: tuple[str, str, str],
    lexicons: PipelineLexicons,
) -> None:
    """Transfer one group's members into its reduction target."""
    source_key, target_key, source_label = move
    moving = groups.pop(source_key)
    target = groups.get(target_key)
    if target is None:
        target_cleaned = strip_safe_modifiers(
            representative_cleaned_for_override(moving), lexicons.modifier_lexicon
        )
        target = VocabularyGroup(key=target_key, canonical_override=target_cleaned)
        groups[target_key] = target
    for raw, member in moving.members.items():
        target.members[raw] = GroupMember(
            cleaned=member.cleaned, source=source_label, rule=member.rule
        )


def representative_cleaned_for_override(group: VocabularyGroup) -> str:
    """Best-effort display text for a synthetic target group."""
    if group.canonical_override is not None:
        return group.canonical_override
    first_member = sorted(group.members)[0]
    return group.members[first_member].cleaned


def merge_groups_by_strip_and_brand(
    groups: dict[str, VocabularyGroup],
    lexicons: PipelineLexicons,
    index: TrainIndex,
) -> dict[str, VocabularyGroup]:
    """Reduce groups via brand resolution and modifier stripping (pass 2).

    Brand targets may create a new group when no natural group exists for
    the generic form; stripped forms only merge into existing groups.
    Runs to a fixpoint within REDUCTION_PASS_LIMIT rounds.

    Args:
        groups: Output of group_strings_mechanically; mutated and returned.
        lexicons: Loaded lexicon bundle.
        index: Train index for representative selection.

    Returns:
        The reduced groups mapping.
    """
    for _ in range(REDUCTION_PASS_LIMIT):
        moves: list[tuple[str, str, str]] = []
        for key in sorted(groups):
            found = _reduction_move(groups[key], lexicons, index)
            if found is None:
                continue
            target_key, source_label, may_create = found
            if may_create or target_key in groups:
                moves.append((key, target_key, source_label))
        if not moves:
            return groups
        for move in moves:
            if move[0] in groups:
                _apply_move(groups, move, lexicons)
    return groups


def generate_candidate_pairs(
    groups: dict[str, VocabularyGroup],
    lexicons: PipelineLexicons,
    index: TrainIndex,
) -> list[PairCandidate]:
    """Pair alias-scope variants with their full-head-phrase base (pass 3).

    A pair forms only when every prefix token is a known modifier and the
    remaining tokens exactly equal another alias-scope group's key; the
    longest available base wins.

    Args:
        groups: Reduced vocabulary groups.
        lexicons: Loaded lexicon bundle.
        index: Train index for frequency scoping.

    Returns:
        Deterministically ordered merge candidates.
    """
    modifier_tokens = (
        lexicons.variant_modifier_tokens
        | lexicons.modifier_lexicon.strip_tokens
        | lexicons.modifier_lexicon.never_strip_tokens
    )
    scope = {
        key
        for key, group in groups.items()
        if group_frequency(group, index) >= ALIAS_SCOPE_MINIMUM_FREQUENCY
    }
    pairs: list[PairCandidate] = []
    for key in sorted(scope):
        tokens = key.split()
        for prefix_length in range(1, len(tokens)):
            if tokens[prefix_length - 1] not in modifier_tokens:
                break
            base_key = " ".join(tokens[prefix_length:])
            if base_key in scope:
                pairs.append(PairCandidate(variant_key=key, base_key=base_key))
                break
    return pairs


def _resolve_redirects(key: str, redirects: dict[str, str]) -> str:
    while key in redirects:
        key = redirects[key]
    return key


def _record_merge(
    groups: dict[str, VocabularyGroup],
    redirects: dict[str, str],
    outcome: tuple[PairCandidate, "GateDecisionLike"],
) -> None:
    pair, decision = outcome
    source_key = _resolve_redirects(pair.variant_key, redirects)
    target_key = _resolve_redirects(pair.base_key, redirects)
    if source_key == target_key or source_key not in groups:
        return
    alias_source = LAYER_TO_ALIAS_SOURCE.get(decision.layer, "statistical_gate")
    moving = groups.pop(source_key)
    target = groups[target_key]
    for raw, member in moving.members.items():
        target.members[raw] = GroupMember(
            cleaned=member.cleaned, source=alias_source, rule=decision.reason
        )
    redirects[source_key] = target_key


GateDecisionLike = object


def apply_pair_outcomes(
    groups: dict[str, VocabularyGroup],
    pairs: list[PairCandidate],
    lexicons: PipelineLexicons,
    index: TrainIndex,
) -> tuple[dict[str, VocabularyGroup], dict[str, str], list[ReviewCandidate]]:
    """Run the merge gate over every pair and apply the verdicts (pass 4).

    Args:
        groups: Reduced vocabulary groups; mutated and returned.
        pairs: Candidates from generate_candidate_pairs.
        lexicons: Loaded lexicon bundle.
        index: Train index for evidence evaluation.

    Returns:
        (groups, preserved, review_candidates) where preserved links each
        gate-preserved variant to its base with the deciding layer and
        evidence, and review_candidates holds every REVIEW outcome for
        queue serialization.
    """
    ordered = sorted(pairs, key=lambda p: (-len(p.variant_key.split()), p.variant_key))
    redirects: dict[str, str] = {}
    preserve_records: list[tuple[str, str, object, MergeEvidence]] = []
    review_candidates: list[ReviewCandidate] = []
    for pair in ordered:
        variant_group = groups[_resolve_redirects(pair.variant_key, redirects)]
        base_group = groups[_resolve_redirects(pair.base_key, redirects)]
        evidence = evaluate_merge_candidate(
            tuple(sorted(variant_group.members)),
            tuple(sorted(base_group.members)),
            index,
        )
        variant_cleaned = representative_cleaned(variant_group, index)
        base_cleaned = representative_cleaned(base_group, index)
        decision = decide_merge(
            variant_cleaned, base_cleaned, evidence, lexicons.gate_lexicons
        )
        if decision.action is GateAction.MERGE:
            _record_merge(groups, redirects, (pair, decision))
        elif decision.action is GateAction.PRESERVE:
            preserve_records.append(
                (pair.variant_key, pair.base_key, decision, evidence)
            )
        else:
            review_candidates.append(
                ReviewCandidate(
                    variant_key=pair.variant_key,
                    base_key=pair.base_key,
                    variant_cleaned=variant_cleaned,
                    base_cleaned=base_cleaned,
                    variant_raw_strings=tuple(sorted(variant_group.members)),
                    evidence=evidence,
                    gate_layer=decision.layer,
                    gate_reason=decision.reason,
                )
            )
    preserved = {
        variant: PreservedVariant(
            base_key=_resolve_redirects(base, redirects),
            layer=decision.layer,
            reason=decision.reason,
            evidence=evidence,
        )
        for variant, base, decision, evidence in preserve_records
    }
    return groups, preserved, review_candidates


def _cuisine_shares_to_dicts(shares) -> list[dict]:
    return [
        {
            "cuisine": share.cuisine,
            "share": round(share.share, EVIDENCE_DECIMAL_PLACES),
            "lift": round(share.lift, EVIDENCE_DECIMAL_PLACES),
        }
        for share in shares
    ]


def _suggested_decision(evidence: MergeEvidence) -> str:
    is_preserve = (
        evidence.jsd_bits >= JSD_FLOOR_BITS
        and evidence.jsd_to_null_ratio >= NULL_MULTIPLIER
    )
    return "preserve" if is_preserve else "merge"


def build_review_queue_entries(
    review_candidates: Iterable[ReviewCandidate], index: TrainIndex
) -> list[dict]:
    """Serialize REVIEW outcomes into self-contained queue entries.

    Every entry carries the full evidence needed to decide it standalone,
    sorted by decision_id with floats rounded for byte-stable regeneration.

    Args:
        review_candidates: REVIEW outcomes from apply_pair_outcomes.
        index: Train index for example recipe ids.

    Returns:
        Queue entries ready for JSONL serialization.
    """
    entries = []
    for candidate in review_candidates:
        evidence = candidate.evidence
        example_ids: set[int] = set()
        for raw in candidate.variant_raw_strings:
            example_ids.update(index.string_to_recipe_ids.get(raw, frozenset()))
        entries.append(
            {
                "decision_id": (
                    f"{candidate.variant_key.replace(' ', '_')}"
                    f"__vs__{candidate.base_key.replace(' ', '_')}"
                ),
                "variant_string": candidate.variant_cleaned,
                "base_string": candidate.base_cleaned,
                "variant_train_frequency": evidence.variant_count,
                "base_train_frequency": evidence.base_count,
                "jsd_bits": round(evidence.jsd_bits, EVIDENCE_DECIMAL_PLACES),
                "null95_bits": round(evidence.null95_bits, EVIDENCE_DECIMAL_PLACES),
                "jsd_to_null_ratio": round(
                    evidence.jsd_to_null_ratio, EVIDENCE_DECIMAL_PLACES
                ),
                "variant_top_cuisines": _cuisine_shares_to_dicts(
                    evidence.variant_top_cuisines
                ),
                "base_top_cuisines": _cuisine_shares_to_dicts(
                    evidence.base_top_cuisines
                ),
                "example_recipe_ids": sorted(example_ids)[:EXAMPLE_RECIPE_ID_LIMIT],
                "gate_route": candidate.gate_layer,
                "suggested_decision": _suggested_decision(evidence),
                "suggestion_reason": candidate.gate_reason,
            }
        )
    return sorted(entries, key=lambda entry: entry["decision_id"])


def build_vocabulary_from_index(
    index: TrainIndex, lexicons: PipelineLexicons
) -> VocabularyBuild:
    """Run the full four-pass vocabulary build over a train index.

    Args:
        index: Train index from build_train_index.
        lexicons: Loaded lexicon bundle.

    Returns:
        VocabularyBuild with final groups, parent links, alias scope, and
        the serialized review queue.
    """
    raw_strings = sorted(index.string_to_recipe_ids)
    groups = group_strings_mechanically(raw_strings, lexicons)
    groups = merge_groups_by_strip_and_brand(groups, lexicons, index)
    pairs = generate_candidate_pairs(groups, lexicons, index)
    groups, preserved, review_candidates = apply_pair_outcomes(
        groups, pairs, lexicons, index
    )
    alias_scope_keys = frozenset(
        key
        for key, group in groups.items()
        if group_frequency(group, index) >= ALIAS_SCOPE_MINIMUM_FREQUENCY
    )
    return VocabularyBuild(
        groups=groups,
        preserved=preserved,
        alias_scope_keys=alias_scope_keys,
        review_candidates=tuple(review_candidates),
        review_entries=build_review_queue_entries(review_candidates, index),
    )


def write_review_queue_to_path(entries: Iterable[dict], path: Path) -> None:
    """Atomically write queue entries as one-JSON-object-per-line.

    Args:
        entries: Queue entries from build_review_queue_entries.
        path: Destination JSONL path; parent directory must exist.
    """
    lines = [
        json.dumps(entry, ensure_ascii=False, sort_keys=True) for entry in entries
    ]
    content = "\n".join(lines) + "\n" if lines else ""
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=path.name, suffix=".tmp"
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temporary_path, path)


def main() -> None:
    """CLI: build the vocabulary from real data and write the review queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lexicons-directory", type=Path, default=DEFAULT_LEXICONS_DIRECTORY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_QUEUE_PATH)
    arguments = parser.parse_args()

    lexicons = load_pipeline_lexicons(arguments.lexicons_directory)
    index = build_train_index(load_train_recipes(RAW_TRAIN_PATH))
    raw_strings = sorted(index.string_to_recipe_ids)
    print(f"raw unique strings: {len(raw_strings)}")

    groups = group_strings_mechanically(raw_strings, lexicons)
    print(f"after mechanical grouping: {len(groups)} groups")
    groups = merge_groups_by_strip_and_brand(groups, lexicons, index)
    print(f"after strip+brand reduction: {len(groups)} groups")

    pairs = generate_candidate_pairs(groups, lexicons, index)
    print(f"candidate pairs: {len(pairs)}")
    groups, preserved, review_candidates = apply_pair_outcomes(
        groups, pairs, lexicons, index
    )
    print(
        f"after gate: {len(groups)} groups, {len(preserved)} preserved variants, "
        f"{len(review_candidates)} for review"
    )

    entries = build_review_queue_entries(review_candidates, index)
    write_review_queue_to_path(entries, arguments.output)
    print(f"wrote {len(entries)} review entries to {arguments.output}")


if __name__ == "__main__":
    main()
