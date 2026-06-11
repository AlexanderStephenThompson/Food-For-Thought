"""Compile the vocabulary build plus merge decisions into staged ingredients.

Takes the VocabularyBuild produced by pipeline.build_vocabulary, applies the
human/LLM verdicts from the merge-decision JSONL, and emits the pinned
staged/ingredients.json artifact: one entry per alias-scope ingredient with
its canonical name, slug id, alias provenance, parent link, and preserve
evidence. A hard validation pass guards every schema invariant (unique alias
ownership, slug format, parent chain depth, alias coverage floor) before the
artifact is written.

The compile is deterministic: sorted iteration everywhere, evidence floats
rounded to a fixed precision, and atomic sorted-key serialization via
pipeline.staged_io.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.build_vocabulary import (
    ALIAS_SCOPE_MINIMUM_FREQUENCY,
    DEFAULT_LEXICONS_DIRECTORY,
    GroupMember,
    PreservedVariant,
    ReviewCandidate,
    VocabularyBuild,
    VocabularyGroup,
    build_vocabulary_from_index,
    group_frequency,
    load_pipeline_lexicons,
    representative_cleaned,
)
from pipeline.load_raw_recipes import (
    RAW_TRAIN_PATH,
    TrainIndex,
    build_train_index,
    load_train_recipes,
)
from pipeline.staged_io import (
    SCHEMA_VERSION,
    compute_build_fingerprint,
    write_staged_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DECISIONS_PATH = PROJECT_ROOT / "lexicons" / "merge_decisions.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "staged" / "ingredients.json"

DECISION_MERGE = "merge"
DECISION_PRESERVE = "preserve"
DECISION_MERGE_INTO_PREFIX = "merge_into:"
DECISION_ID_SEPARATOR = "__vs__"
REQUIRED_DECISION_FIELDS = ("decision_id", "decision", "decided_by", "note")

MANUAL_REVIEW_SOURCE = "manual_review"
MANUAL_REVIEW_LAYER = "manual_review"
CANONICAL_SURFACE_FORM_SOURCE = "canonical_surface_form"

SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
SLUG_REPLACE_PATTERN = re.compile(r"[^a-z0-9]+")

COVERAGE_MINIMUM_RATIO = 0.988
EVIDENCE_DECIMAL_PLACES = 4
PARENT_CHAIN_DEPTH_LIMIT = 2
UNCOVERED_STRING_REPORT_LIMIT = 20


@dataclass(frozen=True)
class CompileStatistics:
    """Coverage and size figures for one validated ingredients payload.

    Attributes:
        ingredient_count: Number of compiled ingredients.
        alias_count: Total alias entries across all ingredients.
        parent_link_count: Ingredients carrying a non-null parent_id.
        covered_mention_count: Train mentions of raw strings that appear
            as an alias of some ingredient.
        total_mention_count: Train mentions across every raw string.
        coverage_ratio: covered_mention_count / total_mention_count.
    """

    ingredient_count: int
    alias_count: int
    parent_link_count: int
    covered_mention_count: int
    total_mention_count: int
    coverage_ratio: float


def make_decision_id(variant_key: str, base_key: str) -> str:
    """Build the canonical decision id for a (variant, base) review pair.

    Args:
        variant_key: Vocabulary group key of the variant.
        base_key: Vocabulary group key of the base.

    Returns:
        '<variant>__vs__<base>' with spaces replaced by underscores.
    """
    return (
        variant_key.replace(" ", "_")
        + DECISION_ID_SEPARATOR
        + base_key.replace(" ", "_")
    )


def load_merge_decisions(path: Path) -> dict[str, dict]:
    """Load and validate the merge-decision JSONL file.

    Args:
        path: JSONL file with one decision object per line, each holding
            exactly the fields decision_id, decision, decided_by, note.

    Returns:
        Mapping of decision_id -> decision record.

    Raises:
        ValueError: On invalid JSON, missing/mistyped/unexpected fields,
            unknown decision values, or duplicate decision ids.
        FileNotFoundError: If the file does not exist.
    """
    decisions: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = _parse_decision_line(line, line_number, path)
            decision_id = record["decision_id"]
            if decision_id in decisions:
                raise ValueError(
                    f"{path} line {line_number}: duplicate decision_id "
                    f"{decision_id!r}"
                )
            decisions[decision_id] = record
    return decisions


def _parse_decision_line(line: str, line_number: int, path: Path) -> dict:
    """Parse one JSONL line into a fully validated decision record."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} line {line_number}: invalid JSON ({error})"
        ) from error
    if not isinstance(record, dict):
        raise ValueError(f"{path} line {line_number}: expected a JSON object")
    for field_name in REQUIRED_DECISION_FIELDS:
        value = record.get(field_name)
        is_required_nonempty = field_name != "note"
        if not isinstance(value, str) or (is_required_nonempty and not value):
            raise ValueError(
                f"{path} line {line_number}: missing or invalid field "
                f"{field_name!r}"
            )
    unexpected_fields = sorted(set(record) - set(REQUIRED_DECISION_FIELDS))
    if unexpected_fields:
        raise ValueError(
            f"{path} line {line_number}: unexpected fields {unexpected_fields}"
        )
    _require_valid_decision_value(record["decision"], line_number, path)
    return record


def _require_valid_decision_value(
    decision: str, line_number: int, path: Path
) -> None:
    """Reject decision values outside merge | preserve | merge_into:<key>."""
    if decision in (DECISION_MERGE, DECISION_PRESERVE):
        return
    if decision.startswith(DECISION_MERGE_INTO_PREFIX):
        target_key = decision[len(DECISION_MERGE_INTO_PREFIX):]
        if target_key.strip():
            return
    raise ValueError(
        f"{path} line {line_number}: unknown decision value {decision!r}"
    )


def apply_merge_decisions(
    build: VocabularyBuild, decisions: dict[str, dict], index: TrainIndex
) -> None:
    """Apply review verdicts to the vocabulary build, in place.

    Merge verdicts absorb the variant group's members into the target group
    as 'manual_review' aliases (rule = the decision note); preserve verdicts
    register the variant in build.preserved with layer 'manual_review' and
    the candidate's gate evidence. Merge verdicts are applied first so a
    preserve whose base was itself merged links to the surviving group.

    Args:
        build: Vocabulary build to mutate.
        decisions: decision_id -> record from load_merge_decisions.
        index: Train index (part of the pinned interface; the absorption
            itself is frequency-independent).

    Raises:
        ValueError: If any review candidate lacks a decision, any decision
            matches no candidate, or a merge target group does not exist.
    """
    candidates = {
        make_decision_id(candidate.variant_key, candidate.base_key): candidate
        for candidate in build.review_candidates
    }
    _require_decision_alignment(candidates, decisions)
    redirects = _apply_merge_verdicts(candidates, decisions, build)
    _apply_preserve_verdicts(candidates, decisions, build, redirects)


def _require_decision_alignment(
    candidates: dict[str, ReviewCandidate], decisions: dict[str, dict]
) -> None:
    """Fail fast when candidates and decisions do not match one-to-one."""
    missing_ids = sorted(set(candidates) - set(decisions))
    orphan_ids = sorted(set(decisions) - set(candidates))
    if not missing_ids and not orphan_ids:
        return
    problems = []
    if missing_ids:
        problems.append(f"review candidates without decisions: {missing_ids}")
    if orphan_ids:
        problems.append(f"decisions without review candidates: {orphan_ids}")
    raise ValueError("; ".join(problems))


def _follow_redirects(key: str, redirects: dict[str, str]) -> str:
    """Chase decision-applied merges to the surviving group key."""
    while key in redirects:
        key = redirects[key]
    return key


def _apply_merge_verdicts(
    candidates: dict[str, ReviewCandidate],
    decisions: dict[str, dict],
    build: VocabularyBuild,
) -> dict[str, str]:
    """Absorb every merge/merge_into variant; return the redirect map."""
    redirects: dict[str, str] = {}
    for decision_id in sorted(candidates):
        record = decisions[decision_id]
        if record["decision"] == DECISION_PRESERVE:
            continue
        candidate = candidates[decision_id]
        target_key = _resolve_merge_target(candidate, record, build, redirects)
        _absorb_variant_group(
            build, candidate.variant_key, target_key, record["note"]
        )
        redirects[candidate.variant_key] = target_key
    return redirects


def _resolve_merge_target(
    candidate: ReviewCandidate,
    record: dict,
    build: VocabularyBuild,
    redirects: dict[str, str],
) -> str:
    """Pick the surviving group a merge verdict absorbs the variant into."""
    decision = record["decision"]
    if decision == DECISION_MERGE:
        named_target = candidate.base_key
    else:
        named_target = decision[len(DECISION_MERGE_INTO_PREFIX):]
    target_key = _follow_redirects(named_target, redirects)
    if target_key == candidate.variant_key or target_key not in build.groups:
        decision_id = make_decision_id(candidate.variant_key, candidate.base_key)
        raise ValueError(
            f"decision {decision_id!r}: merge target {named_target!r} is not "
            "an available vocabulary group"
        )
    return target_key


def _absorb_variant_group(
    build: VocabularyBuild, variant_key: str, target_key: str, note: str
) -> None:
    """Move a variant group's members into the target as review aliases."""
    if variant_key not in build.groups:
        raise ValueError(
            f"variant group {variant_key!r} is not available to merge"
        )
    moving_group = build.groups.pop(variant_key)
    target_group = build.groups[target_key]
    for raw in sorted(moving_group.members):
        member = moving_group.members[raw]
        target_group.members[raw] = GroupMember(
            cleaned=member.cleaned, source=MANUAL_REVIEW_SOURCE, rule=note
        )


def _apply_preserve_verdicts(
    candidates: dict[str, ReviewCandidate],
    decisions: dict[str, dict],
    build: VocabularyBuild,
    redirects: dict[str, str],
) -> None:
    """Register preserve verdicts as manual_review parent links."""
    for decision_id in sorted(candidates):
        record = decisions[decision_id]
        if record["decision"] != DECISION_PRESERVE:
            continue
        candidate = candidates[decision_id]
        build.preserved[candidate.variant_key] = PreservedVariant(
            base_key=_follow_redirects(candidate.base_key, redirects),
            layer=MANUAL_REVIEW_LAYER,
            reason=record["note"],
            evidence=candidate.evidence,
        )


def slugify_ingredient_name(name: str) -> str:
    """Derive the ingredient id slug from a canonical name.

    Args:
        name: Canonical ingredient display name.

    Returns:
        The name with every non-[a-z0-9] run collapsed to '_' and
        leading/trailing underscores stripped.

    Raises:
        ValueError: If the result violates ^[a-z0-9][a-z0-9_]*$.
    """
    slug = SLUG_REPLACE_PATTERN.sub("_", name).strip("_")
    if not SLUG_PATTERN.match(slug):
        raise ValueError(
            f"cannot derive a valid ingredient id from name {name!r}"
        )
    return slug


def compile_ingredients_payload(
    build: VocabularyBuild,
    decisions: dict[str, dict],
    index: TrainIndex,
    fingerprint: dict,
) -> dict:
    """Compile the staged ingredients payload from a vocabulary build.

    Applies the merge decisions (mutating the build), keeps only groups at
    or above the alias-scope frequency floor, and emits the pinned
    staged/ingredients.json structure sorted by id and alias.

    Args:
        build: Vocabulary build; mutated by decision application.
        decisions: decision_id -> record from load_merge_decisions.
        index: Train index for frequencies and representatives.
        fingerprint: Build block from compute_build_fingerprint.

    Returns:
        Payload dict ready for write_staged_json.

    Raises:
        ValueError: On decision/candidate mismatches, missing merge targets,
            preserved parents that were absorbed or fell outside the
            vocabulary, or names that cannot form a valid slug.
    """
    apply_merge_decisions(build, decisions, index)
    eligible_groups = {
        key: build.groups[key]
        for key in sorted(build.groups)
        if group_frequency(build.groups[key], index)
        >= ALIAS_SCOPE_MINIMUM_FREQUENCY
    }
    parent_links = _resolve_parent_links(build, eligible_groups, index)
    ingredients = [
        _build_ingredient_entry(eligible_groups[key], index, parent_links.get(key))
        for key in sorted(eligible_groups)
    ]
    ingredients.sort(key=lambda entry: entry["id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "build": dict(fingerprint),
        "ingredients": ingredients,
    }


def _resolve_parent_links(
    build: VocabularyBuild,
    eligible_groups: dict[str, VocabularyGroup],
    index: TrainIndex,
) -> dict[str, tuple[str, dict]]:
    """Map preserved variant keys to (parent_id, preserve_evidence)."""
    links: dict[str, tuple[str, dict]] = {}
    for variant_key in sorted(build.preserved):
        if variant_key not in eligible_groups:
            continue
        preserved = build.preserved[variant_key]
        base_group = eligible_groups.get(preserved.base_key)
        if base_group is None:
            raise ValueError(
                f"preserved variant {variant_key!r}: base group "
                f"{preserved.base_key!r} was absorbed or sits outside the "
                "vocabulary"
            )
        parent_id = slugify_ingredient_name(
            representative_cleaned(base_group, index)
        )
        links[variant_key] = (parent_id, _preserve_evidence_to_dict(preserved))
    return links


def _preserve_evidence_to_dict(preserved: PreservedVariant) -> dict:
    """Serialize a PreservedVariant into the pinned preserve_evidence shape."""
    evidence = preserved.evidence
    if evidence is None:
        return {
            "layer": preserved.layer,
            "jsd_bits": None,
            "null95_bits": None,
            "variant_count": None,
        }
    return {
        "layer": preserved.layer,
        "jsd_bits": round(evidence.jsd_bits, EVIDENCE_DECIMAL_PLACES),
        "null95_bits": round(evidence.null95_bits, EVIDENCE_DECIMAL_PLACES),
        "variant_count": evidence.variant_count,
    }


def _build_ingredient_entry(
    group: VocabularyGroup,
    index: TrainIndex,
    parent_link: tuple[str, dict] | None,
) -> dict:
    """Assemble one pinned-schema ingredient entry from a vocabulary group."""
    name = representative_cleaned(group, index)
    parent_id, preserve_evidence = parent_link or (None, None)
    return {
        "id": slugify_ingredient_name(name),
        "name": name,
        "category": None,
        "parent_id": parent_id,
        "train_mention_count": group_frequency(group, index),
        "preserve_evidence": preserve_evidence,
        "aliases": _build_alias_entries(group, index),
    }


def _string_frequency(raw: str, index: TrainIndex) -> int:
    """Distinct train recipes containing one raw string."""
    return len(index.string_to_recipe_ids.get(raw, frozenset()))


def _build_alias_entries(group: VocabularyGroup, index: TrainIndex) -> list[dict]:
    """Serialize a group's members as sorted alias entries."""
    representative_raw = min(
        group.members,
        key=lambda raw: (-_string_frequency(raw, index), raw),
    )
    entries = []
    for raw in sorted(group.members):
        member = group.members[raw]
        is_representative = raw == representative_raw
        entries.append(
            {
                "alias": raw,
                "source": (
                    CANONICAL_SURFACE_FORM_SOURCE
                    if is_representative
                    else member.source
                ),
                "rule": member.rule,
                "train_frequency": _string_frequency(raw, index),
            }
        )
    return entries


def validate_compiled_payload(
    payload: dict, index: TrainIndex
) -> CompileStatistics:
    """Run every hard schema gate over a compiled ingredients payload.

    Gates: unique alias ownership, canonical names never shadowing another
    ingredient's alias, slug format and uniqueness, parent existence and
    chain depth, and the alias-coverage floor over train mentions.

    Args:
        payload: Compiled ingredients payload.
        index: Train index supplying string frequencies.

    Returns:
        CompileStatistics with size and coverage figures for reporting.

    Raises:
        ValueError: With a gate-specific message on the first violation.
    """
    ingredients = payload["ingredients"]
    _require_unique_alias_ownership(ingredients)
    _require_names_not_foreign_aliases(ingredients)
    _require_valid_unique_ids(ingredients)
    _require_valid_parent_links(ingredients)
    covered_mentions, total_mentions = _require_alias_coverage(
        ingredients, index
    )
    return CompileStatistics(
        ingredient_count=len(ingredients),
        alias_count=sum(len(entry["aliases"]) for entry in ingredients),
        parent_link_count=sum(
            1 for entry in ingredients if entry["parent_id"] is not None
        ),
        covered_mention_count=covered_mentions,
        total_mention_count=total_mentions,
        coverage_ratio=covered_mentions / total_mentions if total_mentions else 0.0,
    )


def _require_unique_alias_ownership(ingredients: list[dict]) -> None:
    """Gate 1: every alias string belongs to exactly one ingredient."""
    alias_owner_ids: dict[str, list[str]] = {}
    for ingredient in ingredients:
        for entry in ingredient["aliases"]:
            alias_owner_ids.setdefault(entry["alias"], []).append(
                ingredient["id"]
            )
    conflicts = {
        alias: owner_ids
        for alias, owner_ids in sorted(alias_owner_ids.items())
        if len(owner_ids) > 1
    }
    if conflicts:
        raise ValueError(
            f"aliases mapped to more than one ingredient: {conflicts}"
        )


def _require_names_not_foreign_aliases(ingredients: list[dict]) -> None:
    """Gate 2: no canonical name doubles as another ingredient's alias."""
    alias_owner_positions: dict[str, set[int]] = {}
    for position, ingredient in enumerate(ingredients):
        for entry in ingredient["aliases"]:
            alias_owner_positions.setdefault(entry["alias"], set()).add(position)
    for position, ingredient in enumerate(ingredients):
        foreign_positions = alias_owner_positions.get(
            ingredient["name"], set()
        ) - {position}
        if foreign_positions:
            owner_ids = sorted(
                ingredients[owner]["id"] for owner in foreign_positions
            )
            raise ValueError(
                f"ingredient name {ingredient['name']!r} is also an alias "
                f"of {owner_ids}"
            )


def _require_valid_unique_ids(ingredients: list[dict]) -> None:
    """Gate 3: ids match the slug format and never repeat."""
    seen_ids: set[str] = set()
    for ingredient in ingredients:
        ingredient_id = ingredient["id"]
        if not SLUG_PATTERN.match(ingredient_id):
            raise ValueError(
                f"ingredient id {ingredient_id!r} violates the slug format "
                "^[a-z0-9][a-z0-9_]*$"
            )
        if ingredient_id in seen_ids:
            raise ValueError(f"duplicate ingredient id {ingredient_id!r}")
        seen_ids.add(ingredient_id)


def _require_valid_parent_links(ingredients: list[dict]) -> None:
    """Gate 4: parents exist, no cycles, chain depth never exceeds two."""
    parent_id_by_id = {
        ingredient["id"]: ingredient["parent_id"] for ingredient in ingredients
    }
    for ingredient in ingredients:
        parent_id = ingredient["parent_id"]
        if parent_id is None:
            continue
        if parent_id not in parent_id_by_id:
            raise ValueError(
                f"ingredient {ingredient['id']!r} links to missing parent "
                f"{parent_id!r}"
            )
        if parent_id == ingredient["id"]:
            raise ValueError(
                f"ingredient {ingredient['id']!r} is its own parent"
            )
        if parent_id_by_id[parent_id] is not None:
            raise ValueError(
                f"ingredient {ingredient['id']!r} exceeds the parent chain "
                f"depth limit of {PARENT_CHAIN_DEPTH_LIMIT}: parent "
                f"{parent_id!r} has its own parent"
            )


def _require_alias_coverage(
    ingredients: list[dict], index: TrainIndex
) -> tuple[int, int]:
    """Gate 5: frequent strings are aliased and mention coverage holds."""
    covered_aliases = {
        entry["alias"]
        for ingredient in ingredients
        for entry in ingredient["aliases"]
    }
    uncovered_frequent = sorted(
        raw
        for raw, recipe_ids in index.string_to_recipe_ids.items()
        if len(recipe_ids) >= ALIAS_SCOPE_MINIMUM_FREQUENCY
        and raw not in covered_aliases
    )
    if uncovered_frequent:
        raise ValueError(
            "frequent train strings missing from every alias list: "
            f"{uncovered_frequent[:UNCOVERED_STRING_REPORT_LIMIT]}"
        )
    total_mentions = sum(
        len(recipe_ids) for recipe_ids in index.string_to_recipe_ids.values()
    )
    covered_mentions = sum(
        len(recipe_ids)
        for raw, recipe_ids in index.string_to_recipe_ids.items()
        if raw in covered_aliases
    )
    ratio = covered_mentions / total_mentions if total_mentions else 0.0
    if ratio < COVERAGE_MINIMUM_RATIO:
        raise ValueError(
            f"alias coverage {ratio:.4f} is below the "
            f"{COVERAGE_MINIMUM_RATIO} floor"
        )
    return covered_mentions, total_mentions


def main() -> None:
    """CLI: compile, validate, and write staged/ingredients.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lexicons-directory", type=Path, default=DEFAULT_LEXICONS_DIRECTORY
    )
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    lexicons = load_pipeline_lexicons(arguments.lexicons_directory)
    index = build_train_index(load_train_recipes(RAW_TRAIN_PATH))
    build = build_vocabulary_from_index(index, lexicons)
    decisions = load_merge_decisions(arguments.decisions)
    fingerprint = compute_build_fingerprint(
        RAW_TRAIN_PATH, arguments.lexicons_directory
    )
    payload = compile_ingredients_payload(build, decisions, index, fingerprint)
    statistics = validate_compiled_payload(payload, index)
    write_staged_json(payload, arguments.output)
    print(f"ingredients: {statistics.ingredient_count}")
    print(f"aliases: {statistics.alias_count}")
    print(f"parent links: {statistics.parent_link_count}")
    print(
        f"alias coverage: {statistics.covered_mention_count}/"
        f"{statistics.total_mention_count} mentions "
        f"({statistics.coverage_ratio:.4%})"
    )
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
