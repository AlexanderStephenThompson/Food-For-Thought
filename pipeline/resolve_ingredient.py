"""Runtime ingredient resolution: the single tiered fallback chain.

IngredientResolver is the SOLE lookup implementation mapping raw
ingredient strings onto staged vocabulary ids; staging and the future
model/CLI all reuse it. Resolution walks six tiers and returns at the
first hit:

1. exact_alias              raw string, case-sensitive.
2. cleaned_match            cleaned form, then its singularized key.
3. modifier_stripped_match  key of the modifier-stripped form.
4. brand_resolved_match     brand-to-generic on the cleaned AND the
                            stripped form, then key lookup.
5. token_drop_match         drop one (then two) non-final key tokens.
6. unresolved.

The token-drop tier never touches the final head token and never
collapses a key to the bare head alone, so 'fish sauce' can never
resolve to 'sauce'.

Indexes are built once at construction; cross-ingredient collisions
resolve deterministically to the ingredient with the higher
train_mention_count, ties to the lexicographically smaller id.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.build_vocabulary import PipelineLexicons, load_pipeline_lexicons
from pipeline.normalize_text import DegenerateKeyError, clean_ingredient_text
from pipeline.resolve_brands import resolve_brand_to_generic
from pipeline.singularize import make_lookup_key
from pipeline.strip_modifiers import strip_safe_modifiers

METHOD_EXACT_ALIAS = "exact_alias"
METHOD_CLEANED_MATCH = "cleaned_match"
METHOD_MODIFIER_STRIPPED_MATCH = "modifier_stripped_match"
METHOD_BRAND_RESOLVED_MATCH = "brand_resolved_match"
METHOD_TOKEN_DROP_MATCH = "token_drop_match"
METHOD_UNRESOLVED = "unresolved"

INGREDIENTS_FIELD = "ingredients"
INGREDIENT_ID_FIELD = "id"
INGREDIENT_NAME_FIELD = "name"
MENTION_COUNT_FIELD = "train_mention_count"
ALIASES_FIELD = "aliases"
ALIAS_TEXT_FIELD = "alias"
REQUIRED_INGREDIENT_FIELDS = (
    INGREDIENT_ID_FIELD,
    INGREDIENT_NAME_FIELD,
    MENTION_COUNT_FIELD,
    ALIASES_FIELD,
)

TOKEN_SEPARATOR = " "
SINGLE_DROP_COUNT = 1
DOUBLE_DROP_COUNT = 2
# Double drops only fire on keys this long, per the pinned contract.
DOUBLE_DROP_MINIMUM_TOKENS = 4
# A candidate must keep the head plus at least one qualifier token;
# collapsing to the bare head ('fish sauce' -> 'sauce') discards the
# ingredient identity the head alone cannot carry.
MINIMUM_REMAINING_TOKENS = 2


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of resolving one raw ingredient string.

    Attributes:
        ingredient_id: Resolved vocabulary id, or None when unresolved.
        method: Name of the tier that produced the hit; one of the
            METHOD_* constants in this module.
        dropped_tokens: Tokens removed by the token-drop tier, in their
            original order; empty for every other method.
    """

    ingredient_id: str | None
    method: str
    dropped_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class _VocabularyRecord:
    """One staged ingredient reduced to what index construction needs."""

    ingredient_id: str
    name: str
    train_mention_count: int
    alias_strings: tuple[str, ...]


@dataclass(frozen=True)
class _DropCandidate:
    """One token-drop hit awaiting deterministic ranking."""

    remaining_token_count: int
    ingredient_id: str
    dropped_tokens: tuple[str, ...]


class IngredientResolver:
    """Tiered lookup from raw ingredient strings to vocabulary ids.

    Construction builds three deterministic indexes over a staged
    vocabulary payload: raw alias -> id (case-sensitive), cleaned form
    -> id, and singularized lookup key -> id. Collisions in the cleaned
    and key indexes resolve to the ingredient with the higher
    train_mention_count, ties to the lexicographically smaller id.

    Examples:
        >>> resolver = IngredientResolver.from_paths(
        ...     Path("staged/ingredients.json"), Path("lexicons"))
        >>> resolver.resolve("chopped fresh cilantro").ingredient_id
        'cilantro'
    """

    def __init__(self, payload: dict, lexicons: PipelineLexicons) -> None:
        """Build the resolver indexes from a vocabulary payload.

        Args:
            payload: Parsed staged/ingredients.json document (pinned
                schema version 1).
            lexicons: Preloaded pipeline lexicons used for cleaning,
                singularization, modifier stripping, and brand patterns.

        Raises:
            ValueError: If the payload is structurally invalid or holds
                duplicate ingredient ids.
            TypeError: If the payload is not a dict.
        """
        records = _extract_records(payload)
        self._lexicons = lexicons
        self._mention_counts = {
            record.ingredient_id: record.train_mention_count for record in records
        }
        self._alias_to_id: dict[str, str] = {}
        self._cleaned_to_id: dict[str, str] = {}
        self._key_to_id: dict[str, str] = {}
        # Sorted iteration keeps construction order-independent even
        # though the tie-break itself is already a total order.
        for record in sorted(records, key=lambda entry: entry.ingredient_id):
            self._index_record(record)

    @classmethod
    def from_payload(
        cls, payload: dict, lexicons: PipelineLexicons
    ) -> "IngredientResolver":
        """Build a resolver from an already-parsed vocabulary payload.

        Args:
            payload: Parsed staged/ingredients.json document.
            lexicons: Preloaded pipeline lexicons.

        Returns:
            A ready-to-use IngredientResolver.
        """
        return cls(payload, lexicons)

    @classmethod
    def from_paths(
        cls, vocabulary_path: Path, lexicons_directory: Path
    ) -> "IngredientResolver":
        """Build a resolver by loading the payload and lexicons from disk.

        Args:
            vocabulary_path: Path to staged/ingredients.json.
            lexicons_directory: Directory holding the curated lexicons.

        Returns:
            A ready-to-use IngredientResolver.

        Raises:
            FileNotFoundError: If either path does not exist.
            json.JSONDecodeError: If the vocabulary file is not valid JSON.
        """
        payload = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))
        return cls(payload, load_pipeline_lexicons(Path(lexicons_directory)))

    def resolve(self, raw_text: str) -> ResolutionResult:
        """Resolve one raw ingredient string through the fallback chain.

        Args:
            raw_text: The ingredient string exactly as found in a recipe.

        Returns:
            A ResolutionResult with the matched id and tier method, or
            ingredient_id None and method 'unresolved' when no tier hits.

        Raises:
            TypeError: If raw_text is not a string.
            ValueError: If raw_text is empty or whitespace-only.
        """
        _validate_raw_text(raw_text)
        exact_id = self._alias_to_id.get(raw_text)
        if exact_id is not None:
            return ResolutionResult(exact_id, METHOD_EXACT_ALIAS)
        try:
            cleaned = clean_ingredient_text(raw_text)
        except DegenerateKeyError:
            return ResolutionResult(None, METHOD_UNRESOLVED)
        return self._resolve_cleaned(cleaned)

    def _resolve_cleaned(self, cleaned: str) -> ResolutionResult:
        """Run tiers 2-6 over an already-cleaned ingredient string."""
        cleaned_id = self._lookup_cleaned_form(cleaned)
        if cleaned_id is not None:
            return ResolutionResult(cleaned_id, METHOD_CLEANED_MATCH)
        stripped = strip_safe_modifiers(cleaned, self._lexicons.modifier_lexicon)
        stripped_key = self._lookup_key_of(stripped)
        stripped_id = self._key_to_id.get(stripped_key)
        if stripped_id is not None:
            return ResolutionResult(stripped_id, METHOD_MODIFIER_STRIPPED_MATCH)
        brand_id = self._lookup_via_brand(cleaned, stripped)
        if brand_id is not None:
            return ResolutionResult(brand_id, METHOD_BRAND_RESOLVED_MATCH)
        drop_result = self._lookup_via_token_drop(stripped_key)
        if drop_result is not None:
            return drop_result
        return ResolutionResult(None, METHOD_UNRESOLVED)

    def _lookup_cleaned_form(self, cleaned: str) -> str | None:
        """Tier 2: cleaned-form index, then its lookup key in the key index."""
        direct_id = self._cleaned_to_id.get(cleaned)
        if direct_id is not None:
            return direct_id
        return self._key_to_id.get(self._lookup_key_of(cleaned))

    def _lookup_via_brand(self, cleaned: str, stripped: str) -> str | None:
        """Tier 4: brand-to-generic on the cleaned, then the stripped form."""
        candidate_texts = [cleaned]
        if stripped != cleaned:
            candidate_texts.append(stripped)
        for candidate_text in candidate_texts:
            generic = _brand_generic_or_none(
                candidate_text, self._lexicons.brand_lexicon
            )
            if generic is None:
                continue
            generic_id = self._lookup_generic_target(generic)
            if generic_id is not None:
                return generic_id
        return None

    def _lookup_generic_target(self, generic: str) -> str | None:
        """Clean a brand lexicon generic target and look up its key."""
        try:
            generic_cleaned = clean_ingredient_text(generic)
        except DegenerateKeyError:
            return None
        return self._key_to_id.get(self._lookup_key_of(generic_cleaned))

    def _lookup_via_token_drop(self, stripped_key: str) -> ResolutionResult | None:
        """Tier 5: drop one, then two, non-final tokens from the key."""
        tokens = tuple(stripped_key.split(TOKEN_SEPARATOR))
        candidates = self._collect_drop_candidates(tokens, SINGLE_DROP_COUNT)
        if not candidates and len(tokens) >= DOUBLE_DROP_MINIMUM_TOKENS:
            candidates = self._collect_drop_candidates(tokens, DOUBLE_DROP_COUNT)
        if not candidates:
            return None
        best_candidate = min(candidates, key=self._rank_drop_candidate)
        return ResolutionResult(
            best_candidate.ingredient_id,
            METHOD_TOKEN_DROP_MATCH,
            best_candidate.dropped_tokens,
        )

    def _collect_drop_candidates(
        self, tokens: tuple[str, ...], drop_count: int
    ) -> list[_DropCandidate]:
        """Collect key-index hits for every non-final drop combination."""
        if len(tokens) - drop_count < MINIMUM_REMAINING_TOKENS:
            return []
        non_final_positions = range(len(tokens) - 1)
        candidates: list[_DropCandidate] = []
        for drop_positions in itertools.combinations(non_final_positions, drop_count):
            kept_tokens = [
                token
                for position, token in enumerate(tokens)
                if position not in drop_positions
            ]
            ingredient_id = self._key_to_id.get(TOKEN_SEPARATOR.join(kept_tokens))
            if ingredient_id is None:
                continue
            dropped_tokens = tuple(tokens[position] for position in drop_positions)
            candidates.append(
                _DropCandidate(len(kept_tokens), ingredient_id, dropped_tokens)
            )
        return candidates

    def _rank_drop_candidate(
        self, candidate: _DropCandidate
    ) -> tuple[int, int, str, tuple[str, ...]]:
        """Sort key: longest key, highest count, smallest id, then tokens."""
        return (
            -candidate.remaining_token_count,
            -self._mention_counts[candidate.ingredient_id],
            candidate.ingredient_id,
            candidate.dropped_tokens,
        )

    def _index_record(self, record: _VocabularyRecord) -> None:
        """Add one ingredient's surface forms to all three indexes."""
        for alias_text in record.alias_strings:
            self._assign(self._alias_to_id, alias_text, record.ingredient_id)
        for surface_form in (record.name, *record.alias_strings):
            try:
                cleaned = clean_ingredient_text(surface_form)
            except DegenerateKeyError:
                # A degenerate surface can still exact-match through the
                # raw alias index; it just has no cleaned or key form.
                continue
            self._assign(self._cleaned_to_id, cleaned, record.ingredient_id)
            self._assign(
                self._key_to_id, self._lookup_key_of(cleaned), record.ingredient_id
            )

    def _assign(self, index: dict[str, str], text: str, ingredient_id: str) -> None:
        """Write an index entry, resolving collisions deterministically."""
        incumbent_id = index.get(text)
        if incumbent_id is None or self._is_preferred(ingredient_id, incumbent_id):
            index[text] = ingredient_id

    def _is_preferred(self, candidate_id: str, incumbent_id: str) -> bool:
        """Higher train_mention_count wins; ties go to the smaller id."""
        candidate_rank = (-self._mention_counts[candidate_id], candidate_id)
        incumbent_rank = (-self._mention_counts[incumbent_id], incumbent_id)
        return candidate_rank < incumbent_rank

    def _lookup_key_of(self, cleaned_text: str) -> str:
        """Singularize cleaned text into its canonical lookup key."""
        return make_lookup_key(cleaned_text, self._lexicons.singularize_exceptions)


def _validate_raw_text(raw_text: str) -> None:
    """Reject non-string and blank resolution input, fail fast."""
    if not isinstance(raw_text, str):
        raise TypeError(f"raw_text must be str, got {type(raw_text).__name__}")
    if not raw_text.strip():
        raise ValueError("raw_text is empty or whitespace-only")


def _brand_generic_or_none(cleaned_text: str, brand_lexicon: object) -> str | None:
    """Run brand resolution, treating un-resolvable text as no match.

    resolve_brand_to_generic rejects text holding characters outside the
    cleaned alphabet (rare survivors like ';'); such text simply cannot
    name a brand, so the tier reports no match instead of failing.
    """
    try:
        return resolve_brand_to_generic(cleaned_text, brand_lexicon)
    except ValueError:
        return None


def _extract_records(payload: dict) -> tuple[_VocabularyRecord, ...]:
    """Validate the payload shell and reduce it to index-ready records."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"vocabulary payload must be dict, got {type(payload).__name__}"
        )
    ingredients = payload.get(INGREDIENTS_FIELD)
    if not isinstance(ingredients, list):
        raise ValueError(
            f"vocabulary payload needs a list under {INGREDIENTS_FIELD!r}"
        )
    records = tuple(
        _to_record(entry, position) for position, entry in enumerate(ingredients)
    )
    unique_ids = {record.ingredient_id for record in records}
    if len(unique_ids) != len(records):
        raise ValueError("vocabulary payload holds duplicate ingredient ids")
    return records


def _to_record(entry: object, position: int) -> _VocabularyRecord:
    """Validate one ingredient entry and extract its index fields."""
    location = f"{INGREDIENTS_FIELD}[{position}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{location} must be a JSON object")
    missing_fields = [
        field_name
        for field_name in REQUIRED_INGREDIENT_FIELDS
        if field_name not in entry
    ]
    if missing_fields:
        raise ValueError(f"{location} missing fields: {', '.join(missing_fields)}")
    alias_entries = entry[ALIASES_FIELD]
    if not isinstance(alias_entries, list) or not all(
        isinstance(alias_entry, dict) and ALIAS_TEXT_FIELD in alias_entry
        for alias_entry in alias_entries
    ):
        raise ValueError(
            f"{location} aliases must be objects with an {ALIAS_TEXT_FIELD!r} field"
        )
    return _VocabularyRecord(
        ingredient_id=entry[INGREDIENT_ID_FIELD],
        name=entry[INGREDIENT_NAME_FIELD],
        train_mention_count=entry[MENTION_COUNT_FIELD],
        alias_strings=tuple(
            alias_entry[ALIAS_TEXT_FIELD] for alias_entry in alias_entries
        ),
    )
