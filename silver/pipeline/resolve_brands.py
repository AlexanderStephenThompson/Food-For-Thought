"""Brand-to-generic resolution for cleaned ingredient strings.

Loads the curated brand lexicon (silver/lexicons/brand_patterns.json) into a frozen
BrandLexicon and resolves cleaned ingredient text against it. Two pattern
families exist:

- keep_as_ingredient: brands that ARE the ingredient (Marmite, Guinness,
  Old Bay, the liqueur identities) map to themselves-as-generic and are
  checked first so cuisine-discriminative brand signal survives merging.
- patterns: grocery brands whose product duplicates a generic ingredient's
  signal; the WHOLE matched string resolves to the mapped generic target.

Patterns are lowercase word-boundary regexes evaluated against CLEANED text
(lowercase, accents folded, trademark marks and apostrophes removed,
punctuation folded to spaces), e.g. 'hellmanns or best food real mayonnaise'.
Pattern order inside each family is semantic: the first match wins, so
product-specific patterns precede bare-brand fallbacks.

No file I/O happens at import time; call load_brand_lexicon explicitly.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

KEEP_AS_INGREDIENT_KEY = "keep_as_ingredient"
PATTERNS_KEY = "patterns"
REQUIRED_SECTION_KEYS = frozenset({KEEP_AS_INGREDIENT_KEY, PATTERNS_KEY})

PATTERN_FIELD = "pattern"
GENERIC_FIELD = "generic"
REQUIRED_ENTRY_FIELDS = frozenset({PATTERN_FIELD, GENERIC_FIELD})

# Cleaned ingredient text may only contain these characters; anything else
# means the caller skipped the normalize_text chain.
CLEANED_TEXT_PATTERN = re.compile(r"[a-z0-9% ]+")
UPPERCASE_LETTER_PATTERN = re.compile(r"[A-Z]")


@dataclass(frozen=True)
class BrandPattern:
    """One brand resolution rule: a compiled regex and its generic target.

    Attributes:
        regex_source: The lowercase word-boundary regex as written in the
            lexicon JSON, e.g. "\\bkraft\\b(?=.*parmesan)".
        generic_target: The generic ingredient string the whole matched
            text resolves to, e.g. "grated parmesan cheese".
        compiled_regex: regex_source compiled with re.compile.
    """

    regex_source: str
    generic_target: str
    compiled_regex: re.Pattern[str]


@dataclass(frozen=True)
class BrandLexicon:
    """Frozen, ordered brand lexicon loaded from brand_patterns.json.

    Attributes:
        keep_as_ingredient: Rules for brands that are themselves ingredients;
            always checked before patterns.
        patterns: Brand-to-generic rules in first-match-wins order
            (product-specific rules before bare-brand fallbacks).
    """

    keep_as_ingredient: tuple[BrandPattern, ...]
    patterns: tuple[BrandPattern, ...]


def load_brand_lexicon(path: str | Path) -> BrandLexicon:
    """Load and validate a brand lexicon JSON file.

    Args:
        path: Path to a JSON file with exactly two keys,
            "keep_as_ingredient" and "patterns", each a list of
            {"pattern": <lowercase regex>, "generic": <target>} objects.

    Returns:
        A frozen BrandLexicon with every pattern compiled, preserving the
        JSON list order (which is semantic: first match wins).

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the structure is wrong, a pattern fails to compile,
            a pattern contains uppercase letters, or a field is empty.
    """
    lexicon_path = Path(path)
    data = json.loads(lexicon_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"{lexicon_path}: top level must be a JSON object")
    if set(data) != REQUIRED_SECTION_KEYS:
        raise ValueError(
            f"{lexicon_path}: expected exactly keys {sorted(REQUIRED_SECTION_KEYS)},"
            f" got {sorted(data)}"
        )

    return BrandLexicon(
        keep_as_ingredient=_compile_section(data, KEEP_AS_INGREDIENT_KEY),
        patterns=_compile_section(data, PATTERNS_KEY),
    )


def resolve_brand_to_generic(cleaned_text: str, lexicon: BrandLexicon) -> str | None:
    """Resolve a cleaned ingredient string to its generic replacement.

    When any pattern matches, the WHOLE string resolves to that pattern's
    generic target — brand strings are product names, not modifiers, so
    'kraft grated parmesan cheese' resolves to the mapped
    'grated parmesan cheese' rather than to residue text.

    Args:
        cleaned_text: Ingredient text already passed through the cleaning
            chain (lowercase; no trademark marks, apostrophes, or other
            punctuation), e.g. 'hellmanns or best food real mayonnaise'.
        lexicon: Lexicon returned by load_brand_lexicon.

    Returns:
        The generic ingredient string for the first matching rule
        (keep_as_ingredient rules first, then patterns in lexicon order),
        or None when no brand pattern matches.

    Raises:
        TypeError: If cleaned_text is not a string.
        ValueError: If cleaned_text is empty or contains characters that the
            cleaning chain would have removed (uppercase, punctuation, marks).
    """
    _validate_cleaned_text(cleaned_text)

    for rule in lexicon.keep_as_ingredient + lexicon.patterns:
        if rule.compiled_regex.search(cleaned_text):
            return rule.generic_target
    return None


def _validate_cleaned_text(cleaned_text: str) -> None:
    """Reject text that has not been through the cleaning chain.

    Args:
        cleaned_text: Candidate ingredient text.

    Raises:
        TypeError: If cleaned_text is not a string.
        ValueError: If cleaned_text is blank or contains characters outside
            the cleaned alphabet (lowercase letters, digits, '%', spaces).
    """
    if not isinstance(cleaned_text, str):
        raise TypeError(
            f"cleaned_text must be str, got {type(cleaned_text).__name__}"
        )
    if not cleaned_text.strip():
        raise ValueError("cleaned_text is blank; clean and guard input upstream")
    if not CLEANED_TEXT_PATTERN.fullmatch(cleaned_text):
        raise ValueError(
            "cleaned_text contains characters the cleaning chain removes"
            f" (uppercase/punctuation/marks): {cleaned_text!r}"
        )


def _compile_section(data: dict, section_name: str) -> tuple[BrandPattern, ...]:
    """Compile one lexicon section into ordered BrandPattern rules.

    Args:
        data: Parsed top-level lexicon JSON object.
        section_name: "keep_as_ingredient" or "patterns".

    Returns:
        Tuple of compiled rules preserving list order.

    Raises:
        ValueError: If the section is not a list or any entry is invalid.
    """
    entries = data[section_name]
    if not isinstance(entries, list):
        raise ValueError(f"section {section_name!r} must be a JSON list")

    return tuple(
        _compile_entry(entry, section_name, position)
        for position, entry in enumerate(entries)
    )


def _compile_entry(entry: object, section_name: str, position: int) -> BrandPattern:
    """Validate and compile a single {"pattern", "generic"} lexicon entry.

    Args:
        entry: Raw JSON entry from the lexicon list.
        section_name: Section the entry belongs to, for error messages.
        position: Zero-based index in the section, for error messages.

    Returns:
        The compiled BrandPattern.

    Raises:
        ValueError: If fields are missing or empty, the pattern contains
            uppercase letters, or the regex does not compile.
    """
    location = f"{section_name}[{position}]"
    if not isinstance(entry, dict) or set(entry) != REQUIRED_ENTRY_FIELDS:
        raise ValueError(
            f"{location}: each entry needs exactly fields"
            f" {sorted(REQUIRED_ENTRY_FIELDS)}"
        )

    regex_source = entry[PATTERN_FIELD]
    generic_target = entry[GENERIC_FIELD]
    if not isinstance(regex_source, str) or not regex_source:
        raise ValueError(f"{location}: 'pattern' must be a non-empty string")
    if not isinstance(generic_target, str) or not generic_target:
        raise ValueError(f"{location}: 'generic' must be a non-empty string")
    if UPPERCASE_LETTER_PATTERN.search(regex_source):
        raise ValueError(
            f"{location}: pattern {regex_source!r} contains uppercase letters;"
            " cleaned text is lowercase so it could never match"
        )

    try:
        compiled_regex = re.compile(regex_source)
    except re.error as regex_error:
        raise ValueError(
            f"{location}: pattern {regex_source!r} does not compile: {regex_error}"
        ) from regex_error

    return BrandPattern(
        regex_source=regex_source,
        generic_target=generic_target,
        compiled_regex=compiled_regex,
    )
