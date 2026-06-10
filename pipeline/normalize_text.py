"""Deterministic cleaning of raw ingredient strings into stable key text.

Implements the ordered chain validated by the brands-noise and morphology
analyses of raw/kaggle/train.json: mojibake repair, trademark-mark
stripping, casefolding, quote normalization, quantity stripping, accent
folding, ampersand mapping, apostrophe deletion, punctuation folding,
whitespace collapse, and a degenerate-key guard.

Two ordering rules are load-bearing:
  * Trademark marks are stripped BEFORE any unicode normalization, and
    NFKC is never used anywhere -- NFKC turns U+2122 into the glued
    letters 'TM' ('Old El Paso(tm)' would corrupt to 'old el pasotm').
  * Mojibake repair runs first, because the one known mojibake string
    encodes its curly apostrophe through a U+2122 byte that trademark
    stripping would otherwise destroy.
"""

import re
import unicodedata

# --- mojibake repair (UTF-8 text mis-decoded as cp1252) ---
# 'â€' covers the U+20AC-class sequences ('â€™' -> '’'); 'Ã' covers the
# 'Ã©'-class sequences ('Ã©' -> 'é'). Train.json holds exactly one such
# string: "hellmannâ€™ or best food canola cholesterol free mayonnais".
MOJIBAKE_MARKERS: tuple[str, ...] = ("â€", "Ã")
MOJIBAKE_INTERMEDIATE_ENCODING = "cp1252"
MOJIBAKE_TARGET_ENCODING = "utf-8"

# --- trademark marks (® U+00AE, ™ U+2122, © U+00A9) ---
TRADEMARK_MARKS_PATTERN = re.compile(r"[®™©]")

# --- curly quotes (’ U+2019, ‘ U+2018) to straight apostrophe ---
CURLY_QUOTES_TO_STRAIGHT = str.maketrans({"’": "'", "‘": "'"})

# --- leaked recipe-line quantities ---
# Parenthetical ounce sizes, including blank placeholders: '(14.5 oz.)',
# '(    oz.)'. 'ounc' is a Porter-stem artifact present in train.json
# ('8 ounc ziti pasta, cook and drain').
PARENTHETICAL_OUNCE_PATTERN = re.compile(r"\(\s*[\d./\s]*oz\.?\s*\)")
LEADING_QUANTITY_PATTERN = re.compile(
    r"^\s*\d[\d\s./-]*(?:to\s+\d[\d\s./-]*)?"
    r"(?:pounds|pound|lbs|lb|ounces|ounce|ounc|oz|inches|inch)\b\.?\s*"
)

# --- ampersand: always map, never delete ('half & half' must survive) ---
AMPERSAND = "&"
AMPERSAND_REPLACEMENT = " and "

APOSTROPHE = "'"

# Every non-alphanumeric character observed in train.json that should
# fold to a space; '%' is deliberately absent ('1% low-fat milk').
PUNCTUATION_TO_FOLD = "-,.()!/"
PUNCTUATION_TO_SPACE_TRANSLATION = str.maketrans(
    {character: " " for character in PUNCTUATION_TO_FOLD}
)


class DegenerateKeyError(ValueError):
    """Raised when cleaning leaves no token with an alphabetic character.

    Attributes:
        raw_text: The original raw ingredient string that degenerated.
    """

    def __init__(self, raw_text: str) -> None:
        super().__init__(
            f"cleaning left no alphabetic token in raw string {raw_text!r}"
        )
        self.raw_text = raw_text


def repair_mojibake(text: str) -> str:
    """Repair UTF-8 text that was mis-decoded as cp1252.

    Only attempts the round trip when a known mojibake marker sequence
    is present, and falls back to the original text whenever the round
    trip is not byte-clean.

    Args:
        text: A raw ingredient string.

    Returns:
        The repaired string, or ``text`` unchanged when no marker is
        present or the cp1252 -> UTF-8 round trip fails.

    Examples:
        >>> repair_mojibake("hellmannâ€™ or best food")
        'hellmann’ or best food'
    """
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    try:
        encoded_bytes = text.encode(MOJIBAKE_INTERMEDIATE_ENCODING, errors="strict")
        return encoded_bytes.decode(MOJIBAKE_TARGET_ENCODING, errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def strip_trademark_marks(text: str) -> str:
    """Remove ®, ™, and © characters.

    Must run before any unicode compatibility normalization; NFKC would
    instead glue '™' onto the preceding word as literal 'TM'.

    Args:
        text: A raw or partially cleaned ingredient string.

    Returns:
        The string with all trademark marks deleted.
    """
    return TRADEMARK_MARKS_PATTERN.sub("", text)


def normalize_quotes(text: str) -> str:
    """Map curly quotes ’ and ‘ to the straight apostrophe.

    Args:
        text: A casefolded ingredient string.

    Returns:
        The string with curly quotes replaced by ``'``.
    """
    return text.translate(CURLY_QUOTES_TO_STRAIGHT)


def strip_quantities(text: str) -> str:
    """Remove leaked recipe-line quantity expressions.

    Drops parenthetical ounce sizes anywhere ('(14.5 oz.)', '(    oz.)')
    and a single leading digit/fraction quantity with a lb/oz/inch unit
    ('2 1/2 to 3 lb. '). Interior quantities ('pork chops, 1 inch thick')
    and '%' tokens ('1% low-fat milk') are preserved.

    Args:
        text: A casefolded ingredient string.

    Returns:
        The string with quantity expressions removed.
    """
    without_parenthetical_ounces = PARENTHETICAL_OUNCE_PATTERN.sub(" ", text)
    return LEADING_QUANTITY_PATTERN.sub("", without_parenthetical_ounces, count=1)


def fold_accents(text: str) -> str:
    """Fold accented characters to their ASCII base letters.

    Uses NFKD decomposition then drops combining marks; NFKC is never
    used (see module docstring).

    Args:
        text: A casefolded ingredient string.

    Returns:
        The string with accents removed ('purée' -> 'puree').
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )


def map_ampersand_to_and(text: str) -> str:
    """Replace '&' with ' and ' so conjunction keys stay joinable.

    Deleting instead of mapping would break 'half & half' into
    'half half', splitting it from 'half and half'.

    Args:
        text: A casefolded ingredient string.

    Returns:
        The string with every ampersand replaced by ' and '.
    """
    return text.replace(AMPERSAND, AMPERSAND_REPLACEMENT)


def delete_apostrophes(text: str) -> str:
    """Delete apostrophes outright, never replacing them with spaces.

    Deletion makes "hellmann's" -> 'hellmanns' and keeps
    "piment d'espelette" joined as 'piment despelette'.

    Args:
        text: A quote-normalized ingredient string.

    Returns:
        The string with all straight apostrophes removed.
    """
    return text.replace(APOSTROPHE, "")


def punctuation_to_space(text: str) -> str:
    """Fold hyphens, commas, periods, parentheses, '!' and '/' to spaces.

    Args:
        text: A casefolded ingredient string.

    Returns:
        The string with each folded punctuation character replaced by a
        single space ('parmigiano-reggiano' -> 'parmigiano reggiano').
    """
    return text.translate(PUNCTUATION_TO_SPACE_TRANSLATION)


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and trim the ends.

    Args:
        text: An ingredient string in any cleaning stage.

    Returns:
        The string with normalized single-space separation.
    """
    return " ".join(text.split())


def clean_ingredient_text(raw: str) -> str:
    """Clean one raw ingredient string into its deterministic key text.

    Applies, in order: mojibake repair, trademark-mark stripping,
    casefold, quote normalization, quantity stripping, accent folding,
    ampersand mapping, apostrophe deletion, punctuation folding, and
    whitespace collapse. The result is guaranteed to contain at least
    one token with an alphabetic character.

    Args:
        raw: The raw ingredient string exactly as found in the source
            JSON.

    Returns:
        The cleaned, lowercase, single-spaced key text.

    Raises:
        TypeError: If ``raw`` is not a string.
        DegenerateKeyError: If cleaning leaves no alphabetic token
            (for example a pure-numeric string like '14.5').

    Examples:
        >>> clean_ingredient_text("Old El Paso™ taco seasoning")
        'old el paso taco seasoning'
        >>> clean_ingredient_text("(14.5 oz.) diced tomatoes")
        'diced tomatoes'
    """
    if not isinstance(raw, str):
        raise TypeError(f"raw ingredient must be str, got {type(raw).__name__}")
    text = repair_mojibake(raw)
    text = strip_trademark_marks(text)
    text = text.casefold()
    text = normalize_quotes(text)
    text = strip_quantities(text)
    text = fold_accents(text)
    text = map_ampersand_to_and(text)
    text = delete_apostrophes(text)
    text = punctuation_to_space(text)
    text = collapse_whitespace(text)
    if not _has_alphabetic_character(text):
        raise DegenerateKeyError(raw)
    return text


def _has_alphabetic_character(text: str) -> bool:
    """Report whether any character in ``text`` is alphabetic.

    Args:
        text: A fully cleaned ingredient string.

    Returns:
        True when at least one character is alphabetic, which implies
        at least one token carries an alphabetic character.
    """
    return any(character.isalpha() for character in text)
