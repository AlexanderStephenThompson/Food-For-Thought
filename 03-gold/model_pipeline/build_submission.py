"""Build the Kaggle submission from the blends artifact.

The submission is the blend's argmax per test recipe, rendered in the
sample_submission.csv format (id,cuisine header, no quoting).
"""

from __future__ import annotations

from collections.abc import Sequence

SUBMISSION_HEADER = "id,cuisine"
SUBMISSION_FILENAME = "submission.csv"
LINE_SEPARATOR = "\n"


def build_submission_rows(blends_payload: dict) -> list[tuple[int, str]]:
    """Extract (recipe_id, top_cuisine) pairs in blends row order.

    Args:
        blends_payload: The blends artifact from build_blends_payload.

    Returns:
        One (recipe_id, cuisine) pair per test recipe, ascending by id.
    """
    return [
        (row["recipe_id"], row["top_cuisine"]) for row in blends_payload["rows"]
    ]


def render_submission_csv(rows: Sequence[tuple[int, str]]) -> str:
    """Render submission rows as the Kaggle CSV text.

    Args:
        rows: (recipe_id, cuisine) pairs from build_submission_rows.

    Returns:
        The full file content: header line, one row per recipe,
        newline-terminated.
    """
    lines = [SUBMISSION_HEADER]
    lines.extend(f"{recipe_id},{cuisine}" for recipe_id, cuisine in rows)
    return LINE_SEPARATOR.join(lines) + LINE_SEPARATOR
