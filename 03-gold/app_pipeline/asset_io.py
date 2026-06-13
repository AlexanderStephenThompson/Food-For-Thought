"""Serialization contract for app data assets.

App assets are network payloads the browser fetches, so they serialize
compactly (no indentation) — unlike pipeline artifacts, which stay
human-readable. Keys are sorted and the content is newline-terminated so
the idempotency check compares stable bytes.
"""

from __future__ import annotations

import json

COMPACT_SEPARATORS = (",", ":")


def serialize_asset_json(payload: dict) -> str:
    """Serialize a payload to the canonical compact asset format.

    Args:
        payload: JSON-serializable asset content.

    Returns:
        The full file content: compact separators, sorted keys,
        newline-terminated.

    Raises:
        TypeError: If the payload contains non-JSON-serializable values.
    """
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=COMPACT_SEPARATORS,
            sort_keys=True,
        )
        + "\n"
    )
