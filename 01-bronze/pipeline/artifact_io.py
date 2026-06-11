"""Shared I/O contracts for pipeline artifacts: atomic writes and fingerprints.

Every artifact file (silver payloads and reports) embeds a build block
identifying exactly which bronze data and lexicon state produced it, and is
written atomically with deterministic serialization so a rebuild is
byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
BUILD_RANDOM_SEED = 42
ARTIFACT_JSON_INDENT = 2


def sha256_of_file(path: Path) -> str:
    """Return the hex sha256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_build_fingerprint(
    train_path: Path, lexicons_directory: Path
) -> dict:
    """Fingerprint the inputs that determine every pipeline artifact.

    Args:
        train_path: Path to the bronze train JSON.
        lexicons_directory: Directory of curated lexicon files.

    Returns:
        Build block with the train file hash, a combined hash over every
        lexicon file (sorted by name), and the pipeline's random seed.
    """
    lexicon_digest = hashlib.sha256()
    for lexicon_path in sorted(lexicons_directory.glob("*.json*")):
        lexicon_digest.update(lexicon_path.name.encode("utf-8"))
        lexicon_digest.update(bytes.fromhex(sha256_of_file(lexicon_path)))
    return {
        "train_sha256": sha256_of_file(train_path),
        "lexicon_fingerprint": lexicon_digest.hexdigest(),
        "random_seed": BUILD_RANDOM_SEED,
    }


def write_artifact_json(payload: dict, path: Path) -> None:
    """Atomically write a pipeline artifact with deterministic serialization.

    Args:
        payload: JSON-serializable artifact content.
        path: Destination path; parent directory must exist.
    """
    content = json.dumps(
        payload, ensure_ascii=False, indent=ARTIFACT_JSON_INDENT, sort_keys=True
    )
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=path.name, suffix=".tmp"
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content + "\n")
    os.replace(temporary_path, path)
