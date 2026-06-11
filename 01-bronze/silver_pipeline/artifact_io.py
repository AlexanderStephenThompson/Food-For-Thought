"""Shared I/O contracts for pipeline artifacts: serialization, atomic writes, fingerprints.

Every artifact file (silver payloads and reports) embeds a build block
identifying exactly which bronze data and lexicon state produced it, and is
written atomically with deterministic serialization so a rebuild is
byte-identical. This module is the single owner of both halves of that
guarantee: serialize_artifact_json defines the canonical byte format, and
write_text_atomically defines the only write path.
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
ARTIFACT_TEXT_ENCODING = "utf-8"
FILE_HASH_CHUNK_SIZE_BYTES = 65536
TEMPORARY_FILE_SUFFIX = ".tmp"


def sha256_of_file(path: Path) -> str:
    """Return the hex sha256 digest of a file's bytes.

    Args:
        path: File to hash; read in fixed-size chunks.

    Returns:
        64-character lowercase hex digest.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_HASH_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_build_fingerprint(
    train_path: Path, lexicons_directory: Path
) -> dict:
    """Fingerprint the inputs that determine every pipeline artifact.

    Args:
        train_path: Path to the bronze train JSON.
        lexicons_directory: Directory of curated lexicon files
            (every *.json and *.jsonl file participates, sorted by name).

    Returns:
        Build block with the train file hash, a combined hash over every
        lexicon file, and the pipeline's random seed.

    Raises:
        FileNotFoundError: If the train file does not exist.
        OSError: If any input file cannot be read.
    """
    lexicon_digest = hashlib.sha256()
    for lexicon_path in sorted(lexicons_directory.glob("*.json*")):
        lexicon_digest.update(lexicon_path.name.encode(ARTIFACT_TEXT_ENCODING))
        lexicon_digest.update(bytes.fromhex(sha256_of_file(lexicon_path)))
    return {
        "train_sha256": sha256_of_file(train_path),
        "lexicon_fingerprint": lexicon_digest.hexdigest(),
        "random_seed": BUILD_RANDOM_SEED,
    }


def serialize_artifact_json(payload: dict) -> str:
    """Serialize a payload to the canonical artifact byte format.

    This is the single definition of how artifact JSON looks on disk
    (sorted keys, 2-space indent, non-ASCII preserved, trailing newline).
    The idempotency check compares rebuilt payloads against disk through
    this exact serialization.

    Args:
        payload: JSON-serializable artifact content.

    Returns:
        The full file content, newline-terminated.

    Raises:
        TypeError: If the payload contains non-JSON-serializable values.
    """
    return (
        json.dumps(
            payload, ensure_ascii=False, indent=ARTIFACT_JSON_INDENT, sort_keys=True
        )
        + "\n"
    )


def write_text_atomically(content: str, path: Path) -> None:
    """Write text via a sibling temporary file and an atomic rename.

    A reader never observes a partially written file: content lands in a
    temporary sibling first and os.replace swaps it in atomically. The
    temporary file is removed if the write fails partway.

    Args:
        content: Exact file content to write.
        path: Destination path; parent directory must exist.

    Raises:
        OSError: If the temporary file cannot be created, written, or
            renamed onto the destination.
    """
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=path.name, suffix=TEMPORARY_FILE_SUFFIX
    )
    try:
        with os.fdopen(descriptor, "w", encoding=ARTIFACT_TEXT_ENCODING) as handle:
            handle.write(content)
        os.replace(temporary_path, path)
    except OSError:
        Path(temporary_path).unlink(missing_ok=True)
        raise


def write_artifact_json(payload: dict, path: Path) -> None:
    """Atomically write a pipeline artifact with deterministic serialization.

    Args:
        payload: JSON-serializable artifact content.
        path: Destination path; parent directory must exist.

    Raises:
        TypeError: If the payload contains non-JSON-serializable values.
        OSError: If the atomic write fails.
    """
    write_text_atomically(serialize_artifact_json(payload), path)


def find_artifact_mismatches(expected_content_by_path: dict[Path, str]) -> list[str]:
    """Compare expected artifact content against the files on disk.

    Both tier builders use this for their idempotency checks: a rebuild is
    idempotent exactly when every expected serialization matches its file
    byte-for-byte.

    Args:
        expected_content_by_path: Destination path -> exact expected file
            content (the canonical serialization, newline-terminated).

    Returns:
        Names of files that are missing (suffixed " (missing)") or whose
        bytes differ from the expected content; empty when everything
        matches.
    """
    mismatches = []
    for path, expected_content in expected_content_by_path.items():
        if not path.is_file():
            mismatches.append(f"{path.name} (missing)")
            continue
        if path.read_text(encoding=ARTIFACT_TEXT_ENCODING) != expected_content:
            mismatches.append(path.name)
    return mismatches
