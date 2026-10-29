"""File helpers the dataset and segmenter tools share, one copy each (ticket ML-5 review, 2026-10-08).

    sha256_file   a file's sha256, streamed: the identity every photo list and label records
    read_csv      a CSV file's rows as dicts
    rel           a path as the records store it: relative to the repo root
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import BinaryIO, Callable

from coffeecv.config import REPO_ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(path: Path) -> str:
    """sha256 of a file, or of a directory's files (relative path and content), __pycache__ left out: what a
    resume fingerprint holds of the code and data a run reads (ticket ML-13 P2, P3)."""
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for f in sorted(q for q in path.rglob("*") if q.is_file() and "__pycache__" not in q.parts):
        h.update(f"{f.relative_to(path).as_posix()} {sha256_file(f)}\n".encode())
    return h.hexdigest()


def atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """`write(f)` into a temporary file beside `path`, flushed to disk, renamed over `path`, the rename flushed
    too: a reboot leaves the old file or the new one, never part of one (the resume state, ticket ML-13 P2, P3)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def remove_tree(path: Path) -> None:
    """Delete `path`, and its parent once empty."""
    shutil.rmtree(path, ignore_errors=True)
    if path.parent.is_dir() and not any(path.parent.iterdir()):
        path.parent.rmdir()


def differing_keys(a: dict, b: dict) -> list[str]:
    """The keys whose values differ between two fingerprints, one missing from either included."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(Path(path).read_text().splitlines()))


def rel(path: Path | str) -> str:
    """`path` relative to the repo root, with / separators. A relative path is taken as already relative to
    it; an absolute one outside the repo (a test's scratch directory) comes back whole."""
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    return path.as_posix()
