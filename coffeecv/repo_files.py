"""File helpers the dataset and segmenter tools share, one copy each (ticket ML-5 review, 2026-10-08).

    sha256_file   a file's sha256, streamed: the identity every photo list and label records
    read_csv      a CSV file's rows as dicts
    rel           a path as the records store it: relative to the repo root
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from coffeecv.config import REPO_ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(Path(path).read_text().splitlines()))


def rel(path: Path | str) -> str:
    """`path` relative to the repo root, with / separators. A relative path is taken as already relative to
    it; an absolute one outside the repo (a test's scratch directory) comes back whole."""
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    return path.as_posix()
