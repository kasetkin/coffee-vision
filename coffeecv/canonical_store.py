"""Reading a record's photo from the canonical store (ticket ML-19 D18, D33; plan finding 2).

Records (labels.csv, the segmenter dataset, the OOD manifests) key a photo by its original's path under dataset/ and
the original's sha256. Since P6 the tools that feed a model read the canonical photo mapped from that path
(coffeecv.canonical.canonical_path), after checking the original against its sha256. A missing canonical photo is an
error, never a fallback to the original: run its canonicalize@<folder> stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from coffeecv import canonical as cn
from coffeecv.repo_files import sha256_file

REPO = cn.REPO


def canonical_for(path: str | Path, sha256: str | None = None) -> Path:
    """The canonical photo of a record's original (repo-relative or absolute), the original checked first. The
    store is found from the original: <root>/dataset/<rel> -> <root>/data/canonical/<rel>.jxl, <root> the repo's."""
    original = Path(path) if Path(path).is_absolute() else REPO / path
    if sha256 is not None and sha256_file(original) != sha256:
        raise ValueError(f"{path}: sha256 differs from the record")
    dataset = next((a for a in original.parents if a.name == "dataset"), None)
    if dataset is None:
        raise ValueError(f"{path} is not under a dataset/ folder: it has no canonical photo")
    store = dataset.parent / "data" / "canonical"
    canonical = cn.canonical_path(original, dataset, store)
    if not canonical.is_file():
        rel = canonical.relative_to(store)
        raise FileNotFoundError(f"{path}: no canonical photo at {canonical}; run `dvc repro canonicalize@{rel.parts[0]}`")
    return canonical


def load_for_record(path: str | Path, sha256: str | None = None) -> np.ndarray:
    """A record's photo as the models read it: its canonical photo's 8-bit sRGB."""
    return cn.load_canonical(canonical_for(path, sha256).read_bytes())
