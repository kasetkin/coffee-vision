"""Re-key the records that pin dataset photos by sha256 after the metadata strip (ticket ML-3 P3, R8).

    python -m coffeecv.rekey_photos            # rewrite every record below; prints per-file counts
    python -m coffeecv.rekey_photos --check    # change nothing; exit 1 if any record still pins an old hash

The strip changes every photo's bytes, never its pixels; labels/ml3/strip_manifest.csv maps each photo's
sha256 before to after. Every pinned photo hash is read from its own field and must be in the manifest
(as a before hash, or already an after hash), or nothing is written: a hash the manifest does not know is
a photo the strip never saw. Values are replaced in the text, so each file keeps its layout byte for byte
apart from the hashes.

labels/ml2/photo_lists.yaml is not here: its append-only check allows only `seg_lists rekey`, which uses
`load_map` and `rekey_values` below. data/seg_masks/{pretrained,ft_s*}/index.csv are not here either:
they are seg_predict's outputs, rewritten by its re-run on the VM (plan §5).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml

from coffeecv.config import REPO_ROOT

MANIFEST = REPO_ROOT / "labels" / "ml3" / "strip_manifest.csv"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _csv_column(column: str):
    def read(path: Path) -> list[str]:
        rows = csv.DictReader(l for l in path.read_text().splitlines() if not l.startswith("#"))
        return [r[column] for r in rows if r.get(column)]
    return read


def _jsonl_key(key: str):
    def read(path: Path) -> list[str]:
        return [json.loads(l)[key] for l in path.read_text().splitlines() if l.strip()]
    return read


def _fixtures(path: Path) -> list[str]:
    return [l.split()[1] for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def _base_points(path: Path) -> list[str]:
    return [item["sha256"] for item in yaml.safe_load(path.read_text())["items"]]


# (record, reader of its photo hashes). Git-tracked first, then the DVC-tracked ones (`dvc add` them after).
TARGETS = [
    ("webapp/deploy/fixtures.txt", _fixtures),
    ("labels/ml2/verdicts.jsonl", _jsonl_key("photo_sha256")),
    ("labels/ml2/defects/dev.csv", _csv_column("photo_sha256")),
    ("labels/ml2/defects/final.csv", _csv_column("photo_sha256")),
    ("labels/ml2/defects/pilot.csv", _csv_column("photo_sha256")),
    ("labels/ml2/base_points.yaml", _base_points),
    ("data/seg_labels/labels.csv", _csv_column("photo_sha256")),
    ("data/seg_labels/r1/index.csv", _csv_column("photo_sha256")),
    ("data/seg_labels/r2/index.csv", _csv_column("photo_sha256")),
    ("data/seg_labels/r3/index.csv", _csv_column("photo_sha256")),
    ("data/seg_masks/base_points/index.csv", _csv_column("photo_sha256")),
]


def load_map(manifest: Path = MANIFEST) -> tuple[dict[str, str], set[str]]:
    """(before -> after for every photo the strip changed, every after hash)."""
    rows = list(csv.DictReader(l for l in manifest.read_text().splitlines() if not l.startswith("#")))
    before_after = {r["sha256_before"]: r["sha256_after"] for r in rows}
    return {b: a for b, a in before_after.items() if b != a}, set(before_after.values())


def rekey_values(values: list[str], rekey: dict[str, str], current: set[str]) -> tuple[dict[str, str], list[str]]:
    """(old -> new for the values to replace, values the manifest does not know)."""
    todo, unknown = {}, []
    for v in values:
        if v in rekey:
            todo[v] = rekey[v]
        elif v not in current:
            unknown.append(v)
    return todo, unknown


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="change nothing; fail if an old hash is pinned")
    args = ap.parse_args(argv)
    rekey, current = load_map()
    plans, problems = [], []
    for rel, read in TARGETS:
        path = REPO_ROOT / rel
        values = read(path)
        bad = [v for v in values if not _HEX64.match(v)]
        todo, unknown = rekey_values(values, rekey, current)
        problems += [f"{rel}: {v!r} is not a sha256" for v in bad[:3]]
        problems += [f"{rel}: {v} is in no row of the strip manifest" for v in sorted(set(unknown))[:3]]
        plans.append((rel, path, values, todo))
    if problems:
        print("refusing to re-key:\n  " + "\n  ".join(problems))
        return 1
    stale = 0
    for rel, path, values, todo in plans:
        n = sum(1 for v in values if v in todo)
        stale += n
        print(f"  {rel:<40} {len(values):>5} pinned, {n:>5} {'old' if args.check else 're-keyed'}")
        if args.check or not todo:
            continue
        text = path.read_text()
        for old, new in todo.items():
            text = text.replace(old, new)
        path.write_text(text)
    if args.check and stale:
        print(f"{stale} pinned photo hash(es) predate the strip")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
