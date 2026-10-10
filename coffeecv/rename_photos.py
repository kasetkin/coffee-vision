"""Point the live records at a dataset photo's new path after the owner renames it (2026-10-10).

    python -m coffeecv.rename_photos            # rewrite every record below and rename its mask files
    python -m coffeecv.rename_photos --check    # change nothing; exit 1 if any record still names an old path

labels/photo_renames.csv maps each old path to the new one and the photo's sha256, which must be the file at
the new path: a rename moves a photo, never changes it. Only OOD negatives are handled (their batch is the
folder under dataset/ood_negatives/). For each photo the records carry four strings derived from its path,
and each is replaced on token boundaries: the path; the ML-5 id (`seg_dataset.photo_id`); the ML-2 label id
(`seg_base_masks.item_id`), which also names its mask file; and a `batch:` value in YAML. Every other byte of
each file stays as it was, and nothing is written unless every replacement applies.

The run records are not here: data/seg_masks*, data/seg_cache_ml5, outputs/, analysis/, models/*.json and
the verdict logs keep the paths their runs read. labels/ml2/photo_lists.yaml is frozen (ML-5 D3); a rename
is, with `seg_lists rekey`, one of the two ways its entries change. The DVC-tracked records are `dvc add`ed
after, and the stages that read them recorded with `dvc commit`, as ML-3 P3 did.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from coffeecv.config import REPO_ROOT
from coffeecv.repo_files import sha256_file
from coffeecv.seg_base_masks import item_id
from coffeecv.seg_dataset import photo_id

RENAMES = REPO_ROOT / "labels" / "photo_renames.csv"
NEGATIVES = Path("dataset") / "ood_negatives"

# Records naming photos by path, git-tracked first, then the DVC-tracked ones (`dvc add` them after).
RECORDS = [
    "labels/ml5/seg_dataset.yaml",
    "labels/ml2/photo_lists.yaml",
    "data/ml5_labels/pass1/labels.csv",
    "data/ml5_labels/pass2/labels.csv",
    "data/seg_labels/labels.csv",
]
# Folders whose files are named by a label id: <id>.png.
MASK_DIRS = ["data/ml5_labels/pass1/neg", "data/ml5_labels/pass2/neg", "data/seg_labels/neg"]


def load_renames(path: Path = RENAMES) -> list[dict]:
    return list(csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith("#")))


def batch_of(path: str) -> str:
    p = Path(path)
    if p.parts[:2] != NEGATIVES.parts or len(p.parts) < 4:
        raise ValueError(f"{path} is not an OOD negative under {NEGATIVES}/<batch>/")
    return p.parts[2]


def replacements(rows: list[dict]) -> tuple[list[tuple[re.Pattern, str]], dict[str, str]]:
    """([(pattern of an old string, its new string)], {old mask stem: new}) for `rows`. The stems are also
    every old id, so a record still holding one as a substring was missed."""
    subs, stems = [], {}
    for r in rows:
        old, new = r["old_path"], r["new_path"]
        old_batch, new_batch = batch_of(old), batch_of(new)
        old_item = item_id({"batch": old_batch, "path": old})
        new_item = item_id({"batch": new_batch, "path": new})
        stems[photo_id(old)] = photo_id(new)
        stems[old_item] = new_item
        for a, b in ((old, new), (photo_id(old), photo_id(new)), (old_item, new_item)):
            subs.append((re.compile(r"(?<![\w.-])" + re.escape(a) + r"(?![\w-])"), b))
        if old_batch != new_batch:
            subs.append((re.compile(r"^(\s*(?:- )?batch: )" + re.escape(old_batch) + r"(?=\r?$)", re.M),
                         r"\g<1>" + new_batch))
    return list(dict.fromkeys(subs)), stems


def rewrite(text: str, subs: list[tuple[re.Pattern, str]]) -> tuple[str, int]:
    n = 0
    for rx, new in subs:
        text, k = rx.subn(new, text)
        n += k
    return text, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="change nothing; fail if a record names an old path")
    args = ap.parse_args(argv)
    rows = load_renames()
    problems = [f"{r['new_path']}: sha256 is not the manifest's {r['sha256'][:12]}" for r in rows
                if not (REPO_ROOT / r["new_path"]).is_file() or sha256_file(REPO_ROOT / r["new_path"]) != r["sha256"]]
    if problems:
        print("refusing to rename:\n  " + "\n  ".join(problems))
        return 1
    subs, stems = replacements(rows)
    olds = [r["old_path"] for r in rows] + list(stems)
    stale, todo = 0, []
    for rel in RECORDS:
        path = REPO_ROOT / rel
        text, n = rewrite(path.read_bytes().decode(), subs)
        stale += n
        print(f"  {rel:<40} {n:>4} {'old' if args.check else 'renamed'}")
        problems += [f"{rel}: {a} survives the rewrite" for a in olds if a in text]
        todo += [(path, text)] if n else []
    if problems and not args.check:
        print("refusing to rename:\n  " + "\n  ".join(problems))
        return 1
    for path, text in todo if not args.check else ():
        path.write_bytes(text.encode())
    for rel in MASK_DIRS:
        moves = [(REPO_ROOT / rel / f"{a}.png", REPO_ROOT / rel / f"{b}.png") for a, b in stems.items()
                 if (REPO_ROOT / rel / f"{a}.png").exists()]
        stale += len(moves)
        print(f"  {rel:<40} {len(moves):>4} {'old' if args.check else 'renamed'} mask files")
        for a, b in moves if not args.check else ():
            if b.exists():
                raise FileExistsError(b)
            a.rename(b)
    if args.check and stale:
        print(f"{stale} record value(s) or mask file(s) still use an old path")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
