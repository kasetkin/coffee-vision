"""Rig x class photo coverage: print the table, and assert what should stay true.

Producing this table by hand meant cross-referencing eight session directories,
and it needs redoing every time a session or a class is added -- which is
exactly when a quiet inconsistency slips in. Everything here is derived from
disk; nothing is hardcoded except the exclusion list below, which has to be
explicit because those directories are legitimately outside the pipeline rather
than accidentally missing from it.

Checks, in order of how badly they bite:

1. **Class-directory naming drift.** The same class id must use the same
   directory name on every rig. `CLASS_DIR_RE` keys on the numeric id, so
   `class_009__Vietnam` and `class_009__Vietnam_Robusta` train fine today --
   which is why this went unnoticed. It is still the kind of drift that makes a
   grep-based tool silently disagree with the loader.
2. **Sessions not wired into the pipeline.** A session in dataset/ that is
   neither excluded below nor listed in dvc.yaml's `crop` foreach is almost
   certainly a capture someone forgot to wire in.
3. **Undeclared / unphotographed classes**, against dataset/classes.txt.
4. **Rig x class imbalance** -- reported, never fatal, because the dataset is
   legitimately ragged (iPhone is thin, the 08-30 sessions are class_010 only).

Exit status is non-zero only for 1-3. Imbalance is information, not an error.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from coffeecv.config import REPO_ROOT
from coffeecv.dataset import CLASS_DIR_RE

DATASET_DIR = REPO_ROOT / "dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".dng", ".cr2", ".cr3",
              ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}

# Naming drift that is known, understood, and deliberately not fixed. Reported
# as a warning (never silently suppressed) so it stays visible, but it does not
# fail the check -- otherwise this script is permanently red and everyone learns
# to ignore it, which is how the next, real drift would slip through.
#   class_009: box says `class_009__Vietnam`, every other rig says
#   `class_009__Vietnam_Robusta`. Harmless in the loader (CLASS_DIR_RE keys on
#   the numeric id) and renaming it would rewrite a DVC-tracked directory,
#   invalidating the box rig's crop stage and forcing a re-crop -- real cost for
#   a cosmetic gain. Remove this entry if the directory is ever renamed.
ACCEPTED_DRIFT = {"009"}

# Deliberately outside the pipeline -- flat directories with no class_* children,
# absent from dvc.yaml's crop foreach on purpose. Listed here so check 2 does not
# flag them forever; anything NOT here and not in the foreach is a real finding.
EXCLUDED = {
    "2026-07-24__first_pictures",   # pre-project exploratory shots, no class structure
    "2026-08-06__box_pictures",     # superseded by 2026-08-07__box_pictures_all_classes
    "classes_labels_only",          # label reference photos, not training data
}


def sessions_on_disk() -> list[Path]:
    return sorted(d for d in DATASET_DIR.iterdir()
                  if d.is_dir() and d.name not in EXCLUDED)


def sessions_in_dvc() -> set[str]:
    """The `crop` stage's foreach list -- the pipeline's own idea of what exists."""
    dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
    for stage in dvc.get("stages", {}).values():
        if isinstance(stage.get("foreach"), list):
            return set(stage["foreach"])
    return set()


def count_photos(class_dir: Path) -> int:
    return sum(1 for f in class_dir.iterdir()
               if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def scan() -> tuple[dict, dict]:
    """(counts[class_id][session] = n_photos, dirnames[class_id] = {name: [sessions]})"""
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    dirnames: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for session in sessions_on_disk():
        for child in sorted(session.iterdir()):
            if not child.is_dir():
                continue
            m = CLASS_DIR_RE.match(child.name)
            if not m:
                continue
            cid = m.group(1)
            counts[cid][session.name] = count_photos(child)
            dirnames[cid][child.name].append(session.name)
    return counts, dirnames


def declared_classes() -> dict[str, str]:
    out = {}
    for line in (DATASET_DIR / "classes.txt").read_text().splitlines():
        line = line.strip()
        if line and ";" in line:
            cid, label = line.split(";", 1)
            out[cid.strip()] = label.strip()
    return out


def print_table(counts: dict, labels: dict) -> None:
    sessions = sorted({s for per in counts.values() for s in per})
    # "2026-08-07__box_pictures_all_classes" -> "0807 box" : long enough to tell
    # the rigs apart, short enough that ten classes fit on one screen.
    def abbrev(name: str) -> str:
        date, _, rest = name.partition("__")
        return f"{date.replace('2026-', '').replace('-', '')} {rest.split('_')[0][:7]}"
    short = [abbrev(s) for s in sessions]
    width = max(max(len(s) for s in short), 5) if short else 5
    print(f"\n{'class':<28} " + " ".join(f"{s:>{width}}" for s in short) + "   total")
    print("-" * (28 + (width + 1) * len(sessions) + 8))
    for cid in sorted(counts):
        name = f"{cid} {labels.get(cid, '?')}"[:27]
        cells = []
        for s in sessions:
            n = counts[cid].get(s)
            cells.append(f"{n:>{width}}" if n else f"{'-':>{width}}")
        print(f"{name:<28} " + " ".join(cells) + f"   {sum(counts[cid].values()):>5}")
    totals = [sum(counts[c].get(s, 0) for c in counts) for s in sessions]
    print(f"{'TOTAL':<28} " + " ".join(f"{t:>{width}}" for t in totals)
          + f"   {sum(totals):>5}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="findings only, no table")
    args = ap.parse_args()

    counts, dirnames = scan()
    labels = declared_classes()
    if not args.quiet:
        print_table(counts, labels)

    errors, warnings = [], []

    # 1. naming drift
    for cid, names in sorted(dirnames.items()):
        if len(names) > 1:
            detail = "; ".join(f"{n} ({', '.join(sess)})" for n, sess in sorted(names.items()))
            msg = f"class_{cid} uses {len(names)} different directory names: {detail}"
            if cid in ACCEPTED_DRIFT:
                warnings.append(msg + "  [known and accepted, see ACCEPTED_DRIFT]")
            else:
                errors.append(msg)

    # 2. sessions not wired into dvc.yaml
    on_disk = {s.name for s in sessions_on_disk()}
    in_dvc = sessions_in_dvc()
    for name in sorted(on_disk - in_dvc):
        errors.append(f"session {name!r} is on disk but not in dvc.yaml's crop foreach "
                      f"(add it, or add it to EXCLUDED here if it is deliberately outside)")
    for name in sorted(in_dvc - on_disk):
        errors.append(f"session {name!r} is in dvc.yaml's crop foreach but not on disk")

    # 3. classes.txt vs reality
    for cid in sorted(set(counts) - set(labels)):
        errors.append(f"class_{cid} has photos but is not declared in dataset/classes.txt")
    for cid in sorted(set(labels) - set(counts)):
        errors.append(f"class_{cid} ({labels[cid]}) is declared in classes.txt but has no photos")

    # 4a. rig-level coverage. Sessions are the capture unit, but RIGS is what
    # folds hold out, and since 2026-09-03 a rig is a camera model built by
    # merging sessions -- so a class can look uncovered per-session while being
    # perfectly covered per-rig. Reporting only the session view said "class_010
    # has no broad-session coverage, so every fold scores it on nothing", which
    # is now simply false: three of the four cameras carry it.
    merges = {}
    for stage, body in yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())["stages"].items():
        if stage.startswith("merge_") and isinstance(body.get("cmd"), str):
            parts = body["cmd"].split()
            if "--name" in parts and "--sessions" in parts:
                merges[parts[parts.index("--name") + 1]] = parts[parts.index("--sessions") + 1:]
    try:
        from coffeecv.run_folds import RIGS
        rig_names = [Path(r).name for r in RIGS]
    except Exception:
        rig_names = []
    if rig_names:
        print("\nrig-level coverage (what leave-one-rig-out actually holds out):")
        for rn in rig_names:
            srcs = merges.get(rn, [rn])
            ids = {c for c in counts if any(s in counts[c] for s in srcs)}
            miss = sorted(set(labels) - ids)
            print(f"  {rn:<14} {len(ids):>2}/{len(labels)} classes"
                  + (f"   missing {', '.join('class_' + m for m in miss)}" if miss else "   complete"))
            for m in miss:
                warnings.append(f"rig {rn} has no class_{m}: its held-out fold scores "
                                f"{len(ids)} classes, not {len(labels)}")

    # 4. imbalance (informational)
    per_class_totals = {c: sum(v.values()) for c, v in counts.items()}
    if per_class_totals:
        lo, hi = min(per_class_totals.values()), max(per_class_totals.values())
        if hi and lo / hi < 0.5:
            thin = sorted(c for c, n in per_class_totals.items() if n < hi / 2)
            warnings.append(f"class totals span {lo}-{hi} photos; under half of the largest: "
                            + ", ".join(f"class_{c} ({per_class_totals[c]})" for c in thin))
    # Per-class rig coverage, not per-session missing-lists: a session that
    # deliberately carries one class (the 08-30 captures) would otherwise emit a
    # warning naming nine absent classes, every run, drowning the real signal.
    # "Broad" = a session covering more than one class, i.e. a general capture
    # session where a hole means something.
    all_sessions = sorted({s for per in counts.values() for s in per})
    broad = [s for s in all_sessions
             if sum(1 for c in counts if s in counts[c]) > 1]
    cover = {c: sorted(s for s in broad if s in counts[c]) for c in counts}
    full = max((len(v) for v in cover.values()), default=0)
    for cid in sorted(counts):
        have = cover[cid]
        if not have:
            warnings.append(f"class_{cid} appears only on single-class sessions "
                            f"({', '.join(sorted(counts[cid]))}) -- fine if those sessions merge "
                            f"into a camera rig that also carries the other classes, which is what "
                            f"the rig-level section above checks; a problem only if one does not")
        elif len(have) < full:
            absent = [s for s in broad if s not in have]
            warnings.append(f"class_{cid} is on {len(have)} of {full} broad sessions; absent from "
                            + ", ".join(absent))

    for w in warnings:
        print(f"\nWARN  {w}")
    for e in errors:
        print(f"\nFAIL  {e}")
    if not errors and not warnings:
        print("\nOK -- coverage consistent, nothing to report.")
    elif not errors:
        print(f"\nOK -- {len(warnings)} warning(s), no errors.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
