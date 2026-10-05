"""Session x class and capture-dir x class photo coverage: print the tables, and assert what should
stay true.

Producing this table by hand meant cross-referencing eight session directories,
and it needs redoing every time a session or a class is added -- which is
exactly when a quiet inconsistency slips in. Everything here is derived from
disk; nothing is hardcoded except the exclusion list below, which has to be
explicit because those directories are legitimately outside the pipeline rather
than accidentally missing from it.

Checks, in order of how badly they bite:

1. **Class-directory naming drift.** The same class id must use the same
   directory name in every session. `CLASS_DIR_RE` keys on the numeric id, so
   `class_009__Vietnam` and `class_009__Vietnam_Robusta` train fine today --
   which is why this went unnoticed. It is still the kind of drift that makes a
   grep-based tool silently disagree with the loader.
2. **Sessions not wired into the pipeline.** A session in dataset/ that is
   neither excluded below nor listed in dvc.yaml's `segcrop` foreach is almost
   certainly a capture someone forgot to wire in. (The tray heuristic's `crop`
   foreach, read here until ticket ML-3 P2b, is retired.)
3. **Undeclared / unphotographed classes**, against dataset/classes.txt -- and a
   declared class that no training capture dir (`dataset.CAPTURES`) carries, which
   training would refuse outright. Checked per class folder (a coffee): since ticket
   ML-3 a class can pool several folders, and a table per class (a country) is
   printed beside the folder table when one does.
4. **Class balance per capture dir and per session** -- reported, never fatal,
   because the dataset is legitimately ragged (iPhone is thin, the 08-30 sessions
   are class_010 only). Since ticket ML-1 no capture dir is held out, so a dir
   missing a class only means the others supply it; this is class balance, not
   fold coverage.

Exit status is non-zero only for 1-3. Imbalance is information, not an error.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from coffeecv.class_list import load_classes, read_coffees
from coffeecv.config import REPO_ROOT
# A hard import on purpose. This used to be `from coffeecv.run_folds import RIGS` inside a bare
# try/except, which would have silently dropped the whole capture-dir section once that module
# was deleted; a broken import must fail loudly instead.
from coffeecv.dataset import CAPTURES, CLASS_DIR_RE

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
# absent from dvc.yaml's segcrop foreach on purpose. Listed here so check 2 does not
# flag them forever; anything NOT here and not in the foreach is a real finding.
EXCLUDED = {
    "2026-07-24__first_pictures",   # pre-project exploratory shots, no class structure
    "2026-08-06__box_pictures",     # superseded by 2026-08-07__box_pictures_all_classes
    "classes_labels_only",          # label reference photos, not training data
    # The OOD sets (ticket ML-3 P4): read by the OOD tools from their manifests, never segcropped. Their
    # check-2 FAILs dated from ticket ML-1.
    "ood_negatives",
    "ood_positives",
    "ood_positives_internet",
}


def sessions_on_disk() -> list[Path]:
    return sorted(d for d in DATASET_DIR.iterdir()
                  if d.is_dir() and d.name not in EXCLUDED)


def sessions_in_dvc() -> set[str]:
    """The `segcrop` stage's foreach list -- the pipeline's own idea of what exists. By name: the first
    list-valued foreach was the tray heuristic's `crop` until ticket ML-3 P2b, and is seg_finetune's seeds
    now."""
    dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
    return set(dvc["stages"]["segcrop"]["foreach"])


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
    """{class folder id: its classes.txt text}, either format (class_list.read_coffees)."""
    return {c.folder_id: c.label for c in read_coffees(DATASET_DIR / "classes.txt")}


def print_table(counts: dict, labels: dict) -> None:
    sessions = sorted({s for per in counts.values() for s in per})
    # "2026-08-07__box_pictures_all_classes" -> "0807 box" : long enough to tell
    # the rigs apart, short enough that ten classes fit on one screen.
    def abbrev(name: str) -> str:
        date, _, rest = name.partition("__")
        if not rest:                          # an undated session: random_date_raccoon
            return name[:12]
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
        errors.append(f"session {name!r} is on disk but not in dvc.yaml's segcrop foreach "
                      f"(add it, or add it to EXCLUDED here if it is deliberately outside)")
    for name in sorted(in_dvc - on_disk):
        errors.append(f"session {name!r} is in dvc.yaml's segcrop foreach but not on disk")

    # 3. classes.txt vs reality
    for cid in sorted(set(counts) - set(labels)):
        errors.append(f"class_{cid} has photos but is not declared in dataset/classes.txt")
    for cid in sorted(set(labels) - set(counts)):
        errors.append(f"class_{cid} ({labels[cid]}) is declared in classes.txt but has no photos")

    # 4a. class balance per capture dir. Sessions are the capture unit on disk, but
    # training draws from the CAPTURES dirs, each merged from sessions by a merge_segcam_*
    # stage -- so a class can look thin per session and still be well covered per dir.
    # Counted from the raw sessions through dvc.yaml's merge map, so this runs
    # without the segcrop stage having been run.
    merges = {}
    for stage, body in yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())["stages"].items():
        if stage.startswith("merge_segcam_") and isinstance(body.get("cmd"), str):
            parts = body["cmd"].split()
            if "--name" in parts and "--sessions" in parts:
                merges[parts[parts.index("--name") + 1]] = parts[parts.index("--sessions") + 1:]
    capture_names = [Path(c).name for c in CAPTURES]
    per_capture = {cn: {c: sum(counts[c].get(s, 0) for s in merges.get(cn, [cn])) for c in counts}
                   for cn in capture_names}
    if not args.quiet:
        width = max(max(len(cn) for cn in capture_names), 5)
        print("\nclass balance per capture dir (photos; the pools training draws from):")
        print(f"{'class':<28} " + " ".join(f"{cn:>{width}}" for cn in capture_names) + "   total")
        for cid in sorted(labels):
            name = f"{cid} {labels.get(cid, '?')}"[:27]
            cells = [per_capture[cn].get(cid, 0) for cn in capture_names]
            print(f"{name:<28} " + " ".join(f"{n:>{width}}" if n else f"{'-':>{width}}" for n in cells)
                  + f"   {sum(cells):>5}")
        classes = load_classes(DATASET_DIR / "classes.txt")
        if any(len(f) > 1 for f in classes.folders.values()):
            # What training draws from: a class pools its folders in each capture dir (ML-3 D6).
            print("\nper class (what the model predicts; its folders pooled):")
            print(f"{'class':<28} " + " ".join(f"{cn:>{width}}" for cn in capture_names) + "   total")
            for key in classes.keys:
                cells = [sum(per_capture[cn].get(f, 0) for f in classes.folders[key]) for cn in capture_names]
                name = f"{key} ({', '.join(classes.folders[key])})"[:27]
                print(f"{name:<28} " + " ".join(f"{n:>{width}}" if n else f"{'-':>{width}}" for n in cells)
                      + f"   {sum(cells):>5}")
    for cid in sorted(labels):
        if not any(per_capture[cn].get(cid, 0) for cn in capture_names):
            errors.append(f"class_{cid} ({labels[cid]}) is in no training capture dir "
                          f"({', '.join(capture_names)}) -- training would refuse to start")

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
    # "Broad" = a session covering at least half the class folders, i.e. a general
    # capture session where a hole means something. Until ticket ML-3 P4 this was
    # "more than one class"; the 09-24 captures carry two new coffees on purpose.
    all_sessions = sorted({s for per in counts.values() for s in per})
    broad = [s for s in all_sessions
             if 2 * sum(1 for c in counts if s in counts[c]) >= len(counts)]
    cover = {c: sorted(s for s in broad if s in counts[c]) for c in counts}
    full = max((len(v) for v in cover.values()), default=0)
    for cid in sorted(counts):
        have = cover[cid]
        if not have:
            warnings.append(f"class_{cid} appears only on narrow sessions "
                            f"({', '.join(sorted(counts[cid]))}) -- fine if those sessions merge "
                            f"into a capture dir, which the capture-dir table above shows")
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
