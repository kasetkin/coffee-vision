"""Merge two or more already-cropped sessions into one logical rig.

For a rig re-shot on a different day/setup (different framing, lighting, even a
different crop trim) that should still be trained on as *one* rig rather than a
new, separate one -- see dataset/2026-08-27__oneplus_flash.crop.yaml (deleted in
ticket ML-3 P2b; in git history) for why that session and 2026-08-25__oneplus were
first merged rather than kept apart.

This is deliberately a real, tracked pipeline stage (the merge_segcam_* stages in
dvc.yaml), not a manual one-off copy -- this project already has a scar from
treating a step as untracked: crop_tray.py predating the `crop` stage "could not
run anywhere the crops did not already sit in the working tree" (see
EXPERIMENTS_LOG.md Phase 9).

    python -m coffeecv.merge_rig --name cam_sony --sessions 2026-08-09__sony_cam 2026-08-30__sony

Ticket ML-2 (F3): the segmenter's pools merge the same way from data/segcropped (`--root`), and a crop's
bean-region mask (`<stem>__beanmask.png`, dataset.bean_mask_path) travels with it; a D18 fallback crop has
none. Since ticket ML-3 P2b data/segcropped is the default root: the tray heuristic's data/cropped pools,
which this first merged, are retired (Q3).

Ticket ML-3 P5 (D8): `--exclude FILE` leaves photos out of the pool -- the crops the owner declined
(labels/ml3/pool_exclude.csv: session, class_folder, file, reason). The session's own segcrop output keeps
them; the pool, and so every split, does not. A row naming one of this merge's sessions must match a crop
there, or the merge stops; rows for other sessions belong to other pools. The manifest lists what was left
out, and has no `excluded` key without the flag, so pools merged before it are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from coffeecv.config import REPO_ROOT

SEGCROPPED_ROOT = REPO_ROOT / "data" / "segcropped"
CLASS_DIR_RE = re.compile(r"^class_(\d+)__")


def merge_cmd_args(cmd: str) -> tuple[str, list[str]]:
    """(--name, --sessions) of a merge_segcam_* stage's cmd, for readers of dvc.yaml: the session list ends at
    the next flag (--exclude since ML-3 P5), not at the end of the line."""
    parts = cmd.split()
    i = parts.index("--sessions") + 1
    j = next((k for k in range(i, len(parts)) if parts[k].startswith("--")), len(parts))
    return parts[parts.index("--name") + 1], parts[i:j]


def read_exclusions(path: Path) -> set[tuple[str, str, str]]:
    """(session, class folder, raw photo file name) for every row of an exclusion CSV."""
    with open(path, newline="") as f:
        return {(r["session"], r["class_folder"], r["file"]) for r in csv.DictReader(f)}


def merge_rig(name: str, sessions: list[str], root: Path = SEGCROPPED_ROOT,
              exclude: set[tuple[str, str, str]] | None = None) -> dict:
    session_dirs = [root / s for s in sessions]
    # (session, class dir, crop file name) -> the raw photo's name, for the rows naming this merge's sessions.
    left_out = {(s, c, f"{Path(f).stem}__cropped.jpg"): f for s, c, f in (exclude or set()) if s in sessions}
    for s, d in zip(sessions, session_dirs):
        if not d.is_dir():
            raise FileNotFoundError(f"No cropped session at {d}. Run its crop stage first: "
                                     f"`dvc repro segcrop@{s}` (on the VM, at 4 threads).")

    # Union of class directories across all source sessions -- a session missing
    # one class (e.g. iPhone lacking class_008) just contributes nothing for it,
    # same tolerance MultiPhotoPatchDataset already has for a multi-rig train set.
    class_dirs_by_id: dict[str, list[tuple[str, Path]]] = {}
    for session, session_dir in zip(sessions, session_dirs):
        for class_dir in sorted(session_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            m = CLASS_DIR_RE.match(class_dir.name)
            if not m:
                continue
            class_dirs_by_id.setdefault(m.group(1), []).append((session, class_dir))
    unmatched = [k for k in left_out if not (root / k[0] / k[1] / k[2]).is_file()]
    if unmatched:
        raise ValueError(f"{name}: exclusions match no crop in their session: "
                         + ", ".join("/".join((s, c, left_out[(s, c, f)])) for s, c, f in sorted(unmatched)))

    out_root = root / name
    out_root.mkdir(parents=True, exist_ok=True)

    per_class_counts: dict[str, dict[str, int]] = {}
    totals = {"classes": 0, "images": 0}
    for class_id, contributors in sorted(class_dirs_by_id.items()):
        # All contributing dirs for this class share the same "class_XXX__Label"
        # name (classes.txt is global), so any one of them names the output dir.
        out_dir = out_root / contributors[0][1].name
        out_dir.mkdir(parents=True, exist_ok=True)

        seen: dict[str, str] = {}  # filename -> which session it came from
        counts: dict[str, int] = {}
        for session, class_dir in contributors:
            photos = sorted(class_dir.glob("*__cropped.jpg"))
            counts[session] = len(photos)
            for photo in photos:
                if (session, class_dir.name, photo.name) in left_out:
                    counts[session] -= 1
                    continue
                if photo.name in seen:
                    raise ValueError(
                        f"filename collision merging into {out_dir}: {photo.name!r} exists in "
                        f"both {seen[photo.name]!r} and {session!r}. Merge refuses to silently "
                        f"pick one -- rename one of the source files or investigate why two "
                        f"sessions produced the same filename."
                    )
                seen[photo.name] = session
                shutil.copy2(photo, out_dir / photo.name)
                # dataset.bean_mask_path, spelled out to keep this module free of the dataset stack.
                mask = photo.with_name(photo.name.replace("__cropped.jpg", "__beanmask.png"))
                if mask.exists():
                    shutil.copy2(mask, out_dir / mask.name)

        per_class_counts[class_id] = counts
        totals["classes"] += 1
        totals["images"] += len(seen)

    manifest = {
        "name": name,
        "sessions": sessions,
        "per_class_counts": per_class_counts,
        **totals,
    }
    if exclude is not None:
        manifest["excluded"] = sorted("/".join((s, c, f)) for (s, c, _), f in left_out.items())
    (out_root / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{name}: merged {totals['images']} images across {totals['classes']} classes "
          f"from {len(sessions)} sessions ({', '.join(sessions)})"
          + (f", {len(left_out)} excluded" if exclude is not None else ""))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="name of the merged rig, under <root>/<name>")
    p.add_argument("--sessions", required=True, nargs="+", help="cropped session names to merge (>= 2)")
    p.add_argument("--root", type=Path, default=SEGCROPPED_ROOT,
                   help="where the sessions are and the merge goes (default data/segcropped)")
    p.add_argument("--exclude", type=Path, help="CSV of photos to leave out (session, class_folder, file, reason)")
    args = p.parse_args()
    if not args.sessions:
        raise SystemExit("--sessions needs at least one session")
    # A one-session "merge" is a copy, and was previously refused on exactly that
    # ground. It is allowed now because rigs are keyed on camera model, so the rig
    # name has to be stable even when a camera has been shot only once: cam_iphone
    # is one session today and two the moment an iPhone frame-filling session
    # lands, and that must not require editing RIGS, params.yaml and every
    # downstream reference. Paying one directory copy to keep the identifier
    # stable is the cheaper side of that trade.
    merge_rig(args.name, args.sessions, args.root.resolve(), read_exclusions(args.exclude) if args.exclude else None)


if __name__ == "__main__":
    main()
