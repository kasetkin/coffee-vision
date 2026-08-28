"""Merge two or more already-cropped sessions into one logical rig.

For a rig re-shot on a different day/setup (different framing, lighting, even a
different crop trim) that should still be trained on as *one* rig rather than a
new, separate one -- see dataset/2026-08-27__oneplus_flash.crop.yaml for why that
session and 2026-08-25__oneplus are being merged rather than kept apart.

This is deliberately a real, tracked pipeline stage (`merge_oneplus` in dvc.yaml),
not a manual one-off copy -- this project already has a scar from treating a step
as untracked: crop_tray.py predating the `crop` stage "could not run anywhere the
crops did not already sit in the working tree" (see EXPERIMENTS_LOG.md Phase 9).

    python -m coffeecv.merge_rig --name oneplus_combined --sessions 2026-08-25__oneplus 2026-08-27__oneplus_flash
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from coffeecv.config import REPO_ROOT

CROPPED_ROOT = REPO_ROOT / "data" / "cropped"
CLASS_DIR_RE = re.compile(r"^class_(\d+)__")


def merge_rig(name: str, sessions: list[str]) -> dict:
    session_dirs = [CROPPED_ROOT / s for s in sessions]
    for s, d in zip(sessions, session_dirs):
        if not d.is_dir():
            raise FileNotFoundError(f"No cropped session at {d}. Run the crop stage first: "
                                     f"`dvc repro crop` (or `crop_session.py --session {s}`).")

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

    out_root = CROPPED_ROOT / name
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
                if photo.name in seen:
                    raise ValueError(
                        f"filename collision merging into {out_dir}: {photo.name!r} exists in "
                        f"both {seen[photo.name]!r} and {session!r}. Merge refuses to silently "
                        f"pick one -- rename one of the source files or investigate why two "
                        f"sessions produced the same filename."
                    )
                seen[photo.name] = session
                shutil.copy2(photo, out_dir / photo.name)

        per_class_counts[class_id] = counts
        totals["classes"] += 1
        totals["images"] += len(seen)

    manifest = {
        "name": name,
        "sessions": sessions,
        "per_class_counts": per_class_counts,
        **totals,
    }
    (out_root / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{name}: merged {totals['images']} images across {totals['classes']} classes "
          f"from {len(sessions)} sessions ({', '.join(sessions)})")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="name of the merged rig, under data/cropped/<name>")
    p.add_argument("--sessions", required=True, nargs="+", help="cropped session names to merge (>= 2)")
    args = p.parse_args()
    if len(args.sessions) < 2:
        raise SystemExit("--sessions needs at least 2 sessions to merge -- for a single session, "
                          "just use its own data/cropped/<session> directly, no merge needed.")
    merge_rig(args.name, args.sessions)


if __name__ == "__main__":
    main()
