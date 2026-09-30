"""ML-2 P0: pick the photos for the mask-selection look and the latency report (plan §3).

Mask selection: 10 raw photos from the segmenter's TRAIN split (seed 239, D21), never its eval split,
one per capture session plus a second from the tray session with the most varied framing, drawn with a
fixed seed. The split is the head fit's own pooled split (split_photos_by_class) at seed 239, the
same call P1's seg_lists.py will make. Each crop is mapped back to its raw original in dataset/, since
the segmenter sees full photos.

Timing: the first 12 MP pixel photo and the first 50 MP sony photo among the picks' sessions.

    PYTHONPATH=. python analysis/ml2_p0/pick_photos.py     # writes mask_select_photos.txt, timing_photos.txt
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from coffeecv.config import PARAMS_FILE, REPO_ROOT
from coffeecv.dataset import (load_class_labels, pooled_class_photos, resolve_captures,
                              split_photos_by_class)

SEG_SPLIT_SEED = 239          # D21; params.yaml seg_split_seed
HERE = Path(__file__).resolve().parent
EXTRA_FROM = "2026-08-07__box_pictures_all_classes"   # tray on a table, the most varied framing


def raw_index() -> dict[tuple[str, str], Path]:
    """(class_NNN, stem) -> raw photo under dataset/<session>/. Keyed on the class number: the
    label half of a class dir's name differs between sessions. Refuses ambiguous stems."""
    idx: dict[tuple[str, str], Path] = {}
    for f in sorted((REPO_ROOT / "dataset").glob("2026-*/class_*/*")):
        if not f.is_file():
            continue
        key = (f.parent.name.split("__")[0], f.stem)
        if key in idx:
            raise ValueError(f"{f} and {idx[key]} share a stem")
        idx[key] = f
    return idx


def main() -> None:
    params = yaml.safe_load(PARAMS_FILE.read_text())
    captures = resolve_captures([REPO_ROOT / d for d in params["train_capture_dirs"]])
    class_ids = sorted(load_class_labels(REPO_ROOT / params["classes_file"]))
    frac = {s: params[f"{s}_photo_frac"] for s in ("train", "val", "test")}
    idx = raw_index()

    by_session: dict[str, list[Path]] = defaultdict(list)
    for class_idx, class_id in enumerate(class_ids):
        pool, _ = pooled_class_photos(captures, class_id)
        for ph in split_photos_by_class(pool, SEG_SPLIT_SEED, class_idx, frac)["train"]:
            raw = idx[(ph.path.parent.name.split("__")[0], ph.name.removesuffix("__cropped.jpg"))]
            by_session[raw.parts[-3]].append(raw)

    rng = np.random.default_rng([SEG_SPLIT_SEED, 0])
    picks = []
    for session in sorted(by_session):
        pool = sorted(by_session[session])
        k = 2 if session == EXTRA_FROM else 1
        picks += [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]
    rel = [str(p.relative_to(REPO_ROOT)) for p in picks]
    header = "# ML-2 P0 mask-selection photos: seg train split (seed 239), one per session (+1 box_pictures)\n"
    (HERE / "mask_select_photos.txt").write_text(header + "\n".join(rel) + "\n")

    timing = [next(r for r in rel if "__pixel_cam/" in r), next(r for r in rel if "__sony_cam/" in r)]
    (HERE / "timing_photos.txt").write_text("# ML-2 P0 timing: one 12 MP pixel photo, one 50 MP sony photo\n"
                                           + "\n".join(timing) + "\n")
    print("\n".join(rel))


if __name__ == "__main__":
    main()
