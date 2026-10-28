"""Find photos that are copies of another photo by image content, not bytes (2026-10-24).

    python -m coffeecv.find_duplicates NEW...                      # NEW (photos or folders) against dataset/
    python -m coffeecv.find_duplicates NEW... --against DIR...     # against other folders instead

A photo re-sent through a messenger comes back renamed, downscaled, recompressed, stripped of EXIF and with
its rotation baked into the pixels, so neither sha256 nor the file name finds it. Each photo is reduced to a
64x64 grayscale thumbnail of the whole frame, and two photos are compared by the normalized correlation of
their thumbnails, best over the 8 rotations and mirrorings that keep the aspect ratio (`load_rgb_image` and
this decode ignore EXIF Orientation, the copy usually does not). The new photos are also compared with each
other. Exit 1 if any pair reaches --threshold.

Threshold, from the first batch it ran on (15 Telegram re-sends, 960x1280, against 13 Galaxy originals of
2026-10-23 in class_013__Brazil_SulDeMinas): the 6 copies scored 0.9965-0.9996; the closest pair that was not a
copy, two hand-held shots of the same grinder 12 s apart, 0.7119. Not in that set: tripod bursts (a second
shot from a fixed camera may score near a copy), and copies cropped, padded or edited beyond recompression (a
crop changes the whole-frame thumbnail and scores low). Pairs in [--report-above, --threshold) are listed for
a look by eye.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv.config import REPO_ROOT
from coffeecv.dataset import RAW_EXTENSIONS, load_rgb_image
from coffeecv.strip_metadata import PHOTO_EXTENSIONS

THUMB = 64
THRESHOLD = 0.97
REPORT_ABOVE = 0.85
ASPECT_TOLERANCE = 0.03     # a messenger's downscale rounds each side to whole pixels


def signature(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """The photo's 64x64 grayscale thumbnail, zero-mean and unit-norm, and its (width, height)."""
    if path.suffix.lower() in RAW_EXTENSIONS:
        im = Image.fromarray(load_rgb_image(path))
    else:
        im = Image.open(path)
    size = im.size
    im.draft("RGB", (4 * THUMB, 4 * THUMB))   # a JPEG decodes at 1/2-1/8 scale; a no-op for other formats
    g = np.asarray(im.convert("L").resize((THUMB, THUMB), Image.Resampling.BOX), np.float64)
    g = g - g.mean()
    return g / (np.linalg.norm(g) or 1.0), size


def similarity(a: tuple[np.ndarray, tuple[int, int]], b: tuple[np.ndarray, tuple[int, int]]) -> tuple[float, str]:
    """Best correlation of a's thumbnail, rotated and mirrored, with b's; and the transform, as 'rot90 x k[, flip]'.
    A transform whose aspect ratio does not match b's is skipped; -1.0 if none does."""
    (ta, (wa, ha)), (tb, (wb, hb)) = a, b
    best, how = -1.0, ""
    for k in range(4):
        aspect = ha / wa if k % 2 else wa / ha
        if abs(aspect / (wb / hb) - 1) > ASPECT_TOLERANCE:
            continue
        r = np.rot90(ta, k)
        for flip in (False, True):
            c = float((r[:, ::-1] if flip else r).ravel() @ tb.ravel())
            if c > best:
                best, how = c, f"rot90 x {k}" + (", flip" if flip else "")
    return best, how


def photos_in(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        found = sorted(p.rglob("*")) if p.is_dir() else [p]
        out += [f for f in found if f.is_file() and f.suffix.lower() in PHOTO_EXTENSIONS]
    return out


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("new", nargs="+", type=Path, help="photos or folders (searched recursively) to check")
    ap.add_argument("--against", nargs="+", type=Path, default=[REPO_ROOT / "dataset"],
                    help="photos or folders to check them against [dataset/]")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help=f"a copy at or above this [{THRESHOLD}]")
    ap.add_argument("--report-above", type=float, default=REPORT_ABOVE,
                    help=f"also list pairs at or above this, for a look by eye [{REPORT_ABOVE}]")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)

    new = photos_in(args.new)
    if not new:
        ap.error("no photos to check")
    new_set = {p.resolve() for p in new}
    ref = [p for p in photos_in(args.against) if p.resolve() not in new_set]
    print(f"{len(new)} photos against {len(ref)}", file=sys.stderr)
    # spawn: rawpy's LibRaw runs OpenMP, which can deadlock in a forked worker
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        sigs = dict(zip(new + ref, pool.map(signature, new + ref, chunksize=8)))

    copies = 0
    for i, p in enumerate(new):
        others = new[i + 1:] + ref
        scored = sorted(((*similarity(sigs[p], sigs[q]), q) for q in others), key=lambda s: -s[0])
        hits = [s for s in scored if s[0] >= args.report_above]
        nearest = f"nearest {scored[0][0]:.4f} {_rel(scored[0][2])}" if scored else "nothing to compare"
        if not hits:
            print(f"unique  {_rel(p)}  ({nearest})")
        for score, how, q in hits:
            copy = score >= args.threshold
            copies += copy
            print(f"{'COPY  ' if copy else 'look  '}  {_rel(p)}  {score:.4f}  {_rel(q)}  ({how})")
    print(f"{copies} pair(s) at or above {args.threshold}", file=sys.stderr)
    return 1 if copies else 0


if __name__ == "__main__":
    sys.exit(main())
