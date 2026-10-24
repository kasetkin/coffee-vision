"""List the photos whose bean mask is empty or tiny, for the owner to look at by eye (2026-10-24).

    python -m coffeecv.find_tiny_masks                          # every dataset/<session>/class_*/ photo
    python -m coffeecv.find_tiny_masks PATH...                  # these photos or folders instead (recursive)
    python -m coffeecv.find_tiny_masks --max-area-frac 0.10     # a looser cutoff than the D18 fallback's

Each photo goes through `segment_and_crop`, the segmenter serving and the segcrop stage run, with params.yaml's
seg_* fields. A photo is listed when its mask covers less than --max-area-frac of the frame (default
seg_min_area_frac, the D18 fallback cutoff: these are the photos training and serving read whole) or is empty.
Out: one line per listed photo, "<repo-relative path>\\t<mask_area_frac>", smallest mask first.

Nothing is written under data/: the stored masks are the segcrop stage's (docs/agents/experiments.md). Mask bits
move a few pixels with the thread count, so an area within a hair of the cutoff may list differently from the
stage's segcrop_report.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.crop_tray import IMAGE_SUFFIXES, repo_relative
from coffeecv.dataset import load_rgb_image
from coffeecv.segment_beans import BeanSegmenter, seg_params, segment_and_crop

RAW_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUT = REPO_ROOT / "outputs" / "tiny_masks.txt"


def class_photos(raw_root: Path = RAW_ROOT) -> list[Path]:
    """Every photo of every session's class_* folders: what the segcrop stages read, and the retired sessions."""
    return sorted(p for d in raw_root.glob("*/class_*") if d.is_dir()
                  for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def photos_under(paths: list[Path]) -> list[Path]:
    out = []
    for path in paths:
        if path.is_dir():
            out += [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        else:
            out.append(path)
    return sorted(set(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="photos or folders (default: dataset/*/class_*/)")
    ap.add_argument("--max-area-frac", type=float, default=None,
                    help="list masks below this fraction of the frame (default: params.yaml seg_min_area_frac)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the list (default: outputs/tiny_masks.txt)")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch's)")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    p = seg_params(RunConfig.from_params_yaml())
    cutoff = p.min_area_frac if args.max_area_frac is None else args.max_area_frac
    photos = photos_under(args.paths) if args.paths else class_photos()
    if not photos:
        sys.exit("No photos found")
    seg = BeanSegmenter(p)
    print(f"{len(photos)} photos, decoder {p.decoder}, cutoff mask_area_frac < {cutoff} "
          f"({torch.get_num_threads()} threads)", flush=True)

    tiny, t0 = [], time.perf_counter()
    for i, photo in enumerate(photos, 1):
        area = segment_and_crop(load_rgb_image(photo), seg).info["mask_area_frac"]
        if area == 0 or area < cutoff:
            tiny.append((area, repo_relative(photo)))
            print(f"  TINY {area:.4f}  {repo_relative(photo)}", flush=True)
        if i % 50 == 0 or i == len(photos):
            print(f"  {i}/{len(photos)}  {time.perf_counter() - t0:.0f}s", flush=True)

    tiny.sort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{path}\t{area:.6f}\n" for area, path in tiny))
    print(f"{len(tiny)} of {len(photos)} photos below {cutoff} ({sum(a == 0 for a, _ in tiny)} empty) -> "
          f"{repo_relative(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
