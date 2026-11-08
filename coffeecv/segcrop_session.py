"""Ticket ML-2 P3: the `segcrop@<session>` stage (plan §2.4, §6). The segmenter's counterpart of the crop
stage: every raw photo of one session goes through `segment_beans.segment_and_crop`, the function
infer.patches_for_photo runs live, with the segmenter params.yaml names (`seg_params`).

    python -m coffeecv.segcrop_session --session 2026-08-25__iphone --threads 4

In, since ticket ML-19 P6 (D18): data/canonical/<session>/<class dir>/<original file name>.jxl, each original's
canonical photo (the canonicalize@<session> stage), read as 8-bit sRGB by coffeecv.canonical.load_canonical. Until
then the stage read the originals in dataset/<session> through load_rgb_image (no EXIF rotation, no colour profile).

Out, under data/segcropped/<session>/<class dir>/, per photo (<stem>: the original's file name without extension):
    <stem>__cropped.jxl     the filled D4 crop, lossless 8-bit sRGB JPEG XL (coffeecv.crop_io; ML-19 D32), or on a
                            D18 fallback the whole canonical photo, unfilled (JPEG q95 `__cropped.jpg` until ML-19)
    <stem>__beanmask.png    the bean-region mask aligned with that crop (1-bit); absent on a fallback, which is
                            how training knows to accept every patch there (D17/D18)
plus a per-class segcrop_report.json (mask_area_frac, bean_frac_in_crop, retained_frac, fallback, box,
mask_sha256 of the full-frame mask) and a session segcrop_manifest.json (segmenter, thread count, totals).
No timings are written: the outs must be a pure function of the photos, the code and the params.

Mask bits depend on the CPU and the thread count (P0), so the stage runs on one machine (this one since the VM went)
at a pinned thread count, like seg_predict, and nothing recomputes its masks.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import crop_io
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.crop_tray import repo_relative
from coffeecv.segment_beans import BeanCrop, BeanSegmenter, seg_params, segment_and_crop

CANONICAL_ROOT = cn.STORE
SEGCROPPED_ROOT = REPO_ROOT / "data" / "segcropped"


def crop_photo(path: Path, seg: BeanSegmenter) -> BeanCrop:
    """One canonical photo -> the arrays this stage writes, before any encoding."""
    return segment_and_crop(cn.load_canonical(Path(path).read_bytes()), seg)


def original_stem(canonical: Path) -> str:
    """`PXL_1.jpg.jxl` -> `PXL_1`: the original's file name without its extension."""
    return Path(canonical.name[:-len(".jxl")]).stem


def write_crop(crop: BeanCrop, out: Path) -> None:
    crop_io.write_crop(crop.rgb, out)
    if crop.mask is not None:
        Image.fromarray(crop.mask).convert("1").save(crop_io.mask_path(out), optimize=True)


def segcrop_session(session: str, out_root: Path = SEGCROPPED_ROOT, limit_per_class: int | None = None) -> dict:
    cfg = RunConfig.from_params_yaml()
    p = seg_params(cfg)
    in_session = CANONICAL_ROOT / session
    if not in_session.is_dir():
        raise FileNotFoundError(f"No canonical session at {in_session}: `dvc repro canonicalize@{session}` first")
    class_dirs = sorted(d for d in in_session.iterdir() if d.is_dir() and d.name.startswith("class_"))
    if not class_dirs:
        raise FileNotFoundError(f"No class_* directories under {in_session}")

    seg = BeanSegmenter(p)
    out_session = out_root / session
    totals = {"classes": 0, "images": 0, "fallbacks": 0}
    t0 = time.perf_counter()
    for class_dir in class_dirs:
        out_dir = out_session / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        photos = sorted(class_dir.glob("*.jxl"))[:limit_per_class]
        reports = []
        for photo in photos:
            crop = crop_photo(photo, seg)
            out = out_dir / crop_io.crop_name(original_stem(photo))
            write_crop(crop, out)
            info = {k: crop.info[k] for k in ("mask_area_frac", "bean_frac_in_crop", "retained_frac",
                                               "fallback", "box", "mask_sha256")}
            reports.append({"file": photo.name[:-len(".jxl")], "canonical": repo_relative(photo),
                            "out_path": repo_relative(out),
                            "size": [crop.rgb.shape[1], crop.rgb.shape[0]], **info})
            totals["fallbacks"] += info["fallback"]
            if info["fallback"]:
                print(f"  FALLBACK {class_dir.name}/{photo.name}: mask_area_frac {info['mask_area_frac']:.4f} "
                      f"< {p.min_area_frac} -- whole photo, every patch accepted (D18)")
        (out_dir / "segcrop_report.json").write_text(json.dumps(reports, indent=2) + "\n")
        totals["classes"] += 1
        totals["images"] += len(reports)
        print(f"  {class_dir.name}: {len(reports)} photos  {time.perf_counter() - t0:.0f}s", flush=True)

    manifest = {"session": session, "segmenter": {**{k: getattr(p, k) for k in (
                    "weights", "decoder", "prompt", "mask_select", "keep_frac", "fill_rgb", "min_area_frac")},
                    "weights_sha256": seg.weights_sha256, "decoder_sha256": seg.decoder_sha256},
                "threads": torch.get_num_threads(), "limit_per_class": limit_per_class, **totals}
    (out_session / "segcrop_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{session}: {totals['images']} photos across {totals['classes']} classes, "
          f"{totals['fallbacks']} D18 fallbacks")
    return totals


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="session folder name under data/canonical/ (and dataset/)")
    ap.add_argument("--threads", type=int, required=True,
                    help="torch.set_num_threads; pinned in dvc.yaml, because it changes mask bits")
    ap.add_argument("--out-root", type=Path, default=SEGCROPPED_ROOT,
                    help="for smoke runs only; the stage writes data/segcropped")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="first N photos per class (smoke runs only)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    segcrop_session(args.session, args.out_root, args.limit_per_class)


if __name__ == "__main__":
    main()
