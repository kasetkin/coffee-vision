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

Since ticket ML-26 the stage crops only the photos new or changed since its last run, when nothing else changed
(coffeecv.incremental reads dvc.lock: the code, the segmenter, its params and the thread count in the command): a
photo's crop and mask on disk as dvc.lock recorded them stay, with their entry of the recorded report, and the outputs
of a photo that is gone are deleted. Anything else, or --full, rebuilds the session's crops from nothing. The output
is the same bytes either way: a photo's crop depends on that photo alone.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import crop_io
from coffeecv import incremental as inc
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


def recorded_run(session: str, out_root: Path, limit_per_class: int | None, full: bool,
                 root: Path) -> tuple[inc.Record | None, dict[str, dict[str, dict]], str]:
    """(the last run's Record, its report entries {class dir: {file: entry}}, how) when the stage may keep its outputs
    (ML-26); (None, {}, why) for a full rebuild: --full, a smoke run, or anything but the photos changed."""
    if full:
        return None, {}, "full rebuild (--full)"
    if limit_per_class is not None or out_root.resolve() != (root / "data" / "segcropped").resolve() \
            or not (root / "dvc.lock").exists():
        return None, {}, "full rebuild (not a repository's crops)"
    record, why = inc.recorded(f"segcrop@{session}", f"data/canonical/{session}", f"data/segcropped/{session}", root)
    if record is None:
        return None, {}, f"full rebuild ({why})"
    reports = {}
    for rel, md5 in record.outputs.items():
        if rel.endswith("/segcrop_report.json"):
            data = inc.recorded_output(root, md5)
            if data is None:
                return None, {}, f"full rebuild ({rel}'s recorded report is not in the cache)"
            reports[rel.rsplit("/", 1)[0]] = {e["file"]: e for e in json.loads(data)}
    return record, reports, "incremental"


def kept_entry(record: inc.Record | None, reports: dict, out_session: Path, class_dir: Path, photo: Path) -> dict | None:
    """The recorded report entry of `photo` when its crop and mask stand as recorded; else None (crop it)."""
    if record is None or f"{class_dir.name}/{photo.name}" not in record.inputs:
        return None
    entry = reports.get(class_dir.name, {}).get(photo.name[:-len(".jxl")])
    crop = f"{class_dir.name}/{crop_io.crop_name(original_stem(photo))}"
    mask = crop_io.mask_path(Path(crop)).as_posix()
    mask_ok = inc.keeps(record, out_session, mask) if mask in record.outputs else not (out_session / mask).exists()
    return entry if entry is not None and inc.keeps(record, out_session, crop) and mask_ok else None


def segcrop_session(session: str, out_root: Path = SEGCROPPED_ROOT, limit_per_class: int | None = None,
                    full: bool = False, root: Path = REPO_ROOT, canonical_root: Path = CANONICAL_ROOT) -> dict:
    cfg = RunConfig.from_params_yaml(root / "params.yaml")
    p = seg_params(cfg)
    in_session = canonical_root / session
    if not in_session.is_dir():
        raise FileNotFoundError(f"No canonical session at {in_session}: `dvc repro canonicalize@{session}` first")
    class_dirs = sorted(d for d in in_session.iterdir() if d.is_dir() and d.name.startswith("class_"))
    if not class_dirs:
        raise FileNotFoundError(f"No class_* directories under {in_session}")

    out_session = out_root / session
    record, recorded_reports, how = recorded_run(session, out_root, limit_per_class, full, root)
    if record is None:
        shutil.rmtree(out_session, ignore_errors=True)
    seg = BeanSegmenter(p)
    totals = {"classes": 0, "images": 0, "fallbacks": 0}
    t0 = time.perf_counter()
    want, done, kept = {"segcrop_manifest.json"}, 0, 0
    for class_dir in class_dirs:
        out_dir = out_session / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        photos = sorted(class_dir.glob("*.jxl"))[:limit_per_class]
        reports = []
        want.add(f"{class_dir.name}/segcrop_report.json")
        for photo in photos:
            name = crop_io.crop_name(original_stem(photo))
            want |= {f"{class_dir.name}/{name}", crop_io.mask_path(Path(class_dir.name) / name).as_posix()}
            entry = kept_entry(record, recorded_reports, out_session, class_dir, photo)
            if entry is not None:
                reports.append(entry)
                totals["fallbacks"] += entry["fallback"]
                kept += 1
                continue
            done += 1
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
        # (kept entries are the recorded report's own, so the file is the bytes a full run writes)
        totals["classes"] += 1
        totals["images"] += len(reports)
        print(f"  {class_dir.name}: {len(reports)} photos  {time.perf_counter() - t0:.0f}s", flush=True)

    manifest = {"session": session, "segmenter": {**{k: getattr(p, k) for k in (
                    "weights", "decoder", "prompt", "mask_select", "keep_frac", "fill_rgb", "min_area_frac")},
                    "weights_sha256": seg.weights_sha256, "decoder_sha256": seg.decoder_sha256},
                "threads": torch.get_num_threads(), "limit_per_class": limit_per_class, **totals}
    (out_session / "segcrop_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    removed = 0
    for f in sorted(out_session.rglob("*"), reverse=True):   # files first, then the folders they leave empty
        if f.is_file() and f.relative_to(out_session).as_posix() not in want:
            f.unlink()
            removed += 1
        elif f.is_dir() and not any(f.iterdir()):
            f.rmdir()
    print(f"{session}: {how}: {done} photos cropped, {kept} kept, {removed} stale files removed")
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
    ap.add_argument("--full", action="store_true", help="rebuild the session's crops from nothing (ticket ML-26)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    segcrop_session(args.session, args.out_root, args.limit_per_class, args.full)


if __name__ == "__main__":
    main()
