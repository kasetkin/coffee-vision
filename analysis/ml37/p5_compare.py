"""ML-37 P5's checks: the store re-canonicalized (D1) against the store before it, R1 (D20), D13's masks and P2's input
check repeated (docs/ml-0037_plan_linear_canonical_photo.html, P5).

    python analysis/ml37/p5_compare.py keep               # before the re-run: today's store kept, by hard link
    python analysis/ml37/p5_compare.py before [--jobs 4]  # before or after it: today's side and the reference
    python analysis/ml37/p5_compare.py after  [--jobs 4]  # after it: the new side
    python analysis/ml37/p5_compare.py report             # -> analysis/ml37/p5_report.txt

keep    Every canonical photo dvc.lock records for a canonicalize stage, hard-linked from its DVC cache object (read
        only, never written in place) into tmp/ml37/p5/before_store/<folder>/..., its md5 checked against the listing
        and the workspace's file. The store's re-run deletes its folders (no incremental keep after a code change).
before  Per photo, from the kept file and the original (tmp/ml37/p5/photos/<path>.before.npz):
          today's model picture, the form before ML-37 D1 read as it was read before P3 (4b0a9bc's rec2020_to_srgb:
            LittleCMS from rec2020_bt709.icc to sRGB, uint8, no optimization): its 32 x 32 block means (ML-29 D27);
          today's mask: the live segmenter (params.yaml's, xl0_v7r_s123, box prompt, multi3; torch at 2 threads, as
            P2's check) on today's model picture with today's 8-bit sam_input (P2's today_sam_input);
          the uncoded reference (ML-37 D20): the original through the store's colour path and tone map as built
            (coffeecv.canonical.decode, gain map applied; to_rec2020's steps without its rounding to 16 bits), read
            back by coffeecv.readback's maths before rounding: float sRGB levels; its block means;
          the camera (EXIF make and model: tmp/ml29_p8/scan/files.csv, else the file's own EXIF), whether a gain map
            was applied, the kept file's size.
        The kept file's header must say the form before (transfer BT709): a photo already linear is refused.
after   Per photo, from data/canonical (tmp/ml37/p5/photos/<path>.after.npz): the new model picture
        (canonical.load_canonical, which refuses anything but D1's form) and its block means; the mask with the input
        as built (float sam_input, P2); the file's size and header.
report  R1: each form against the reference, per photo the mean |block difference| over blocks and channels and the
        largest; per camera, gain-map photos apart; D27 (mean at most 1.0, no block over 6) per photo, and the stop:
        a camera group where the new form breaks D27 and today's does not. The new form against today's, no bar. File
        sizes. D13: mask IoU, the median per camera at least 0.99. P2's 13 photos: the new inputs (new store, float
        resizes) against today's (today's store, 8-bit resizes), within 2.02 levels (D11's research).

Reads dataset/, data/canonical and models_pretrained (not in git); writes tmp/ml37/p5/ and the report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

STORE = REPO / "data" / "canonical"
DATASET = REPO / "dataset"
WORK = REPO / "tmp" / "ml37" / "p5"
KEPT = WORK / "before_store"
PHOTOS = WORK / "photos"
SCAN = REPO / "tmp" / "ml29_p8" / "scan" / "files.csv"
OUT = Path(__file__).with_name("p5_report.txt")
BLOCK = 32
THREADS = 2                   # torch's, and every decoder's, per worker (P2's check ran at 2)
SEGMENTER = "xl0_v7r_s123"    # params.yaml's seg_decoder (checked below)
D27_MEAN, D27_BLOCK = 1.0, 6.0
D13_IOU = 0.99
P2_BOUND = 2.02


# ----- which photos -----

def store_photos() -> list[Path]:
    """Every canonical photo, as a path under the store (the same under the kept store)."""
    return sorted(p.relative_to(STORE) for p in STORE.rglob("*.jxl"))


def original(rel: Path) -> Path:
    return DATASET / rel.parent / rel.name[:-len(".jxl")]


def record(rel: Path, side: str) -> Path:
    return PHOTOS / rel.parent / f"{rel.name}.{side}.npz"


# ----- keep -----

def run_keep(_: argparse.Namespace) -> None:
    lock = yaml.safe_load((REPO / "dvc.lock").read_text())["stages"]
    cache = REPO / ".dvc" / "cache" / "files" / "md5"
    n = 0
    for stage, entry in sorted(lock.items()):
        if not stage.startswith("canonicalize@"):
            continue
        (out,) = entry["outs"]
        listing = json.loads((cache / out["md5"][:2] / out["md5"][2:]).read_text())
        for e in listing:
            obj = cache / e["md5"][:2] / e["md5"][2:]
            target = KEPT / Path(out["path"]).relative_to("data/canonical") / e["relpath"]
            live = REPO / out["path"] / e["relpath"]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(obj, target)
            got = hashlib.md5(target.read_bytes()).hexdigest()
            if got != e["md5"] or hashlib.md5(live.read_bytes()).hexdigest() != e["md5"]:
                raise SystemExit(f"{target}: md5 {got}, the listing's {e['md5']}, or the workspace's file differs")
            n += 1
    print(f"{n} canonical photos kept under {KEPT}, each md5 as dvc.lock lists it and as the workspace holds it")


# ----- today's form, read as before P3 -----

def before_model_picture(data: bytes) -> np.ndarray:
    """A canonical photo of the form before ML-37 D1 as every model read it until P3 (4b0a9bc's load_canonical)."""
    import imagecodecs as ic
    from coffeecv.jxl_header import read_header
    h = read_header(data)
    if h.colour.transfer != "BT709":
        raise ValueError(f"transfer {h.colour.transfer}: not the form before ML-37 D1")
    profile = (REPO / "coffeecv" / "data" / "rec2020_bt709.icc").read_bytes()
    srgb = ic.cms_profile("srgb")
    rec = ic.jpegxl_decode(data, numthreads=THREADS)
    out = np.empty(rec.shape, np.uint8)
    for y in range(0, rec.shape[0], 128):
        ic.cms_transform(rec[y:y + 128], profile, srgb, intent=ic.CMS.INTENT.RELATIVE_COLORIMETRIC,
                         flags=ic.CMS.FLAGS.NOOPTIMIZE, outdtype=np.uint8, out=out[y:y + 128])
    return out


def blocks(a: np.ndarray) -> np.ndarray:
    """ML-29 D27's 32 x 32 block means (analysis/ml29/exports_tolerance.blocks), float64."""
    h, w = (a.shape[0] // BLOCK) * BLOCK, (a.shape[1] // BLOCK) * BLOCK
    return a[:h, :w].astype(np.float64).reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, 3).mean((1, 3))


# ----- the uncoded reference (D20) -----

def reference_blocks(path: Path) -> tuple[np.ndarray, tuple[int, int], bool, float]:
    """The original through the store's colour path and tone map, not encoded and not rounded: its float sRGB levels'
    block means, the picture's size, whether a gain map was applied, and the largest difference of its block means
    from the same picture rounded to 16 bits as to_rec2020 writes it (a check on this copy of to_rec2020's steps)."""
    import imagecodecs as ic
    from PIL import Image
    from coffeecv import canonical as cn
    from coffeecv import readback
    from coffeecv import tone_map as tm
    from coffeecv.canonicalize_folder import DATASET_MAX_PIXELS
    Image.MAX_IMAGE_PIXELS = DATASET_MAX_PIXELS
    data = path.read_bytes()
    source = cn.decode(data, path.name)
    pixels, full = source.pixels, source.full
    h, w = pixels.shape[:2]
    peak = source.peak()
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    alpha_at = {2: 1, 4: 3}.get(channels)
    colour = 1 if channels <= 2 else 3
    fill = None
    if alpha_at is not None:
        fill = ic.cms_transform(np.array([[cn.FILL_SRGB]], np.float32) / 255, cn.SRGB, cn.LINEAR_REC2020,
                                intent=cn.RELATIVE, outdtype=np.float32)[0, 0]
    rows = []
    worst16 = 0.0
    strip = cn.STRIP_ROWS                     # 256, a multiple of BLOCK: the strips' blocks are the picture's
    for y in range(0, (h // BLOCK) * BLOCK, strip):
        part, f = cn._light(pixels[y:y + strip], full, source.linearize, colour)
        c = ic.cms_transform(f, source.profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE,
                             outdtype=np.float32)
        if peak is not None and peak > 1:
            tm.eetf_in_place(c, peak)
        if part.dtype.kind == "f":
            np.nan_to_num(c, copy=False, nan=0.0)
        np.clip(c, 0, 1, out=c)
        if fill is not None:
            a = part[..., alpha_at:alpha_at + 1].astype(np.float32) / np.float32(full)
            c = c * a + fill * (1 - a)
        c *= 65535                            # srgb_levels takes samples scaled to 65535; float, so not rounded
        n = (c.shape[0] // BLOCK) * BLOCK
        b = blocks(readback.srgb_levels(c[:n]))
        b16 = blocks(readback.srgb_levels(np.rint(c[:n]).astype(np.uint16)))
        worst16 = max(worst16, float(np.abs(b - b16).max()))
        rows.append(b)
    return np.concatenate(rows).astype(np.float32), (h, w), "+" in source.decoder, worst16


# ----- the camera -----

_SCAN: dict[str, dict] | None = None


def camera(path: Path) -> str:
    global _SCAN
    if _SCAN is None:
        _SCAN = {r["file"]: r for r in csv.DictReader(SCAN.open())} if SCAN.exists() else {}
    r = _SCAN.get(str(path))
    if r is None:
        try:
            from PIL import Image
            exif = Image.open(path).getexif()
            r = {"make": str(exif.get(0x010F, "")).strip(), "model": str(exif.get(0x0110, "")).strip()}
        except Exception:
            r = {}
    return " ".join(x for x in (r.get("make", ""), r.get("model", "")) if x) or "unknown"


# ----- masks -----

_SEG = None


def segmenter():
    global _SEG
    if _SEG is None:
        import torch
        from coffeecv.segment_beans import BeanSegmenter, named_params
        torch.set_num_threads(THREADS)
        _SEG = BeanSegmenter(named_params(SEGMENTER))
    return _SEG


def mask_today(rgb: np.ndarray) -> np.ndarray:
    """The live segmenter with today's 8-bit sam_input (before ML-37 P2)."""
    from coffeecv import model_io
    from p2_inputs_and_masks import today_sam_input
    with mock.patch.object(model_io, "sam_input", today_sam_input):
        return segmenter().predict_mask(rgb)


def packed(mask: np.ndarray) -> dict:
    return {"mask": np.packbits(mask.astype(bool)), "mask_shape": np.array(mask.shape)}


def unpacked(z) -> np.ndarray:
    shape = tuple(int(v) for v in z["mask_shape"])
    return np.unpackbits(z["mask"], count=shape[0] * shape[1]).reshape(shape).astype(bool)


# ----- before / after -----

def before_one(rel: Path) -> str:
    out = record(rel, "before")
    if out.exists():
        return ""
    try:
        data = (KEPT / rel).read_bytes()
        rgb = before_model_picture(data)
        ref, (h, w), gain, worst16 = reference_blocks(original(rel))
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"today's photo {rgb.shape[:2]}, the original as decoded {(h, w)}")
        mask = mask_today(rgb)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez_compressed(tmp, blocks=blocks(rgb).astype(np.float32), ref=ref, gain_map=gain, worst16=worst16,
                            camera=camera(original(rel)), size=len(data), **packed(mask))
        os.replace(tmp, out)
        return ""
    except Exception as exc:
        return f"{rel}: {type(exc).__name__}: {exc}"


def after_one(rel: Path) -> str:
    out = record(rel, "after")
    if out.exists():
        return ""
    try:
        from coffeecv import canonical as cn
        from coffeecv.jxl_header import read_header
        data = (STORE / rel).read_bytes()
        h = read_header(data)
        rgb = cn.load_canonical(data, numthreads=THREADS)
        mask = segmenter().predict_mask(rgb)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez_compressed(tmp, blocks=blocks(rgb).astype(np.float32), size=len(data),
                            header=f"{h.colour.primaries} {h.colour.transfer} {h.bits_per_sample}-bit "
                                   f"{'float' if h.float_sample else 'integer'} "
                                   f"{'lossy' if h.xyb_encoded else 'lossless'} {h.intensity_target:g}",
                            **packed(mask))
        os.replace(tmp, out)
        return ""
    except Exception as exc:
        return f"{rel}: {type(exc).__name__}: {exc}"


def run_pool(fn, a: argparse.Namespace) -> None:
    seg = yaml.safe_load((REPO / "params.yaml").read_text())["seg_decoder"]
    if Path(seg).stem != SEGMENTER:
        raise SystemExit(f"params.yaml's seg_decoder is {seg}, not {SEGMENTER}")
    todo = [r for r in store_photos() if not record(r, "before" if fn is before_one else "after").exists()]
    todo = todo[:a.limit] if a.limit else todo
    print(f"{len(todo)} photos to do, {a.jobs} jobs", flush=True)
    errors = []
    with ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, err in enumerate(pool.map(fn, todo, chunksize=1), 1):
            if err:
                errors.append(err)
                print(err, file=sys.stderr, flush=True)
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done, {len(errors)} errors", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} photos failed")


# ----- P2's 13 photos -----

def p2_check() -> list[str]:
    """P2's check (analysis/ml37/p2_inputs_and_masks.py) repeated across the re-run: its 13 photos and patch draws,
    the new inputs (the new store read back, the float resizes as built) against today's (today's store read as
    before P3, the 8-bit resizes before P2)."""
    from PIL import Image
    from coffeecv import canonical as cn
    from coffeecv import model_io
    import d11_resize_parity as d11
    import p2_inputs_and_masks as p2
    d11.REPO = REPO
    rng = np.random.default_rng(37011)
    lines = [f"{'folder':<38}{'size':>11}{'seg max':>9}{'seg mean':>9}{'dino max':>9}{'dino mean':>10}"]
    seg_max = dino_max = 0.0
    seg_means, dino_means = [], []
    photos = d11.sample_photos()
    assert len(photos) == d11.N_PHOTOS
    for path in photos:
        rel = path.relative_to(STORE)
        new = cn.load_canonical(path.read_bytes())
        old = before_model_picture((KEPT / rel).read_bytes())
        h, w = new.shape[:2]
        nh, nw = model_io.sam_preprocess_shape(h, w, p2.SIDE)
        today = np.asarray(Image.fromarray(old).resize((nw, nh), Image.BILINEAR)).astype(np.float32)
        d = np.abs(model_io.sam_resize(new, p2.SIDE) * 255 - today)
        s_max, s_mean = float(d.max()), float(d.mean())
        dd = []
        for _ in range(p2.PATCHES):
            side = int(round(np.exp(rng.uniform(np.log(224), np.log(min(1100, h, w))))))
            y, x = int(rng.integers(0, h - side + 1)), int(rng.integers(0, w - side + 1))
            p_new = Image.fromarray(new[y:y + side, x:x + side]).resize((p2.STORE, p2.STORE), Image.BILINEAR)
            p_old = Image.fromarray(old[y:y + side, x:x + side]).resize((p2.STORE, p2.STORE), Image.BILINEAR)
            o = np.asarray(p_old.resize((p2.RESIZE, p2.RESIZE), Image.BILINEAR)).astype(np.float32)
            dd.append(np.abs(model_io.eval_resize(p_new, p2.RESIZE) * 255 - o))
        dd = np.stack(dd)
        p_max, p_mean = float(dd.max()), float(dd.mean())
        lines.append(f"{rel.parts[0]:<38}{f'{w}x{h}':>11}{s_max:>9.3f}{s_mean:>9.4f}{p_max:>9.3f}{p_mean:>10.4f}")
        print(lines[-1], file=sys.stderr, flush=True)
        seg_max, dino_max = max(seg_max, s_max), max(dino_max, p_max)
        seg_means.append(s_mean)
        dino_means.append(p_mean)
    return lines + ["", f"segmenter input: max {seg_max:.3f} levels, mean {np.mean(seg_means):.4f}",
                    f"DINOv3 input:    max {dino_max:.3f} levels, mean {np.mean(dino_means):.4f}",
                    f"within {P2_BOUND} levels (D11's research bound, P2's check): "
                    f"{max(seg_max, dino_max) <= P2_BOUND}"]


# ----- report -----

def diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.mean()), float(d.max())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def breaks(mean: float, largest: float) -> bool:
    return mean > D27_MEAN or largest > D27_BLOCK


def run_report(a: argparse.Namespace) -> None:
    rows = []
    for rel in store_photos():
        zb, za = np.load(record(rel, "before")), np.load(record(rel, "after"))
        r = {"rel": rel, "camera": str(zb["camera"]), "gain": bool(zb["gain_map"]),
             "old_size": int(zb["size"]), "new_size": int(za["size"]), "header": str(za["header"]),
             "worst16": float(zb["worst16"])}
        r["old_mean"], r["old_max"] = diff(zb["blocks"], zb["ref"])
        r["new_mean"], r["new_max"] = diff(za["blocks"], zb["ref"])
        r["nvo_mean"], r["nvo_max"] = diff(za["blocks"], zb["blocks"])
        r["iou"] = iou(unpacked(zb), unpacked(za))
        r["old_breaks"], r["new_breaks"] = breaks(r["old_mean"], r["old_max"]), breaks(r["new_mean"], r["new_max"])
        rows.append(r)

    groups: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["camera"], r["gain"])].append(r)
    order = sorted(groups, key=lambda k: (k[1], -len(groups[k]), k[0]))
    stops = [k for k in order if any(r["new_breaks"] for r in groups[k]) and not any(r["old_breaks"]
                                                                                       for r in groups[k])]
    cams: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cams[r["camera"]].append(r)
    low_iou = sorted(c for c, rs in cams.items() if np.median([r["iou"] for r in rs]) < D13_IOU)

    headers = sorted({r["header"] for r in rows})
    lines = [
        "ML-37 P5: the store re-canonicalized (D1) against the store before it. analysis/ml37/p5_compare.py",
        f"{len(rows)} canonical photos, every one in data/canonical; the store before it kept by hard link from the "
        "DVC cache (tmp/ml37/p5/before_store).",
        "",
        "Headers of the new store (primaries, transfer, sample, coding, intensity target): "
        + "; ".join(f"{h} ({sum(r['header'] == h for r in rows)})" for h in headers),
        "",
        "R1 (ML-37 D20). Each form's model picture against the uncoded reference: the original through the store's "
        "colour path and tone map as built, not encoded, read back by the read-back's maths before rounding. ML-29 "
        "D27's measure, 32 x 32 block means, |difference| in 8-bit levels: per photo 'mean' over blocks and channels, "
        "'max' the largest block. A photo breaks D27 with a mean over 1.0 or a block over 6. 'today' is the store "
        "before (BT.709 curve, intensity target 255, read through rec2020_bt709.icc as before P3), 'new' is D1's "
        "(linear, 203, read by coffeecv.readback). By camera, gain-map photos apart (the gain map applied and "
        "tone-mapped, ML-29 D33).",
        f"The reference's own 16-bit rounding (to_rec2020's) moves a block mean by at most "
        f"{max(r['worst16'] for r in rows):.4f} level on any photo.",
        "",
        f"{'camera':<36}{'gain':>5}{'n':>6}{'today mean':>11}{'(max)':>7}{'today blk':>10}{'brk':>5}"
        f"{'new mean':>10}{'(max)':>7}{'new blk':>9}{'brk':>5}{'new-today':>10}{'blk':>7}",
    ]
    for k in order:
        rs = groups[k]
        lines.append(
            f"{k[0][:35]:<36}{'yes' if k[1] else 'no':>5}{len(rs):>6}"
            f"{np.mean([r['old_mean'] for r in rs]):>11.3f}{max(r['old_mean'] for r in rs):>7.3f}"
            f"{max(r['old_max'] for r in rs):>10.2f}{sum(r['old_breaks'] for r in rs):>5}"
            f"{np.mean([r['new_mean'] for r in rs]):>10.3f}{max(r['new_mean'] for r in rs):>7.3f}"
            f"{max(r['new_max'] for r in rs):>9.2f}{sum(r['new_breaks'] for r in rs):>5}"
            f"{np.mean([r['nvo_mean'] for r in rs]):>10.3f}{max(r['nvo_max'] for r in rs):>7.2f}")
    allr = rows
    lines += [
        f"{'every photo':<36}{'':>5}{len(allr):>6}"
        f"{np.mean([r['old_mean'] for r in allr]):>11.3f}{max(r['old_mean'] for r in allr):>7.3f}"
        f"{max(r['old_max'] for r in allr):>10.2f}{sum(r['old_breaks'] for r in allr):>5}"
        f"{np.mean([r['new_mean'] for r in allr]):>10.3f}{max(r['new_mean'] for r in allr):>7.3f}"
        f"{max(r['new_max'] for r in allr):>9.2f}{sum(r['new_breaks'] for r in allr):>5}"
        f"{np.mean([r['nvo_mean'] for r in allr]):>10.3f}{max(r['nvo_max'] for r in allr):>7.2f}",
        "",
        "Columns: 'mean' the mean of the photos' means, '(max)' the largest photo mean, 'blk' the largest block, "
        "'brk' photos breaking D27; 'new-today' the new form against today's, no bar (D20).",
        "",
        "D20's stop (the new form breaks D27 on a camera group where today's does not): "
        + ("none" if not stops else "STOP: " + "; ".join(f"{c} (gain map {'yes' if g else 'no'})" for c, g in stops)),
    ]
    broken = sorted((r for r in rows if r["old_breaks"] or r["new_breaks"]), key=lambda r: str(r["rel"]))
    if broken:
        lines += ["", "Photos breaking D27 in either form (today mean/max, new mean/max):"]
        lines += [f"  {r['rel']}  ({r['camera']}{', gain map' if r['gain'] else ''}): "
                  f"{r['old_mean']:.3f}/{r['old_max']:.2f}, {r['new_mean']:.3f}/{r['new_max']:.2f}" for r in broken]

    lines += ["", "File sizes, the store's setting (quality 95, effort 7), new against today's:",
              f"{'camera':<36}{'gain':>5}{'n':>6}{'today MB':>10}{'new MB':>10}{'change':>9}"]
    for k in order:
        rs = groups[k]
        o, n = sum(r["old_size"] for r in rs), sum(r["new_size"] for r in rs)
        lines.append(f"{k[0][:35]:<36}{'yes' if k[1] else 'no':>5}{len(rs):>6}{o / 2**20:>10.1f}{n / 2**20:>10.1f}"
                     f"{(n / o - 1) * 100:>8.1f}%")
    o, n = sum(r["old_size"] for r in rows), sum(r["new_size"] for r in rows)
    lines += [f"{'every photo':<36}{'':>5}{len(rows):>6}{o / 2**20:>10.1f}{n / 2**20:>10.1f}{(n / o - 1) * 100:>8.1f}%",
              "The front's setting (quality 90, effort 3) was measured in P3 (analysis/ml37/p3_front_cost.jsonl): "
              "files 6% smaller."]

    lines += ["", f"D13's mask check: the live segmenter ({SEGMENTER}, box prompt, multi3, torch at {THREADS} threads) "
              "on the new input (the new store read back, float sam_input as built) against today's (today's store "
              "read as before P3, 8-bit sam_input before P2). IoU per photo; the bar is a median of at least "
              f"{D13_IOU} per camera.",
              f"{'camera':<36}{'n':>6}{'median':>9}{'least':>9}{'<0.99':>7}{'<0.9':>6}"]
    for c in sorted(cams, key=lambda c: (-len(cams[c]), c)):
        v = [r["iou"] for r in cams[c]]
        lines.append(f"{c[:35]:<36}{len(v):>6}{np.median(v):>9.5f}{min(v):>9.5f}{sum(x < 0.99 for x in v):>7}"
                     f"{sum(x < 0.9 for x in v):>6}")
    v = [r["iou"] for r in rows]
    lines += [f"{'every photo':<36}{len(v):>6}{np.median(v):>9.5f}{min(v):>9.5f}{sum(x < 0.99 for x in v):>7}"
              f"{sum(x < 0.9 for x in v):>6}",
              "D13's stop (a camera's median under 0.99): " + ("none" if not low_iou else "STOP: " + "; ".join(low_iou))]
    worst = sorted(rows, key=lambda r: r["iou"])[:10]
    lines += ["The ten lowest IoUs:"] + [f"  {r['iou']:.4f}  {r['rel']} ({r['camera']})" for r in worst]

    lines += ["", "P2's check repeated across the re-run (D11's 13 photos, P2's patch draws): the models' inputs from "
              "the new store with the float resizes against today's (today's store, 8-bit resizes), "
              "|float x 255 - today's uint8| in levels.", *p2_check()]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["keep", "before", "after", "report"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="at most this many photos (a toy run)")
    a = ap.parse_args()
    if a.step == "keep":
        run_keep(a)
    elif a.step == "before":
        run_pool(before_one, a)
    elif a.step == "after":
        run_pool(after_one, a)
    else:
        run_report(a)


if __name__ == "__main__":
    main()
