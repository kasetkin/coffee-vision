"""ML-34 experiment B: does the frozen DINOv3 head read PU21-encoded crops as well as today's sRGB ones?

    python -I analysis/ml34/b_pu21_probe.py /workspace build --arm pu21_sdr      # a scratch pool of re-encoded crops
    python -I analysis/ml34/b_pu21_probe.py /workspace probe --arm base pu21_sdr --seeds 42 123 7
    python -I analysis/ml34/b_pu21_probe.py /workspace report

A diagnostic, not a recorded run (docs/agents/experiments.md): the head is fitted unaugmented only (no views), by
fit_frozen_head's own pieces (frozen_config, build_fold_datasets, embed, fit_arm at C = 0.1), so each arm and seed
costs one embedding of the ~9,200 patches. The arms differ in their crops alone: each scratch pool is
data/segcropped/pool with every `__cropped.jxl` re-encoded by the arm's curve, masks linked, the same names, so the
same split and the same patch boxes.

Arms (`ENCODERS`): every crop's bean pixels, 8-bit sRGB, decoded to linear light, put on an absolute scale (white =
`WHITE` cd/m2), PU21-encoded per channel, and scaled to 8 bits so that `top` cd/m2 is 255. Pixels outside the mask
keep the fill grey (the fill is applied in the model's input space, as an HDR pipeline would apply it).

- base:            today's crops (the pool's own files, linked).
- pu21_sdr:        top = white: PU21 in place of the sRGB curve, no headroom kept. Is the curve itself a cost?
- pu21_h4:         top = 4 x white (2 stops of headroom kept, the canonical photo's light above white would land in
                   the top of the range): the cost of reserving headroom for an HDR input to an SDR-trained backbone.
- pu21_h4_hdr:     as pu21_h4, but each crop of a photo with an applied gain map is cut, at its segcrop_report box,
                   from the original's light with the map applied and no tone map (build_hdr, _one_hdr).
- pu21_h4_tm:      the control for pu21_h4_hdr: the same originals and boxes, tone-mapped by D33's EETF as the store
                   does, then pu21_h4's curve.
- pu21_pmax_hdr:   as pu21_h4_hdr with top = 100 x white (PU21 of 10,000 cd/m2 = 255): Chubarau et al.'s full-range
                   scaling (research_pu21_papers.md).

Writes outputs/ml34_b/<arm>/pool (scratch, gitignored), outputs/ml34_b/results/<arm>_s<seed>.json. The 2026-11-22/23
runs' results are kept in analysis/ml34/b_results/ and b_pu21_probe_report.txt (the scratch pools were deleted;
`build` remakes them from the tracked pool and the originals, research_hdr_pu21.md section 6).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "ml34_b"
TRACKED = REPO / "data" / "segcropped" / "pool"

# PU21, Mantiuk & Azimi 2021, the 'banding_glare' parameters (the authors' recommended set), from the reference
# implementation (github.com/gfxdisp/pu21, pu21_encoder.m): V = p7 * (((p1 + p2 Y^p4) / (1 + p3 Y^p4))^p5 - p6),
# Y in cd/m2 clamped to [0.005, 10000].
PU21_BANDING_GLARE = (0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204,
                      596.3148142)
Y_MIN, Y_MAX = 0.005, 10000.0
WHITE = 100.0          # cd/m2 of sRGB white: the display PU21 is anchored to for SDR content (see the research note)
BLACK = 0.0            # the display's black (cd/m2), added before PU21: 0 means PU21's own floor, Y_MIN
FILL = np.array((124, 116, 104), np.uint8)   # segment_beans.SegmentConfig.fill_rgb


def pu21(y: np.ndarray, p=PU21_BANDING_GLARE) -> np.ndarray:
    y = np.clip(y, Y_MIN, Y_MAX)
    yp = y ** p[3]
    return p[6] * (((p[0] + p[1] * yp) / (1 + p[2] * yp)) ** p[4] - p[5])


def srgb_to_linear(u: np.ndarray) -> np.ndarray:
    return np.where(u > 0.04045, ((u + 0.055) / 1.055) ** 2.4, u / 12.92)


ENCODERS = {"pu21_sdr": 1.0, "pu21_h4": 4.0}     # arm -> top, in multiples of white


def encode(rgb: np.ndarray, top: float) -> np.ndarray:
    lin = srgb_to_linear(rgb.astype(np.float32) / 255)
    lo, hi = pu21(np.float32(BLACK + 0)), pu21(np.float32(BLACK + WHITE * top))
    v = (pu21(BLACK + WHITE * lin) - lo) / (hi - lo)
    return np.clip(np.rint(v * 255), 0, 255).astype(np.uint8)


def write_fast(rgb: np.ndarray, path: Path) -> None:
    """crop_io.write_crop at effort 1: lossless, so the same pixels at any effort (crop_io's docstring), 5x faster."""
    import imagecodecs as ic
    data = ic.jpegxl_encode(np.ascontiguousarray(rgb), lossless=True, effort=1, numthreads=1)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _one(job: tuple[str, str, float]) -> str:
    from PIL import Image
    from coffeecv import crop_io
    src, dst, top = Path(job[0]), Path(job[1]), job[2]
    if dst.exists():
        return "kept"
    rgb = crop_io.load_crop(src)
    out = encode(rgb, top)
    m = crop_io.mask_path(src)
    if m.exists():
        with Image.open(m) as im:
            mask = np.asarray(im.convert("L")) > 127
        out[~mask] = FILL
    write_fast(out, dst)
    return "made"


# linear BT.2020 -> linear BT.709/sRGB primaries, both D65 (ITU-R BT.2087 inverse, as LittleCMS relative
# colorimetric computes it); out-of-gamut light is clipped at 0, as the read-back clips.
REC2020_TO_709 = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]],
                          np.float32)


def class_id(class_dir: str) -> str:
    """`class_013` of `class_013__Brazil_Decaff` (as analysis/ml29/p10_pool.py matches sessions to the pool)."""
    return class_dir.split("__")[0]


def session_index() -> dict[tuple[str, str], dict]:
    """(class id, crop file name) -> its segcrop_report entry, from every session's report."""
    idx = {}
    for rep in (REPO / "data" / "segcropped").glob("*/*/segcrop_report.json"):
        if rep.parts[-3] == "pool":
            continue
        for e in json.loads(rep.read_text()):
            out = Path(e["out_path"])
            idx[(class_id(rep.parent.name), out.name)] = e
    return idx


def _one_hdr(job: tuple[str, str, str, list, float]) -> dict:
    """A gain-map photo's crop from its HDR light: the original decoded as the store decodes it (map applied, no
    tone map), cut at the crop's box, PU21 as `encode` with white 1.0 = WHITE. Else (no map applied) as _one."""
    from PIL import Image
    from coffeecv import canonical as cn
    Image.MAX_IMAGE_PIXELS = None                    # as the store's stages run (a 200 MP original is in the pool)
    from coffeecv import crop_io
    src, dst, original, box, top = Path(job[0]), Path(job[1]), Path(job[2]), job[3], job[4]
    tone_map = len(job) > 5 and job[5]
    if dst.exists():
        return {"crop": str(src), "kind": "kept"}
    stored = crop_io.load_crop(src)
    m = crop_io.mask_path(src)
    mask = None
    if m.exists():
        with Image.open(m) as im:
            mask = np.asarray(im.convert("L")) > 127
    rec = {"crop": str(src), "kind": "sdr"}
    s = cn.decode(original.read_bytes(), original.name) if original.exists() and box else None
    if s is not None and s.tone_map:
        x0, y0, w, h = box
        light = np.array(s.pixels[y0:y0 + h, x0:x0 + w], np.float32)
        if tone_map:                                 # the control: D33's EETF at the whole picture's peak, as encode
            from coffeecv import tone_map as tm
            tm.eetf_in_place(light, s.peak())
        if light.shape[:2] != stored.shape[:2]:
            rec["kind"] = f"shape mismatch {light.shape[:2]} vs {stored.shape[:2]}"
        else:
            lin = np.clip(light @ REC2020_TO_709.T, 0, None)
            lo, hi = pu21(np.float32(BLACK)), pu21(np.float32(BLACK + WHITE * top))
            out = np.clip(np.rint((pu21(BLACK + WHITE * lin) - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
            # check: the HDR light rendered as standard range (clip at white, sRGB curve) against the stored crop,
            # on bean pixels the tone map leaves alone (max below 0.3 of white): only lossy coding should differ.
            sdr = np.clip(lin, 0, 1)
            sdr = np.where(sdr > 0.0031308, 1.055 * sdr ** (1 / 2.4) - 0.055, 12.92 * sdr) * 255
            calm = (lin.max(-1) < 0.3) & (mask if mask is not None else True)
            rec.update(kind="hdr", check_mean_abs=float(np.abs(sdr[calm] - stored[calm]).mean()) if calm.any() else None,
                       above_white=float((lin.max(-1)[mask] > 1).mean()) if mask is not None else None)
            if mask is not None:
                out[~mask] = FILL
            write_fast(out, dst)
            return rec
    out = encode(stored, top)
    if mask is not None:
        out[~mask] = FILL
    write_fast(out, dst)
    return rec


def build_hdr(arm: str, jobs: int) -> Path:
    top = HDR_ARMS[arm]
    pool = OUT / arm / "pool"
    idx = session_index()
    work = []
    for f in sorted(TRACKED.rglob("*")):
        if f.is_dir():
            continue
        dst = pool / f.relative_to(TRACKED)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.name.endswith("__cropped.jxl"):
            e = idx.get((class_id(f.parent.name), f.name))
            original, box = Path("/nonexistent"), None
            if e and Path(REPO / e["out_path"]).exists() and (REPO / e["out_path"]).stat().st_size == f.stat().st_size:
                c = Path(e["canonical"])                     # data/canonical/<session>/<class>/<file>.jxl
                original = REPO / "dataset" / Path(*c.parts[2:]).with_suffix("")
                box = e["box"]
            work.append((str(f), str(dst), str(original), box, top, arm in TONE_MAPPED))
        elif not dst.exists():
            os.symlink(f.resolve(), dst)
    t, recs = time.time(), []
    with ProcessPoolExecutor(jobs) as ex:
        for i, r in enumerate(ex.map(_one_hdr, work, chunksize=2)):
            recs.append(r)
            if i % 100 == 0:
                print(f"{arm}: {i}/{len(work)} ({time.time() - t:.0f}s)", flush=True)
    (OUT / arm / "build.json").write_text(json.dumps(recs, indent=0))
    kinds = {}
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    checks = [r["check_mean_abs"] for r in recs if r.get("check_mean_abs") is not None]
    print(arm, kinds, "check mean |hdr->sdr - stored| on calm bean pixels: median",
          f"{np.median(checks):.2f}, max {np.max(checks):.2f} levels" if checks else "-")
    return pool


HDR_ARMS = {"pu21_h4_hdr": 4.0, "pu21_h4_tm": 4.0, "pu21_pmax_hdr": 100.0}   # pmax: PU21(10,000 cd/m2) = 255, Chubarau et al.'s full-range scaling
TONE_MAPPED = {"pu21_h4_tm"}   # the control for pu21_h4_hdr: same decode, the light tone-mapped as the store does


def build(arm: str, jobs: int) -> Path:
    if arm in HDR_ARMS:
        return build_hdr(arm, jobs)
    pool = OUT / arm / "pool"
    work = []
    for f in sorted(TRACKED.rglob("*")):
        if f.is_dir():
            continue
        dst = pool / f.relative_to(TRACKED)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.name.endswith("__cropped.jxl") and arm != "base":
            work.append((str(f), str(dst), ENCODERS[arm]))
        elif not dst.exists():
            os.symlink(f.resolve(), dst)
    t = time.time()
    with ProcessPoolExecutor(jobs) as ex:
        for i, r in enumerate(ex.map(_one, work, chunksize=4)):
            if i % 250 == 0:
                print(f"{arm}: {i}/{len(work)} ({time.time() - t:.0f}s)", flush=True)
    return pool


def probe(arm: str, seed: int, threads: int) -> dict:
    import torch
    from dataclasses import asdict  # noqa: F401
    from coffeecv import fit_frozen_head as ffh
    from coffeecv.backbones import build_backbone
    from coffeecv.fold_data import build_fold_datasets
    from coffeecv.transforms import build_eval_transform
    torch.set_num_threads(threads)
    res_path = OUT / "results" / f"{arm}_s{seed}.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    pool = OUT / arm / "pool"
    bb = build_backbone("dinov3_vitb16")
    cfg = ffh.frozen_config(bb.name, seed, False, str(pool.relative_to(REPO)))
    ffh.set_seed(seed)
    tf = build_eval_transform(cfg.patch_resize)
    X, y, n = {}, {}, {}
    class_ids = class_labels = None
    t = time.time()
    for split in ("train", "val", "test"):
        fold = build_fold_datasets(cfg, tf, tf, only=(split,))
        ds = getattr(fold, split)
        X[split], y[split], ms = ffh.embed(ds, bb, "cls_mean")
        n[split] = len(ds)
        class_ids, class_labels = fold.class_ids, fold.class_labels
        print(f"{arm} s{seed}: {split} {len(ds)} patches at {ms:.1f} ms ({time.time() - t:.0f}s)", flush=True)
    feats = OUT / "features"
    feats.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feats / f"{arm}_s{seed}.npz", **{f"X_{s}": X[s] for s in X}, **{f"y_{s}": y[s] for s in y})
    fit = ffh.fit_arm(X["train"], y["train"], None, X, y, ffh.DEFAULT_C, class_ids, class_labels)
    rec = {"arm": arm, "seed": seed, "sizes": n, "class_ids": class_ids,
           "val_macro_f1": fit.record["splits"]["val"]["macro_f1"],
           "test_macro_f1": fit.record["splits"]["test"]["macro_f1"],
           "per_class_f1": {s: {c: v["f1"] for c, v in fit.record["splits"][s]["per_class"].items()}
                            for s in ("val", "test")},
           "seconds": round(time.time() - t)}
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("arm", "seed", "val_macro_f1", "test_macro_f1", "seconds")}), flush=True)
    return rec


def report() -> None:
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "results").glob("*.json"))]
    by = {(r["arm"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != "base", a))
    print("unaugmented head, C = 0.1, patch macro-F1 (val / test)")
    print(f"{'arm':12} " + " ".join(f"{'s' + str(s):>17}" for s in seeds) + "   paired vs base (val; test)")
    for arm in arms:
        cells, dv, dt = [], [], []
        for s in seeds:
            r = by.get((arm, s))
            cells.append(f"{r['val_macro_f1']:.4f} / {r['test_macro_f1']:.4f}" if r else f"{'-':>17}")
            b = by.get(("base", s))
            if r and b:
                dv.append(r["val_macro_f1"] - b["val_macro_f1"])
                dt.append(r["test_macro_f1"] - b["test_macro_f1"])
        d = (f"{np.mean(dv):+.4f} ({min(dv):+.4f}..{max(dv):+.4f}); {np.mean(dt):+.4f} ({min(dt):+.4f}..{max(dt):+.4f})"
             if dv and arm != "base" else "")
        print(f"{arm:12} " + " ".join(cells) + "   " + d)
    for arm in arms:
        if arm == "base":
            continue
        print(f"\nper class, {arm} - base, mean over seeds (test; val)")
        classes = by[(arm, seeds[0])]["per_class_f1"]["test"].keys() if (arm, seeds[0]) in by else []
        for c in classes:
            dt = [by[(arm, s)]["per_class_f1"]["test"][c] - by[("base", s)]["per_class_f1"]["test"][c]
                  for s in seeds if (arm, s) in by and ("base", s) in by]
            dv = [by[(arm, s)]["per_class_f1"]["val"][c] - by[("base", s)]["per_class_f1"]["val"][c]
                  for s in seeds if (arm, s) in by and ("base", s) in by]
            print(f"  {c:12} {np.mean(dt):+.4f}; {np.mean(dv):+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("cmd", choices=("build", "probe", "report"))
    ap.add_argument("--arm", nargs="+", default=["base"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "build":
        for arm in a.arm:
            build(arm, a.jobs)
    elif a.cmd == "probe":
        for seed in a.seeds:
            for arm in a.arm:
                probe(arm, seed, a.threads)
    else:
        report()


if __name__ == "__main__":
    main()
