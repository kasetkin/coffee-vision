"""ML-37 D6: how close each read-back option comes to today's models' input, on gain-map photos.

    cd /workspace && OMP_NUM_THREADS=2 PYTHONPATH=/workspace uv run python analysis/ml37/d6_readback_options.py

Research only: reads dataset/ and data/ (never writes there), writes tmp/ml37/d6/readback_rows.json and prints the
tables kept in d6_readback_options.txt.

The sample: 24 gain-map photos from ML-34's experiment A (tmp/ml34_a/a_light_above_white.csv; photos with a bean
mask in data/seg_masks_ml19/xl0_v7r_s123), spread per camera over the share of bean pixels above the EETF's knee
(the median, the most and the spread between), at most 24.5 MP so that one photo stays under ~700 MB; plus 6
standard-range photos (no gain map) for option (c)'s effect on them.

For each photo the original is decoded as the store decodes it (canonical.decode, STORE_SETTINGS, gain map applied):
linear Rec.2020, 1.0 = reference white, light above white kept. That is D1's canonical photo before its lossy
coding, so the comparison isolates D6 (the lossy half-float coding is R1/R2's, and the same for every option).
Negative light is clipped at 0 first (D7 (a)). The options, all float32, as float 8-bit levels (x 255):

  a2020     EETF (BT.2408 Annex 5 on max(R, G, B), tone_map.eetf_in_place) in linear Rec.2020 at the picture's
            Rec.2020 peak (D33's, today's), then the 3x3 matrix to linear sRGB, clip [0, 1], sRGB curve. Today's maths
            with no 8-bit step and no lossy integer canonical photo.
  aSRGB     the matrix first, then the EETF on max(R, G, B) of linear sRGB at the peak measured there (99.99th
            percentile of the sRGB max, at least 1), clip, curve: the owner's recipe read literally.
  aSRGBp    as aSRGB but the peak measured in Rec.2020 (the ordering's effect on the max alone).
  b         the matrix, clip [0, 1], curve.
  c2, c4    a fixed roll-off: the same EETF on the sRGB max at a fixed peak of 2 or 4 x white, for every picture.

The reference T is today's models' input: canonical.load_canonical of the photo's canonical photo in data/canonical
(uint16 Rec.2020 with the BT.709 curve, lossy; LittleCMS to 8-bit sRGB). Metrics against T: ML-29 D27's 32 x 32 block
means (mean |d| over blocks and channels, worst block, signed shift), over the whole photo and over bean blocks (half
their pixels bean, as analysis/ml29/tone_map_measure.py). Per bean pixel, each option against a2020 (today's maths
without the lossy integer photo, so the codec's per-pixel noise is out; a2020 against T is that noise): mean |d|, 99th
percentile and largest of the largest channel's |d|, share of bean pixels with a channel off by more than 2 levels.
And for the ordering: quantiles of max(R, G, B) in linear sRGB over max(R, G, B) in linear Rec.2020 per bean pixel,
and the share of bean pixels that a2020 leaves above 1.0 in sRGB after its Rec.2020 EETF (clipped per channel, as
today's LittleCMS conversion clips them).
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import tone_map as tm

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tmp" / "ml37" / "d6"
MASKS = REPO / "data" / "seg_masks_ml19" / "xl0_v7r_s123"
STRIP = 256           # rows; a multiple of the 32-pixel block
BLOCK = 32
MAX_PIXELS = 24_500_000
THREADS = 2
OPTIONS = ("a2020", "aSRGB", "aSRGBp", "b", "c2", "c4")


def xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz(primaries) -> np.ndarray:
    white = xy_to_xyz(0.3127, 0.3290)
    p = np.stack([xy_to_xyz(*xy) for xy in primaries], axis=1)
    return p * np.linalg.solve(p, white)


REC2020_XYZ = rgb_to_xyz([(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)])     # BT.2020 Table 3
SRGB_XYZ = rgb_to_xyz([(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)])         # BT.709 / IEC 61966-2-1
M = np.linalg.solve(SRGB_XYZ, REC2020_XYZ).astype(np.float32)                    # linear Rec.2020 -> linear sRGB


def srgb_curve(x: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1's encoding on [0, 1], float32."""
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def levels(lin_srgb: np.ndarray) -> np.ndarray:
    np.clip(lin_srgb, 0, 1, out=lin_srgb)
    return (srgb_curve(lin_srgb) * 255).astype(np.float32)


def eetf(c: np.ndarray, peak: float) -> np.ndarray:
    c = c.copy()
    tm.eetf_in_place(c, peak)
    return c


def options(rec: np.ndarray, peak2020: float, peak_srgb: float) -> dict[str, np.ndarray]:
    s = rec @ M.T
    np.maximum(s, 0, out=s)    # negative sRGB (outside its gamut) is clipped before the max: the operator's own order
    return {"a2020": levels(eetf(rec, peak2020) @ M.T),
            "aSRGB": levels(eetf(s, peak_srgb)),
            "aSRGBp": levels(eetf(s, peak2020)),
            "b": levels(s.copy()),
            "c2": levels(eetf(s, 2.0)),
            "c4": levels(eetf(s, 4.0))}


def block_means(a: np.ndarray) -> np.ndarray:
    h, w = (a.shape[0] // BLOCK) * BLOCK, (a.shape[1] // BLOCK) * BLOCK
    return a[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, -1).mean((1, 3))


def pick_gainmap_photos() -> list[dict]:
    rows = [r for r in csv.DictReader((REPO / "tmp" / "ml34_a" / "a_light_above_white.csv").open())
            if not r["skipped"] and int(r["h"]) * int(r["w"]) <= MAX_PIXELS]
    quota = {"JPEG Google Pixel 9 Pro": 5, "JPEG GM1911 (OnePlus7Pro) OnePlus": 5,
             "JPEG samsung Galaxy Z Fold8 Ultra": 3, "HEIC Apple iPhone 15": 3, "HEIC Apple iPhone 15 Pro Max": 3,
             "HEIC Apple iPhone 16 Pro Max": 2, "JPEG Google Pixel 7": 1, "HEIC Apple iPhone 15 Pro": 1}
    out = []
    for cam, n in quota.items():
        rs = sorted((r for r in rows if r["camera"] == cam), key=lambda r: float(r["bean_above_knee"]))
        idx = sorted({round(q * (len(rs) - 1)) for q in np.linspace(0, 1, n + 2)[1:]})[-n:]
        if len(rs) - 1 not in idx:
            idx[-1] = len(rs) - 1
        out += [rs[i] | {"kind": "gainmap"} for i in idx]
    return out


def pick_sdr_photos(n: int = 6) -> list[dict]:
    rows = [r for r in csv.DictReader((REPO / "tmp" / "ml29_scan" / "files.csv").open())
            if not r["gain_map_detail"] and r["type"] in ("JPEG", "HEIC") and float(r["megapixels"] or 99) <= 24.5]
    rows = [r for r in rows if cn.canonical_path(Path(r["file"])).exists()]
    by_cam: dict[str, list] = {}
    for r in rows:
        by_cam.setdefault(f"{r['type']} {r['make']} {r['model']}", []).append(r)
    out = []
    for cam, rs in sorted(by_cam.items(), key=lambda kv: -len(kv[1])):
        out.append({"photo": rs[len(rs) // 2]["file"], "camera": cam, "kind": "sdr"})
        if len(out) == n:
            break
    return out


def measure(row: dict) -> dict:
    photo = Path(row["photo"])
    t0 = time.perf_counter()
    src = cn.decode(photo.read_bytes(), photo.name)
    t_decode = time.perf_counter() - t0
    if src.pixels.dtype != np.float32 or src.pixels.ndim != 3 or not src.tone_map:
        # a standard-range photo: its light in linear Rec.2020, as the new canonicalization would make it
        h, w = src.pixels.shape[:2]
        light = np.empty((h, w, 3), np.float32)
        for y in range(0, h, STRIP):
            _, f = cn._light(src.pixels[y:y + STRIP], src.full, src.linearize,
                             1 if src.pixels.ndim == 2 or src.pixels.shape[2] <= 2 else 3)
            light[y:y + STRIP] = cn.ic.cms_transform(f, src.profile or cn.SRGB, cn.LINEAR_REC2020,
                                                     intent=cn.RELATIVE, outdtype=np.float32)
    else:
        light = src.pixels
    del src
    np.maximum(light, 0, out=light)                                  # D7 (a)
    h, w = light.shape[:2]
    t0 = time.perf_counter()
    peak2020 = tm.picture_peak(light.max(axis=-1).ravel())           # the peak pass: one float32 per pixel
    t_peak = time.perf_counter() - t0
    smax = np.empty(h * w, np.float32)
    for y in range(0, h, STRIP):
        s = light[y:y + STRIP] @ M.T
        np.maximum(s, 0, out=s)
        smax[y * w:(y + s.shape[0]) * w] = s.max(axis=-1).ravel()
    peak_srgb = tm.picture_peak(smax)
    del smax
    rec_today = cn.decode_canonical(cn.canonical_path(photo).read_bytes(), numthreads=THREADS)
    if rec_today.shape[:2] != (h, w):
        return {**row, "skipped": f"shape {rec_today.shape[:2]} != {(h, w)}"}
    mask = None
    mask_path = MASKS / ("__".join(Path(str(photo).split("dataset/")[1]).with_suffix("").parts) + ".png")
    if mask_path.exists():
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert("L").resize((w, h), Image.NEAREST)) > 127
    hb, wb = h // BLOCK, w // BLOCK
    blk = {k: np.zeros((hb, wb, 3), np.float64) for k in ("T",) + OPTIONS}
    bean_frac = np.zeros((hb, wb), np.float64)
    hist_edges = np.arange(0, 256.05, 0.1)
    acc = {k: {"sum": 0.0, "n": 0, "over2": 0, "hist": np.zeros(len(hist_edges) - 1)}
           for k in ("a2020_vs_T",) + OPTIONS[1:]}
    ratio_edges = np.arange(0.5, 3.0001, 0.01)
    ratio_hist = np.zeros(len(ratio_edges) - 1)
    clipped_2020 = 0
    knees = {"c2": tm.eetf_knee(2.0), "c4": tm.eetf_knee(4.0)}
    above = {k: 0 for k in knees} | {"white": 0}
    t_read = 0.0
    for y in range(0, h, STRIP):
        rec = light[y:y + STRIP]
        today = cn.rec2020_to_srgb(rec_today[y:y + STRIP], numthreads=1).astype(np.float32)
        t0 = time.perf_counter()
        opts = options(rec, peak2020, peak_srgb)
        t_read += time.perf_counter() - t0
        rows_in_blocks = (min(y + STRIP, hb * BLOCK) - y)
        if rows_in_blocks > 0:
            b0 = y // BLOCK
            for k, a in (("T", today), *opts.items()):
                blk[k][b0:b0 + rows_in_blocks // BLOCK] = block_means(a[:rows_in_blocks])
            if mask is not None:
                bean_frac[b0:b0 + rows_in_blocks // BLOCK] = block_means(
                    mask[y:y + rows_in_blocks, :, None].astype(np.float32))[..., 0]
        m = mask[y:y + STRIP] if mask is not None else np.ones(rec.shape[:2], bool)
        if not m.any():
            continue
        s = rec[m] @ M.T
        np.maximum(s, 0, out=s)
        smax, rmax = s.max(-1), rec[m].max(-1)
        lit = rmax > 0.01
        ratio_hist += np.histogram(np.clip(smax[lit] / rmax[lit], 0.5, 2.999), ratio_edges)[0]
        clipped_2020 += int(((eetf(rec[m], peak2020) @ M.T).max(-1) > 1).sum())
        for k in knees:
            above[k] += int((smax > knees[k]).sum())
        above["white"] += int((smax > 1).sum())
        ref2020 = opts["a2020"][m]
        pairs = [("a2020_vs_T", ref2020, today[m])] + [(k, opts[k][m], ref2020) for k in OPTIONS[1:]]
        for k, got, ref in pairs:
            d = np.abs(got - ref)
            acc[k]["sum"] += float(d.sum())
            acc[k]["n"] += d.size
            dm = d.max(-1)
            acc[k]["over2"] += int((dm > 2).sum())
            acc[k]["hist"] += np.histogram(np.minimum(dm, 255.9), hist_edges)[0]
    del light, rec_today
    out = {**row, "skipped": "", "h": h, "w": w, "mask": mask is not None, "peak2020": peak2020,
           "peak_srgb": peak_srgb, "t_decode_s": t_decode, "t_peak_s": t_peak, "t_readback_s": t_read}
    beans = np.repeat((bean_frac >= 0.5)[..., None], 3, axis=2)
    for k in OPTIONS:
        d = blk[k] - blk["T"]
        out[f"{k}_all_mean"], out[f"{k}_all_worst"], out[f"{k}_all_shift"] = (
            float(np.abs(d).mean()), float(np.abs(d).max()), float(d.mean()))
        if mask is not None and beans.any():
            db = d[beans]
            out[f"{k}_bblk_mean"], out[f"{k}_bblk_worst"], out[f"{k}_bblk_shift"] = (
                float(np.abs(db).mean()), float(np.abs(db).max()), float(db.mean()))
    for k, a in acc.items():
        if a["n"]:
            cdf = np.cumsum(a["hist"]) / a["hist"].sum()
            out[f"{k}_px_mean"] = a["sum"] / a["n"]
            out[f"{k}_px_p99"] = float(hist_edges[1:][np.searchsorted(cdf, 0.99)])
            out[f"{k}_px_max"] = float(hist_edges[1:][np.nonzero(a["hist"])[0].max()])
            out[f"{k}_px_over2"] = a["over2"] / (a["n"] / 3)
    npx = acc["b"]["n"] / 3
    rc = np.cumsum(ratio_hist) / max(ratio_hist.sum(), 1)
    for q in (0.05, 0.5, 0.95):
        out[f"srgb_over_2020_max_q{int(q * 100)}"] = float(ratio_edges[1:][np.searchsorted(rc, q)])
    out["region_a2020_srgb_clipped"] = clipped_2020 / max(npx, 1)
    for k in above:
        out[f"region_above_{k}"] = above[k] / max(npx, 1)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = pick_gainmap_photos() + pick_sdr_photos()
    print(f"rec2020->srgb matrix:\n{M}", flush=True)
    rows = []
    for i, r in enumerate(jobs):
        row = {k: r[k] for k in ("photo", "camera", "kind")}
        try:
            res = measure(row)
        except Exception as exc:            # noqa: BLE001 - a research run records and goes on
            res = {**row, "skipped": f"{type(exc).__name__}: {exc}"}
        rows.append(res)
        print(f"{i + 1}/{len(jobs)} {res['camera']} {res.get('skipped') or ''} "
              f"peak {res.get('peak2020', 0):.3f}/{res.get('peak_srgb', 0):.3f}", flush=True)
        (OUT / "readback_rows.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
