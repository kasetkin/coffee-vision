"""ML-37: linear uint16 (I) against linear half-float (H) and today's uint16 BT.709 canonical photo, lossy JPEG XL.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/canon_int_measure.py [--n 20] [--scale 2]
        [--workers 3] [--out tmp/ml37/canon_int]

The photo sample, the decode into D1's light and the half-scale area average are D8's (d8_intensity_target.sample,
light_of: seed 0, stratified by camera and gain map), so the numbers line up with research_d8_intensity_target.md.
From that light (linear Rec.2020, 1.0 = white, light above white kept, negatives at 0) and its peak:

  sdr     = clip(EETF_peak(light), 0, 1): the picture tone-mapped at canonicalization, as today (ML-29 D33, ADR 0041)
  today   = d8.today_u16(light, peak): sdr through LittleCMS to coffeecv's REC2020 (BT.709 curve), rounded to uint16
  I       = rint(sdr * 65535) as uint16, linear transfer
  Hs      = sdr as float16, linear transfer (the same content as I)
  Hh      = light as float16, linear transfer, light above white kept (D1)

Each is encoded at the store's (q95, e7) and the front's (q90, e3) settings, at intensity target 255 (libjxl's default,
what today's file declares) and 203 (D8's recommendation), with the repo's chunked encoder's settings
(d8_intensity_target.encode, extended here to a linear uint16 form). Each file is decoded with
imagecodecs.jpegxl_decode (uint16 or float16, as the file says) and read back as D2 says, in float32: linear Rec.2020
-> (EETF at the source's peak, Hh only) -> matrix -> clip -> sRGB curve -> x 255; today's file through LittleCMS
REC2020 -> linear Rec.2020 first. Then rounded once to uint8: the models' input.

The reference is the uncompressed light through the same read-back, rounded to uint8 (one reference for every form;
read_back(light, peak) equals read_back(sdr, 1) up to float rounding). Measures on the uint8 result, in 8-bit levels:
pixel RMS; ML-29 D27's 32 x 32 block means (mean and largest absolute block difference, all channels); dark pixels
(reference luma < 0.1 of full scale, 25.5 levels): RMS; the darkest tenth of blocks by reference luma: mean block
difference; the share of uint8 values (pixel x channel) that differ from the reference, overall and in dark pixels.
Also the float block mean before rounding (D8's measure), and "uncoded" rows: each form's own rounding only.

Writes <out>/canon_int_rows.csv and prints the summary (kept as analysis/ml37/canon_int_measure.txt).
"""
from __future__ import annotations

import argparse
import csv
import os
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagecodecs as ic
import numpy as np

from analysis.ml37 import d8_intensity_target as d8
from coffeecv import jxl_encode as je
from coffeecv import jxl_header as jh
from coffeecv import tone_map as tm

THREADS = d8.THREADS     # 2, as D8
SETTINGS = [("store", 95, 7), ("front", 90, 3)]
TARGETS = (255.0, 203.0)
LUMA = d8.SRGB_LUMA.astype(np.float64)


def encode_linear_u16(a: np.ndarray, *, level: float, effort: int, intensity_target: float) -> bytes:
    """d8.encode's chunked encoder with JXL_TYPE_UINT16 samples and the linear transfer (the (I) form).

    d8.encode ties the transfer to the pixel type (float16 -> linear, uint16 -> BT.709); here the colour encoding is
    patched for the one call. Same libjxl calls otherwise: BT2100 primaries, D65, relative intent, no container,
    uses_original_profile 0, distance from quality, effort, frame header duration 1."""
    orig = je.JXL_TRANSFER_FUNCTION_709
    try:
        je.JXL_TRANSFER_FUNCTION_709 = d8.JXL_TRANSFER_FUNCTION_LINEAR
        return d8.encode(a, level=level, effort=effort, intensity_target=intensity_target, half=False)
    finally:
        je.JXL_TRANSFER_FUNCTION_709 = orig


def to8(levels: np.ndarray) -> np.ndarray:
    return np.rint(levels).astype(np.uint8)


def measures(got_f: np.ndarray, ref_f: np.ndarray, ref8: np.ndarray, dark_px: np.ndarray, dark_blk: np.ndarray,
             ref8_blocks: np.ndarray) -> dict:
    """got_f: float levels of the read-back (before rounding). Everything else is precomputed from the reference."""
    got8 = to8(got_f)
    diff = got8.astype(np.int16) - ref8.astype(np.int16)
    d = np.abs(d8.blocks(got8.astype(np.float64)) - ref8_blocks)
    dfl = np.abs(d8.blocks(got_f.astype(np.float64)) - d8.blocks(ref_f.astype(np.float64)))
    ne = diff != 0
    out = {"rms8": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
           "blk_mean8": float(d.mean()), "blk_max8": float(d.max()),
           "blk_mean_float": float(dfl.mean()), "blk_max_float": float(dfl.max()),
           "dark_blk10_mean8": float(d[dark_blk].mean()),
           "differ_share": float(ne.mean()),
           "dark_px_share": float(dark_px.mean())}
    if dark_px.any():
        dd = diff[dark_px].astype(np.float64)
        out["dark_rms8"] = float(np.sqrt(np.mean(dd ** 2)))
        out["dark_differ_share"] = float(ne[dark_px].mean())
        out["dark_bias8"] = float(dd.mean())
    else:
        out["dark_rms8"] = out["dark_differ_share"] = out["dark_bias8"] = float("nan")
    return out


def run_photo(args: tuple) -> list[dict]:
    k, p, scale = args
    t0 = time.time()
    light, mapped = d8.light_of(p["file"], scale)
    peak = tm.picture_peak(light.max(axis=-1).ravel().copy())
    ref_f = d8.read_back(light, peak)
    ref8 = to8(ref_f)
    ref8_blocks = d8.blocks(ref8.astype(np.float64))
    dark_px = (ref_f @ d8.SRGB_LUMA) < 25.5
    blk_luma = ref8_blocks @ LUMA
    dark_blk = blk_luma <= np.quantile(blk_luma, 0.10)

    sdr = light.copy()
    if peak > 1:
        for y in range(0, sdr.shape[0], 256):
            tm.eetf_in_place(sdr[y:y + 256], peak)
    np.clip(sdr, 0, 1, out=sdr)
    forms = {
        "today": d8.today_u16(light, peak),
        "I": np.rint(sdr * 65535).astype(np.uint16),
        "Hs": sdr.astype(np.float16),
        "Hh": light.astype(np.float16),
    }
    sdr = None

    def read(form: str, decoded: np.ndarray) -> np.ndarray:
        if form == "today":
            return d8.read_back(d8.u16_to_linear(decoded), 1.0)
        if form == "I":
            return d8.read_back(decoded.astype(np.float32) / 65535, 1.0)
        if form == "Hs":
            return d8.read_back(decoded.astype(np.float32), 1.0)
        return d8.read_back(decoded.astype(np.float32), peak)

    base = {"k": k, "photo": Path(p["file"]).name, "camera": p["_k"][0], "gain_map": mapped,
            "h": light.shape[0], "w": light.shape[1], "peak": round(peak, 4),
            "above_white": float((light.max(axis=-1) > 1).mean())}
    rows = []
    for form, a in forms.items():
        rows.append({**base, "setting": "-", "form": form, "it": 0.0, "bytes": 0,
                     **measures(read(form, a), ref_f, ref8, dark_px, dark_blk, ref8_blocks)})
    for name, q, e in SETTINGS:
        for form, a in forms.items():
            for it in TARGETS:
                t1 = time.perf_counter()
                if form == "today" and it == 255.0:     # today's encoder, unchanged
                    data = je.encode_rec2020(a.shape[0], a.shape[1], lambda y, x, ys, xs: a[y:y + ys, x:x + xs],
                                             level=q, effort=e, numthreads=THREADS, transfer="BT709", intensity_target=255)
                elif form == "today":
                    data = d8.encode(a, level=q, effort=e, intensity_target=it, half=False)
                elif form == "I":
                    data = encode_linear_u16(a, level=q, effort=e, intensity_target=it)
                else:
                    data = d8.encode(a, level=q, effort=e, intensity_target=it, half=True)
                t_enc = time.perf_counter() - t1
                t1 = time.perf_counter()
                got = ic.jpegxl_decode(data, numthreads=THREADS)
                t_dec = time.perf_counter() - t1
                want = np.float16 if form.startswith("H") else np.uint16
                assert got.dtype == want, (form, got.dtype)
                hdr = jh.read_header(data)
                rows.append({**base, "setting": name, "form": form, "it": it, "bytes": len(data),
                             "bpp": len(data) * 8 / (a.shape[0] * a.shape[1]),
                             "t_enc": t_enc, "t_dec": t_dec,
                             "hdr": f"{hdr.bits_per_sample}{'f' if hdr.float_sample else 'u'} "
                                    f"{hdr.colour.primaries} {hdr.colour.transfer} xyb={hdr.xyb_encoded}",
                             "declared_it": d8.basic_info(data).intensity_target,
                             **measures(read(form, got), ref_f, ref8, dark_px, dark_blk, ref8_blocks)})
                got = None
    print(f"{k + 1:2d} {base['photo']} {base['camera']} map={mapped} {light.shape[1]}x{light.shape[0]} "
          f"peak {peak:.2f} {time.time() - t0:.0f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB",
          flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="tmp/ml37/canon_int")
    ap.add_argument("--summarise", action="store_true", help="only re-print the summary of <out>/canon_int_rows.csv")
    args = ap.parse_args()
    out = Path(args.out)
    if args.summarise:
        summarise([_typed(r) for r in csv.DictReader(open(out / "canon_int_rows.csv"))])
        return
    out.mkdir(parents=True, exist_ok=True)
    photos = d8.sample(args.n)
    print(f"# {len(photos)} photos (D8's sample), scale 1/{args.scale}, {args.workers} workers x {THREADS} threads, "
          f"load {os.getloadavg()[0]:.2f}", flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_photo, [(k, p, args.scale) for k, p in enumerate(photos)]):
            rows.extend(r)
    keys = sorted({k for r in rows for k in r})
    with open(out / "canon_int_rows.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    summarise(rows)
    print(f"\nload after {os.getloadavg()[0]:.2f}")


def _typed(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if k in ("photo", "camera", "setting", "form", "hdr"):
            out[k] = v
        elif k == "gain_map":
            out[k] = v == "True"
        else:
            out[k] = float(v) if v != "" else float("nan")
    if out["it"] != out["it"]:      # the first run wrote nan for the uncoded rows
        out["it"] = 0.0
    return out


FORM_ORDER = {"today": 0, "I": 1, "Hs": 2, "Hh": 3}


def summarise(rows: list[dict]) -> None:
    print("\n## headers (jxl_header.read_header) and declared intensity target, per form and target")
    seen = sorted({(r["form"], r["it"], r["hdr"], r["declared_it"]) for r in rows if r["setting"] != "-"},
                  key=lambda t: (FORM_ORDER[t[0]], t[1]))
    for f, it, h, d in seen:
        print(f"  {f:5s} IT={it:g}: {h}, declared {d:g}")

    def table(sel, title):
        print(f"\n## {title}")
        print(f"{'setting':7s} {'form':5s} {'IT':>4s} {'MB':>6s} {'bpp':>6s} {'vs today':>16s} {'vs Hs':>6s} "
              f"{'rms8':>6s} {'blk8':>6s} {'blkmx8':>6s} {'blkF':>6s} {'drk10':>6s} {'drkrms':>6s} {'drkbias':>7s} "
              f"{'differ':>6s} {'drkdif':>6s}")
        g = defaultdict(list)
        for r in rows:
            if sel(r):
                g[(r["setting"], r["form"], r["it"])].append(r)
        def per_photo(s, it, form):
            return {r["photo"]: r for r in g.get((s, form, it), [])}
        for (s, f, it), rs in sorted(g.items(), key=lambda kv: (kv[0][0] != "-", kv[0][0], -kv[0][2],
                                                                 FORM_ORDER[kv[0][1]])):
            mb = sum(r["bytes"] for r in rs) / 1e6
            bpp = np.mean([r["bpp"] for r in rs]) if s != "-" else 0.0
            rel_t = rel_h = ""
            if s != "-":
                t = per_photo(s, 255.0, "today")
                rt = [r["bytes"] / t[r["photo"]]["bytes"] for r in rs if r["photo"] in t]
                rel_t = f"{np.mean(rt):.3f}({min(rt):.2f}-{max(rt):.2f})"
                hs = per_photo(s, it, "Hs")
                rh = [r["bytes"] / hs[r["photo"]]["bytes"] for r in rs if r["photo"] in hs]
                rel_h = f"{np.mean(rh):.3f}"
            itl = "-" if s == "-" else f"{it:g}"
            print(f"{s:7s} {f:5s} {itl:>4s} {mb:6.2f} {bpp:6.3f} {rel_t:>16s} {rel_h:>6s} "
                  f"{np.mean([r['rms8'] for r in rs]):6.3f} {np.mean([r['blk_mean8'] for r in rs]):6.3f} "
                  f"{max(r['blk_max8'] for r in rs):6.2f} {np.mean([r['blk_mean_float'] for r in rs]):6.3f} "
                  f"{np.mean([r['dark_blk10_mean8'] for r in rs]):6.3f} {np.nanmean([r['dark_rms8'] for r in rs]):6.3f} "
                  f"{np.nanmean([r['dark_bias8'] for r in rs]):7.3f} "
                  f"{np.mean([r['differ_share'] for r in rs]):6.3f} {np.nanmean([r['dark_differ_share'] for r in rs]):6.3f}")

    table(lambda r: True, "all photos (means over photos; blkmx8 = largest block over all photos)")
    table(lambda r: r["gain_map"], "gain-map photos")
    table(lambda r: not r["gain_map"], "photos without a gain map")

    print("\n## per-photo ratios, I against Hs and Hh and today (same setting and target): mean (min-max)")
    idx = {(r["photo"], r["setting"], r["form"], r["it"]): r for r in rows if r["setting"] != "-"}
    for s, _, _ in SETTINGS:
        for it in TARGETS:
            for other in ("Hs", "Hh", "today"):
                for m in ("bytes", "rms8", "blk_mean8", "dark_rms8", "dark_blk10_mean8"):
                    v = [idx[(p, s, "I", it)][m] / idx[(p, s, other, it)][m] for (p, s2, f, i) in idx
                         if s2 == s and f == "I" and i == it and idx[(p, s, other, it)][m] > 0
                         and idx[(p, s, other, it)][m] == idx[(p, s, other, it)][m]]
                    print(f"  {s} IT={it:g} I/{other:5s} {m:16s} {np.mean(v):.3f} ({min(v):.3f}-{max(v):.3f})")
    print("\n## time per photo at this scale (s, mean over photos; workers share the machine)")
    for s, _, _ in SETTINGS:
        for f in FORM_ORDER:
            rs = [r for r in rows if r["setting"] == s and r["form"] == f]
            print(f"  {s} {f:5s} encode {np.mean([r['t_enc'] for r in rs]):6.2f}  decode {np.mean([r['t_dec'] for r in rs]):5.2f}")


if __name__ == "__main__":
    sys.exit(main())
