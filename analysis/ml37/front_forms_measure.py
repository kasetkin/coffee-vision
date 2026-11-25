"""ML-37: three forms the front could send to the host, timed and sized on dataset photos.

    cd /workspace && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/workspace \
        taskset -c 0,1 uv run python analysis/ml37/front_forms_measure.py [--n 41] [--runs 3] [--out tmp/ml37/front_forms]
    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/front_forms_measure.py --summarise

Forms, each made from the uploaded original with the front's settings (coffeecv.canonical.FRONT_SETTINGS: quality 90,
effort 3, over 18 MP area-averaged to at most 15 MP, gain maps applied) and the front's pixel ceilings
(webapp/front/check_service.py: 60 MP, RAW 16 MP; a photo over its ceiling is still measured, flagged):
  A    lossy JPEG XL, Rec.2100 primaries, D65, linear transfer, uint16, intensity target 203 (ML-37 D1): the store's
       float colour path (canonical._light, LittleCMS on float32 to linear Rec.2020), ML-29 D33's EETF at the
       picture's peak for a tone-mapped source, clip [0, 1], rint(x * 65535); written by the repo's ctypes strip
       encoder with the linear transfer and the target (canon_int_measure.encode_linear_u16, as research_canonical_
       int_vs_half.md wrote form I).
  B    lossy JPEG XL (imagecodecs, uint8, sRGB primaries and curve, quality 90, effort 3) of the 8-bit sRGB picture.
  C1   lossless JPEG XL of the same picture, effort 1;  C3: effort 3.
The 8-bit picture is made directly from the same clipped light A is rounded from: D2's read-back in numpy float32
(matrix linear Rec.2020 -> linear sRGB, clip, sRGB curve, x 255, rint), ticket D9 (a). It is checked against the
read-back of A's uncompressed uint16 picture (the same steps from rint(x * 65535) / 65535).

Timing: wall clock of each step (decode incl. gain map and cap; peak pass; colour; tone map; A's uint16 rounding;
B/C's read-back; each encode), one process, libjxl 2 threads, the whole process pinned to two CPUs (taskset). One
untimed warm-up on the first photo, then --runs runs per photo; the per-step median is kept. Errors (first run):
each file decoded (imagecodecs, 2 threads), A read back as above and rounded to uint8 once; against the uncompressed
8-bit picture: pixel RMS in 8-bit levels, ML-29 D27's 32 x 32 block means (mean, largest |difference|), the share of
values differing. Files are written to <out>/<form>/, rows to <out>/rows.csv, the sample to <out>/sample.txt.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

import imagecodecs as ic
import numpy as np

from analysis.ml37 import canon_int_measure as cm
from analysis.ml37 import d8_intensity_target as d8
from coffeecv import canonical as cn
from coffeecv import tone_map as tm

SET = cn.FRONT_SETTINGS
MAX_PIXELS, RAW_MAX_PIXELS = 60_000_000, 16_000_000      # webapp/front/check_service.py L78-79
LINK = 2.1e6                                             # bit/s, the front -> yappy link
THREADS = 2
M = d8.REC2020_TO_SRGB
FORMS = ("A", "B", "C1", "C3")

# (label, type, gain map, make filter, megapixel test, count)
STRATA = [
    ("JPEG Sony 17 MP", "JPEG", False, "Sony", None, 3),
    ("JPEG OnePlus", "JPEG", False, "OnePlus", None, 2),
    ("JPEG Google", "JPEG", False, "Google", None, 2),
    ("JPEG no make", "JPEG", False, "", None, 2),
    ("JPEG Xiaomi", "JPEG", False, "Xiaomi", None, 1),
    ("JPEG samsung", "JPEG", False, "samsung", None, 1),
    ("gain-map JPEG Google >=48 MP", "JPEG", True, "Google", lambda mp: mp >= 48, 3),
    ("gain-map JPEG samsung >=48 MP", "JPEG", True, "samsung", lambda mp: 48 <= mp <= 60, 2),
    ("gain-map JPEG samsung 16-48 MP", "JPEG", True, "samsung", lambda mp: 16 < mp < 48, 1),
    ("gain-map JPEG Google 12 MP", "JPEG", True, "Google", lambda mp: mp <= 16, 3),
    ("gain-map JPEG OnePlus 7 Pro", "JPEG", True, "GM1911 (OnePlus7Pro)", None, 2),
    ("gain-map JPEG samsung 12 MP", "JPEG", True, "samsung", lambda mp: mp <= 16, 1),
    ("gain-map HEIC Apple >=48 MP", "HEIC", True, "Apple", lambda mp: mp >= 48, 2),
    ("gain-map HEIC Apple 16-48 MP", "HEIC", True, "Apple", lambda mp: 16 < mp < 48, 2),
    ("gain-map HEIC Apple 12 MP", "HEIC", True, "Apple", lambda mp: mp <= 16, 2),
    ("HEIC Apple, no map", "HEIC", False, "Apple", None, 1),
    ("DNG Google", "DNG", False, "Google", None, 2),
    ("DNG Apple", "DNG", False, "Apple", None, 1),
    ("DNG OnePlus", "DNG", False, "OnePlus", None, 1),
    ("CR3 Canon", "CR3", False, "Canon", None, 2),
    ("PNG RGB", "PNG", False, "", lambda mp: True, 1),
    ("PNG grey", "PNG", False, "", lambda mp: True, 1),
    ("JPEG XL 12-bit", "JXL Codestream", False, "", None, 2),
]


def sample(seed: int = 0) -> list[dict]:
    rows = [r for r in csv.DictReader(open("tmp/ml29_scan/files.csv")) if Path(r["file"]).exists()]
    rng = random.Random(seed)
    picked = []
    for label, typ, gm, make, mp_ok, n in STRATA:
        cands = []
        for r in rows:
            if r["type"] != typ or bool(r["gain_map"]) != gm or r["make"] != make:
                continue
            if typ == "PNG" and (("Alpha" in r["bits"]) or (("Grayscale" in r["bits"]) != (label == "PNG grey"))):
                continue
            try:
                mp = float(r["megapixels"] or 0)
            except ValueError:
                mp = 0.0
            if mp_ok is not None and not mp_ok(mp):
                continue
            if typ not in ("DNG", "CR3") and mp > MAX_PIXELS / 1e6:   # refused by the front: never sent
                continue
            cands.append(r)
        cands.sort(key=lambda r: r["file"])
        for r in rng.sample(cands, min(n, len(cands))):
            picked.append({**r, "stratum": label})
    return picked


def light_strips(src: cn.Source, peak: float | None, tim: dict):
    """Yields each strip of the clipped light (linear Rec.2020, float32), timing colour and tone map."""
    h = src.pixels.shape[0]
    channels = 1 if src.pixels.ndim == 2 else src.pixels.shape[2]
    if channels in (2, 4):
        raise ValueError("transparency is not in this sample")
    for y in range(0, h, cn.STRIP_ROWS):
        t = time.perf_counter()
        part, f = cn._light(src.pixels[y:y + cn.STRIP_ROWS], src.full, src.linearize, 1 if channels == 1 else 3)
        c = ic.cms_transform(f, src.profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.float32)
        if part.dtype.kind == "f":
            np.nan_to_num(c, copy=False, nan=0.0)
        t1 = time.perf_counter()
        tim["colour"] += t1 - t
        if peak is not None and peak > 1:
            tm.eetf_in_place(c, peak)
        np.clip(c, 0, 1, out=c)
        tim["tonemap"] += time.perf_counter() - t1
        yield y, c


def read_back8(lin: np.ndarray) -> np.ndarray:
    """D2/D3/D9 (a): linear Rec.2020 [0, 1] float32 -> matrix -> clip -> sRGB curve -> rint uint8."""
    c = lin @ M.T
    np.clip(c, 0, 1, out=c)
    c = d8.srgb_curve(c)
    c *= 255
    np.rint(c, out=c)
    return c.astype(np.uint8)


def make(path: str, max_pixels: int | None, keep: bool):
    """One run: every step timed. Returns timings, the four files and (if keep) the arrays."""
    tim = dict.fromkeys(("decode", "peak", "colour", "tonemap", "quant16", "readback", "encA", "encB", "encC1",
                         "encC3"), 0.0)
    data = Path(path).read_bytes()
    t = time.perf_counter()
    src = cn.decode(data, Path(path).name, max_pixels, settings=SET)
    tim["decode"] = time.perf_counter() - t
    data = None
    t = time.perf_counter()
    peak = src.peak()
    tim["peak"] = time.perf_counter() - t
    h, w = src.pixels.shape[:2]
    u16 = np.empty((h, w, 3), np.uint16)
    u8 = np.empty((h, w, 3), np.uint8)
    for y, c in light_strips(src, peak, tim):
        t = time.perf_counter()
        u16[y:y + c.shape[0]] = np.rint(c * 65535)
        t1 = time.perf_counter()
        u8[y:y + c.shape[0]] = read_back8(c)
        tim["readback"] += time.perf_counter() - t1
        tim["quant16"] += t1 - t
    info = {"w": w, "h": h, "decoder": src.decoder, "peak": peak or 0.0, "bits": src.pixels.dtype.itemsize * 8}
    src = None
    files = {}
    t = time.perf_counter()
    files["A"] = cm.encode_linear_u16(u16, level=SET.quality, effort=SET.effort, intensity_target=203.0)
    tim["encA"] = time.perf_counter() - t
    t = time.perf_counter()
    files["B"] = ic.jpegxl_encode(u8, level=SET.quality, effort=SET.effort, numthreads=THREADS)
    tim["encB"] = time.perf_counter() - t
    for e in (1, 3):
        t = time.perf_counter()
        files[f"C{e}"] = ic.jpegxl_encode(u8, lossless=True, effort=e, numthreads=THREADS)
        tim[f"encC{e}"] = time.perf_counter() - t
    return tim, files, info, ((u16, u8) if keep else None)


def blocks8(a: np.ndarray) -> np.ndarray:
    return d8.blocks(a.astype(np.float32))


def errors(got8: np.ndarray, ref8: np.ndarray, ref_blocks: np.ndarray) -> dict:
    diff = got8.astype(np.int16) - ref8.astype(np.int16)
    d = np.abs(blocks8(got8) - ref_blocks)
    return {"rms8": float(np.sqrt(np.mean(diff.astype(np.float32) ** 2))), "blk_mean": float(d.mean()),
            "blk_max": float(d.max()), "differ": float((diff != 0).mean())}


def gen_time(m: dict, form: str) -> float:
    common = m["decode"] + m["peak"] + m["colour"] + m["tonemap"]
    return common + (m["quant16"] + m["encA"] if form == "A" else m["readback"] + m[f"enc{form}"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="tmp/ml37/front_forms")
    ap.add_argument("--limit", type=int, default=0, help="first N photos only (a trial)")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if args.summarise:
        return summarise(list(csv.DictReader(open(out / "rows.csv"))))
    for f in FORMS:
        (out / f).mkdir(parents=True, exist_ok=True)
    photos = sample()
    if args.limit:
        photos = photos[:args.limit]
    with open(out / "sample.txt", "w") as fh:
        for i, p in enumerate(photos):
            fh.write(f"{i:02d}\t{p['stratum']}\t{os.path.relpath(p['file'], '/workspace')}\n")
    print(f"# {len(photos)} photos; affinity {sorted(os.sched_getaffinity(0))}; uptime load {os.getloadavg()}; "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC", flush=True)
    make(photos[0]["file"], None, False)            # warm-up, untimed
    done = {}
    if (out / "rows.csv").exists():                 # resume: rows already measured for the same photo and index
        done = {(r["i"], r["file"]): r for r in csv.DictReader(open(out / "rows.csv"))}
    rows = []
    for i, p in enumerate(photos):
        old = done.get((str(i), os.path.relpath(p["file"], "/workspace")))
        if old is not None:
            rows.append(old)
            continue
        raw = p["type"] in ("DNG", "CR3")
        ceiling = RAW_MAX_PIXELS if raw else MAX_PIXELS
        accepted = _accepted(p["file"], ceiling) if raw else True   # the dataset's non-RAW photos are all <= 50 MP
        runs = []
        kept = None
        for r in range(args.runs):
            tim, files, info, arrays = make(p["file"], None, r == 0)
            runs.append(tim)
            if r == 0:
                kept, first_files = arrays, files
        med = {k: st.median(t[k] for t in runs) for k in runs[0]}
        u16, u8 = kept
        kept = None
        stem = f"{i:02d}_{Path(p['file']).stem}"
        for f in FORMS:
            (out / f / f"{stem}.jxl").write_bytes(first_files[f])
        # B/C's input against the read-back of A's uncompressed picture
        alt = np.empty_like(u8)
        for y in range(0, u16.shape[0], cn.STRIP_ROWS):
            alt[y:y + cn.STRIP_ROWS] = read_back8(u16[y:y + cn.STRIP_ROWS].astype(np.float32) / 65535)
        dd = np.abs(alt.astype(np.int16) - u8.astype(np.int16))
        ref_blocks = blocks8(u8)
        row = {"i": i, "stratum": p["stratum"], "type": p["type"], "file": os.path.relpath(p["file"], "/workspace"),
               "front_accepts": accepted, **info, "mp": info["w"] * info["h"] / 1e6,
               "u8_vs_A_readback_max": int(dd.max()), "u8_vs_A_readback_differ": float((dd != 0).mean())}
        alt = dd = None
        row.update({f"t_{k}": v for k, v in med.items()})
        row["runs"] = args.runs
        for f in FORMS:
            data = first_files[f]
            t = time.perf_counter()
            dec = ic.jpegxl_decode(data, numthreads=THREADS)
            row[f"tdec_{f}"] = time.perf_counter() - t
            if f == "A":
                got = np.empty(u8.shape, np.uint8)
                for y in range(0, dec.shape[0], cn.STRIP_ROWS):
                    got[y:y + cn.STRIP_ROWS] = read_back8(dec[y:y + cn.STRIP_ROWS].astype(np.float32) / 65535)
                dec = got
                got = None
            e = errors(dec, u8, ref_blocks)
            dec = None
            row.update({f"{k}_{f}": v for k, v in e.items()})
            row[f"bytes_{f}"] = len(data)
            row[f"gen_{f}"] = gen_time(med, f)
        u16 = u8 = first_files = None
        rows.append(row)
        print(f"{i:2d} {p['stratum'][:30]:30s} {row['mp']:5.1f} MP acc={accepted!s:5s} "
              f"gen A {row['gen_A']:5.2f} B {row['gen_B']:5.2f} C1 {row['gen_C1']:5.2f} C3 {row['gen_C3']:5.2f} | "
              f"kB A {row['bytes_A'] / 1e3:6.0f} B {row['bytes_B'] / 1e3:6.0f} C1 {row['bytes_C1'] / 1e3:6.0f} "
              f"C3 {row['bytes_C3'] / 1e3:6.0f} | rms A {row['rms8_A']:.2f} B {row['rms8_B']:.2f} "
              f"C {row['rms8_C1']:.2f} | load {os.getloadavg()[0]:.2f}", flush=True)
        with open(out / "rows.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print(f"# end {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC; load {os.getloadavg()}; "
          f"peak RSS {d8_status('VmHWM'):.0f} MB", flush=True)
    summarise([{k: str(v) for k, v in r.items()} for r in rows])


def _accepted(path: str, ceiling: int) -> bool:
    """Whether the front takes the photo: its decoder's size check against the ceiling (TooLarge before pixels)."""
    try:
        cn.decode(Path(path).read_bytes(), Path(path).name, ceiling, settings=SET)
        return True
    except cn.TooLarge:
        return False


def d8_status(field: str) -> float:
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024
    return float("nan")


def ms(vals: list[float], fmt: str = "{:.2f}") -> str:
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(m)} ± {fmt.format(s)}"


def summarise(rows: list[dict]) -> None:
    def f(r, k):
        return float(r[k])

    def table(sel: list[dict], title: str) -> None:
        print(f"\n## {title} ({len(sel)} photos, {st.mean(f(r, 'mp') for r in sel):.1f} MP mean as sent)")
        print("| form | generate s | MB | bytes/px | transfer s at 2.1 Mbit/s | total s | pixel RMS (8-bit) | "
              "block mean | block max | values differing |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for fm in FORMS:
            gen = [f(r, f"gen_{fm}") for r in sel]
            mb = [f(r, f"bytes_{fm}") / 1e6 for r in sel]
            bpp = [f(r, f"bytes_{fm}") / (f(r, "mp") * 1e6) for r in sel]
            tx = [f(r, f"bytes_{fm}") * 8 / LINK for r in sel]
            tot = [g + x for g, x in zip(gen, tx)]
            print(f"| {fm} | {ms(gen)} | {ms(mb)} | {ms(bpp, '{:.3f}')} | {ms(tx)} | {ms(tot)} | "
                  f"{ms([f(r, f'rms8_{fm}') for r in sel], '{:.3f}')} | "
                  f"{ms([f(r, f'blk_mean_{fm}') for r in sel], '{:.3f}')} | "
                  f"{max(f(r, f'blk_max_{fm}') for r in sel):.2f} | "
                  f"{st.mean(f(r, f'differ_{fm}') for r in sel):.1%} |")

    table(rows, "All photos")
    acc = [r for r in rows if r["front_accepts"] == "True"]
    table(acc, "Photos the live front accepts (under its ceilings)")
    print("\n## Generation broken down (s, mean ± std over photos of the per-photo median)")
    keys = ("decode", "peak", "colour", "tonemap", "quant16", "readback", "encA", "encB", "encC1", "encC3")
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    print("| " + " | ".join(ms([f(r, f't_{k}') for r in rows]) for k in keys) + " |")
    print("\n## Decode on the host side (s, imagecodecs, 2 threads; A without its read-back)")
    print(" ".join(f"{fm}: {ms([f(r, f'tdec_{fm}') for r in rows])}" for fm in FORMS))
    print("\n## B/C's input vs the read-back of A's uncompressed uint16 picture")
    print(f"largest |difference| over photos: {max(int(r['u8_vs_A_readback_max']) for r in rows)} level(s); share "
          f"of values differing: mean {st.mean(f(r, 'u8_vs_A_readback_differ') for r in rows):.4%}, max "
          f"{max(f(r, 'u8_vs_A_readback_differ') for r in rows):.4%}")
    print("\n## By format (means; generate s / kB / transfer s / pixel RMS / block mean)")
    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = {"JXL Codestream": "JPEG XL"}.get(r["type"], r["type"])
        key = f"gain-map {key}" if r["stratum"].startswith("gain-map") else key
        groups.setdefault(key, []).append(r)
    print("| format | n | MP | " + " | ".join(FORMS) + " |")
    print("|---|---|---|" + "---|" * len(FORMS))
    for k, sel in groups.items():
        cells = []
        for fm in FORMS:
            cells.append(f"{st.mean(f(r, f'gen_{fm}') for r in sel):.2f} s / "
                         f"{st.mean(f(r, f'bytes_{fm}') for r in sel) / 1e3:.0f} kB / "
                         f"{st.mean(f(r, f'bytes_{fm}') * 8 / LINK for r in sel):.2f} s / "
                         f"{st.mean(f(r, f'rms8_{fm}') for r in sel):.2f} / "
                         f"{st.mean(f(r, f'blk_mean_{fm}') for r in sel):.3f}")
        print(f"| {k} | {len(sel)} | {st.mean(f(r, 'mp') for r in sel):.1f} | " + " | ".join(cells) + " |")
    print("\n## Per photo: stratum, MP, accepted, generate A/B/C1/C3 s, bytes A/B/C1/C3")
    for r in rows:
        print(f"{int(r['i']):2d} {r['stratum'][:32]:32s} {f(r, 'mp'):5.1f} {r['front_accepts']:5s} "
              f"{f(r, 'gen_A'):5.2f} {f(r, 'gen_B'):5.2f} {f(r, 'gen_C1'):5.2f} {f(r, 'gen_C3'):5.2f}  "
              f"{int(r['bytes_A']):9d} {int(r['bytes_B']):9d} {int(r['bytes_C1']):9d} {int(r['bytes_C3']):9d}")


if __name__ == "__main__":
    sys.exit(main())
