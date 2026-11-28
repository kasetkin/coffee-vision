"""ML-37 D10: size, encode and decode time, and exactness of a crop stored three ways, on a sample of today's crops.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d10_crop_forms.py [--n 26] [--probe]

The sample: `--n` crops of data/segcropped/<session> that are in the pool (data/segcropped/pool), drawn per session
(seed 3710) in proportion to the session's crops in the pool, at least one each, no crop over --max-mp megapixels
(memory: this machine is shared), and one D18 fallback. Read only; nothing under data/ is written.

Per crop, the stand-in for ML-37's float read-back: the crop's canonical photo decoded (canonical.decode_canonical,
uint16 Rec.2020, BT.709 curve), the crop's box (segcrop_report.json) cut out, converted to sRGB by the same LittleCMS
transform load_canonical uses (canonical.rec2020_to_srgb's, NOOPTIMIZE, relative) but with float32 out, clipped to
[0, 1], and filled outside its __beanmask.png with fill_rgb / 255. Rounded to 8 bits it should be the stored crop
(checked: share of samples equal, largest difference). It is today's canonical photo, not D1's half-float one: the
lossy coding noise in its low bits is today's (ML-37 R2 measures the new one).

Forms, each lossless JPEG XL through imagecodecs at crop_io.EFFORT (5), 2 threads (crop_io uses 4):
  u8   today's: the stored file's size; re-encoded from the stored pixels for the time (and its size compared)
  u16  (a): round(x * 65535) as uint16, tagged sRGB primaries and transfer
  f16  (b): x as float16, tagged the same
Each decoded and compared bit for bit. On the first --sweep crops, efforts 1, 3 and 7 as well.
For (c), the time to rebuild the crop at fit time: the canonical decode plus the box's float read-back plus the fill.
Also: the crop's fine-scale variation (robust sigma of the green channel's second difference, bean pixels, in 8-bit
levels), the rounding error of each form, and two 448 px float32 patches from each of the first --patches crops
(boxes of 600-1000 px, >= 80% bean, PIL "F" bilinear per channel) saved to tmp/ml37/d10/patches.npz for
d10_embed_precision.py.

Writes tmp/ml37/d10/rows.jsonl (one row per crop) and analysis/ml37/d10_crop_forms.txt (the summary and the
extrapolation to every crop by session).
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import imagecodecs as ic
import numpy as np
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import crop_io
from coffeecv.config import REPO_ROOT

SEG = REPO_ROOT / "data" / "segcropped"
SCRATCH = REPO_ROOT / "tmp" / "ml37" / "d10"
OUT_TXT = Path(__file__).with_suffix(".txt")
FILL = np.array([124, 116, 104], np.float32) / 255     # segment_beans.SegParams.fill_rgb
TAG = dict(primaries=ic.JPEGXL.PRIMARIES.SRGB, transfer=ic.JPEGXL.TRANSFER_FUNCTION.SRGB)
THREADS = 2
STRIP = 128
WINDOW_PX = 8_000_000      # crops over this are measured on a centre band of rows of this many pixels (memory)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def all_entries() -> list[dict]:
    """Every session crop with its report entry, session and whether the pool holds it."""
    pool = {p.name for p in (SEG / "pool").rglob("*" + crop_io.CROP_SUFFIX)}
    rows = []
    for rep in sorted(SEG.glob("*/class_*/segcrop_report.json")):
        session = rep.parent.parent.name
        if session == "pool":
            continue
        for e in json.loads(rep.read_text()):
            out = REPO_ROOT / e["out_path"]
            rows.append({**e, "session": session, "in_pool": out.name in pool, "bytes_u8": out.stat().st_size,
                         "mp": e["size"][0] * e["size"][1] / 1e6})
    return rows


def draw(rows: list[dict], n: int, max_mp: float, seed: int = 3710) -> list[dict]:
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        if r["in_pool"] and r["mp"] <= max_mp and not r["fallback"]:
            by[r["session"]].append(r)
    total = sum(len(v) for v in by.values())
    out = []
    for s in sorted(by):
        k = max(1, round((n - 1) * len(by[s]) / total))
        idx = rng.choice(len(by[s]), size=min(k, len(by[s])), replace=False)
        out += [by[s][i] for i in sorted(idx)]
    fb = [r for r in rows if r["in_pool"] and r["fallback"] and r["mp"] <= max_mp]
    if fb:
        out.append(fb[int(rng.integers(len(fb)))])
    return out


def float_readback(rec: np.ndarray) -> np.ndarray:
    """Rec.2020/BT.709 uint16 -> sRGB float32 in [0, 1]: load_canonical's transform with float output, clipped."""
    out = np.empty(rec.shape, np.float32)
    for y in range(0, rec.shape[0], STRIP):
        ic.cms_transform(rec[y:y + STRIP], cn.REC2020, cn.SRGB, intent=cn.RELATIVE, flags=cn.NOOPTIMIZE,
                         outdtype=np.float32, out=out[y:y + STRIP])
    np.clip(out, 0.0, 1.0, out=out)
    return out


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t


def encode_check(arr: np.ndarray, effort: int, **tag) -> dict:
    data, t_enc = timed(ic.jpegxl_encode, arr, lossless=True, effort=effort, numthreads=THREADS, **tag)
    back, t_dec = timed(ic.jpegxl_decode, data, numthreads=THREADS)
    if arr.dtype == np.float16:
        exact = back.dtype == np.float16 and np.array_equal(back.view(np.uint16), arr.view(np.uint16))
    else:
        exact = back.dtype == arr.dtype and np.array_equal(back, arr)
    return {"bytes": len(data), "enc_s": t_enc, "dec_s": t_dec, "exact": bool(exact)}


def fine_sigma(green: np.ndarray, mask: np.ndarray | None) -> float:
    """Robust sigma (1.4826 MAD / sqrt 6) of the horizontal second difference, bean pixels, in 8-bit levels."""
    d = green[:, 2:] - 2 * green[:, 1:-1] + green[:, :-2]
    if mask is not None:
        d = d[mask[:, 1:-1] & mask[:, 2:] & mask[:, :-2]]
    d = d.ravel()[::7]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(6) * 255)


def patches_of(x: np.ndarray, mask: np.ndarray | None, rng, k: int = 2) -> list[np.ndarray]:
    h, w = x.shape[:2]
    out = []
    for _ in range(200):
        if len(out) == k:
            break
        side = int(rng.integers(600, min(1000, h, w) + 1))
        y0, x0 = int(rng.integers(0, h - side + 1)), int(rng.integers(0, w - side + 1))
        if mask is not None and mask[y0:y0 + side, x0:x0 + side].mean() < 0.8:
            continue
        p = x[y0:y0 + side, x0:x0 + side]
        out.append(np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(p[..., c])).resize(
            (448, 448), Image.BILINEAR)) for c in range(3)], axis=-1).astype(np.float32))
    return out


def measure(e: dict, sweep: bool, keep_patches: bool, rng) -> tuple[dict, list[np.ndarray]]:
    crop_path = REPO_ROOT / e["out_path"]
    data = (REPO_ROOT / e["canonical"]).read_bytes()
    rec, t_decode = timed(cn.decode_canonical, data, THREADS)
    if e["box"] is not None:
        x0, y0, w, h = e["box"]
        region = np.ascontiguousarray(rec[y0:y0 + h, x0:x0 + w])
    else:
        region = rec
    canon_mp = rec.shape[0] * rec.shape[1] / 1e6
    del rec
    rows_all = region.shape[0]
    keep = min(rows_all, int(WINDOW_PX // region.shape[1]))     # a centre band of rows, the same for every form
    r0 = (rows_all - keep) // 2
    region = np.ascontiguousarray(region[r0:r0 + keep])
    win_mp = region.shape[0] * region.shape[1] / 1e6
    x, t_readback = timed(float_readback, region)
    del region
    t0 = time.perf_counter()
    mask = None
    if not e["fallback"]:
        mask = np.asarray(Image.open(crop_io.mask_path(crop_path)).convert("1"), dtype=bool)[r0:r0 + keep]
        x[~mask] = FILL
    t_fill = time.perf_counter() - t0

    stored, t_dec_u8 = timed(crop_io.load_crop, crop_path)
    t_dec_u8 *= win_mp / e["mp"]          # per window, to compare like with like
    stored = np.ascontiguousarray(stored[r0:r0 + keep])
    u8 = np.rint(x * 255).astype(np.uint8)
    diff = np.abs(u8.astype(np.int16) - stored.astype(np.int16))
    row = {"file": crop_path.name, "session": e["session"], "fallback": e["fallback"], "crop_mp": e["mp"],
           "win_mp": win_mp, "windowed": keep < rows_all,
           "canonical_mp": canon_mp, "canonical_bytes": len(data), "t_canonical_decode": t_decode,
           "t_readback_box": t_readback, "t_fill": t_fill, "t_decode_stored_u8": t_dec_u8,
           "u8_equal_share": float(np.mean(diff == 0)), "u8_max_diff": int(diff.max()),
           "fine_sigma_levels": fine_sigma(x[..., 1], mask)}
    del diff
    u8_enc = encode_check(stored, crop_io.EFFORT)
    row["u8"] = {**u8_enc, "stored_bytes": e["bytes_u8"]}    # bytes: the window re-encoded as crop_io does
    del u8, stored
    u16 = np.rint(x * 65535).astype(np.uint16)
    row["err_u16_max"] = float(np.max(np.abs(u16 / 65535 - x)))
    row["u16"] = encode_check(u16, crop_io.EFFORT, **TAG)
    f16 = x.astype(np.float16)
    row["err_f16_max"] = float(np.max(np.abs(f16.astype(np.float32) - x)))
    pats = patches_of(x, mask, rng) if keep_patches else []
    del x
    row["f16"] = encode_check(f16, crop_io.EFFORT, **TAG)
    if sweep:
        for eff in (1, 3, 7):
            row[f"u16_e{eff}"] = encode_check(u16, eff, **TAG)
            row[f"f16_e{eff}"] = encode_check(f16, eff, **TAG)
            u8b = np.rint(f16.astype(np.float32) * 255).astype(np.uint8)
            row[f"u8_e{eff}"] = encode_check(u8b, eff)
    row["peak_rss_mb"] = rss_mb()
    return row, pats


def summarise(rows: list[dict], entries: list[dict]) -> list[str]:
    """Sizes and times per pixel of what was encoded (the window), u8 = today's encoder on the same window."""
    L = []
    n = len(rows)
    mp = sum(r["win_mp"] for r in rows)
    full = [r for r in rows if not r["windowed"]]
    L.append(f"{n} crops ({sum(r['windowed'] for r in rows)} measured on a centre band of 8 MP), {mp:.0f} MP encoded, "
             f"{sum(r['crop_mp'] for r in rows):.0f} MP of crop; {len(set(r['session'] for r in rows))} sessions; "
             f"peak RSS {max(r['peak_rss_mb'] for r in rows):.0f} MB")
    L.append(f"stand-in check: rounded to 8 bits, share of samples equal to the stored crop: min "
             f"{min(r['u8_equal_share'] for r in rows):.6f}, mean {np.mean([r['u8_equal_share'] for r in rows]):.6f};"
             f" largest difference {max(r['u8_max_diff'] for r in rows)} level(s)")
    L.append(f"whole crops: today's encoder at 2 threads gives the stored file's size in "
             f"{sum(r['u8']['bytes'] == r['u8']['stored_bytes'] for r in full)} of {len(full)}")
    L.append(f"bit-exact: u8 {sum(r['u8']['exact'] for r in rows)}/{n}, u16 {sum(r['u16']['exact'] for r in rows)}/"
             f"{n}, f16 {sum(r['f16']['exact'] for r in rows)}/{n}")
    L.append("")
    L.append("form  bytes/px  x u8   enc s/MP  dec s/MP   (effort 5, 2 threads; sums over the sample)")
    u8b = sum(r["u8"]["bytes"] for r in rows)
    for f in ("u8", "u16", "f16"):
        b = sum(r[f]["bytes"] for r in rows)
        L.append(f"{f:4s}  {b / (mp * 1e6):7.3f}  {b / u8b:5.2f}  {sum(r[f]['enc_s'] for r in rows) / mp:8.3f}  "
                 f"{sum(r[f]['dec_s'] for r in rows) / mp:8.3f}")
    for f in ("u16", "f16"):
        v = [r[f]["bytes"] / r["u8"]["bytes"] for r in rows]
        L.append(f"per-crop size ratio {f}/u8: min {min(v):.2f}, median {np.median(v):.2f}, max {max(v):.2f}")
    L.append("per session (ratio of sums): " + "; ".join(
        f"{s} {sum(r['u16']['bytes'] for r in rows if r['session'] == s) / sum(r['u8']['bytes'] for r in rows if r['session'] == s):.2f}"
        f"/{sum(r['f16']['bytes'] for r in rows if r['session'] == s) / sum(r['u8']['bytes'] for r in rows if r['session'] == s):.2f}"
        for s in sorted(set(r["session"] for r in rows))) + "  (u16/f16)")
    sw = [r for r in rows if "u16_e1" in r]
    if sw:
        smp = sum(r["win_mp"] for r in sw)
        L.append("")
        L.append(f"effort sweep, {len(sw)} crops, {smp:.0f} MP (u8 at efforts 1, 3, 7 = the f16 crop rounded to 8 bits, "
                 f"untagged; at 5 the stored pixels)")
        L.append("form effort  bytes/px  enc s/MP  dec s/MP  exact")
        for f in ("u8", "u16", "f16"):
            for eff in (1, 3, 5, 7):
                k = f if eff == 5 else f"{f}_e{eff}"
                b = sum(r[k]["bytes"] for r in sw)
                L.append(f"{f:4s} {eff:6d}  {b / (smp * 1e6):8.3f}  {sum(r[k]['enc_s'] for r in sw) / smp:8.3f}  "
                         f"{sum(r[k]['dec_s'] for r in sw) / smp:8.3f}  {all(r[k]['exact'] for r in sw)}")
    L.append("")
    cmp_ = sum(r["crop_mp"] for r in rows)
    t_c = sum(r["t_canonical_decode"] + (r["t_readback_box"] + r["t_fill"]) * r["crop_mp"] / r["win_mp"] for r in rows)
    t_u8 = sum(r["t_decode_stored_u8"] * r["crop_mp"] / r["win_mp"] for r in rows)
    L.append("(c) rebuilding the crop at fit time, per photo (2 threads): canonical decode (whole photo) + float "
             "read-back of the box + fill, scaled to the whole crop")
    L.append(f"  decode {sum(r['t_canonical_decode'] for r in rows) / n:.2f} s, read-back "
             f"{sum(r['t_readback_box'] / r['win_mp'] for r in rows) / n:.3f} s/MP, fill "
             f"{sum(r['t_fill'] / r['win_mp'] for r in rows) / n:.3f} s/MP; all in {t_c / n:.2f} s per photo, "
             f"{t_c / cmp_:.3f} s/MP of crop")
    L.append(f"  vs decoding today's stored 8-bit crop {t_u8 / n:.2f} s per photo ({t_u8 / cmp_:.3f} s/MP); u16 crop "
             f"{sum(r['u16']['dec_s'] * r['crop_mp'] / r['win_mp'] for r in rows) / n:.2f} s, f16 crop "
             f"{sum(r['f16']['dec_s'] * r['crop_mp'] / r['win_mp'] for r in rows) / n:.2f} s per photo")
    L.append(f"  canonical photo {sum(r['canonical_bytes'] for r in rows) / n / 1e6:.2f} MB per photo, "
             f"{sum(r['canonical_mp'] for r in rows) / n:.1f} MP")
    L.append("")
    L.append(f"rounding: largest |u16/65535 - x| {max(r['err_u16_max'] for r in rows):.3e}; largest |f16 - x| "
             f"{max(r['err_f16_max'] for r in rows):.3e}")
    fs = [r["fine_sigma_levels"] for r in rows]
    L.append(f"fine-scale variation (green, bean pixels, 8-bit levels): min {min(fs):.2f}, median {np.median(fs):.2f},"
             f" max {max(fs):.2f}")
    L.append("")
    ratio = defaultdict(dict)
    for s in set(r["session"] for r in rows):
        for f in ("u16", "f16"):
            ratio[s][f] = (sum(r[f]["bytes"] for r in rows if r["session"] == s)
                           / sum(r["u8"]["bytes"] for r in rows if r["session"] == s))
    overall = {f: sum(r[f]["bytes"] for r in rows) / u8b for f in ("u16", "f16")}
    tot, pool = defaultdict(float), defaultdict(float)
    for e in entries:
        for f in ("u16", "f16"):
            v = e["bytes_u8"] * ratio.get(e["session"], overall).get(f, overall[f])
            tot[f] += v
            pool[f] += v if e["in_pool"] else 0
        tot["u8"] += e["bytes_u8"]
        pool["u8"] += e["bytes_u8"] if e["in_pool"] else 0
        tot["n"] += 1
        pool["n"] += e["in_pool"]
    L.append(f"extrapolated, each session's stored 8-bit bytes x its sample's ratio: {int(tot['n'])} session crops, "
             f"{int(pool['n'])} of them in the pool")
    for f in ("u8", "u16", "f16"):
        L.append(f"  {f:4s} sessions {tot[f] / 1e9:6.2f} GB   pool {pool[f] / 1e9:6.2f} GB")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=26)   # 30 crops: one per session at least, and a fallback
    ap.add_argument("--max-mp", type=float, default=19.0)
    ap.add_argument("--sweep", type=int, default=5)
    ap.add_argument("--patches", type=int, default=12)
    ap.add_argument("--probe", action="store_true", help="the largest sampled crop only, for its peak memory")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    entries = all_entries()
    rows_path = SCRATCH / "rows.jsonl"
    if not args.summary_only:
        sample = draw(entries, args.n, args.max_mp)
        if args.probe:
            sample = [max(sample, key=lambda e: e["mp"])]
        print(f"{len(sample)} crops: " + ", ".join(f"{e['session']}:{e['mp']:.1f}MP" for e in sample), flush=True)
        rng = np.random.default_rng(37)
        rows, pats = [], []
        with open(rows_path if not args.probe else SCRATCH / "probe.jsonl", "w") as fh:
            for i, e in enumerate(sample):
                row, p = measure(e, i < args.sweep and not args.probe, i < args.patches and not args.probe, rng)
                pats += p
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"{i + 1}/{len(sample)} {row['session']} {row['crop_mp']:.1f} MP ({row['win_mp']:.1f}): u8 {row['u8']['bytes']/1e6:.1f}"
                      f" MB, u16 {row['u16']['bytes']/1e6:.1f} MB {row['u16']['enc_s']:.1f}s, f16 "
                      f"{row['f16']['bytes']/1e6:.1f} MB {row['f16']['enc_s']:.1f}s; rss {row['peak_rss_mb']:.0f} MB",
                      flush=True)
        if args.probe:
            return
        np.savez(SCRATCH / "patches.npz", patches=np.stack(pats))
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    lines = summarise(rows, entries)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
