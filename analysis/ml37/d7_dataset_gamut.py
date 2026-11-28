"""ML-37 D7: how much of the dataset lies outside Rec.2020 (negative linear samples) when canonicalization converts it,
and what values above 1.0 the conversion itself makes.

    OMP_NUM_THREADS=2 PYTHONPATH=. uv run python analysis/ml37/d7_dataset_gamut.py profiles
    OMP_NUM_THREADS=2 PYTHONPATH=. uv run python analysis/ml37/d7_dataset_gamut.py photos
    OMP_NUM_THREADS=2 PYTHONPATH=. uv run python analysis/ml37/d7_dataset_gamut.py raw
    OMP_NUM_THREADS=2 PYTHONPATH=. uv run python analysis/ml37/d7_dataset_gamut.py raw_jxl

profiles  Every 8-bit RGB triple (2^24) through each colour source the dataset carries (analysis/ml29/README.md: the
          ICC profiles, untagged sRGB, the JPEG XL sRGB enumeration), taken from one photo each by
          coffeecv.canonical.decode, so the profile and the curve undone are canonicalization's own; LittleCMS float,
          relative colorimetric, into linear Rec.2020 (canonical.LINEAR_REC2020, D1's target), unclipped. ProPhoto
          (not in the dataset) as the wide-gamut reference. Exhaustive for 8-bit sources.
photos    A sample of dataset photos (SAMPLE below, chosen by camera and colour source from the ML-29 scan) decoded as
          the store does (STORE_SETTINGS: gain maps applied), converted strip by strip into linear Rec.2020 unclipped:
          the share of pixels with a channel below 0 (and below -1e-3), and above 1.0. Then the read-back: D2's matrix
          to linear sRGB with (a) Rec.2020 clipped at 0 first, (b) negatives kept; clip to [0, 1]; sRGB curve; the
          largest difference in 8-bit levels. And D33's peak (99.99th percentile of max(R, G, B)) in Rec.2020 and
          after the matrix, in linear sRGB.
raw       The RAW photos of the sample: LibRaw clips its matrix output to 0-65535 (convert_to_rgb_loop), so the share
          LibRaw would have made negative is measured from its camera-RGB output (output_color raw, the same
          demosaic and white balance) times its own out_cam (rec2020_rgb x rgb_cam); half_size to stay in memory.
          The JPEG XL LinearRaw DNGs go through coffeecv.linear_dng's steps without the clip.
No photo names or metadata are printed beyond camera make and colour source.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import imagecodecs as ic
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coffeecv import canonical as cn  # noqa: E402
from coffeecv import linear_dng  # noqa: E402
from coffeecv import tone_map as tm  # noqa: E402

DS = REPO / "dataset"
NO_MAPS = cn.Settings("probe", "float", 7, 95, cn.KEEP_MAX_PIXELS, cn.CAP_PIXELS, gain_maps=False)


def rgb_to_xyz(prim, white=(0.3127, 0.3290)):
    xr, yr, xg, yg, xb, yb = prim
    xw, yw = white
    m = np.array([[xr / yr, xg / yg, xb / yb], [1, 1, 1], [(1 - xr - yr) / yr, (1 - xg - yg) / yg, (1 - xb - yb) / yb]])
    return m * np.linalg.solve(m, np.array([xw / yw, 1, (1 - xw - yw) / yw]))


SRGB_FROM_2020 = (np.linalg.inv(rgb_to_xyz((0.64, 0.33, 0.30, 0.60, 0.15, 0.06)))
                  @ rgb_to_xyz((0.708, 0.292, 0.170, 0.797, 0.131, 0.046))).astype(np.float32)


def srgb_curve(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def readback(rec: np.ndarray) -> np.ndarray:
    """D2 with a clip (no tone map): linear Rec.2020 -> matrix -> clip [0, 1] -> sRGB curve, in 8-bit levels."""
    return srgb_curve(np.clip(rec @ SRGB_FROM_2020.T, 0, 1)) * 255


# ----- profiles -----

def sources() -> dict[str, Path]:
    """One dataset photo per colour source (the ML-29 scan's list), found by reading, not by name."""
    import csv
    rows = list(csv.DictReader(open(REPO / "tmp/ml29_scan/files.csv")))
    want = {}
    for r in rows:
        key = r["read_today_as"]
        if key.startswith("developed") or not Path(r["file"]).exists() or key in want:
            continue
        want[key] = Path(r["file"])
    return want


def cube_stats(profile, linearize) -> dict:
    neg = neg3 = neg4 = over = over4 = 0
    lo, hi = np.inf, -np.inf
    n = 1 << 24
    chunk = 1 << 20
    idx = np.arange(chunk, dtype=np.uint32)
    for s in range(0, n, chunk):
        v = idx + s
        f = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.float32)[None] / 255
        if linearize is not None:
            linearize(f)
        c = ic.cms_transform(f, profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.float32)[0]
        mn, mx = c.min(-1), c.max(-1)
        neg += int((mn < 0).sum())
        neg4 += int((mn < -1e-4).sum())
        neg3 += int((mn < -1e-3).sum())
        over += int((mx > 1).sum())
        over4 += int((mx > 1 + 1e-4).sum())
        lo, hi = min(lo, float(mn.min())), max(hi, float(mx.max()))
    return dict(neg=neg / n, neg4=neg4 / n, neg3=neg3 / n, over=over / n, over4=over4 / n, lo=lo, hi=hi)


def run_profiles():
    print("| colour source (one dataset photo each) | triples < 0 | < -1e-4 | < -1e-3 | lowest | > 1 | > 1.0001 "
          "| highest |")
    print("|---|---|---|---|---|---|---|---|")
    rows = list(sources().items())
    prophoto = ic.cms_profile("rgb", whitepoint=(0.3457, 0.3585),
                              primaries=(0.7347, 0.2653, 0.1596, 0.8404, 0.0366, 0.0001), gamma=1.8)
    for key, path in rows + [("ProPhoto RGB (reference, not in the dataset)", None)]:
        if path is None:
            profile, linearize = prophoto, None
        else:
            src = cn.decode(path.read_bytes(), path.name, settings=NO_MAPS)
            profile, linearize = src.profile, src.linearize
            del src
            gc.collect()
        s = cube_stats(profile, linearize)
        print(f"| {key} | {s['neg']:.4%} | {s['neg4']:.4%} | {s['neg3']:.4%} | {s['lo']:+.6f} | {s['over']:.4%} "
              f"| {s['over4']:.4%} | {s['hi']:.6f} |", flush=True)


# ----- photos -----

# (glob under dataset/, how many): chosen from the ML-29 scan's colour sources and cameras, files at or under 25 MP
SAMPLE = [
    ("2026-08-30__pixel/**/PXL_*.jpg", 4),          # Google, Display P3 ICC, Ultra HDR gain map
    ("2026-09-24__pixel/**/PXL_*.jpg", 2),
    ("2026-08-30__oneplus/**/*.jpg", 3),            # OnePlus 7 Pro (sRGB ICC, gain map) / other OnePlus (untagged)
    ("2026-09-24__oneplus/**/*.jpg", 2),
    ("2026-08-30__sony/**/*.JPG", 3),               # Sony G8441, untagged sRGB
    ("2026-08-25__iphone/**/*.HEIC", 4),            # Apple HEIC, Display P3, gain map
    ("ood_positives_internet/**/*.jpg", 4),         # web photos: c2, Adobe RGB, Display P3 sRGB-transfer, sRGB, ...
    ("massimo/**/*.jpg", 2),
    ("tasty_coffee/**/*.jpg", 2),
    ("random_date_raccoon/**/*Galaxy*.jpg", 2),     # Samsung, DCI-P3 D65 ICC, gain map with ISO 21496-1
]


def pick() -> list[Path]:
    import csv
    meta = {r["file"]: r for r in csv.DictReader(open(REPO / "tmp/ml29_scan/files.csv"))}
    out = []
    for pattern, n in SAMPLE:
        files = sorted(DS.glob(pattern))
        files = [f for f in files if float(meta.get(str(f), {}).get("megapixels") or 0) <= 25]
        if not files:
            continue
        step = max(1, len(files) // n)
        out += files[::step][:n]
    # the less common colour sources and cameras, one photo each, whatever folder they are in
    seen = set()
    for f, r in meta.items():
        src = r["read_today_as"]
        if r["make"] in ("OnePlus", "Xiaomi", "samsung") and not r["gain_map"]:
            src = f"{src} {r['make']}"
        elif not src.startswith("ICC"):
            continue
        if src not in seen and Path(f).exists() and float(r["megapixels"] or 0) <= 25:
            seen.add(src)
            if Path(f) not in out:
                out.append(Path(f))
    return out


def photo_stats(path: Path) -> dict:
    src = cn.decode(path.read_bytes(), path.name)            # STORE_SETTINGS: gain maps applied
    h, w = src.pixels.shape[:2]
    channels = 1 if src.pixels.ndim == 2 else src.pixels.shape[2]
    colour = 1 if channels <= 2 else 3
    n = neg = neg3 = over = srgb_over = differ = 0
    lo, worst = 0.0, 0.0
    max2020 = np.empty(h * w, np.float32)
    maxsrgb = np.empty(h * w, np.float32)
    for y in range(0, h, cn.STRIP_ROWS):
        _, f = cn._light(src.pixels[y:y + cn.STRIP_ROWS], src.full, src.linearize, colour)
        if src.profile is cn.LINEAR_REC2020 and src.pixels.dtype == np.float32:
            c = f                                            # gain-map light: already linear Rec.2020
        else:
            c = ic.cms_transform(f, src.profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE,
                                 outdtype=np.float32)
        c = c.reshape(-1, 3)
        mn, mx = c.min(-1), c.max(-1)
        n += len(c)
        neg += int((mn < 0).sum())
        neg3 += int((mn < -1e-3).sum())
        over += int((mx > 1.001).sum())
        lo = min(lo, float(mn.min()))
        rows = slice(y * w, y * w + len(c))
        max2020[rows] = mx
        s = c @ SRGB_FROM_2020.T
        maxsrgb[rows] = s.max(-1)
        srgb_over += int((s.max(-1) > 1.001).sum())
        bad = mn < 0
        if bad.any():
            a, b = readback(np.maximum(c[bad], 0)), readback(c[bad])
            d = np.abs(a - b).max(-1)
            differ += int((d >= 0.5).sum())
            worst = max(worst, float(d.max()))
    out = dict(kind=f"{src.decoder}", tone_map=src.tone_map, n=n, neg=neg / n, neg3=neg3 / n, lo=lo,
               over=over / n, srgb_over=srgb_over / n, differ=differ / n, worst=worst,
               peak2020=tm.picture_peak(max2020), peaksrgb=tm.picture_peak(maxsrgb))
    del src, max2020, maxsrgb
    gc.collect()
    return out


def describe(path: Path) -> str:
    import csv
    meta = {r["file"]: r for r in csv.DictReader(open(REPO / "tmp/ml29_scan/files.csv"))}
    r = meta.get(str(path))
    if r is None:
        return f"{path.suffix.lower()} (after the ML-29 scan)"
    return f"{r['type']} {r['make'] or '-'}, {r['read_today_as'][:42]}{', gain map' if r['gain_map'] else ''}"


def run_photos():
    files = pick()
    print(f"{len(files)} photos")
    print("| photo (camera, colour source) | decoder | px < 0 | px < -1e-3 | lowest | px > 1.001 (Rec.2020) "
          "| px > 1.001 after the matrix (sRGB) | peak Rec.2020 | peak after matrix | px whose read-back differs >= 0.5 "
          "level, (a) vs (b) | largest difference (levels) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        try:
            s = photo_stats(f)
        except Exception as exc:                             # report and go on
            print(f"| {describe(f)} | failed: {type(exc).__name__}: {exc} |")
            continue
        print(f"| {describe(f)} | {s['kind']} | {s['neg']:.4%} | {s['neg3']:.4%} | {s['lo']:+.5f} | {s['over']:.3%} "
              f"| {s['srgb_over']:.3%} | {s['peak2020']:.4f} | {s['peaksrgb']:.4f} | {s['differ']:.4%} "
              f"| {s['worst']:.2f} |", flush=True)


# ----- RAW -----

RAW_SAMPLE = [
    ("2026-09-24__pixel/**/*.dng", 3), ("segmenter_positives/*.dng", 2), ("**/*iPhone*.dng", 0),
    ("random_date_raccoon/**/*Pixel*.dng", 1), ("random_date_raccoon/**/*Galaxy*.dng", 2), ("**/*.CR3", 2),
    ("**/*.cr3", 0),
]
XYZ_FROM_SRGB = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])   # LibRaw's xyz_rgb (colorconst.cpp), dcraw's
REC2020_RGB = np.array([[0.627452, 0.329249, 0.043299], [0.069109, 0.919531, 0.011360],
                        [0.016398, 0.088030, 0.895572]])   # LibRaw 0.22.1 src/tables/colorconst.cpp


def raw_files() -> list[Path]:
    import csv
    meta = {r["file"]: r for r in csv.DictReader(open(REPO / "tmp/ml29_scan/files.csv"))}
    out = []
    for pattern, n in RAW_SAMPLE:
        fs = sorted(DS.glob(pattern))
        if n:
            out += [f for f in fs[::max(1, len(fs) // n)][:n] if f not in out]
    for make in ("Apple", "OnePlus"):                       # one of each other camera
        for f, r in meta.items():
            if r["type"] == "DNG" and r["make"] == make and Path(f).exists():
                if Path(f) not in out:
                    out.append(Path(f))
                break
    return out


def raw_stats(path: Path) -> tuple[str, dict]:
    import io

    import rawpy
    data = path.read_bytes()
    dng = linear_dng.parse(data) if path.suffix.lower() == ".dng" else None
    if dng is not None:                                    # linear_dng.develop's steps 1-3 without the clip
        light = linear_dng._stored(dng)
        mult = 1 / dng.neutral
        light *= (mult / mult.min()).astype(np.float32)
        np.minimum(light, 1, out=light)
        cfo = dng.colour_matrix @ linear_dng.XYZ_FROM["rec2020"]
        cfo /= cfo.sum(axis=1, keepdims=True)
        m = np.linalg.inv(cfo).astype(np.float32)
        light = light[::2, ::2]                            # every other pixel, as half_size keeps a quarter
        flat = light.reshape(-1, 3)
        kind = "dngjxl (linear_dng)"
    else:
        with rawpy.imread(io.BytesIO(data)) as raw:
            cam = raw.postprocess(output_bps=16, use_camera_wb=True, output_color=rawpy.ColorSpace.raw,
                                  gamma=(1, 1), no_auto_bright=True, half_size=True)
            rgb_cam = np.array(raw.color_matrix, np.float64)[:, :3]
            if rgb_cam[0, 0] <= 0.125:                 # LibRaw's own test (identify.cpp): no usable file matrix, so
                cam_xyz = np.array(raw.rgb_xyz_matrix, np.float64)[:3]   # rgb_cam from its table (cam_xyz_coeff)
                cam_rgb = cam_xyz @ XYZ_FROM_SRGB
                cam_rgb /= cam_rgb.sum(axis=1, keepdims=True)
                rgb_cam = np.linalg.inv(cam_rgb)
            colors = raw.num_colors
        if colors != 3:
            return "rawpy, 4 colours", {}
        m = (REC2020_RGB @ rgb_cam).astype(np.float32)
        flat = cam.reshape(-1, 3).astype(np.float32) / 65535
        kind = "rawpy (LibRaw)"
    n = len(flat)
    neg = neg3 = differ = 0
    lo = worst = 0.0
    # LibRaw's auto-bright (canonicalization's rawpy call keeps it on): white is the largest per-channel level with 1%
    # of the pixels above it, on the clipped output (dcraw's auto_bright_thr 0.01); measured here, then every pixel is
    # divided by it before the read-back, as LibRaw's gamma_curve(..., t_white << 3) scales its output
    white = max(float(np.percentile(np.clip(flat @ m[c], 0, 1), 99)) for c in range(3)) if dng is None else 1.0
    for s in range(0, n, 1 << 20):
        o = flat[s:s + (1 << 20)] @ m.T
        mn = o.min(-1)
        neg += int((mn < 0).sum())
        neg3 += int((mn < -1e-3).sum())
        lo = min(lo, float(mn.min()))
        bad = mn < 0
        if bad.any():
            o = o[bad] / white
            d = np.abs(readback(np.maximum(o, 0)) - readback(o)).max(-1)
            differ += int((d >= 0.5).sum())
            worst = max(worst, float(d.max()))
    return kind, dict(neg=neg / n, neg3=neg3 / n, lo=lo, n=n, differ=differ / n, worst=worst, white=white)


def run_raw():
    files = raw_files()
    print(f"{len(files)} RAW photos (half size)")
    print("| RAW (camera) | developer | px with a channel < 0 before the clip | < -1e-3 | lowest (1.0 = white before "
          "auto-bright) | auto-bright white | px whose read-back differs >= 0.5 level, (a) vs (b) | largest (levels) |")
    print("|---|---|---|---|---|---|---|---|")
    for f in files:
        try:
            kind, s = raw_stats(f)
        except Exception as exc:
            print(f"| {describe(f)} | failed: {type(exc).__name__}: {exc} |")
            continue
        if not s:
            print(f"| {describe(f)} | {kind} | - | - | - | - | - | - |")
            continue
        print(f"| {describe(f)} | {kind} | {s['neg']:.3%} | {s['neg3']:.3%} | {s['lo']:+.4f} | {s['white']:.3f} "
              f"| {s['differ']:.3%} | {s['worst']:.1f} |", flush=True)
        gc.collect()


if __name__ == "__main__" and sys.argv[1] != "raw_jxl":
    {"profiles": run_profiles, "photos": run_photos, "raw": run_raw}[sys.argv[1]]()


# ----- RAW through the canonical photo's encode -----

OPSIN = np.array([[0.30, 1 - 0.078 - 0.30, 0.078], [0.23, 1 - 0.078 - 0.23, 0.078],
                  [0.24342268924547819, 0.20476744424496821, 1 - 0.24342268924547819 - 0.20476744424496821]])
BIAS = 0.0037930732552754493                               # libjxl 0.12 cms/opsin_params.h; enc_xyb.cc clamps at 0


def clamped(rec: np.ndarray, intensity_target: float) -> np.ndarray:
    """The pixels whose opsin 'mixed' value libjxl's encoder sets to 0 (enc_xyb.cc LinearRGBRowToXYB)."""
    return ((rec @ SRGB_FROM_2020.T) @ OPSIN.T * (intensity_target / 255) + BIAS < 0).any(-1)


def run_raw_jxl():
    """The RAW photos with the most negatives: the 768x768 tile (half size) holding most of them, linear Rec.2020
    after auto-bright, encoded as D1's canonical photo (float16, BT2100 primaries, linear, level 95, effort 7) with its
    negatives (a) clipped at 0 and (b) kept; both decoded and read back (D2 with a clip, 8-bit sRGB levels)."""
    import io

    import rawpy
    files = [f for f in raw_files() if describe(f).startswith(("CR3", "DNG Google"))]
    print("| RAW | tile px < 0 | px libjxl clamps (intensity target 255 / 203) | read-back, decoded (b) vs unencoded (b): "
          "mean, px >= 2 levels | decoded (a) vs unencoded (a): mean, px >= 2 | decoded (b) vs decoded (a): mean, "
          "px >= 2, max | unencoded (b) vs (a): px >= 2 |")
    print("|---|---|---|---|---|---|---|")
    for f in files:
        with rawpy.imread(io.BytesIO(f.read_bytes())) as raw:
            cam = raw.postprocess(output_bps=16, use_camera_wb=True, output_color=rawpy.ColorSpace.raw,
                                  gamma=(1, 1), no_auto_bright=True, half_size=True)
            rgb_cam = np.array(raw.color_matrix, np.float64)[:, :3]
            if rgb_cam[0, 0] <= 0.125:
                cam_rgb = np.array(raw.rgb_xyz_matrix, np.float64)[:3] @ XYZ_FROM_SRGB
                cam_rgb /= cam_rgb.sum(axis=1, keepdims=True)
                rgb_cam = np.linalg.inv(cam_rgb)
        m = (REC2020_RGB @ rgb_cam).astype(np.float32)
        o = cam.astype(np.float32) / 65535 @ m.T
        del cam
        white = max(float(np.percentile(np.clip(o[..., c], 0, 1), 99)) for c in range(3))
        o /= white
        neg = (o < 0).any(-1).astype(np.float32)
        t = 768
        h, w = neg.shape
        best, at = -1.0, (0, 0)
        for y in range(0, h - t + 1, t // 2):
            for x in range(0, w - t + 1, t // 2):
                s = float(neg[y:y + t, x:x + t].sum())
                if s > best:
                    best, at = s, (y, x)
        b = np.ascontiguousarray(o[at[0]:at[0] + t, at[1]:at[1] + t])
        del o, neg
        gc.collect()
        a = np.maximum(b, 0)
        enc = dict(level=95, effort=7, photometric="RGB", primaries="BT2100", numthreads=2)
        db = ic.jpegxl_decode(ic.jpegxl_encode(b.astype(np.float16), **enc), numthreads=2).astype(np.float32)
        da = ic.jpegxl_decode(ic.jpegxl_encode(a.astype(np.float16), **enc), numthreads=2).astype(np.float32)
        flat = [v.reshape(-1, 3).astype(np.float64) for v in (b, a, db, da)]
        rb, ra, rdb, rda = (readback(v) for v in flat)

        def cmp(x, y):
            d = np.abs(x - y).max(-1)
            return float(d.mean()), float((d >= 2).mean()), float(d.max())
        e_b, e_a, e_ba, i_ba = cmp(rdb, rb), cmp(rda, ra), cmp(rdb, rda), cmp(rb, ra)
        def blocks(r):                                     # 32x32 block means of the read-back, as ML-29's "Error"
            return r.reshape(t // 32, 32, t // 32, 32, 3).mean((1, 3)).reshape(-1, 3)
        again = ic.jpegxl_decode(ic.jpegxl_encode(a.astype(np.float16), **enc), numthreads=2).astype(np.float32)
        deterministic = np.array_equal(again, da)
        bb = [cmp(blocks(x), blocks(y)) for x, y in ((rdb, rb), (rda, ra), (rdb, rda), (rb, ra))]
        c255, c203 = float(clamped(flat[0], 255).mean()), float(clamped(flat[0], 203).mean())
        print(f"| {describe(f)} | {(flat[0] < 0).any(-1).mean():.2%} | {c255:.2%} / {c203:.2%} "
              f"| {e_b[0]:.2f}, {e_b[1]:.2%} | {e_a[0]:.2f}, {e_a[1]:.2%} | {e_ba[0]:.2f}, {e_ba[1]:.2%}, {e_ba[2]:.0f} "
              f"| {i_ba[1]:.2%} |", flush=True)
        print(f"|  32x32 block means (levels: mean, share >= 2, max); encode deterministic {deterministic} | | "
              + " | ".join(f"{m:.2f}, {p:.1%}, {x:.1f}" for m, p, x in bb[:3]) + f" | {bb[3][0]:.2f}, {bb[3][2]:.1f} |",
              flush=True)
        gc.collect()


if __name__ == "__main__" and sys.argv[1] == "raw_jxl":
    run_raw_jxl()
