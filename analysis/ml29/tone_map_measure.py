"""P4's measurement: each tone-mapping candidate (coffeecv.tone_map) on three sets, as the models read the result
(ticket ML-29, D11, D27).

    python analysis/ml29/tone_map_measure.py darktable    # (i)   about 4 min
    python analysis/ml29/tone_map_measure.py libjxl       # (i)'s JPEG XL files as libjxl itself renders them
    python analysis/ml29/tone_map_measure.py public       # (ii)  under a minute
    python analysis/ml29/tone_map_measure.py bands        # the light per band behind the note's prose
    python analysis/ml29/tone_map_measure.py gainmaps     # (iii) about 30 min
    python analysis/ml29/tone_map_measure.py gainmaps-table   # (iii)'s summary again, from its rows
    python analysis/ml29/tone_map_measure.py page         # the owner's page, from the crops the three wrote

Every file's samples are read from its bytes (imagecodecs for JPEG XL and AVIF, pillow-heif for HEIC) and its colour
declaration by coffeecv.colour_declaration, as P5 will read them; the curve is undone by coffeecv.tone_map's own
BT.2100 functions under a reference-white rule, the primaries taken to Rec.2020 by matrix (Bradford), then each
candidate brings the light into [0, 1]. The result goes through the canonical encode as linear Rec.2020 at 16 bits
(canonical.Source(..., LINEAR_REC2020) and canonical.encode, store settings, 4 threads) and is read back with
load_canonical, 8-bit sRGB. Its reference, a standard-range file of the same picture, takes the same path with no
operator. The metric is D27's: 32 x 32 block means of the 8-bit read-back, "mean" the mean absolute difference over
every block and channel, "worst" the largest; "shift" the mean signed difference (positive: brighter).

(i) darktable: the PQ and HLG JPEG XL exports (6984 x 4660) and the PQ and HLG files made by make_test_files.py
(1746 x 1164), against the 8-bit sRGB JPEG XL export. Every full-size file, the reference too, is decoded, tone-mapped
at its own size and area-averaged in linear light to a quarter (the bottom 4 rows dropped), the way the made files
were made, so all are compared at 1746 x 1164.
(ii) public: libavif's tests/data samples in tmp/ml29_samples (research_tone_mapping.md lists their URLs and
licences): the Seine PQ AVIFs against the standard-range JPEG of the same picture, the colours PQ AVIFs against
colors_sdr_srgb.avif. Small pictures: 108 and 36 blocks.
(iii) gainmaps: 5 dataset photos per camera with a hand-checked bean mask (data/seg_masks_ml19), spread over the
gain-map boost measured by gainmap_in_beans.py (its 10th to 90th percentile), the map applied at full weight by a
throwaway reader (Ultra HDR's formula for JPEG, Apple's published one for HEIC), each candidate run, against today's
canonical photo (data/canonical, read only). A block is in the bean region if half its pixels are. "null" is the
same path with no map: what the path itself changes.

Per-file and per-photo rows go to tmp/ml29_tm/ (local); crops for the owner to tmp/ml29_tonemap_crops/ (never
committed). Prints the tables research_tone_mapping.md records. Light is converted in strips, one photo at a
time, 4 threads.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imagecodecs as ic  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from coffeecv import canonical as cn  # noqa: E402
from coffeecv import colour_declaration as cd  # noqa: E402
from coffeecv import tone_map as tm  # noqa: E402
from coffeecv.jxl_header import read_header  # noqa: E402
from exports_tolerance import BLOCK, blocks  # noqa: E402

THREADS = 4
OUT = REPO / "tmp" / "ml29_tm"
CROPS = REPO / "tmp" / "ml29_tonemap_crops"
DARKTABLE = REPO / "tmp" / "darktable_exported" / "jxl"
STEM = "20261113_172701_Canon_EOS_90D_Brazil-q90-"
MADE = REPO / "tmp" / "ml29_made"
SAMPLES = REPO / "tmp" / "ml29_samples"
SCALE = 4
STRIP = 256                     # rows at a time, a multiple of SCALE
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
D65 = (0.3127, 0.3290)

# The candidates of the main table, all under BT.2408's reference white (PQ 203 cd/m2; HLG 75% with the OOTF at
# 1000 cd/m2): (name, operator, which peak). "measured": the picture's own, the 99.99th percentile of max(R, G, B),
# at least white; "declared": the file's (a JPEG XL's intensity target, a coded tag's MaxCLL, else PQ's 10,000 or
# HLG's 1000 cd/m2; a gain map's largest boost), so that "c-declared" is libjxl's own linear reading;
# "format": libultrahdr's (10,000 / 203 PQ, 1000 / 203 HLG; a gain map's largest boost).
CANDIDATES = [("a clip", "clip", "measured"), ("b EETF", "eetf", "measured"), ("c scale", "scale", "measured"),
              ("c-declared", "scale", "declared"), ("d Reinhard", "reinhard", "format")]


# ----- colour -----

def xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz(primaries, white) -> np.ndarray:
    """SMPTE RP 177: the RGB-to-XYZ matrix of six chromaticities (R x, y, G x, y, B x, y) and a white point."""
    m = np.array([xy_to_xyz(*primaries[i:i + 2]) for i in (0, 2, 4)]).T
    return m * np.linalg.solve(m, xy_to_xyz(*white))


def to_rec2020(primaries, white) -> np.ndarray:
    """Linear RGB in these primaries to linear Rec.2020 (D65), white to white (Bradford)."""
    s, d = BRADFORD @ xy_to_xyz(*white), BRADFORD @ xy_to_xyz(*D65)
    adapt = np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD
    return np.linalg.inv(rgb_to_xyz(cd.REC2020_PRIMARIES, D65)) @ adapt @ rgb_to_xyz(primaries, white)


REC2020_Y = rgb_to_xyz(cd.REC2020_PRIMARIES, D65)[1]


def srgb_to_linear(u: np.ndarray) -> np.ndarray:
    return np.where(u > 0.04045, ((u + 0.055) / 1.055) ** 2.4, u / 12.92)


def bt709_to_linear(u: np.ndarray) -> np.ndarray:
    return np.where(u < 0.081, u / 4.5, ((u + 0.099) / 1.099) ** (1 / 0.45))


# ----- reading a file -----

class Signal:
    """A file's samples as read (0..1 per channel, not yet linear), its declaration, and the peak it declares."""

    def __init__(self, path: Path):
        data = path.read_bytes()
        self.name, self.declared_nits = path.name, None
        if path.suffix == ".jxl":
            h = read_header(data)
            self.decl = cd.jxl_declaration(h.colour)
            a = ic.jpegxl_decode(data, numthreads=THREADS)
            self.full = 1.0 if a.dtype == np.float32 else float(2 ** h.bits_per_sample - 1)
            from make_test_files import jxl_decode_libjxl
            self.declared_nits = jxl_decode_libjxl(data)[0]["intensity_target"]
        elif path.suffix == ".avif":
            self.decl = cd.declare(data, "avif")
            a = ic.avif_decode(data, numthreads=THREADS)
            self.full = 255.0 if a.dtype == np.uint8 else float(2 ** cd.primary_bit_depth(data) - 1)
        elif path.suffix == ".heic":
            import pillow_heif
            heif = pillow_heif.open_heif(io.BytesIO(data), convert_hdr_to_8bit=False)
            depth = heif.info.get("bit_depth", 8)
            self.decl = cd.declare(data, "heif", heif.info.get("icc_profile"))
            a = np.asarray(heif)
            self.full = float((2**depth - 1) << (16 - depth)) if a.dtype == np.uint16 else 255.0
        else:
            raise SystemExit(f"{path.name}: not a file this script reads")
        if path.suffix in (".avif", ".heic"):
            clli = next((b for k, b in cd.primary_properties(data) if k == b"clli"), None)
            if clli:
                self.declared_nits = float(struct.unpack(">H", clli[:2])[0])
        self.samples = a
        self.matrix = to_rec2020(self.decl.primaries, self.decl.white)
        self.weights = rgb_to_xyz(self.decl.primaries, self.decl.white)[1]   # the file's own luminance, for HLG's OOTF

    @property
    def curve(self) -> str:
        return self.decl.curve

    def relative(self, rows: slice, rule: str = "bt2408") -> np.ndarray:
        """Linear Rec.2020 of these rows, 1.0 = reference white by `rule`: "bt2408" (PQ 203 cd/m2; HLG with the OOTF
        at 1000 cd/m2, white 203), "white10000" (PQ's top as white), "measured" (tone_map.measured_pq_white),
        "scene" (HLG without the OOTF, 75% white), "darktable" (darktable up to 5.6.2: HLG as the inverse OETF of
        0.96 E' + 0.04, white at signal 1); a standard-range curve ignores the rule."""
        v = self.samples[rows].astype(np.float64) / self.full
        if self.curve == "PQ":
            white = {"bt2408": tm.REFERENCE_WHITE, "white10000": tm.PQ_PEAK,
                     "measured": getattr(self, "measured_white", tm.REFERENCE_WHITE)}[rule]
            lin = tm.pq_to_relative(v, white)
        elif self.curve == "HLG":
            if rule == "darktable":
                lin = tm.hlg_inverse_oetf(0.96 * v + 0.04)
            else:
                lin = tm.hlg_to_relative(v, ootf=rule != "scene", weights=self.weights)
        elif self.curve == "sRGB":
            lin = srgb_to_linear(np.clip(v, 0, 1))
        else:
            raise SystemExit(f"{self.name}: curve {self.curve} not measured here")
        return lin @ self.matrix.T

    def peaks(self, rule: str = "bt2408") -> dict:
        """The measured, declared and format peaks, relative to white; and the share of pixels above white, the median
        and 99.9th percentile luminance in cd/m2 (luminance by BT.2100's weights)."""
        h = self.samples.shape[0]
        mx, y = [], []
        for s in range(0, h, STRIP):
            lin = self.relative(slice(s, s + STRIP), rule)
            mx.append(lin.max(-1).astype(np.float32).ravel())
            y.append((lin @ REC2020_Y).astype(np.float32).ravel())
        mx, y = np.concatenate(mx), np.concatenate(y)
        hlg = tm.hlg_peak()
        nominal = tm.PQ_PEAK / tm.REFERENCE_WHITE if self.curve == "PQ" else hlg
        declared = self.declared_nits / tm.REFERENCE_WHITE if self.declared_nits else nominal
        return {"measured": max(1.0, float(np.percentile(mx, 99.99))), "declared": declared, "format": nominal,
                "above_white": float((y > 1).mean()), "median_nits": float(np.median(y) * tm.REFERENCE_WHITE),
                "p999_nits": float(np.percentile(y, 99.9) * tm.REFERENCE_WHITE)}


def rendered(sig: Signal, op, peak: float, rule: str = "bt2408", scale: int = 1) -> np.ndarray:
    """The file's light through an operator, as linear Rec.2020 in [0, 1], area-averaged by `scale`."""
    h, w = sig.samples.shape[:2]
    h, w = (h // scale) * scale, (w // scale) * scale
    out = np.empty((h // scale, w // scale, 3), np.float32)
    for s in range(0, h, STRIP):
        part = op(sig.relative(slice(s, min(s + STRIP, h)), rule), peak)[:, :w]
        n = part.shape[0] // scale
        out[s // scale:s // scale + n] = part.reshape(n, scale, w // scale, scale, 3).mean((1, 3))
    return out


def models_read(lin: np.ndarray) -> np.ndarray:
    """Linear Rec.2020 [0, 1] through the canonical encode and back, as the models read it (8-bit sRGB)."""
    u16 = np.rint(np.clip(lin, 0, 1) * 65535).astype(np.uint16)
    src = cn.Source(u16, 65535.0, cn.LINEAR_REC2020, "tone_map_measure")
    return cn.load_canonical(cn.encode(src, numthreads=THREADS), numthreads=THREADS)


def compare(got: np.ndarray, ref: np.ndarray, sel: np.ndarray | None = None) -> dict:
    d = blocks(got.astype(np.float64)) - blocks(ref.astype(np.float64))
    if sel is not None:
        d = d[sel]
    if d.size == 0:
        return {"mean": float("nan"), "worst": float("nan"), "shift": float("nan"), "blocks": 0}
    return {"mean": float(np.abs(d).mean()), "worst": float(np.abs(d).max()), "shift": float(d.mean()),
            "blocks": int(d.size // 3)}


def save_png(a: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def operator(name: str):
    return tm.OPERATORS[name]


def run_pair(label: str, sig: Signal, ref: np.ndarray, scale: int, crop: tuple[slice, slice], set_name: str,
             rows: list[dict], extra_rules: list[tuple[str, str]]) -> None:
    """Every candidate on one file against its reference; the white rules with clip_at_white beside them."""
    t = time.monotonic()
    pk = sig.peaks()
    if sig.curve == "PQ":
        sig.measured_white = tm.measured_pq_white(np.array([pk["median_nits"]]))
    stats = {k: round(v, 4) for k, v in pk.items()}
    print(f"  {label}: {sig.curve}, peaks (x white) measured {pk['measured']:.2f}, declared {pk['declared']:.2f}; "
          f"above white {100 * pk['above_white']:.1f}%, median {pk['median_nits']:.0f} cd/m2, "
          f"99.9th pct {pk['p999_nits']:.0f} cd/m2", flush=True)
    save_png(ref[crop], CROPS / set_name / f"{label}__reference.png")
    runs = [(n, op, pk[p], "bt2408") for n, op, p in CANDIDATES]
    runs += [(n, "clip", 1.0, rule) for n, rule in extra_rules]
    for name, op, peak, rule in runs:
        got = models_read(rendered(sig, operator(op), peak, rule, scale))
        if got.shape[1:] != ref.shape[1:] or abs(got.shape[0] - ref.shape[0]) > 1:
            raise SystemExit(f"{label}: {got.shape} against the reference's {ref.shape}")
        rows_ = min(got.shape[0], ref.shape[0])      # a made file lost the export's bottom 4 rows: 1164, not 1165
        r = compare(got[:rows_], ref[:rows_])
        rows.append({"set": set_name, "file": label, "candidate": name, "rule": rule, "peak": round(peak, 3),
                     **{k: round(v, 3) for k, v in r.items()}, **stats})
        save_png(got[crop], CROPS / set_name / f"{label}__{name.replace(' ', '_')}.png")
        print(f"    {name:28s} mean {r['mean']:6.2f}  worst {r['worst']:6.1f}  shift {r['shift']:+6.2f}", flush=True)
    print(f"    ({time.monotonic() - t:.0f} s)", flush=True)


# ----- the sets -----

def darktable() -> None:
    ref_sig = Signal(DARKTABLE / f"{STEM}8bit-sRGB.jxl")
    ref_lin = rendered(ref_sig, tm.clip_at_white, 1.0, scale=SCALE)
    del ref_sig
    ref = models_read(ref_lin)
    print(f"reference: the 8-bit sRGB export, {ref.shape[1]}x{ref.shape[0]} after the quarter")
    # The owner's crop: the 384 x 384 window whose reference holds the brightest block mean
    yb = blocks(np.repeat((ref_lin @ REC2020_Y)[..., None], 3, axis=2))[..., 0]
    by, bx = np.unravel_index(np.argmax(yb), yb.shape)
    cy = int(np.clip(by * BLOCK + BLOCK // 2 - 192, 0, ref.shape[0] - 384))
    cx = int(np.clip(bx * BLOCK + BLOCK // 2 - 192, 0, ref.shape[1] - 384))
    crop = (slice(cy, cy + 384), slice(cx, cx + 384))
    del ref_lin
    files = [DARKTABLE / f"{STEM}8bit-PQRec2020.jxl", DARKTABLE / f"{STEM}8bit-HLGRec2020.jxl",
             *(MADE / n for n in ("pq-p3-16bit.jxl", "pq-rec2020-highlights-to-812nits-16bit.jxl",
                                  "pq-rec2020-10bit.avif", "pq-rec2020-12bit.heic", "hlg-p3-16bit.jxl",
                                  "hlg-rec2020-10bit.avif", "hlg-rec2020-12bit.heic"))]
    rows: list[dict] = []
    for path in files:
        sig = Signal(path)
        label = path.stem.replace(STEM, "darktable-")
        scale = SCALE if sig.samples.shape[1] > ref.shape[1] else 1
        extra = ([("PQ white 10000 cd/m2, clip", "white10000"), ("PQ measured white, clip", "measured")]
                 if sig.curve == "PQ" else
                 [("HLG scene light, clip", "scene"), ("HLG darktable <=5.6.2, clip", "darktable")])
        run_pair(label, sig, ref, scale, crop, "darktable", rows, extra)
        del sig
    write_rows(rows, OUT / "darktable.csv")
    table(rows)


def public() -> None:
    sdr = Image.open(SAMPLES / "seine_sdr_gainmap_srgb.jpg")
    icc = sdr.info.get("icc_profile")
    a = np.asarray(sdr.convert("RGB"), np.float32) / 255
    lin = ic.cms_transform(a, icc, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.float32)
    refs = {"seine": models_read(np.clip(lin, 0, 1))}
    csig = Signal(SAMPLES / "colors_sdr_srgb.avif")
    refs["colors"] = models_read(rendered(csig, tm.clip_at_white, 1.0))
    rows: list[dict] = []
    for name in ("seine_hdr_rec2020.avif", "seine_hdr_srgb.avif", "seine_hdr_gainmap_srgb.avif",
                 "colors_hdr_rec2020.avif", "colors_hdr_p3.avif", "colors_hdr_srgb.avif"):
        sig = Signal(SAMPLES / name)
        ref = refs[name.split("_")[0]]
        run_pair(Path(name).stem, sig, ref, 1, (slice(None), slice(None)), "public", rows,
                 [("PQ white 10000 cd/m2, clip", "white10000"), ("PQ measured white, clip", "measured")])
    for name, ref in refs.items():
        save_png(ref, CROPS / "public" / f"{name}__reference.png")
    write_rows(rows, OUT / "public.csv")
    table(rows)


def table(rows: list[dict]) -> None:
    print("\n| File | " + " | ".join(n for n, _, _ in CANDIDATES) + " |")
    print("|---|" + "---|" * len(CANDIDATES))
    for f in dict.fromkeys(r["file"] for r in rows):
        cells = {r["candidate"]: r for r in rows if r["file"] == f}
        print(f"| {f} | " + " | ".join(f"{cells[n]['mean']:.2f}; {cells[n]['worst']:.1f}" for n, _, _ in CANDIDATES)
              + " |")
    print("\nWhite rules, clip_at_white:")
    for r in rows:
        if r["candidate"] not in {n for n, _, _ in CANDIDATES}:
            print(f"  {r['file']:45s} {r['candidate']:30s} mean {r['mean']:.2f}  worst {r['worst']:.1f}  "
                  f"shift {r['shift']:+.2f}")


# ----- libjxl's own standard-range rendering of the JPEG XL files in (i) -----

def libjxl_rendering(data: bytes, desired: float) -> np.ndarray:
    """libjxl's decode to linear Rec.2020 float with JxlDecoderSetDesiredIntensityTarget(desired): its ToneMapping
    stage (lib/jxl/render_pipeline/stage_tone_mapping.cc), BT.2408's EETF on luminance from [0, intensity target]
    to [0, desired] for PQ, an HLG OOTF towards `desired` for HLG, then its gamut map; 1.0 = `desired` cd/m2."""
    import ctypes as C

    import make_test_files as m
    f = m._jxl.JxlDecoderSetDesiredIntensityTarget
    f.restype, f.argtypes = C.c_int, [C.c_void_p, C.c_float]
    dec = m._jxl.JxlDecoderCreate(None)
    try:
        m._jxl.JxlDecoderSubscribeEvents(dec, m.DEC_BASIC_INFO | m.DEC_COLOR_ENCODING | m.DEC_FULL_IMAGE)
        if f(dec, desired):
            raise RuntimeError("libjxl refused the desired intensity target")
        m._jxl.JxlDecoderSetInput(dec, data, len(data))
        m._jxl.JxlDecoderCloseInput(dec)
        info, fmt, out = m.BasicInfo(), m.PixelFormat(3, m.JXL_TYPE_FLOAT, 0, 0), None
        while True:
            status = m._jxl.JxlDecoderProcessInput(dec)
            if status == m.DEC_BASIC_INFO:
                m._jxl.JxlDecoderGetBasicInfo(dec, C.byref(info))
            elif status == m.DEC_COLOR_ENCODING:
                if m._jxl.JxlDecoderSetPreferredColorProfile(dec, C.byref(m.ColorEncoding(**m.LINEAR_REC2020))):
                    raise RuntimeError("libjxl refused linear Rec.2020")
            elif status == m.DEC_NEED_IMAGE_OUT_BUFFER:
                n = C.c_size_t()
                m._jxl.JxlDecoderImageOutBufferSize(dec, C.byref(fmt), C.byref(n))
                out = np.empty((info.ysize, info.xsize, 3), np.float32)
                m._jxl.JxlDecoderSetImageOutBuffer(dec, C.byref(fmt), out.ctypes.data, n.value)
            elif status == m.DEC_FULL_IMAGE:
                continue
            elif status == 0:
                return out
            else:
                raise RuntimeError(f"JxlDecoderProcessInput: {status}")
    finally:
        m._jxl.JxlDecoderDestroy(dec)


def libjxl() -> None:
    """(i)'s JPEG XL files as libjxl renders them for a standard-range display of 203 and of 255 cd/m2 (libjxl's
    default SDR intensity target), clipped, against the same reference."""
    ref_sig = Signal(DARKTABLE / f"{STEM}8bit-sRGB.jxl")
    ref = models_read(rendered(ref_sig, tm.clip_at_white, 1.0, scale=SCALE))
    del ref_sig
    rows = []
    for path in (DARKTABLE / f"{STEM}8bit-PQRec2020.jxl", DARKTABLE / f"{STEM}8bit-HLGRec2020.jxl",
                 MADE / "pq-p3-16bit.jxl", MADE / "pq-rec2020-highlights-to-812nits-16bit.jxl",
                 MADE / "hlg-p3-16bit.jxl"):
        for desired in (203.0, 255.0):
            lin = libjxl_rendering(path.read_bytes(), desired)
            h, w = lin.shape[:2]
            if w > ref.shape[1]:
                h, w = (h // SCALE) * SCALE, (w // SCALE) * SCALE
                small = np.empty((h // SCALE, w // SCALE, 3), np.float32)
                for s in range(0, h, STRIP):
                    part = np.clip(lin[s:s + STRIP, :w], 0, 1)
                    small[s // SCALE:(s + part.shape[0]) // SCALE] = part.reshape(
                        part.shape[0] // SCALE, SCALE, w // SCALE, SCALE, 3).mean((1, 3))
                lin = small
            got = models_read(lin)
            rows_ = min(got.shape[0], ref.shape[0])
            r = compare(got[:rows_], ref[:rows_])
            label = path.stem.replace(STEM, "darktable-")
            rows.append({"file": label, "desired": desired, **{k: round(v, 3) for k, v in r.items()}})
            print(f"  {label:45s} libjxl to {desired:.0f} cd/m2: mean {r['mean']:6.2f}  worst {r['worst']:6.1f}  "
                  f"shift {r['shift']:+6.2f}", flush=True)
            del lin
    write_rows(rows, OUT / "libjxl.csv")


# ----- light per band: how each pair relates, behind the note's prose -----

def bands() -> None:
    """Per band of the standard-range reference's luminance, the median ratio of the other file's light to it: the
    owner's HLG export read by darktable <= 5.6.2's curve; Seine and the colours in cd/m2 per unit of standard-range
    light. And the share of the owner's PQ and HLG samples at the top of the signal, and of the sRGB export at white."""
    cuts = [0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.01]

    def show(title: str, y_ref: np.ndarray, y: np.ndarray) -> None:
        print(title)
        for lo, hi in zip(cuts, cuts[1:]):
            k = (y_ref > lo) & (y_ref < hi)
            if k.sum() > 20:
                print(f"  reference Y {lo:5.3f}-{hi:4.2f}: {k.sum():8d} px, ratio {np.median(y[k] / y_ref[k]):8.3f}, "
                      f"other's median {np.median(y[k]):8.3f}")

    ref = Signal(DARKTABLE / f"{STEM}8bit-sRGB.jxl")
    y_ref = rendered(ref, tm.clip_at_white, 1.0, scale=SCALE) @ REC2020_Y
    print(f"sRGB export at white (max channel >= 254.5/255): {np.mean(ref.samples.max(-1) >= 255):.2e}")
    del ref
    for name, rule in (("PQRec2020", "white10000"), ("HLGRec2020", "darktable")):
        sig = Signal(DARKTABLE / f"{STEM}8bit-{name}.jxl")
        print(f"{name}: samples at 254 or more {np.mean(sig.samples.max(-1) >= 254):.2e}")
        show(f"{name} read as rule {rule!r}, ratio to the sRGB export",
             y_ref, rendered(sig, lambda x, p: x, 1.0, rule, scale=SCALE) @ REC2020_Y)
        del sig
    sdr = Image.open(SAMPLES / "seine_sdr_gainmap_srgb.jpg")
    a = np.asarray(sdr.convert("RGB"), np.float32) / 255
    y_seine = ic.cms_transform(a, sdr.info.get("icc_profile"), cn.LINEAR_REC2020, intent=cn.RELATIVE,
                               outdtype=np.float32).astype(np.float64) @ REC2020_Y
    y_colors = rendered(Signal(SAMPLES / "colors_sdr_srgb.avif"), tm.clip_at_white, 1.0) @ REC2020_Y
    for name, y_sdr in (("seine_hdr_rec2020.avif", y_seine), ("colors_hdr_rec2020.avif", y_colors)):
        nits = Signal(SAMPLES / name).relative(slice(None)) @ REC2020_Y * tm.REFERENCE_WHITE
        show(f"{name}: cd/m2 per unit of standard-range light", y_sdr, nits)


# ----- (iii) gain maps -----

def chosen_photos(per_camera: int = 5) -> list[dict]:
    rows = list(csv.DictReader((REPO / "tmp" / "ml29_scan" / "gainmap_in_beans.csv").open()))
    picked = []
    for make in sorted({r["make"] for r in rows}):
        g = sorted((r for r in rows if r["make"] == make), key=lambda r: (float(r["bean_mean"]), r["file"]))
        idx = sorted({int(round(q * (len(g) - 1))) for q in np.linspace(0.1, 0.9, per_camera)})
        picked += [g[i] for i in idx]
    return picked


def first(value: str | None, default: float) -> float:
    return float(str(value).split(",")[0]) if value not in (None, "") else default


def gain(path: Path, kind: str, detail: dict, shape: tuple[int, int]) -> tuple[np.ndarray, float, float, float]:
    """The map's gain as a linear factor at the main image's size and orientation, its largest gain, and the two
    offsets: Ultra HDR (developer.android.com, "Display": log_boost = min + (max - min) * map ** (1 / gamma)) or Apple
    ("Applying Apple HDR effect to your photos": 1 + (headroom - 1) * map, the map undone by the Rec.709 curve)."""
    d = path.read_bytes()
    if kind == "JPEG":
        at = d.find(b"hdrgm:GainMapM")
        soi = d.rfind(b"\xff\xd8\xff", 0, at)
        with Image.open(io.BytesIO(d[soi:])) as im:
            rec = np.asarray(im.convert("L"), np.float32) / 255
        lo, hi = first(detail.get("GainMapMin"), 0.0), first(detail.get("GainMapMax"), 1.0)
        g = 2.0 ** (lo + (hi - lo) * rec ** (1 / first(detail.get("Gamma"), 1.0)))
        g = cn.orient(g, cn.applied_orientation(d, path.name))
        k_sdr, k_hdr, top = first(detail.get("OffsetSDR"), 1 / 64), first(detail.get("OffsetHDR"), 1 / 64), 2.0**hi
    else:
        import pillow_heif
        heif = pillow_heif.open_heif(io.BytesIO(d), convert_hdr_to_8bit=True)
        ids = next(v for k, v in heif.info["aux"].items() if k.endswith("hdrgainmap"))
        rec = np.asarray(heif.get_aux_image(ids[0]), np.float32) / 255
        headroom = float(detail["headroom"])
        g, k_sdr, k_hdr, top = 1 + (headroom - 1) * bt709_to_linear(rec), 0.0, 0.0, headroom
    if abs(g.shape[1] / g.shape[0] - shape[1] / shape[0]) > 0.02:
        raise ValueError("map and photo differ in shape")
    g = np.asarray(Image.fromarray(np.ascontiguousarray(g, np.float32)).resize((shape[1], shape[0]), Image.BILINEAR))
    return g, float(top), k_sdr, k_hdr


def main_linear(data: bytes, name: str) -> np.ndarray:
    """The main image as canonicalization reads it (decode, upright, its declaration), as linear Rec.2020 float32."""
    src = cn.decode(data, name)
    h, w = src.pixels.shape[:2]
    out = np.empty((h, w, 3), np.float32)
    for s in range(0, h, STRIP):
        f = src.pixels[s:s + STRIP, :, :3].astype(np.float32) / src.full
        if src.linearize is not None:
            src.linearize(f)
        out[s:s + STRIP] = ic.cms_transform(f, src.profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE,
                                            outdtype=np.float32)
    return out


def gainmaps() -> None:
    masks = REPO / "data" / "seg_masks_ml19" / "xl0_v7r_s123"
    rows, skipped = [], []
    runs = [("null", None, None)] + [(n, op, p) for n, op, p in CANDIDATES if n != "c-declared"] + \
        [("c-declared", "scale", "declared")]
    for i, r in enumerate(chosen_photos()):
        path = Path(r["file"])
        rel = path.relative_to(REPO / "dataset")
        label = f"{r['make'].split()[0]}-{i:02d}"
        canonical = cn.canonical_path(path)
        mask_path = masks / ("__".join(rel.with_suffix("").parts) + ".png")
        if not canonical.exists() or not mask_path.exists():
            skipped.append((label, "no canonical photo or mask"))
            continue
        t = time.monotonic()
        try:
            today = cn.load_canonical(canonical.read_bytes(), numthreads=THREADS)
            lin = main_linear(path.read_bytes(), path.name)
            g, top, k_sdr, k_hdr = gain(path, r["type"], json.loads(
                next(x for x in csv.DictReader((REPO / "tmp" / "ml29_scan" / "files.csv").open())
                     if x["file"] == r["file"])["gain_map_detail"]), lin.shape[:2])
        except (ValueError, OSError, cn.Unreadable, StopIteration) as exc:
            skipped.append((label, type(exc).__name__ + ": " + str(exc)[:80]))
            continue
        if today.shape != lin.shape:
            skipped.append((label, f"canonical {today.shape} against decoded {lin.shape}"))
            continue
        with Image.open(mask_path) as im:
            if abs(im.width / im.height - lin.shape[1] / lin.shape[0]) > 0.02:
                skipped.append((label, "mask and photo differ in shape"))
                continue
            mask = np.asarray(im.convert("L").resize((lin.shape[1], lin.shape[0]), Image.BILINEAR)) > 127
        frac = blocks(np.repeat(mask[..., None].astype(np.float32), 3, axis=2))[..., 0]
        bean_blocks = np.repeat((frac >= 0.5)[..., None], 3, axis=2)
        hdr_max = []
        for s in range(0, lin.shape[0], STRIP):
            part = (lin[s:s + STRIP] + k_sdr) * g[s:s + STRIP, :, None] - k_hdr
            hdr_max.append(part.max(-1).ravel())
        peaks = {"measured": max(1.0, float(np.percentile(np.concatenate(hdr_max), 99.99))), "declared": top,
                 "format": top}
        del hdr_max
        ys, xs = np.nonzero(mask)
        cy, cx = (int(np.median(ys)), int(np.median(xs))) if ys.size else (lin.shape[0] // 2, lin.shape[1] // 2)
        size = min(768, *lin.shape[:2])
        cy, cx = int(np.clip(cy - size // 2, 0, lin.shape[0] - size)), int(np.clip(cx - size // 2, 0, lin.shape[1] - size))
        crop = (slice(cy, cy + size), slice(cx, cx + size))
        save_png(today[crop], CROPS / "gainmaps" / f"{label}__today.png")
        for name, op, which in runs:
            out = np.empty(lin.shape, np.float32)
            for s in range(0, lin.shape[0], STRIP):
                if op is None:
                    out[s:s + STRIP] = np.clip(lin[s:s + STRIP], 0, 1)
                else:
                    hdr = (lin[s:s + STRIP] + k_sdr) * g[s:s + STRIP, :, None] - k_hdr
                    out[s:s + STRIP] = operator(op)(hdr, peaks[which])
            got = models_read(out)
            del out
            bean, rest = compare(got, today, bean_blocks), compare(got, today, ~bean_blocks)
            rows.append({"photo": label, "make": r["make"], "candidate": name,
                         "peak": round(peaks[which], 3) if which else "",
                         **{f"bean_{k}": round(v, 3) for k, v in bean.items()},
                         **{f"rest_{k}": round(v, 3) for k, v in rest.items()}})
            save_png(got[crop], CROPS / "gainmaps" / f"{label}__{name.replace(' ', '_')}.png")
            print(f"  {label} {name:12s} beans: mean {bean['mean']:5.2f} worst {bean['worst']:5.1f} "
                  f"shift {bean['shift']:+6.2f} | rest: mean {rest['mean']:5.2f} shift {rest['shift']:+6.2f}",
                  flush=True)
        print(f"  {label}: map's top {top:.2f}, measured peak {peaks['measured']:.2f} ({time.monotonic() - t:.0f} s)",
              flush=True)
        del lin, g, today
    write_rows(rows, OUT / "gainmaps.csv")
    print(f"\nskipped: {skipped}")
    gainmap_table(rows)


def gainmap_table(rows: list[dict] | None = None) -> None:
    """(iii)'s summary per camera and candidate: the median photo's mean |change| and mean shift in the bean region,
    the worst block of any photo there, and the same outside it (a photo whose blocks are all bean has no outside)."""
    if rows is None:
        rows = list(csv.DictReader((OUT / "gainmaps.csv").open()))
    print("\n| Camera | Candidate | Photos | Beans: mean | shift | worst | Outside: mean | shift |")
    print("|---|---|---|---|---|---|---|---|")
    for make in dict.fromkeys(r["make"] for r in rows):
        for name in dict.fromkeys(r["candidate"] for r in rows):
            g = [r for r in rows if r["make"] == make and r["candidate"] == name]
            med = lambda k: float(np.nanmedian([float(r[k]) for r in g]))  # noqa: E731
            print(f"| {make} | {name} | {len(g)} | {med('bean_mean'):.2f} | {med('bean_shift'):+.2f} | "
                  f"{max(float(r['bean_worst']) for r in g):.1f} | {med('rest_mean'):.2f} | {med('rest_shift'):+.2f} |")


# ----- the owner's page -----

def page() -> None:
    from html import escape
    from urllib.parse import quote
    parts = ["<!doctype html><meta charset=utf-8><title>ML-29 P4: tone-mapping candidates</title>",
             "<style>body{font-family:sans-serif;background:#222;color:#ddd}img{width:256px;image-rendering:auto}"
             "td{vertical-align:top;font-size:12px;padding:2px}</style>",
             "<h1>ML-29 P4: tone-mapping candidates, as the models read them</h1>",
             "<p>Local only (tmp/), never committed. Each row: the reference (the standard-range file, or today's "
             "canonical photo), then each candidate. Numbers: tmp/ml29_tm/*.csv; the note: "
             "analysis/ml29/research_tone_mapping.md.</p>"]
    for set_name in ("darktable", "public", "gainmaps"):
        d = CROPS / set_name
        if not d.exists():
            continue
        csv_rows = list(csv.DictReader((OUT / f"{set_name}.csv").open())) if (OUT / f"{set_name}.csv").exists() else []
        parts.append(f"<h2>{set_name}</h2><table>")
        for label in sorted({p.name.split("__")[0] for p in d.glob("*.png")}):
            imgs = sorted(d.glob(f"{label}__*.png"), key=lambda p: (not p.stem.endswith(("reference", "today")), p.name))
            if set_name == "public" and not any(not p.stem.endswith("reference") for p in imgs):
                continue
            if set_name == "public":
                imgs = [d / f"{label.split('_')[0]}__reference.png"] + [p for p in imgs if "reference" not in p.stem]
            cells = []
            for p in imgs:
                cand = p.stem.split("__")[1].replace("_", " ")
                m = next((r for r in csv_rows if r.get("file", r.get("photo")) == label and r["candidate"] == cand),
                         None)
                note = ""
                if m:
                    note = (f"mean {m['mean']}, worst {m['worst']}" if "mean" in m
                            else f"beans {m['bean_mean']} ({m['bean_shift']}), rest {m['rest_mean']}")
                cells.append(f"<td><img src='{set_name}/{quote(p.name)}'><br>{escape(cand)}<br>{note}</td>")
            parts.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
        parts.append("</table>")
    (CROPS / "index.html").write_text("\n".join(parts))
    print(CROPS / "index.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", choices=("darktable", "libjxl", "public", "bands", "gainmaps", "gainmaps-table", "page"))
    a = ap.parse_args()
    {"darktable": darktable, "libjxl": libjxl, "public": public, "bands": bands, "gainmaps": gainmaps,
     "gainmaps-table": gainmap_table, "page": page}[a.set]()
