"""Test files for ML-29 (D25): colour declarations darktable did not export, made from its exports' own pixels.

    python analysis/ml29/make_test_files.py [--src tmp/darktable_exported/jxl] [--out tmp/ml29_made] [--scale 4]

The owner gives no new exports (D25), so the declarations the owner's 22 darktable exports lack are made here from
the 16-bit linear Rec.2020 JPEG XL export (header BT2100, Linear, D65), area-averaged in linear light by --scale
(4: 6984x4656 to 1746x1164, the bottom 4 rows dropped so both sides stay even) and re-encoded:
- JPEG XL, 16-bit integer, lossy at quality 90: linear Rec.709, Rec.709 (BT.709 transfer), PQ P3, HLG P3 through
  imagecodecs.jpegxl_encode's primaries=/transfer=; linear ProPhoto (custom primaries and a D50 white, which
  jpegxl_encode refuses) through libjxl's JxlEncoderSetColorEncoding over ctypes, the libjxl inside imagecodecs'
  wheel, as coffeecv/jxl_encode.py reaches it; and a PQ Rec.2020 file with highlights above reference white.
- AVIF (imagecodecs.avif_encode, 10-bit) and HEIC (pillow-heif's x265, 12-bit), PQ and HLG, Rec.2020 primaries,
  BT.2020 non-constant-luminance matrix, full range, 4:2:0, quality 90.

The standard-range picture (white = 1) goes into PQ and HLG with its diffuse white at BT.2408's reference, 203 cd/m2,
as display light: PQ is BT.2100's inverse EOTF of 203 * L / 10000; HLG is BT.2100's inverse OOTF (nominal peak
1000 cd/m2, black 0, system gamma 1.2, luminance from the file's own primaries) then its OETF, which puts white at
75% signal. The highlights file first raises each pixel by 4 ** t, t = clip((Y - 0.25) / 0.75, 0, 1), Y its Rec.2020
luminance: below Y 0.25 nothing changes, white goes to 4 (812 cd/m2).

Each file is then read back from its bytes: the JPEG XL header (coffeecv.jxl_header, and libjxl's decoder for custom
chromaticities and the intensity target), the AVIF/HEIC primary item's colour boxes and pixi (scan_dataset.heif).
Its samples are decoded to linear Rec.2020 by this file's own maths, driven by the declaration read back, not by
what was asked for, and compared with what was meant (the source after the target's gamut clip); errors are in
linear light, white 1. For JPEG XL, libjxl's own conversion of the same file to linear Rec.2020 (scaled by its
intensity target / 203 for PQ and HLG) is compared with this file's maths, at the 99.9th percentile: libjxl's float
output keeps the codec's overshoot past 0 and 1, which the integer samples clip. Prints the table
analysis/ml29/README.md records. Writes only under --out (local, never committed); about 30 s.
"""
from __future__ import annotations

import argparse
import ctypes as C
import io
import sys
from pathlib import Path

import imagecodecs
import numpy as np
import pillow_heif

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coffeecv.jxl_encode import BasicInfo, ColorEncoding, PixelFormat, _jxl  # noqa: E402
from coffeecv.jxl_header import read_header  # noqa: E402
from scan_dataset import heif  # noqa: E402

# ----- colour maths, from the chromaticities -----

D65, D50 = (0.3127, 0.3290), (0.3457, 0.3585)
PRIMARIES = {"srgb": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
             "p3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
             "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
             "prophoto": ((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001))}
WHITE = {"srgb": D65, "p3": D65, "rec2020": D65, "prophoto": D50}
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
NCLX_PRIMARIES = {1: "srgb", 9: "rec2020", 12: "p3"}          # ITU-T H.273 ColourPrimaries
NCLX_TRANSFERS = {1: "bt709", 8: "linear", 16: "pq", 18: "hlg"}  # H.273 TransferCharacteristics
JXL_PRIMARIES = {"sRGB": "srgb", "BT2100": "rec2020", "P3": "p3"}
JXL_TRANSFERS = {"BT709": "bt709", "Linear": "linear", "PQ": "pq", "HLG": "hlg"}
REF_WHITE = 203.0       # cd/m2, BT.2408
HLG_PEAK, HLG_GAMMA = 1000.0, 1.2   # BT.2100 Table 5: gamma 1.2 at a 1000 cd/m2 nominal peak


def xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz(prim, white) -> np.ndarray:
    """The RGB-to-XYZ matrix of a set of primaries and a white point (SMPTE RP 177)."""
    m = np.array([xy_to_xyz(*p) for p in prim]).T
    return m * np.linalg.solve(m, xy_to_xyz(*white))


def adapt(src_white, dst_white) -> np.ndarray:
    """Bradford chromatic adaptation, XYZ to XYZ."""
    s, d = BRADFORD @ xy_to_xyz(*src_white), BRADFORD @ xy_to_xyz(*dst_white)
    return np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD


def rec2020_to(prim, white) -> np.ndarray:
    """Linear Rec.2020 (D65) to linear RGB in other primaries; white stays white (Bradford, relative intent)."""
    return np.linalg.inv(rgb_to_xyz(prim, white)) @ adapt(D65, white) @ rgb_to_xyz(PRIMARIES["rec2020"], D65)


def apply(m: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    return rgb @ m.T


A_HLG = 0.17883277
B_HLG = 1 - 4 * A_HLG
C_HLG = 0.5 - A_HLG * np.log(4 * A_HLG)
M1, M2 = 2610 / 16384, 2523 / 4096 * 128
C1, C2, C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_inverse_eotf(nits):
    y = (np.clip(nits, 0, 10000) / 10000) ** M1
    return ((C1 + C2 * y) / (1 + C3 * y)) ** M2


def pq_eotf(e):
    p = np.clip(e, 0, 1) ** (1 / M2)
    return 10000 * (np.maximum(p - C1, 0) / (C2 - C3 * p)) ** (1 / M1)


def hlg_oetf(e):
    e = np.clip(e, 0, 1)
    return np.where(e <= 1 / 12, np.sqrt(3 * e), A_HLG * np.log(np.maximum(12 * e - B_HLG, 1e-12)) + C_HLG)


def hlg_inverse_oetf(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.5, v * v / 3, (np.exp((v - C_HLG) / A_HLG) + B_HLG) / 12)


def hlg_ootf(e, lum):
    """Scene light (0..1) to display light in cd/m2: alpha * Ys**(gamma-1) * E, alpha = peak, black 0."""
    ys = np.maximum(e @ lum, 1e-12)[..., None]
    return HLG_PEAK * ys ** (HLG_GAMMA - 1) * e


def hlg_inverse_ootf(fd, lum):
    yd = np.maximum(fd @ lum, 1e-12)[..., None]
    return fd / HLG_PEAK * (yd / HLG_PEAK) ** ((1 - HLG_GAMMA) / HLG_GAMMA)


def bt709_oetf(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin < 0.018, 4.5 * lin, 1.099 * lin ** 0.45 - 0.099)


def bt709_inverse_oetf(v):
    v = np.clip(v, 0, 1)
    return np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45))


def encode_signal(rel: np.ndarray, transfer: str, lum: np.ndarray) -> np.ndarray:
    """Relative display light (white = 1, in the file's primaries) to the 0..1 signal."""
    if transfer == "linear":
        return np.clip(rel, 0, 1)
    if transfer == "bt709":
        return bt709_oetf(rel)
    if transfer == "pq":
        return pq_inverse_eotf(REF_WHITE * rel)
    if transfer == "hlg":
        return hlg_oetf(hlg_inverse_ootf(REF_WHITE * np.maximum(rel, 0), lum))
    raise ValueError(transfer)


def decode_signal(sig: np.ndarray, transfer: str, lum: np.ndarray) -> np.ndarray:
    if transfer == "linear":
        return sig
    if transfer == "bt709":
        return bt709_inverse_oetf(sig)
    if transfer == "pq":
        return pq_eotf(sig) / REF_WHITE
    if transfer == "hlg":
        return hlg_ootf(hlg_inverse_oetf(sig), lum) / REF_WHITE
    raise ValueError(transfer)


# ----- libjxl over ctypes: an encoder for custom chromaticities, a reader of what libjxl itself decodes -----

_P = C.c_void_p
for _name, _res, _args in [
        ("JxlEncoderCreate", _P, [_P]), ("JxlEncoderDestroy", None, [_P]),
        ("JxlEncoderInitBasicInfo", None, [C.POINTER(BasicInfo)]),
        ("JxlEncoderSetBasicInfo", C.c_int, [_P, C.POINTER(BasicInfo)]),
        ("JxlEncoderSetColorEncoding", C.c_int, [_P, C.POINTER(ColorEncoding)]),
        ("JxlEncoderUseContainer", C.c_int, [_P, C.c_int]),
        ("JxlEncoderFrameSettingsCreate", _P, [_P, _P]),
        ("JxlEncoderSetFrameDistance", C.c_int, [_P, C.c_float]),
        ("JxlEncoderDistanceFromQuality", C.c_float, [C.c_float]),
        ("JxlEncoderAddImageFrame", C.c_int, [_P, C.POINTER(PixelFormat), _P, C.c_size_t]),
        ("JxlEncoderCloseInput", None, [_P]),
        ("JxlEncoderProcessOutput", C.c_int, [_P, C.POINTER(C.POINTER(C.c_uint8)), C.POINTER(C.c_size_t)]),
        ("JxlDecoderCreate", _P, [_P]), ("JxlDecoderDestroy", None, [_P]),
        ("JxlDecoderSubscribeEvents", C.c_int, [_P, C.c_int]),
        ("JxlDecoderSetInput", C.c_int, [_P, C.c_char_p, C.c_size_t]), ("JxlDecoderCloseInput", None, [_P]),
        ("JxlDecoderProcessInput", C.c_int, [_P]),
        ("JxlDecoderGetBasicInfo", C.c_int, [_P, C.POINTER(BasicInfo)]),
        ("JxlDecoderGetColorAsEncodedProfile", C.c_int, [_P, C.c_int, C.POINTER(ColorEncoding)]),
        ("JxlDecoderSetPreferredColorProfile", C.c_int, [_P, C.POINTER(ColorEncoding)]),
        ("JxlDecoderImageOutBufferSize", C.c_int, [_P, C.POINTER(PixelFormat), C.POINTER(C.c_size_t)]),
        ("JxlDecoderSetImageOutBuffer", C.c_int, [_P, C.POINTER(PixelFormat), _P, C.c_size_t])]:
    _f = getattr(_jxl, _name)
    _f.restype, _f.argtypes = _res, _args
JXL_TYPE_FLOAT, JXL_TYPE_UINT16 = 0, 3
DEC_BASIC_INFO, DEC_COLOR_ENCODING, DEC_NEED_IMAGE_OUT_BUFFER, DEC_FULL_IMAGE = 0x40, 0x100, 5, 0x1000
LINEAR_REC2020 = dict(color_space=0, white_point=1, primaries=9, transfer_function=8, rendering_intent=1)


def jxl_encode_custom(a: np.ndarray, prim, white, transfer: int, quality: float) -> bytes:
    """uint16 RGB to a bare lossy JPEG XL codestream with custom primaries and white point."""
    a = np.ascontiguousarray(a, np.uint16)
    enc = _jxl.JxlEncoderCreate(None)
    try:
        info = BasicInfo()
        _jxl.JxlEncoderInitBasicInfo(C.byref(info))
        info.xsize, info.ysize, info.num_color_channels, info.bits_per_sample = a.shape[1], a.shape[0], 3, 16
        info.uses_original_profile = 0
        ce = ColorEncoding(color_space=0, white_point=2, white_point_xy=white, primaries=2,
                           primaries_red_xy=prim[0], primaries_green_xy=prim[1], primaries_blue_xy=prim[2],
                           transfer_function=transfer, rendering_intent=1)
        for status, what in [(_jxl.JxlEncoderSetBasicInfo(enc, C.byref(info)), "basic info"),
                             (_jxl.JxlEncoderSetColorEncoding(enc, C.byref(ce)), "colour encoding"),
                             (_jxl.JxlEncoderUseContainer(enc, 0), "container")]:
            if status:
                raise RuntimeError(f"libjxl refused the {what}")
        settings = _jxl.JxlEncoderFrameSettingsCreate(enc, None)
        _jxl.JxlEncoderSetFrameDistance(settings, _jxl.JxlEncoderDistanceFromQuality(quality))
        fmt = PixelFormat(3, JXL_TYPE_UINT16, 0, 0)
        if _jxl.JxlEncoderAddImageFrame(settings, C.byref(fmt), a.ctypes.data, a.nbytes):
            raise RuntimeError("JxlEncoderAddImageFrame failed")
        _jxl.JxlEncoderCloseInput(enc)
        out, chunk = bytearray(), (C.c_uint8 * (1 << 20))()
        while True:
            nxt, avail = C.cast(chunk, C.POINTER(C.c_uint8)), C.c_size_t(len(chunk))
            status = _jxl.JxlEncoderProcessOutput(enc, C.byref(nxt), C.byref(avail))
            out += bytes(chunk[:len(chunk) - avail.value])
            if status == 0:
                return bytes(out)
            if status != 2:          # JXL_ENC_NEED_MORE_OUTPUT
                raise RuntimeError(f"JxlEncoderProcessOutput: {status}")
    finally:
        _jxl.JxlEncoderDestroy(enc)


def jxl_decode_libjxl(data: bytes, prefer: dict | None = None) -> tuple[dict, np.ndarray | None]:
    """libjxl's reading of a file: its colour encoding and intensity target, and, with `prefer`, the float pixels
    libjxl itself converts to that encoding."""
    dec = _jxl.JxlDecoderCreate(None)
    try:
        _jxl.JxlDecoderSubscribeEvents(dec, DEC_BASIC_INFO | DEC_COLOR_ENCODING | (DEC_FULL_IMAGE if prefer else 0))
        _jxl.JxlDecoderSetInput(dec, data, len(data))
        _jxl.JxlDecoderCloseInput(dec)
        info, ce, fmt, out, meta = BasicInfo(), ColorEncoding(), PixelFormat(3, JXL_TYPE_FLOAT, 0, 0), None, {}
        while True:
            status = _jxl.JxlDecoderProcessInput(dec)
            if status == DEC_BASIC_INFO:
                _jxl.JxlDecoderGetBasicInfo(dec, C.byref(info))
                meta["intensity_target"] = info.intensity_target
            elif status == DEC_COLOR_ENCODING:
                _jxl.JxlDecoderGetColorAsEncodedProfile(dec, 0, C.byref(ce))   # JXL_COLOR_PROFILE_TARGET_ORIGINAL
                meta.update(white=tuple(ce.white_point_xy), primaries=(tuple(ce.primaries_red_xy),
                            tuple(ce.primaries_green_xy), tuple(ce.primaries_blue_xy)))
                if not prefer:
                    return meta, None
                if _jxl.JxlDecoderSetPreferredColorProfile(dec, C.byref(ColorEncoding(**prefer))):
                    raise RuntimeError("libjxl refused the preferred colour profile")
            elif status == DEC_NEED_IMAGE_OUT_BUFFER:
                n = C.c_size_t()
                _jxl.JxlDecoderImageOutBufferSize(dec, C.byref(fmt), C.byref(n))
                out = np.empty((info.ysize, info.xsize, 3), np.float32)
                _jxl.JxlDecoderSetImageOutBuffer(dec, C.byref(fmt), out.ctypes.data, n.value)
            elif status == DEC_FULL_IMAGE:
                continue
            elif status == 0:
                return meta, out
            else:
                raise RuntimeError(f"JxlDecoderProcessInput: {status}")
    finally:
        _jxl.JxlDecoderDestroy(dec)


# ----- the files -----

# name, container, primaries, transfer, bits, highlights
FILES = [("linear-rec709-16bit.jxl", "jxl", "srgb", "linear", 16, False),
         ("rec709-16bit.jxl", "jxl", "srgb", "bt709", 16, False),
         ("pq-p3-16bit.jxl", "jxl", "p3", "pq", 16, False),
         ("hlg-p3-16bit.jxl", "jxl", "p3", "hlg", 16, False),
         ("linear-prophoto-d50-16bit.jxl", "jxl", "prophoto", "linear", 16, False),
         ("pq-rec2020-highlights-to-812nits-16bit.jxl", "jxl", "rec2020", "pq", 16, True),
         ("pq-rec2020-10bit.avif", "avif", "rec2020", "pq", 10, False),
         ("hlg-rec2020-10bit.avif", "avif", "rec2020", "hlg", 10, False),
         ("pq-rec2020-12bit.heic", "heic", "rec2020", "pq", 12, False),
         ("hlg-rec2020-12bit.heic", "heic", "rec2020", "hlg", 12, False)]
JXL_CODES = {"srgb": 1, "p3": 11, "rec2020": 9, "linear": 8, "bt709": 1, "pq": 16, "hlg": 18}
NCLX_CODES = {"rec2020": 9, "pq": 16, "hlg": 18}
QUALITY = 90


def luminance(prim_name: str) -> np.ndarray:
    return rgb_to_xyz(PRIMARIES[prim_name], WHITE[prim_name])[1]


def load_source(src: Path, scale: int) -> np.ndarray:
    path = next(src.glob("*-16bit-linearRec2020.jxl"))
    data = path.read_bytes()
    c = read_header(data).colour
    if (c.primaries, c.transfer, c.white_point) != ("BT2100", "Linear", "D65"):
        raise SystemExit(f"{path.name}: not linear Rec.2020 ({c})")
    a = imagecodecs.jpegxl_decode(data, numthreads=8).astype(np.float64) / 65535
    h, w = (a.shape[0] // (2 * scale)) * 2 * scale, (a.shape[1] // (2 * scale)) * 2 * scale
    return a[:h, :w].reshape(h // scale, scale, w // scale, scale, 3).mean(axis=(1, 3))


def highlights(lin2020: np.ndarray) -> np.ndarray:
    t = np.clip((lin2020 @ luminance("rec2020") - 0.25) / 0.75, 0, 1)
    return lin2020 * (4.0 ** t)[..., None]


def make(spec, src: np.ndarray) -> tuple[bytes, np.ndarray, float]:
    """The file's bytes, what it is meant to hold (linear Rec.2020, white 1) and the share of samples clipped."""
    name, kind, prim, transfer, bits, hi = spec
    lin = highlights(src) if hi else src
    to_file = rec2020_to(PRIMARIES[prim], WHITE[prim])
    rel = apply(to_file, lin)
    top = np.inf if transfer in ("pq", "hlg") else 1.0
    clipped = float(np.mean((rel < 0) | (rel > top)))
    rel = np.clip(rel, 0, top)
    meant = apply(np.linalg.inv(to_file), rel)
    sig = encode_signal(rel, transfer, luminance(prim))
    q = np.round(sig * (2 ** bits - 1)).astype(np.uint16)
    if kind == "jxl" and prim == "prophoto":
        data = jxl_encode_custom(q, PRIMARIES[prim], WHITE[prim], JXL_CODES[transfer], QUALITY)
    elif kind == "jxl":
        data = imagecodecs.jpegxl_encode(q, level=QUALITY, primaries=JXL_CODES[prim], transfer=JXL_CODES[transfer],
                                         numthreads=8)
    elif kind == "avif":
        data = imagecodecs.avif_encode(q, level=QUALITY, bitspersample=bits, primaries=NCLX_CODES[prim],
                                       transfer=NCLX_CODES[transfer], matrix=9, numthreads=8)
    else:   # pillow-heif takes 16-bit samples and keeps the top `bits` of them
        buf = io.BytesIO()
        pillow_heif.encode("RGB;16", (q.shape[1], q.shape[0]), (q << (16 - bits)).astype("<u2").tobytes(), buf,
                           quality=QUALITY, bit_depth=bits, color_primaries=NCLX_CODES[prim],
                           transfer_characteristics=NCLX_CODES[transfer], matrix_coefficients=9, full_range_flag=1,
                           tile_size=0)
        data = buf.getvalue()
    return data, meant, clipped


def read_back(kind: str, data: bytes) -> tuple[str, int, str, str, tuple, tuple, np.ndarray, dict]:
    """From the bytes alone: (declaration text, bits, primaries, transfer, primaries xy, white xy, samples 0..1,
    libjxl's reading)."""
    if kind == "jxl":
        h = read_header(data)
        c = h.colour
        meta, _ = jxl_decode_libjxl(data)
        prim_xy, white_xy = meta["primaries"], meta["white"]
        prim = JXL_PRIMARIES.get(c.primaries, "custom")
        text = f"{c.primaries} primaries, {c.white_point} white, {c.transfer or f'gamma {c.gamma}'}"
        if prim == "custom" or c.white_point == "Custom":
            text += " (R %.4f,%.4f G %.4f,%.4f B %.4f,%.4f; W %.4f,%.4f)" % (*sum(prim_xy, ()), *white_xy)
        text += f"; intensity target {meta['intensity_target']:g}"
        a = imagecodecs.jpegxl_decode(data, numthreads=8)
        return (text, h.bits_per_sample, prim, JXL_TRANSFERS[c.transfer], prim_xy, white_xy,
                a / (2 ** h.bits_per_sample - 1), meta)
    item = heif(data)["primary"]
    colr = item["colr"]
    bits = int(item["bits"].split("/")[0])
    p, t, m, full = _nclx(data)
    prim, transfer = NCLX_PRIMARIES[p], NCLX_TRANSFERS[t]
    text = f"colr {', '.join(colr)} (nclx {p}/{t}/{m}, full range {full})"
    if kind == "avif":
        a = imagecodecs.avif_decode(data, numthreads=8)
    else:
        im = pillow_heif.open_heif(data, convert_hdr_to_8bit=False)
        a = np.asarray(im) >> (16 - bits)
    return (text, bits, prim, transfer, PRIMARIES[prim], WHITE[prim], a / (2 ** bits - 1), {})


def _nclx(data: bytes) -> tuple[int, int, int, int]:
    """The first nclx colr box's four values (scan_dataset.heif names only primaries and transfer)."""
    at = data.find(b"colrnclx")
    p, t, m = (int.from_bytes(data[at + 8 + 2 * i:at + 10 + 2 * i], "big") for i in range(3))
    return p, t, m, data[at + 14] >> 7


def errors(got: np.ndarray, meant: np.ndarray, block: int = 8) -> tuple[float, float]:
    """Mean absolute difference per sample, and the largest over block x block means (linear light, white 1)."""
    d = got - meant
    h, w = (d.shape[0] // block) * block, (d.shape[1] // block) * block
    b = d[:h, :w].reshape(h // block, block, w // block, block, 3).mean(axis=(1, 3))
    return float(np.abs(d).mean()), float(np.abs(b).max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=REPO / "tmp/darktable_exported/jxl")
    ap.add_argument("--out", type=Path, default=REPO / "tmp/ml29_made")
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src = load_source(args.src, args.scale)
    print(f"source {src.shape[1]}x{src.shape[0]}, linear Rec.2020, max {src.max():.3f}\n")
    print("| File | Declaration read back | Bits | Clipped | Error: mean per sample; largest 8x8 block mean "
          "| Own maths against libjxl's conversion, 99.9th percentile |")
    print("|---|---|---|---|---|---|")
    for spec in FILES:
        name, kind, prim, transfer, bits, hi = spec
        data, meant, clipped = make(spec, src)
        (args.out / name).write_bytes(data)
        text, got_bits, got_prim, got_tf, prim_xy, white_xy, sig, _ = read_back(kind, data)
        if (got_prim if got_prim != "custom" else "prophoto", got_tf, got_bits) != (prim, transfer, bits):
            raise SystemExit(f"{name}: read back {got_prim}/{got_tf}/{got_bits}, meant {prim}/{transfer}/{bits}")
        # back to linear Rec.2020 from the declaration read back, not from the spec
        lum = rgb_to_xyz(prim_xy, white_xy)[1]
        rel = decode_signal(sig, got_tf, lum)
        got = apply(np.linalg.inv(rec2020_to(prim_xy, white_xy)), rel)
        mean, block = errors(got, meant)
        lj = ""
        if kind == "jxl":
            meta, f = jxl_decode_libjxl(data, LINEAR_REC2020)
            k = meta["intensity_target"] / REF_WHITE if got_tf in ("pq", "hlg") else 1.0
            lj = f"{np.percentile(np.abs(got - f * k), 99.9):.0e}"
        extra = ""
        if hi:
            y = meant @ luminance("rec2020")
            extra = f" ({100 * np.mean(y > 1):.1f}% of pixels above 203 cd/m2, peak {REF_WHITE * y.max():.0f})"
        print(f"| `{name}`{extra} | {text} | {got_bits} | {100 * clipped:.2f}% | {mean:.4f}; {block:.4f} | {lj} |")


if __name__ == "__main__":
    main()
