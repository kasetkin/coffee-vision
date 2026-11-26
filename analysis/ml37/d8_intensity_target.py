"""ML-37 D8: does the JPEG XL intensity target change the lossy linear half-float canonical photo?

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d8_intensity_target.py [--scale 2] [--n 20]
        [--targets 0,80,203,255,1000,10000] [--out tmp/ml37/d8]

A sample of dataset originals (tmp/ml29_scan/files.csv, ML-29's scan; seed 0, stratified by camera and gain map, at
most 12.5 MP) is decoded as the store decodes it (coffeecv.canonical.decode, STORE_SETTINGS: colour declaration
honoured, gain map applied) and turned into D1's light: linear Rec.2020, 1.0 = reference white, light above white
kept, negatives clipped at 0 (D7 a). It is area-averaged by --scale on each side (the encode time; said in the note).

That light is encoded at the store's (quality 95, effort 7) and the front's (90, 3) settings:
  - "f16 IT=<t>": lossy half-float JPEG XL, linear transfer, BT2100 primaries, D65, intensity_target t (0: libjxl's
    default), through libjxl 0.12.0 from imagecodecs' wheel via ctypes (the struct layouts of coffeecv.jxl_encode,
    the same chunked encoder, only the pixel type, transfer and intensity target differ).
  - "u16 709": today's canonical photo: the same light through D33's EETF at its peak (if above white), clipped,
    LittleCMS to coffeecv's REC2020 (BT.709 curve), rounded to uint16, coffeecv.jxl_encode.encode_rec2020 unchanged.
  - "u16 709 IT=203": the same with intensity_target 203 (does the target move today's encode too?).
  - rate-distortion neighbours: f16 IT=255 at quality q-1, q-2 and IT=203 at q+1, q+2 (is the target only a
    quality knob?).

Each file is decoded (imagecodecs.jpegxl_decode) and read back in float32 as D2 says: linear Rec.2020 -> 3x3 matrix
-> linear sRGB -> D6 (a), the EETF at the source's peak, in Rec.2020 before the matrix -> clip [0, 1] -> sRGB curve ->
x 255 (float levels, no rounding: D3). The source goes through the same read-back. Error (ML-29 D27's measure on 32 x 32
block means, here on float levels): mean and largest absolute block difference over all blocks and channels; also the
pixel RMSE, and the block mean error on dark blocks (block mean luma < 64 levels). The intensity target each file
declares is read back with libjxl's decoder (JxlDecoderGetBasicInfo).

Writes <out>/d8_rows.csv (one row per photo, setting and variant) and prints the summary (kept as
analysis/ml37/d8_intensity_target.txt). Reads data/ and dataset/ only.
"""
from __future__ import annotations

import argparse
import csv
import ctypes as C
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import imagecodecs as ic
import numpy as np

from coffeecv import canonical as cn
from coffeecv import jxl_encode as je
from coffeecv import tone_map as tm

JXL_TYPE_FLOAT16 = 5                     # lib/include/jxl/types.h
JXL_TRANSFER_FUNCTION_LINEAR = 8         # lib/include/jxl/color_encoding.h
JXL_DEC_BASIC_INFO, JXL_DEC_SUCCESS = 0x40, 0
THREADS = 2

_jxl = je._jxl
_P = C.c_void_p
je._sig(_jxl, "JxlDecoderCreate", _P, _P)
je._sig(_jxl, "JxlDecoderDestroy", None, _P)
je._sig(_jxl, "JxlDecoderSubscribeEvents", C.c_int, _P, C.c_int)
je._sig(_jxl, "JxlDecoderSetInput", C.c_int, _P, C.c_char_p, C.c_size_t)
je._sig(_jxl, "JxlDecoderCloseInput", None, _P)
je._sig(_jxl, "JxlDecoderProcessInput", C.c_int, _P)
je._sig(_jxl, "JxlDecoderGetBasicInfo", C.c_int, _P, C.POINTER(je.BasicInfo))


def basic_info(data: bytes) -> je.BasicInfo:
    dec = _jxl.JxlDecoderCreate(None)
    try:
        assert _jxl.JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO) == JXL_DEC_SUCCESS
        assert _jxl.JxlDecoderSetInput(dec, data, len(data)) == JXL_DEC_SUCCESS
        _jxl.JxlDecoderCloseInput(dec)
        status = _jxl.JxlDecoderProcessInput(dec)
        assert status == JXL_DEC_BASIC_INFO, status
        info = je.BasicInfo()
        assert _jxl.JxlDecoderGetBasicInfo(dec, C.byref(info)) == JXL_DEC_SUCCESS
        return info
    finally:
        _jxl.JxlDecoderDestroy(dec)


def encode(a: np.ndarray, *, level: float, effort: int, intensity_target: float, half: bool) -> bytes:
    """coffeecv.jxl_encode.encode_rec2020's chunked encode, with the pixel type (float16 or uint16), the transfer
    (linear or BT.709) and the intensity target (0: libjxl's default) as parameters. The whole array is in memory."""
    h, w = a.shape[:2]
    a = np.ascontiguousarray(a)
    out = bytearray()
    pos = [0]
    outbuf: list = [None]
    errors: list[BaseException] = []
    held: dict[int, np.ndarray] = {}
    lock = threading.Lock()
    dtype_code = JXL_TYPE_FLOAT16 if half else je.JXL_TYPE_UINT16

    def get_format(_o, fmt):
        fmt.contents.num_channels, fmt.contents.data_type = 3, dtype_code
        fmt.contents.endianness, fmt.contents.align = je.JXL_NATIVE_ENDIAN, 0

    def get_data(_o, xpos, ypos, xsize, ysize, row_offset):
        try:
            r = np.ascontiguousarray(a[ypos:ypos + ysize, xpos:xpos + xsize])
            row_offset[0] = r.strides[0]
            with lock:
                held[r.ctypes.data] = r
            return r.ctypes.data
        except BaseException as e:
            errors.append(e)
            return None

    def get_ec_format(_o, _i, fmt):
        get_format(None, fmt)

    def get_ec_data(*_args):
        errors.append(RuntimeError("extra channel asked for"))
        return None

    def release_input(_o, buf):
        with lock:
            held.pop(buf, None)

    def get_buffer(_o, size):
        n = max(int(size[0]), 1 << 16)
        outbuf[0] = (C.c_uint8 * n)()
        size[0] = n
        return C.addressof(outbuf[0])

    def release_output(_o, written):
        end = pos[0] + written
        if end > len(out):
            out.extend(bytes(end - len(out)))
        out[pos[0]:end] = C.string_at(C.addressof(outbuf[0]), written)
        pos[0] = end

    def seek(_o, position):
        pos[0] = position

    def finalized(_o, _p):
        pass

    cbs = (je.GET_FORMAT(get_format), je.GET_DATA(get_data), je.GET_EC_FORMAT(get_ec_format),
           je.GET_EC_DATA(get_ec_data), je.RELEASE_INPUT(release_input), je.GET_BUFFER(get_buffer),
           je.RELEASE_OUTPUT(release_output), je.SEEK(seek), je.FINALIZED(finalized))
    enc = _jxl.JxlEncoderCreate(None)
    runner = je._threads.JxlThreadParallelRunnerCreate(None, THREADS)
    try:
        je._check(_jxl.JxlEncoderSetParallelRunner(enc, je._RUNNER, runner), "runner", enc)
        info = je.BasicInfo()
        _jxl.JxlEncoderInitBasicInfo(C.byref(info))
        info.xsize, info.ysize, info.num_color_channels, info.num_extra_channels = w, h, 3, 0
        info.bits_per_sample, info.exponent_bits_per_sample = 16, (5 if half else 0)
        info.uses_original_profile = 0
        info.intensity_target = intensity_target
        je._check(_jxl.JxlEncoderSetBasicInfo(enc, C.byref(info)), "basic info", enc)
        ce = je.ColorEncoding(color_space=je.JXL_COLOR_SPACE_RGB, white_point=je.JXL_WHITE_POINT_D65,
                              primaries=je.JXL_PRIMARIES_2100,
                              transfer_function=JXL_TRANSFER_FUNCTION_LINEAR if half else je.JXL_TRANSFER_FUNCTION_709,
                              rendering_intent=je.JXL_RENDERING_INTENT_RELATIVE)
        je._check(_jxl.JxlEncoderSetColorEncoding(enc, C.byref(ce)), "colour", enc)
        je._check(_jxl.JxlEncoderUseContainer(enc, 0), "container", enc)
        s = _jxl.JxlEncoderFrameSettingsCreate(enc, None)
        je._check(_jxl.JxlEncoderSetFrameDistance(s, _jxl.JxlEncoderDistanceFromQuality(level)), "distance", enc)
        je._check(_jxl.JxlEncoderFrameSettingsSetOption(s, je.JXL_ENC_FRAME_SETTING_EFFORT, effort), "effort", enc)
        fh = je.FrameHeader()
        _jxl.JxlEncoderInitFrameHeader(C.byref(fh))
        fh.duration = 1
        je._check(_jxl.JxlEncoderSetFrameHeader(s, C.byref(fh)), "frame header", enc)
        je._check(_jxl.JxlEncoderSetOutputProcessor(enc, je.OutputProcessor(None, *cbs[5:])), "output", enc)
        st = _jxl.JxlEncoderAddChunkedFrame(s, 1, je.ChunkedInput(None, *cbs[:5]))
        if errors:
            raise errors[0]
        je._check(st, "add frame", enc)
        je._check(_jxl.JxlEncoderFlushInput(enc), "flush", enc)
    finally:
        _jxl.JxlEncoderDestroy(enc)
        je._threads.JxlThreadParallelRunnerDestroy(runner)
    return bytes(out)


# ----- colour maths (float32) -----

def _xyz_matrix(prim, white=(0.3127, 0.3290)):
    def xyz(x, y):
        return np.array([x / y, 1.0, (1 - x - y) / y])
    m = np.stack([xyz(*p) for p in prim], axis=1)
    return m * np.linalg.solve(m, xyz(*white))


REC2020_PRIM = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))
SRGB_PRIM = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
REC2020_TO_SRGB = (np.linalg.inv(_xyz_matrix(SRGB_PRIM)) @ _xyz_matrix(REC2020_PRIM)).astype(np.float32)
SRGB_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def srgb_curve(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def read_back(lin2020: np.ndarray, peak: float) -> np.ndarray:
    """D2 in float32: (D6 a: EETF at the source's peak) -> matrix -> clip -> sRGB curve -> levels 0-255."""
    c = lin2020.astype(np.float32, copy=True)
    np.nan_to_num(c, copy=False, nan=0.0)
    if peak > 1:
        tm.eetf_in_place(c, peak)
    c = c @ REC2020_TO_SRGB.T
    np.clip(c, 0, 1, out=c)
    return (srgb_curve(c) * 255).astype(np.float32)


def blocks(a: np.ndarray, b: int = 32) -> np.ndarray:
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b, -1).mean(axis=(1, 3))


def errors(got: np.ndarray, ref: np.ndarray) -> dict:
    bg, br = blocks(got.astype(np.float64)), blocks(ref.astype(np.float64))
    d = np.abs(bg - br)
    dark = (br @ SRGB_LUMA.astype(np.float64)) < 64
    return {"block_mean": float(d.mean()), "block_max": float(d.max()),
            "dark_block_mean": float(d[dark].mean()) if dark.any() else float("nan"),
            "dark_share": float(dark.mean()),
            "pixel_rmse": float(np.sqrt(np.mean((got.astype(np.float64) - ref) ** 2)))}


# ----- the sample -----

def sample(n: int, seed: int = 0) -> list[dict]:
    rows = list(csv.DictReader(open("tmp/ml29_scan/files.csv")))
    groups = defaultdict(list)
    for r in rows:
        try:
            mp = float(r["megapixels"] or 0)
        except ValueError:
            continue
        if r["type"] not in ("JPEG", "HEIC") or not (5 <= mp <= 17.5) or not Path(r["file"]).exists():
            continue
        has_map = bool(r["gain_map"])
        groups[(r["make"] + " " + r["model"], has_map)].append(r)
    rng = random.Random(seed)
    keys = sorted(groups)
    for k in keys:
        rng.shuffle(groups[k])
    picked: list[dict] = []
    gm_keys = [k for k in keys if k[1]]
    sdr_keys = [k for k in keys if not k[1]]
    i = 0
    while len(picked) < n and i < 10:   # round robin, gain-map and not alternating
        for ks in (gm_keys, sdr_keys):
            for k in ks:
                if len(picked) < n and i < len(groups[k]) and sum(1 for p in picked if p["_k"] == k) < 3:
                    picked.append({**groups[k][i], "_k": k})
        i += 1
    return picked[:n]


def light_of(path: str, scale: int) -> tuple[np.ndarray, bool]:
    """D1's light, linear Rec.2020 float32, 1.0 white, above white kept, negatives at 0; and whether a map applied."""
    data = Path(path).read_bytes()
    src = cn.decode(data, Path(path).name)
    if src.tone_map and src.profile == cn.LINEAR_REC2020:     # a gain map applied: already linear Rec.2020 light
        light = src.pixels
    else:
        h, w = src.pixels.shape[:2]
        channels = 1 if src.pixels.ndim == 2 else src.pixels.shape[2]
        light = np.empty((h, w, 3), np.float32)
        for y in range(0, h, cn.STRIP_ROWS):
            _, f = cn._light(src.pixels[y:y + cn.STRIP_ROWS], src.full, src.linearize, 1 if channels <= 2 else 3)
            light[y:y + cn.STRIP_ROWS] = ic.cms_transform(f, src.profile or cn.SRGB, cn.LINEAR_REC2020,
                                                          intent=cn.RELATIVE, outdtype=np.float32)
    mapped = bool(src.tone_map)
    del src
    np.maximum(light, 0, out=light)
    if scale > 1:
        light = cv2.resize(light, (light.shape[1] // scale, light.shape[0] // scale), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(light), mapped


def today_u16(light: np.ndarray, peak: float) -> np.ndarray:
    """canonical.to_rec2020's float path from linear Rec.2020: EETF at the peak, LittleCMS to REC2020, clip, round."""
    out = np.empty(light.shape, np.uint16)
    for y in range(0, light.shape[0], cn.STRIP_ROWS):
        c = light[y:y + cn.STRIP_ROWS].copy()
        if peak > 1:
            tm.eetf_in_place(c, peak)
        c = ic.cms_transform(c, cn.LINEAR_REC2020, cn.REC2020, intent=cn.RELATIVE, outdtype=np.float32)
        np.clip(c, 0, 1, out=c)
        out[y:y + cn.STRIP_ROWS] = np.rint(c * 65535)
    return out


def u16_to_linear(u: np.ndarray) -> np.ndarray:
    out = np.empty(u.shape, np.float32)
    for y in range(0, u.shape[0], cn.STRIP_ROWS):
        out[y:y + cn.STRIP_ROWS] = ic.cms_transform(u[y:y + cn.STRIP_ROWS], cn.REC2020, cn.LINEAR_REC2020,
                                                     intent=cn.RELATIVE, flags=cn.NOOPTIMIZE, outdtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--targets", default="0,80,203,255,1000,10000")
    ap.add_argument("--out", default="tmp/ml37/d8")
    ap.add_argument("--save", action="store_true", help="keep the encoded files in <out>/files/")
    args = ap.parse_args()
    targets = [float(t) for t in args.targets.split(",")]
    out = Path(args.out)
    (out / "files").mkdir(parents=True, exist_ok=True)
    settings = [("store", 95, 7), ("front", 90, 3)]
    rows: list[dict] = []
    photos = sample(args.n)
    print(f"# {len(photos)} photos, scale 1/{args.scale}, targets {targets}", flush=True)
    for k, p in enumerate(photos):
        t0 = time.time()
        light, mapped = light_of(p["file"], args.scale)
        peak = tm.picture_peak(light.max(axis=-1).ravel().copy())
        ref = read_back(light, peak)
        f16 = light.astype(np.float16)
        base = {"photo": Path(p["file"]).name, "camera": p["_k"][0], "gain_map": mapped,
                "h": light.shape[0], "w": light.shape[1], "peak": round(peak, 4),
                "above_white": float((light.max(axis=-1) > 1).mean())}
        # f16 rounding alone (no coding), the floor any half-float file starts from
        rows.append({**base, "setting": "-", "variant": "f16 uncoded", "bytes": 0, "declared_it": float("nan"),
                     **errors(read_back(f16.astype(np.float32), peak), ref)})
        u16 = today_u16(light, peak)
        ref_u16 = read_back(u16_to_linear(u16), 1.0)     # today's picture, its own source (tone map already in it)
        for name, q, e in settings:
            for it in targets:
                data = encode(f16, level=q, effort=e, intensity_target=it, half=True)
                got = ic.jpegxl_decode(data, numthreads=THREADS)
                assert got.dtype == np.float16, got.dtype
                rows.append({**base, "setting": name, "variant": f"f16 IT={int(it)}", "bytes": len(data),
                             "declared_it": basic_info(data).intensity_target,
                             **errors(read_back(got.astype(np.float32), peak), ref)})
                if args.save:
                    (out / "files" / f"{k:02d}_{name}_f16_it{int(it)}.jxl").write_bytes(data)
            for it, dq in ((255.0, -1), (255.0, -2), (203.0, 1), (203.0, 2)):   # rate-distortion neighbours
                data = encode(f16, level=q + dq, effort=e, intensity_target=it, half=True)
                got = ic.jpegxl_decode(data, numthreads=THREADS)
                rows.append({**base, "setting": name, "variant": f"f16 IT={int(it)} q{q + dq}", "bytes": len(data),
                             "declared_it": basic_info(data).intensity_target,
                             **errors(read_back(got.astype(np.float32), peak), ref)})
            for label, it in (("u16 709", 0.0), ("u16 709 IT=203", 203.0)):
                data = (je.encode_rec2020(u16.shape[0], u16.shape[1],
                                          lambda y, x, ys, xs: u16[y:y + ys, x:x + xs],
                                          level=q, effort=e, numthreads=THREADS, transfer="BT709", intensity_target=255)
                        if it == 0 else encode(u16, level=q, effort=e, intensity_target=it, half=False))
                got = ic.jpegxl_decode(data, numthreads=THREADS)
                assert got.dtype == np.uint16
                rows.append({**base, "setting": name, "variant": label, "bytes": len(data),
                             "declared_it": basic_info(data).intensity_target,
                             # against today's own source (the u16 picture) and against D1's light read back
                             **errors(read_back(u16_to_linear(got), 1.0), ref_u16),
                             "vs_light_block_mean": errors(read_back(u16_to_linear(got), 1.0), ref)["block_mean"]})
        print(f"{k + 1:2d}/{len(photos)} {base['photo']} {base['camera']} map={mapped} {light.shape[1]}x"
              f"{light.shape[0]} peak {peak:.2f} {time.time() - t0:.0f}s", flush=True)
        light = f16 = u16 = ref = ref_u16 = None
    keys = sorted({k for r in rows for k in r})
    with open(out / "d8_rows.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    summarise(rows)
    import resource
    print(f"\npeak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


def summarise(rows: list[dict]) -> None:
    def table(sel, title):
        print(f"\n## {title}")
        print(f"{'setting':7s} {'variant':20s} {'decl IT':>8s} {'MB total':>9s} {'size vs IT=255':>15s} "
              f"{'blk mean':>9s} {'blk max':>8s} {'dark blk':>9s} {'pix RMSE':>9s}")
        groups = defaultdict(list)
        for r in rows:
            if sel(r):
                groups[(r["setting"], r["variant"])].append(r)
        ref_bytes = {s: {r["photo"]: r["bytes"] for r in groups.get((s, "f16 IT=255"), [])} for s in ("store", "front")}
        for (s, v), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], _order(kv[0][1]))):
            mb = sum(r["bytes"] for r in rs) / 1e6
            rel = [r["bytes"] / ref_bytes[s][r["photo"]] for r in rs if s in ref_bytes and r["photo"] in ref_bytes[s]]
            rel_s = f"{np.mean(rel):.3f} ({min(rel):.3f}-{max(rel):.3f})" if rel else ""
            dit = sorted({r["declared_it"] for r in rs if r["declared_it"] == r["declared_it"]})
            print(f"{s:7s} {v:20s} {','.join(f'{d:g}' for d in dit):>8s} {mb:9.2f} {rel_s:>15s} "
                  f"{np.mean([r['block_mean'] for r in rs]):9.3f} {max(r['block_max'] for r in rs):8.2f} "
                  f"{np.nanmean([r['dark_block_mean'] for r in rs]):9.3f} "
                  f"{np.mean([r['pixel_rmse'] for r in rs]):9.3f}")
    table(lambda r: True, "all photos (mean over photos; blk max = largest block over all photos)")
    table(lambda r: r["gain_map"], "gain-map photos (light above white kept in f16)")
    table(lambda r: not r["gain_map"], "photos without a gain map")
    u = [r for r in rows if r["variant"] == "u16 709"]
    for s in ("store", "front"):
        print(f"u16 709 {s}: block mean vs D1's light (incl. the EETF already baked in): "
              f"{np.mean([r['vs_light_block_mean'] for r in u if r['setting'] == s]):.3f}")


def _order(v: str) -> tuple:
    if v.startswith("f16 IT="):
        t = v.split("=")[1].split()
        return (1, float(t[0]), t[1] if len(t) > 1 else "")
    return (0 if v == "f16 uncoded" else 2, 0.0, v)


if __name__ == "__main__":
    sys.exit(main())
