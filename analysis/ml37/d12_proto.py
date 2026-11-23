"""ML-37 D12 prototype: the front's canonical photo as linear Rec.2020 half-float (ticket ML-37 D1), by four colour
paths. Research code, imported by d12_front_cost.py and d12_accuracy.py only; nothing in coffeecv/ imports it, and
it changes nothing there.

The colour paths, each a function (pixels, full, profile, linearize) -> float32 linear Rec.2020, 1.0 = white:
- float   the store's float path (canonical.to_rec2020 without the clip at 1, the tone map or the BT.709 curve):
          canonical._light, then LittleCMS on float32 into linear Rec.2020 (D12 (a)).
- lcms16  LittleCMS's default (optimized) transform from the source's own 8/16-bit samples into linear Rec.2020
          uint16, then / 65535 (the integer path of ADR 0036 with a linear target profile).
- lcmsf   LittleCMS's default transform from the source's own 8/16-bit samples straight to float32. lcms2's
          AllocEmptyTransform makes any transform with a float side a FloatXFORM (cmsxform.c), so none of its
          16-bit optimizations run: this is the float path without numpy's copy.
- lut     a per-channel table built once per (profile, depth) through the float path: T(v e_c) - T(0) for every
          code v of channel c, so out = T(0) + LUT_r[r] + LUT_g[g] + LUT_b[b]. Exact for a matrix-shaper profile,
          where T is a matrix of per-channel curves; checked against the float path on random colours when built, and
          the float path is used when it does not hold.
Every path clips below 0 (ML-37 D7 (a)) and keeps light above 1 (D1). The gain map is applied as
canonical._with_gain_map does, but its light is kept in float16 (the canonical photo's own type) and already clipped,
so the encode reads slices of it.

The encode: chunked (copied from coffeecv/jxl_encode.encode_rec2020, JXL_TYPE_FLOAT16, exponent bits 5, transfer
LINEAR) or whole (imagecodecs.jpegxl_encode on one float16 array). The intensity target is left at libjxl's default
(0 -> 255 for a linear transfer, lib/jxl/luminance.cc): ML-37 D8 is open.
"""
from __future__ import annotations

import ctypes as C
import threading

import imagecodecs as ic
import numpy as np

from coffeecv import canonical as cn
from coffeecv import gain_map
from coffeecv import jxl_encode as je

JXL_TYPE_FLOAT16 = 5                 # libjxl 0.12.0 lib/include/jxl/types.h
JXL_TRANSFER_FUNCTION_LINEAR = 8     # lib/include/jxl/color_encoding.h
STRIP = cn.STRIP_ROWS


# ----- the colour paths -----

def _rgb_only(pixels: np.ndarray) -> None:
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise NotImplementedError("the prototype takes RGB without alpha (every photo measured)")


def light_float(part, full, profile, linearize):
    _rgb_only(part)
    _, f = cn._light(part, full, linearize, 3)
    return ic.cms_transform(f, profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.float32)


def _scaled(part, full):
    """canonical._to_rec2020_integer's scaling: pillow-heif's 10/12-bit samples moved so `full` is the dtype's top."""
    top = np.iinfo(part.dtype).max
    if full != top:
        f = int(full)
        part = np.minimum((part.astype(np.uint32) * top + f // 2) // f, top).astype(part.dtype)
    return np.ascontiguousarray(part)


def light_lcms16(part, full, profile, linearize):
    if part.dtype.kind != "u" or linearize is not None:
        return light_float(part, full, profile, linearize)
    _rgb_only(part)
    u = ic.cms_transform(_scaled(part, full), profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE,
                         outdtype=np.uint16)
    f = u.astype(np.float32)
    f *= np.float32(1 / 65535)
    return f


def light_lcmsf(part, full, profile, linearize):
    if part.dtype.kind != "u" or linearize is not None:
        return light_float(part, full, profile, linearize)
    _rgb_only(part)
    return ic.cms_transform(_scaled(part, full), profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE,
                            outdtype=np.float32)


_LUTS: dict = {}
LUT_CHECK_TOL = 1e-5                 # linear light; an 8-bit sRGB level near black is 3e-4


def lut_for(profile, dtype, full, linearize):
    key = (profile, np.dtype(dtype).str, float(full), id(linearize))
    if key not in _LUTS:
        n = np.iinfo(dtype).max + 1
        codes = np.arange(n, dtype=dtype)
        zero = np.zeros((1, 1, 3), dtype)
        t0 = light_float(zero, full, profile, linearize)[0, 0]
        tables = []
        for c in range(3):
            ramp = np.zeros((n, 1, 3), dtype)
            ramp[:, 0, c] = codes
            tables.append(light_float(ramp, full, profile, linearize)[:, 0, :] - t0)
        tables[0] += t0
        rng = np.random.default_rng(0)
        probe = rng.integers(0, n, size=(4096, 1, 3)).astype(dtype)
        got = tables[0][probe[..., 0]] + tables[1][probe[..., 1]] + tables[2][probe[..., 2]]
        err = float(np.abs(got - light_float(probe, full, profile, linearize)).max())
        _LUTS[key] = (tables if err <= LUT_CHECK_TOL else None, err)
    return _LUTS[key]


def light_lut(part, full, profile, linearize):
    if part.dtype.kind != "u":
        return light_float(part, full, profile, linearize)
    _rgb_only(part)
    tables, _ = lut_for(profile, part.dtype, full, linearize)
    if tables is None:
        return light_float(part, full, profile, linearize)
    out = np.take(tables[0], part[..., 0], axis=0)
    out += np.take(tables[1], part[..., 1], axis=0)
    out += np.take(tables[2], part[..., 2], axis=0)
    return out


PATHS = {"float": light_float, "lcms16": light_lcms16, "lcmsf": light_lcmsf, "lut": light_lut}


def finish(c: np.ndarray, out_dtype=np.float16) -> np.ndarray:
    """D7 (a): below 0 clipped, a NaN black; light above 1 kept. As the canonical photo's samples."""
    np.nan_to_num(c, copy=False, nan=0.0)
    np.maximum(c, 0, out=c)
    return c.astype(out_dtype)


# ----- decoding, with the gain map applied into float16 light -----

class Decoded:
    """An original as the prototype encodes it: `light` (float16 or float32 H x W x 3, final samples) when a gain map
    was applied, else the decoder's `source` (canonical.Source) converted per rectangle by `path`."""

    def __init__(self, source, light, form):
        self.source, self.light, self.form = source, light, form

    @property
    def shape(self):
        return (self.light if self.light is not None else self.source.pixels).shape[:2]


def decode(data: bytes, name: str, max_pixels, path, light_dtype=np.float16) -> Decoded:
    """canonical.decode with FRONT_SETTINGS, its gain-map step copied (_with_gain_map) with `path` as the colour
    conversion, no tone map, and the light stored as `light_dtype`."""
    settings = cn.FRONT_SETTINGS
    source = cn._decode(data, name, max_pixels, settings)
    g = gain_map.read(data, cn.MAP_MAX_SAMPLES)
    channels = 1 if source.pixels.ndim == 2 else source.pixels.shape[2]
    if g is None or not g.alike or channels in (2, 4):
        return Decoded(source, None, None)
    h, w = source.pixels.shape[:2]
    map_rows = gain_map.stretched(np.ascontiguousarray(cn.orient(g.pixels, cn.applied_orientation(data, name))), h, w)
    light = np.empty((h, w, 3), light_dtype)
    for y in range(0, h, STRIP):
        strip = path(source.pixels[y:y + STRIP], source.full, source.profile, source.linearize)
        gain_map.apply_in_place(strip, g.metadata, map_rows[y:y + STRIP])
        light[y:y + STRIP] = finish(strip, light_dtype)
    return Decoded(None, light, g.form)


def region_fn(d: Decoded, path):
    """The rectangle callback: final float16 samples of (y, x, ys, xs)."""
    if d.light is not None:
        return lambda y, x, ys, xs: d.light[y:y + ys, x:x + xs].astype(np.float16, copy=False)
    s = d.source

    def region(y, x, ys, xs):   # in strips of STRIP rows, as canonical.to_rec2020: float32 temporaries stay small
        out = np.empty((ys, xs, 3), np.float16)
        for r in range(0, ys, STRIP):
            out[r:r + STRIP] = finish(path(s.pixels[y + r:y + min(r + STRIP, ys), x:x + xs], s.full, s.profile,
                                           s.linearize))
        return out
    return region


def whole(d: Decoded, path) -> np.ndarray:
    """The whole picture as one float16 array, made a strip at a time."""
    if d.light is not None:
        return d.light.astype(np.float16, copy=False)
    h, w = d.shape
    out = np.empty((h, w, 3), np.float16)
    r = region_fn(d, path)
    for y in range(0, h, STRIP):
        out[y:y + STRIP] = r(y, 0, min(STRIP, h - y), w)
    return out


# ----- the encoders -----

def encode_whole(a: np.ndarray, level: int, effort: int, numthreads: int) -> bytes:
    return bytes(ic.jpegxl_encode(a, level=level, effort=effort, numthreads=numthreads,
                                  primaries=ic.JPEGXL.PRIMARIES.BT2100, transfer=ic.JPEGXL.TRANSFER_FUNCTION.LINEAR))


def encode_chunked(height: int, width: int, region, *, level: float, effort: int, numthreads: int) -> bytes:
    """coffeecv/jxl_encode.encode_rec2020, copied, for float16 samples and the linear transfer (ML-37 D1)."""
    _jxl, _threads = je._jxl, je._threads
    held: dict[int, np.ndarray] = {}
    lock = threading.Lock()
    out = bytearray()
    pos = [0]
    outbuf: list = [None]
    errors: list[BaseException] = []

    def get_format(_opaque, fmt):
        fmt.contents.num_channels, fmt.contents.data_type = 3, JXL_TYPE_FLOAT16
        fmt.contents.endianness, fmt.contents.align = je.JXL_NATIVE_ENDIAN, 0

    def get_data(_opaque, xpos, ypos, xsize, ysize, row_offset):
        try:
            a = np.ascontiguousarray(region(ypos, xpos, ysize, xsize), dtype=np.float16)
            if a.shape != (ysize, xsize, 3):
                raise ValueError(f"region gave {a.shape}, not {(ysize, xsize, 3)}")
            row_offset[0] = a.strides[0]
            ptr = a.ctypes.data
            with lock:
                held[ptr] = a
            return ptr
        except BaseException as e:
            errors.append(e)
            return None

    def get_ec_format(_opaque, _index, fmt):
        get_format(None, fmt)

    def get_ec_data(_opaque, _index, _x, _y, _xs, _ys, _row_offset):
        errors.append(RuntimeError("libjxl asked for an extra channel; the frame has none"))
        return None

    def release_input(_opaque, buf):
        with lock:
            held.pop(buf, None)

    def get_buffer(_opaque, size):
        try:
            n = max(int(size[0]), 1 << 16)
            outbuf[0] = (C.c_uint8 * n)()
            size[0] = n
            return C.addressof(outbuf[0])
        except BaseException as e:
            errors.append(e)
            size[0] = 0
            return None

    def release_output(_opaque, written):
        try:
            end = pos[0] + written
            if end > len(out):
                out.extend(bytes(end - len(out)))
            out[pos[0]:end] = C.string_at(C.addressof(outbuf[0]), written)
            pos[0] = end
            outbuf[0] = None
        except BaseException as e:
            errors.append(e)

    def seek(_opaque, position):
        pos[0] = position

    def finalized(_opaque, _position):
        pass

    callbacks = (je.GET_FORMAT(get_format), je.GET_DATA(get_data), je.GET_EC_FORMAT(get_ec_format),
                 je.GET_EC_DATA(get_ec_data), je.RELEASE_INPUT(release_input), je.GET_BUFFER(get_buffer),
                 je.RELEASE_OUTPUT(release_output), je.SEEK(seek), je.FINALIZED(finalized))
    enc = _jxl.JxlEncoderCreate(None)
    runner = _threads.JxlThreadParallelRunnerCreate(None, numthreads if numthreads > 1 else 0)
    try:
        je._check(_jxl.JxlEncoderSetParallelRunner(enc, je._RUNNER, runner), "JxlEncoderSetParallelRunner", enc)
        info = je.BasicInfo()
        _jxl.JxlEncoderInitBasicInfo(C.byref(info))
        info.xsize, info.ysize, info.num_color_channels, info.num_extra_channels = width, height, 3, 0
        info.bits_per_sample, info.exponent_bits_per_sample, info.uses_original_profile = 16, 5, 0
        je._check(_jxl.JxlEncoderSetBasicInfo(enc, C.byref(info)), "JxlEncoderSetBasicInfo", enc)
        ce = je.ColorEncoding(color_space=je.JXL_COLOR_SPACE_RGB, white_point=je.JXL_WHITE_POINT_D65,
                              primaries=je.JXL_PRIMARIES_2100, transfer_function=JXL_TRANSFER_FUNCTION_LINEAR,
                              rendering_intent=je.JXL_RENDERING_INTENT_RELATIVE)
        je._check(_jxl.JxlEncoderSetColorEncoding(enc, C.byref(ce)), "JxlEncoderSetColorEncoding", enc)
        je._check(_jxl.JxlEncoderUseContainer(enc, 0), "JxlEncoderUseContainer", enc)
        settings = _jxl.JxlEncoderFrameSettingsCreate(enc, None)
        je._check(_jxl.JxlEncoderSetFrameDistance(settings, _jxl.JxlEncoderDistanceFromQuality(level)),
                  "JxlEncoderSetFrameDistance", enc)
        je._check(_jxl.JxlEncoderFrameSettingsSetOption(settings, je.JXL_ENC_FRAME_SETTING_EFFORT, effort), "effort",
                  enc)
        header = je.FrameHeader()
        _jxl.JxlEncoderInitFrameHeader(C.byref(header))
        header.duration = 1
        je._check(_jxl.JxlEncoderSetFrameHeader(settings, C.byref(header)), "JxlEncoderSetFrameHeader", enc)
        je._check(_jxl.JxlEncoderSetOutputProcessor(enc, je.OutputProcessor(None, *callbacks[5:])),
                  "JxlEncoderSetOutputProcessor", enc)
        status = _jxl.JxlEncoderAddChunkedFrame(settings, 1, je.ChunkedInput(None, *callbacks[:5]))
        if errors:
            raise errors[0]
        je._check(status, "JxlEncoderAddChunkedFrame", enc)
        je._check(_jxl.JxlEncoderFlushInput(enc), "JxlEncoderFlushInput", enc)
        if errors:
            raise errors[0]
    finally:
        _jxl.JxlEncoderDestroy(enc)
        _threads.JxlThreadParallelRunnerDestroy(runner)
    return bytes(out)


# ----- the read-back, for the accuracy comparison (ML-37 D2 with D6 (b), a clip; D9 (a), numpy) -----

def _rgb_to_xyz(prim, white=(0.3127, 0.3290)):
    xy = np.array(prim, np.float64).reshape(3, 2)
    cols = np.stack([xy[:, 0] / xy[:, 1], np.ones(3), (1 - xy[:, 0] - xy[:, 1]) / xy[:, 1]])
    wx, wy = white
    W = np.array([wx / wy, 1.0, (1 - wx - wy) / wy])
    return cols * np.linalg.solve(cols, W)


REC2020_TO_SRGB = (np.linalg.inv(_rgb_to_xyz((0.64, 0.33, 0.30, 0.60, 0.15, 0.06)))
                   @ _rgb_to_xyz((0.708, 0.292, 0.170, 0.797, 0.131, 0.046))).astype(np.float32)


def readback_levels(lin: np.ndarray) -> np.ndarray:
    """Linear Rec.2020 (any float) -> sRGB in float 8-bit levels (0-255), clipped to [0, 1] before the curve."""
    x = lin.astype(np.float32) @ REC2020_TO_SRGB.T
    np.clip(x, 0, 1, out=x)
    lo = x <= 0.0031308
    y = np.where(lo, 12.92 * x, 1.055 * np.power(x, 1 / 2.4, dtype=np.float32) - 0.055)
    return y * 255
