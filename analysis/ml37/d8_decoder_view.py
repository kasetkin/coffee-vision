"""ML-37 D8: what libjxl 0.12.0's decoder makes of the intensity target in a linear half-float file.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d8_decoder_view.py

A flat-patch test image (linear Rec.2020 grey at 0.18, 1.0, 2.0, 4.0 and 8.0 x reference white) is encoded as the
lossy half-float canonical photo (quality 95, effort 7, linear transfer, BT2100 primaries) with intensity_target 203
and 255, through d8_intensity_target.encode. Each file is decoded through libjxl's decoder (ctypes, the library inside
imagecodecs' wheel) four ways, and the centre of each patch printed:
  1. as stored (float32, the file's own linear encoding): what imagecodecs.jpegxl_decode and the read-back see;
  2. the same with JxlDecoderSetDesiredIntensityTarget(100) (a 100-nit display asked for);
  3. to PQ (JxlDecoderSetPreferredColorProfile, BT2100 primaries, PQ transfer), shown as cd/m2 through BT.2100's PQ
     EOTF: where an HDR viewer that asks libjxl for PQ would put each patch;
  4. to PQ with a desired intensity target of 1000.
Writes nothing but stdout (kept as analysis/ml37/d8_decoder_view.txt).
"""
from __future__ import annotations

import ctypes as C
import importlib.util
from pathlib import Path

import numpy as np

from coffeecv import jxl_encode as je
from coffeecv import tone_map as tm

spec = importlib.util.spec_from_file_location("d8", Path(__file__).with_name("d8_intensity_target.py"))
d8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d8)

_jxl = je._jxl
_P = C.c_void_p
JXL_DEC_SUCCESS, JXL_DEC_ERROR, JXL_DEC_NEED_MORE_INPUT = 0, 1, 2
JXL_DEC_NEED_IMAGE_OUT_BUFFER = 5
JXL_DEC_BASIC_INFO, JXL_DEC_COLOR_ENCODING, JXL_DEC_FULL_IMAGE = 0x40, 0x100, 0x1000
JXL_TYPE_FLOAT = 0
JXL_TRANSFER_FUNCTION_PQ = 16
for name, res, args in (
        ("JxlDecoderSetPreferredColorProfile", C.c_int, (_P, C.POINTER(je.ColorEncoding))),
        ("JxlDecoderSetDesiredIntensityTarget", C.c_int, (_P, C.c_float)),
        ("JxlDecoderImageOutBufferSize", C.c_int, (_P, C.POINTER(je.PixelFormat), C.POINTER(C.c_size_t))),
        ("JxlDecoderSetImageOutBuffer", C.c_int, (_P, C.POINTER(je.PixelFormat), _P, C.c_size_t))):
    je._sig(_jxl, name, res, *args)

PATCHES = (0.18, 1.0, 2.0, 4.0, 8.0)
SIDE = 64


def test_image() -> np.ndarray:
    a = np.zeros((SIDE, SIDE * len(PATCHES), 3), np.float16)
    for i, v in enumerate(PATCHES):
        a[:, i * SIDE:(i + 1) * SIDE] = v
    return a


def decode(data: bytes, pq: bool, desired: float | None) -> np.ndarray:
    dec = _jxl.JxlDecoderCreate(None)
    try:
        assert _jxl.JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING
                                              | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS
        if desired:
            assert _jxl.JxlDecoderSetDesiredIntensityTarget(dec, desired) == JXL_DEC_SUCCESS
        assert _jxl.JxlDecoderSetInput(dec, data, len(data)) == JXL_DEC_SUCCESS
        _jxl.JxlDecoderCloseInput(dec)
        fmt = je.PixelFormat(3, JXL_TYPE_FLOAT, je.JXL_NATIVE_ENDIAN, 0)
        info = je.BasicInfo()
        out = None
        while True:
            st = _jxl.JxlDecoderProcessInput(dec)
            if st == JXL_DEC_BASIC_INFO:
                assert _jxl.JxlDecoderGetBasicInfo(dec, C.byref(info)) == JXL_DEC_SUCCESS
            elif st == JXL_DEC_COLOR_ENCODING:
                if pq:
                    ce = je.ColorEncoding(color_space=je.JXL_COLOR_SPACE_RGB, white_point=je.JXL_WHITE_POINT_D65,
                                          primaries=je.JXL_PRIMARIES_2100, transfer_function=JXL_TRANSFER_FUNCTION_PQ,
                                          rendering_intent=je.JXL_RENDERING_INTENT_RELATIVE)
                    assert _jxl.JxlDecoderSetPreferredColorProfile(dec, C.byref(ce)) == JXL_DEC_SUCCESS
            elif st == JXL_DEC_NEED_IMAGE_OUT_BUFFER:
                size = C.c_size_t()
                assert _jxl.JxlDecoderImageOutBufferSize(dec, C.byref(fmt), C.byref(size)) == JXL_DEC_SUCCESS
                out = np.empty((info.ysize, info.xsize, 3), np.float32)
                assert out.nbytes == size.value
                assert _jxl.JxlDecoderSetImageOutBuffer(dec, C.byref(fmt), out.ctypes.data, size.value) == 0
            elif st == JXL_DEC_FULL_IMAGE:
                pass
            elif st == JXL_DEC_SUCCESS:
                return out
            else:
                raise RuntimeError(f"decoder status {st}")
    finally:
        _jxl.JxlDecoderDestroy(dec)


def centres(a: np.ndarray) -> list[float]:
    return [float(a[SIDE // 2, i * SIDE + SIDE // 2, 1]) for i in range(len(PATCHES))]


def main() -> None:
    img = test_image()
    print("patches (x reference white):", PATCHES)
    for it in (203.0, 255.0):
        data = d8.encode(img, level=95, effort=7, intensity_target=it, half=True)
        print(f"\nintensity_target {it:g}: declared {d8.basic_info(data).intensity_target:g}, {len(data)} bytes")
        for label, pq, desired in (("as stored, linear", False, None), ("as stored, desired IT 100", False, 100.0),
                                   ("to PQ, cd/m2", True, None), ("to PQ, desired IT 1000, cd/m2", True, 1000.0)):
            v = np.array(centres(decode(data, pq, desired)))
            if pq:
                v = tm.pq_eotf(v)
            print(f"  {label:30s} " + "  ".join(f"{x:9.3f}" for x in v))


if __name__ == "__main__":
    main()
