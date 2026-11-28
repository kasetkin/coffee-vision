"""ML-37: can the pinned stack write the linear uint16 canonical photo (I)? What do the headers say?

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/canon_int_probe.py

On a synthetic picture (a ramp 0..1 in each channel, plus noise and a dark corner) and on the first photo of D8's
sample at 1/4 scale (tone-mapped light, clipped to [0, 1], rounded to uint16), encoded at q95/e7 and q90/e3:
  - imagecodecs 2026.10.10 jpegxl_encode(uint16, primaries=BT2100, transfer=LINEAR) and with transfer left unset;
  - the repo's chunked ctypes encoder with JXL_TYPE_UINT16 and the linear transfer
    (canon_int_measure.encode_linear_u16), intensity target 0 (libjxl's default), 255 and 203.
Reports: whether imagecodecs' and the ctypes bytes are identical, the header (coffeecv.jxl_header.read_header), the
declared intensity target and bit depth (libjxl's JxlDecoderGetBasicInfo), and the decoded dtype and range.
"""
from __future__ import annotations

import sys

import imagecodecs as ic
import numpy as np

from analysis.ml37 import canon_int_measure as cm
from analysis.ml37 import d8_intensity_target as d8
from coffeecv import jxl_header as jh
from coffeecv import tone_map as tm


def describe(label: str, data: bytes) -> None:
    h = jh.read_header(data)
    bi = d8.basic_info(data)
    got = ic.jpegxl_decode(data, numthreads=2)
    print(f"  {label:44s} {len(data):8d} B  header: {h.bits_per_sample}-bit {'float' if h.float_sample else 'int'}, "
          f"{h.colour.colour_space} {h.colour.white_point} {h.colour.primaries} {h.colour.transfer} "
          f"{h.colour.rendering_intent}, xyb={h.xyb_encoded}, icc={h.colour.want_icc}; basic info: bits "
          f"{bi.bits_per_sample}/exp {bi.exponent_bits_per_sample}, IT {bi.intensity_target:g}, "
          f"original profile {bi.uses_original_profile}; decoded {got.dtype} {got.min()}..{got.max()}")


def main() -> None:
    print(f"imagecodecs {ic.__version__}, {ic.jpegxl_version()}")
    rng = np.random.default_rng(0)
    h, w = 512, 768
    ramp = np.linspace(0, 1, w, dtype=np.float32)[None, :, None].repeat(h, 0).repeat(3, 2)
    ramp[:, :, 1] *= 0.7
    ramp += rng.normal(0, 0.002, ramp.shape).astype(np.float32)
    ramp[: h // 4, : w // 4] *= 0.01
    synth = np.rint(np.clip(ramp, 0, 1) * 65535).astype(np.uint16)

    p = d8.sample(20)[0]
    light, _ = d8.light_of(p["file"], 4)
    peak = tm.picture_peak(light.max(axis=-1).ravel().copy())
    if peak > 1:
        tm.eetf_in_place(light, peak)
    photo = np.rint(np.clip(light, 0, 1) * 65535).astype(np.uint16)

    for name, a in (("synthetic ramp 768x512", synth), (f"photo 1 of D8's sample at 1/4, {a_shape(photo)}", photo)):
        print(f"\n## {name}")
        for q, e in ((95, 7), (90, 3)):
            ic_lin = ic.jpegxl_encode(a, level=q, effort=e, primaries="BT2100", transfer="LINEAR", numthreads=2)
            ic_unset = ic.jpegxl_encode(a, level=q, effort=e, primaries="BT2100", numthreads=2)
            ct = {it: cm.encode_linear_u16(a, level=q, effort=e, intensity_target=it) for it in (0.0, 255.0, 203.0)}
            print(f" q{q}/e{e}: imagecodecs LINEAR == ctypes IT=0: {ic_lin == ct[0.0]}; "
                  f"imagecodecs transfer unset == LINEAR: {ic_unset == ic_lin}; ctypes IT=0 == IT=255: "
                  f"{ct[0.0] == ct[255.0]}")
            describe("imagecodecs uint16 BT2100 LINEAR", ic_lin)
            for it, data in ct.items():
                describe(f"ctypes uint16 linear IT={it:g}", data)


def a_shape(a: np.ndarray) -> str:
    return f"{a.shape[1]}x{a.shape[0]}"


if __name__ == "__main__":
    sys.exit(main())
