"""ML-37 D9: the sRGB curve, IEC 61966-2-1's closed form against what LittleCMS 2.19 evaluates from its built-in sRGB
profile (a type-4 `para` curve whose parameters are stored as s15.16, inverted analytically as type -4).

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d9_curve.py > analysis/ml37/d9_curve.txt
"""
from __future__ import annotations

import imagecodecs as ic
import numpy as np

from coffeecv import canonical as cn

D65 = (0.3127, 0.3290)
LIN_SRGB = ic.cms_profile("rgb", whitepoint=D65, primaries=(0.64, 0.33, 0.30, 0.60, 0.15, 0.06), gamma=1.0)


def iec_encode(lin):
    """IEC 61966-2-1 linear -> sRGB, float64: 12.92 L at or below 0.0031308, else 1.055 L^(1/2.4) - 0.055."""
    lin = np.asarray(lin, np.float64)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)


def iec_encode32(lin):
    """The same in float32 throughout, as a numpy read-back would run it."""
    lin = np.asarray(lin, np.float32)
    p = np.float32(1.055) * np.power(np.maximum(lin, np.float32(0)), np.float32(1 / 2.4)) - np.float32(0.055)
    return np.where(lin <= np.float32(0.0031308), np.float32(12.92) * lin, p).astype(np.float32)


def lcms_encode(lin, dtype=np.float64):
    """LittleCMS: linear sRGB (gamma-1 profile, same colorants) -> its built-in sRGB, float, greys on all channels."""
    a = np.repeat(np.asarray(lin, dtype)[:, None], 3, axis=1)[None]
    return ic.cms_transform(a, LIN_SRGB, cn.SRGB, intent=cn.RELATIVE, outdtype=dtype)[0]


def main() -> None:
    np.set_printoptions(precision=10, linewidth=140)
    g, a, b, c, d = 2.399993896484375, 0.9478607177734375, 0.0521392822265625, 0.077392578125, 0.0404510498046875
    print("# LittleCMS built-in sRGB `para` type 4, as stored (s15.16): g a b c d =", g, a, b, c, d)
    print("  exact (cmsvirt.c Build_sRGBGamma): 2.4, 1/1.055 =", 1 / 1.055, " 0.055/1.055 =", 0.055 / 1.055,
          " 1/12.92 =", 1 / 12.92, " 0.04045")
    print("  type -4 threshold in linear, (a d + b)^g, stored params:", (a * d + b) ** g,
          " exact params:", ((0.04045 + 0.055) / 1.055) ** 2.4, " IEC encode threshold: 0.0031308",
          " 0.04045/12.92 =", 0.04045 / 12.92)
    print("  the stored linear-segment slope 1/c =", 1 / c, " (IEC 12.92)")
    t = 0.0031308
    print("# IEC closed form at its own threshold: 12.92 * 0.0031308 =", 12.92 * t, "; 1.055 * t^(1/2.4) - 0.055 =",
          1.055 * t ** (1 / 2.4) - 0.055, " (gap", 1.055 * t ** (1 / 2.4) - 0.055 - 12.92 * t, ")")
    v = 0.04045
    print("# IEC decode at 0.04045: v/12.92 =", v / 12.92, "; ((v+0.055)/1.055)^2.4 =", ((v + 0.055) / 1.055) ** 2.4)

    ramp = np.linspace(0, 1, 2_000_001)
    ref = iec_encode(ramp)
    for name, out in (("LittleCMS float64 out", lcms_encode(ramp)[:, 0]),
                      ("LittleCMS float32 out", lcms_encode(ramp.astype(np.float32), np.float32)[:, 0]),
                      ("numpy float32 closed form", iec_encode32(ramp))):
        dd = out.astype(np.float64) - ref
        k = np.argmax(np.abs(dd))
        lo = ramp < 0.0031308
        print(f"\n{name}: vs IEC float64 over [0, 1] (2,000,001 points): max |diff| {np.abs(dd).max():.3e} at "
              f"L = {ramp[k]:.7f} ({np.abs(dd).max() * 255:.5f} 8-bit levels, {np.abs(dd).max() * 65535:.3f} 16-bit "
              f"codes); mean {np.abs(dd).mean():.3e}; on the linear segment max {np.abs(dd[lo]).max():.3e}, above it "
              f"{np.abs(dd[~lo]).max():.3e}")
        r8 = np.rint(out * 255) != np.rint(ref * 255)
        r16 = np.rint(out * 65535) != np.rint(ref * 65535)
        print(f"  rounds to another 8-bit level at {r8.sum()} of {ramp.size} points; another 16-bit code at "
              f"{r16.sum()} ({r16.mean():.2%})")
    lcm1 = lcms_encode(np.array([1.0]))[0, 0]
    print("\nLittleCMS sRGB curve at 1.0:", lcm1, " at 4.0:", lcms_encode(np.array([4.0]))[0, 0],
          " IEC formula at 4.0:", iec_encode(4.0), " at -0.01:", lcms_encode(np.array([-0.01]))[0, 0])
    # 8-bit output as today's read-back makes it (LittleCMS rounding to uint8) against the closed form + rint
    a8 = np.repeat(np.linspace(0, 1, 1_000_001)[:, None], 3, 1)[None]
    o8 = ic.cms_transform(a8, LIN_SRGB, cn.SRGB, intent=cn.RELATIVE, outdtype=np.uint8)[0, :, 0]
    want = np.rint(np.clip(iec_encode(a8[0, :, 0]), 0, 1) * 255)
    print("LittleCMS uint8 out vs rint(IEC * 255) on 1,000,001 linear greys: differ at", int((o8 != want).sum()),
          "points; max", int(np.abs(o8.astype(int) - want).max()), "level")


if __name__ == "__main__":
    main()
