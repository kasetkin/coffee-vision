"""ML-37 D9: the linear Rec.2020 -> linear sRGB matrix, four ways, and what their differences do to sRGB levels.

(1) exact, float64, from BT.2020's and BT.709's primaries and D65 (the SMPTE RP 177 construction, the one
    tests/test_canonical.py already writes); (2) from IEC 61966-2-1's own 7-digit XYZ -> sRGB matrix (as quoted by the
    ICC's "How to interpret the sRGB color space", 2015); (3) BT.2087's convention, the 709 -> 2020 matrix rounded to
    four decimals (its NOTE 3), inverted; (4) what LittleCMS 2.19 (imagecodecs 2026.10.10) applies, read back from float
    transforms between linear profiles and computed from the profiles' s15.16 colorant tags.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d9_matrix.py > analysis/ml37/d9_matrix.txt
"""
from __future__ import annotations

import os
import struct

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import imagecodecs as ic
import numpy as np

from coffeecv import canonical as cn

D65 = (0.3127, 0.3290)
P709 = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
P2020 = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))
D50_ICC = np.array([0.9642, 1.0, 0.8249])                       # ICC PCS white (ISO 15076-1; LittleCMS cmsD50_XYZ)
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
# IEC 61966-2-1 (Amd. 1) XYZ(D65) -> linear sRGB, as printed in the ICC's sRGB note, A.7
IEC_XYZ_TO_SRGB = np.array([[3.2406255, -1.5372080, -0.4986286],
                            [-0.9689307, 1.8757561, 0.0415175],
                            [0.0557101, -0.2040211, 1.0569959]])


def npm(primaries, white=D65) -> np.ndarray:
    """SMPTE RP 177's normalized primary matrix: linear RGB -> XYZ, white (1,1,1) -> the white's XYZ with Y = 1."""
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in primaries]).T
    wx, wy = white
    s = np.linalg.solve(xyz, np.array([wx / wy, 1.0, (1 - wx - wy) / wy]))
    return xyz * s


def icc_tags(profile: bytes) -> dict[bytes, bytes]:
    n = struct.unpack(">I", profile[128:132])[0]
    out = {}
    for i in range(n):
        sig, off, size = struct.unpack(">4sII", profile[132 + 12 * i:144 + 12 * i])
        out[sig] = profile[off:off + size]
    return out


def s15(b: bytes) -> float:
    return struct.unpack(">i", b)[0] / 65536.0


def colorants(profile: bytes) -> np.ndarray:
    t = icc_tags(profile)
    cols = [[s15(t[k][8 + 4 * j:12 + 4 * j]) for j in range(3)] for k in (b"rXYZ", b"gXYZ", b"bXYZ")]
    return np.array(cols).T


def para(profile: bytes, tag: bytes = b"rTRC"):
    d = icc_tags(profile)[tag]
    if d[:4] != b"para":
        return d[:4].decode(), None
    ftype = struct.unpack(">H", d[8:10])[0]
    n = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[ftype]
    return ftype, [s15(d[12 + 4 * i:16 + 4 * i]) for i in range(n)]


def lcms_matrix(src: bytes, dst: bytes, unit: float = 1.0) -> np.ndarray:
    """The matrix a float LittleCMS transform applies, read from three unit inputs (float64 in, float64 out)."""
    e = np.eye(3, dtype=np.float64)[None] * unit
    out = ic.cms_transform(e, src, dst, intent=cn.RELATIVE, outdtype=np.float64)[0]
    return out.T / unit


def srgb_encode(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def main() -> None:
    np.set_printoptions(precision=10, suppress=True, linewidth=140)
    n709, n2020 = npm(P709), npm(P2020)
    exact = np.linalg.inv(n709) @ n2020
    print("# 1. exact, float64 (SMPTE RP 177 construction from BT.2020 / BT.709 primaries, D65 0.3127, 0.3290)")
    print("NPM(709) row Y (BT.709 item 3.2: 0.2126 0.7152 0.0722):", n709[1])
    print("NPM(2020) row Y (BT.2020 Table 4: 0.2627 0.6780 0.0593):", n2020[1])
    print("M exact = inv(NPM709) @ NPM2020:\n", exact)
    print("M exact @ (1,1,1):", exact @ np.ones(3), " row sums - 1:", exact.sum(1) - 1)
    print("M exact as float32:\n", exact.astype(np.float32))
    print("max |float32 - float64| element:", np.abs(exact.astype(np.float32) - exact).max())

    print("\n# 2. from IEC 61966-2-1's 7-digit XYZ -> sRGB matrix")
    iec = IEC_XYZ_TO_SRGB @ n2020
    print("IEC matrix vs inv(NPM709), max |diff|:", np.abs(IEC_XYZ_TO_SRGB - np.linalg.inv(n709)).max())
    print("M_iec = IEC @ NPM2020:\n", iec)
    print("max |M_iec - exact|:", np.abs(iec - exact).max(), " M_iec @ 1:", iec @ np.ones(3))

    print("\n# 3. BT.2087 convention: 709 -> 2020 rounded to 4 decimals (its NOTE 3), inverted")
    fwd = np.linalg.inv(exact)
    fwd4 = np.round(fwd, 4)
    print("709 -> 2020, exact:\n", fwd, "\nrounded to 4 decimals:\n", fwd4)
    bt2087 = np.linalg.inv(fwd4)
    print("inverse of the rounded one:\n", bt2087)
    print("max |diff| vs exact:", np.abs(bt2087 - exact).max(), " white ->", bt2087 @ np.ones(3))
    exact4 = np.round(exact, 4)
    print("2020 -> 709 rounded to 4 decimals:\n", exact4, "\n white ->", exact4 @ np.ones(3))

    print("\n# 4. LittleCMS 2.19 (", ic.cms_version(), ")")
    lin_srgb = ic.cms_profile("rgb", whitepoint=D65, primaries=(0.64, 0.33, 0.30, 0.60, 0.15, 0.06), gamma=1.0)
    c_srgb, c_lin2020, c_rec2020 = colorants(cn.SRGB), colorants(cn.LINEAR_REC2020), colorants(cn.REC2020)
    print("colorant tags (D50-adapted, s15.16), columns R G B:")
    print("SRGB (LittleCMS built-in):\n", c_srgb, "\n sum:", c_srgb.sum(1))
    print("LINEAR_SRGB (imagecodecs rgb, gamma 1) equal to SRGB's:", np.array_equal(colorants(lin_srgb), c_srgb))
    print("LINEAR_REC2020 (imagecodecs rgb, gamma 1):\n", c_lin2020, "\n sum:", c_lin2020.sum(1))
    print("REC2020 (committed libjxl profile):\n", c_rec2020, "\n sum:", c_rec2020.sum(1))
    print("TRCs: SRGB", para(cn.SRGB), " LINEAR_REC2020", para(cn.LINEAR_REC2020), " REC2020", para(cn.REC2020),
          " LINEAR_SRGB", para(lin_srgb))
    bradford_d65_d50 = None
    w65 = np.array([D65[0] / D65[1], 1.0, (1 - sum(D65)) / D65[1]])
    lms_s, lms_d = BRADFORD @ w65, BRADFORD @ D50_ICC
    bradford_d65_d50 = np.linalg.inv(BRADFORD) @ np.diag(lms_d / lms_s) @ BRADFORD
    print("Bradford D65 -> D50 (computed as LittleCMS does):\n", bradford_d65_d50)
    print("theory: inv(B @ NPM709) @ (B @ NPM2020) - exact, max:",
          np.abs(np.linalg.inv(bradford_d65_d50 @ n709) @ (bradford_d65_d50 @ n2020) - exact).max())
    print("unquantized B@NPM709 vs SRGB tags, max |diff|:", np.abs(bradford_d65_d50 @ n709 - c_srgb).max(),
          "(s15.16 step 1.5e-5)")
    print("unquantized B@NPM2020 vs LINEAR_REC2020 tags:", np.abs(bradford_d65_d50 @ n2020 - c_lin2020).max(),
          " vs committed REC2020 tags:", np.abs(bradford_d65_d50 @ n2020 - c_rec2020).max())
    tag_lin = np.linalg.inv(c_srgb) @ c_lin2020
    tag_rec = np.linalg.inv(c_srgb) @ c_rec2020
    emp_lin = lcms_matrix(cn.LINEAR_REC2020, lin_srgb)
    curve1 = ic.cms_transform(np.ones((1, 1, 3)), cn.REC2020, cn.LINEAR_REC2020, intent=cn.RELATIVE,
                              outdtype=np.float64)[0, 0]
    emp_rec = lcms_matrix(cn.REC2020, lin_srgb)
    print("REC2020's curve at 1.0 (via LINEAR_REC2020):", curve1)
    for name, tag, emp in (("LINEAR_REC2020 -> LINEAR_SRGB", tag_lin, emp_lin),
                           ("REC2020 (committed) -> LINEAR_SRGB", tag_rec, emp_rec)):
        print(f"\n{name}: from the tags inv(C_srgb) @ C_src:\n", tag, "\n read back from the float transform:\n", emp)
        print(" max |transform - tags|:", np.abs(emp - tag).max(), " max |transform - exact|:",
              np.abs(emp - exact).max(), " white ->", emp @ np.ones(3))
    big = lcms_matrix(cn.LINEAR_REC2020, lin_srgb, unit=4.0)
    print("\nlinear in, 4.0 units: matrix same as at 1.0, max |diff|:", np.abs(big - emp_lin).max(),
          "(values above 1 pass through unclipped)")
    neg = ic.cms_transform(np.array([[[0.0, 1.0, 0.0]]]), cn.LINEAR_REC2020, lin_srgb, intent=cn.RELATIVE,
                           outdtype=np.float64)[0, 0]
    print("Rec.2020 green (0,1,0) -> linear sRGB:", neg, "(negatives kept)")

    print("\n# 5. effect on sRGB levels: 2,000,000 random linear Rec.2020 colours in [0,1]^3, and the subset inside sRGB")
    rng = np.random.default_rng(37)
    x = rng.random((2_000_000, 3))
    x = x ** 2.2       # spread like real light: more dark values
    ref = srgb_encode(x @ exact.T)
    inside = np.all((x @ exact.T >= 0) & (x @ exact.T <= 1), axis=1)
    print("inside sRGB:", inside.mean())
    for name, m in (("float32(exact)", exact.astype(np.float32).astype(np.float64)), ("IEC 7-digit", iec),
                    ("BT.2087 4-digit inverted", bt2087), ("2020->709 4-digit", exact4),
                    ("LittleCMS LINEAR_REC2020", emp_lin), ("LittleCMS REC2020 committed", emp_rec)):
        d = np.abs(srgb_encode(x @ m.T) - ref)
        di = d[inside]
        lin_d = np.abs(np.clip(x @ m.T, 0, 1) - np.clip(x @ exact.T, 0, 1))
        print(f"{name:30s} max |M - exact| {np.abs(m - exact).max():.2e}; sRGB-encoded max {d.max() * 255:.4f} levels "
              f"(8-bit), {d.max() * 65535:.2f} codes (16-bit); mean {d.mean() * 65535:.3f} codes; inside sRGB max "
              f"{di.max() * 255:.4f} levels; linear max {lin_d.max():.2e}")


if __name__ == "__main__":
    main()
