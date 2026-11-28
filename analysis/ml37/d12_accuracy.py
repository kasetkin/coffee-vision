"""ML-37 D12: how far each faster colour path lands from D12 (a), the store's float path, in sRGB levels after the
read-back, before the encode and after it; and what each path costs per megapixel on one thread.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d12_accuracy.py [--cpus 0,1]

Before the encode: each photo decoded once (FRONT_SETTINGS' cap; its gain map applied when it has one), every
colour path run on the same 256-row strips, each strip's float16 samples read back (linear Rec.2020 -> 3x3 matrix ->
clip to [0, 1] -> sRGB curve, d12_proto.readback_levels, in float 8-bit levels) and compared with the float path's:
largest and mean absolute difference over every sample, and the share over 0.5 and over 1 level. The colour step
alone is timed per strip (best of the strips' sum on one thread; the gain map's own maths not included).

After the encode: the canonical photos d12_front_cost.py wrote to tmp/ml37/d12/out/, decoded and read back the same
way, each path's against a_chunked's; a_whole's bytes against a_chunked's; today's canonical photo through
canonical.load_canonical (8-bit sRGB) against a_chunked's read-back, for scale (a gain-map photo's differ by the EETF
as well, which a clip does not apply). Writes d12_accuracy.txt next to this script.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import imagecodecs as ic
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d12_front_cost as FC   # noqa: E402
import d12_proto as P         # noqa: E402
from coffeecv import canonical as cn, gain_map  # noqa: E402
from coffeecv.jxl_header import read_header     # noqa: E402

PATHS = ("float", "lcms16", "lcmsf", "lut")


class Diff:
    def __init__(self):
        self.max = 0.0
        self.sum = 0.0
        self.n = 0
        self.over_half = 0
        self.over_one = 0

    def add(self, a, b):
        d = np.abs(a.astype(np.float32) - b.astype(np.float32))
        self.max = max(self.max, float(d.max()))
        self.sum += float(d.sum(dtype=np.float64))
        self.n += d.size
        self.over_half += int((d > 0.5).sum())
        self.over_one += int((d > 1.0).sum())

    def row(self):
        return (f"{self.max:.3f} | {self.sum / self.n:.4f} | {100 * self.over_half / self.n:.3f}% | "
                f"{100 * self.over_one / self.n:.4f}%")


def profile_name(profile: bytes | None) -> str:
    if profile is None:
        return "none (sRGB, or the decoder's linear Rec.2020)"
    from PIL import ImageCms
    try:
        return ImageCms.ImageCmsProfile(io.BytesIO(profile)).profile.profile_description.strip()
    except Exception as exc:   # noqa: BLE001
        return f"unreadable ({type(exc).__name__})"


def readback(lin):
    return P.readback_levels(lin)


def before_encode(key: str) -> list[str]:
    photo = FC.photo_path(key)
    data, name = photo.read_bytes(), photo.name
    from check_service import MAX_PIXELS, RAW_MAX_PIXELS
    limit = RAW_MAX_PIXELS if cn.accepted_format(data, name).is_raw else MAX_PIXELS
    src = cn._decode(data, name, limit, cn.FRONT_SETTINGS)
    g = gain_map.read(data, cn.MAP_MAX_SAMPLES)
    h, w = src.pixels.shape[:2]
    map_rows = None
    if g is not None and g.alike:
        map_rows = gain_map.stretched(np.ascontiguousarray(cn.orient(g.pixels, cn.applied_orientation(data, name))),
                                      h, w)
    diffs = {p: Diff() for p in PATHS[1:]}
    secs = dict.fromkeys(PATHS, 0.0)
    noopt = Diff()
    above = 0
    for y in range(0, h, P.STRIP):
        part = src.pixels[y:y + P.STRIP]
        ref = None
        for p in PATHS:
            t = time.perf_counter()
            lin = P.PATHS[p](part, src.full, src.profile, src.linearize)
            secs[p] += time.perf_counter() - t
            if map_rows is not None:
                gain_map.apply_in_place(lin, g.metadata, map_rows[y:y + P.STRIP].copy())
            f16 = P.finish(lin)
            lv = readback(f16)
            if p == "float":
                ref = lv
                above += int((f16.max(axis=-1) > 1).sum())
            else:
                diffs[p].add(lv, ref)
        if y == 0 and part.dtype.kind == "u":
            # LittleCMS with its optimization against without, into linear uint16: is the 16-bit transform optimized?
            a = P._scaled(np.ascontiguousarray(part), src.full)
            o = ic.cms_transform(a, src.profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.uint16)
            n = ic.cms_transform(a, src.profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.uint16,
                                 flags=cn.NOOPTIMIZE)
            noopt.add(o, n)
    mp = h * w / 1e6
    luts = [f"{v[1]:.1e}{'' if v[0] is not None else ' (not held: float path used)'}" for v in P._LUTS.values()]
    P._LUTS.clear()
    lines = [f"### {FC.PHOTOS[key][0]}",
             f"decoded {w} x {h} {src.pixels.dtype} ({src.decoder}), full {src.full:g}, profile: "
             f"{profile_name(src.profile)}; gain map {'applied' if map_rows is not None else 'none'}; "
             f"pixels above white (float path): {100 * above / (h * w):.2f}%",
             f"LUT additivity check, largest error in linear light: {', '.join(luts) or 'n/a'}",
             f"LittleCMS uint16 linear output, optimized vs NOOPTIMIZE, first strip: largest {noopt.max:.0f} of 65535"
             if noopt.n else "LittleCMS optimized vs NOOPTIMIZE: n/a (float source)",
             "",
             "| Path | Colour step, s per MP (1 thread) | vs float path: max levels | mean | > 0.5 | > 1 |",
             "|---|---|---|---|---|---|",
             f"| float (D12 a) | {secs['float'] / mp:.3f} | - | - | - | - |"]
    for p in PATHS[1:]:
        lines.append(f"| {p} | {secs[p] / mp:.3f} | {diffs[p].row()} |")
    return lines


def after_encode(key: str) -> list[str]:
    out = FC.OUT
    ref_b = (out / f"{key}_a_chunked.jxl").read_bytes()
    hdr = read_header(ref_b)
    c = hdr.colour
    lines = [f"a_chunked header: {c.colour_space} {c.white_point} {c.primaries} {c.transfer} {c.rendering_intent}, "
             f"{hdr.bits_per_sample}-bit{' float' if hdr.float_sample else ''}, "
             f"{'lossy (XYB)' if hdr.xyb_encoded else 'lossless'}"]
    whole = out / f"{key}_a_whole.jxl"
    if whole.exists():
        lines.append(f"a_whole bytes == a_chunked bytes: {whole.read_bytes() == ref_b}")
    ref = ic.jpegxl_decode(ref_b, numthreads=2)
    lines += ["", "| After the encode, against a_chunked | max levels | mean | > 0.5 | > 1 | size |",
              "|---|---|---|---|---|---|"]
    for v in ("a_chunked32", "b_lcms16", "b_lcmsf", "b_lut"):
        f = out / f"{key}_{v}.jxl"
        if not f.exists():
            continue
        b = f.read_bytes()
        if b == ref_b:
            lines.append(f"| {v} | identical bytes | | | | {len(b):,} |")
            continue
        other = ic.jpegxl_decode(b, numthreads=2)
        d = Diff()
        for y in range(0, ref.shape[0], P.STRIP):
            d.add(readback(other[y:y + P.STRIP]), readback(ref[y:y + P.STRIP]))
        lines.append(f"| {v} | {d.row()} | {len(b):,} |")
        del other
    t = out / f"{key}_today.jxl"
    if t.exists():
        tb = t.read_bytes()
        today = cn.load_canonical(tb, numthreads=2)
        d = Diff()
        for y in range(0, ref.shape[0], P.STRIP):
            d.add(today[y:y + P.STRIP], readback(ref[y:y + P.STRIP]))
        lines.append(f"| today (load_canonical, uint8) | {d.row()} | {len(tb):,} |")
    lines.append(f"| a_chunked itself | | | | | {len(ref_b):,} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", nargs="*", default=list(FC.PHOTOS))
    args = ap.parse_args()
    sys.path.insert(0, str(FC.REPO / "webapp" / "front"))
    report = [f"ML-37 D12 accuracy, {time.strftime('%Y-%m-%d %H:%M')}, load average {os.getloadavg()[0]:.1f}, "
              f"CPUs {sorted(os.sched_getaffinity(0))}", ""]
    for key in args.photos:
        report += before_encode(key) + [""] + after_encode(key) + [""]
        print("\n".join(report[-30:]), file=sys.stderr)
    Path(__file__).with_name("d12_accuracy.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
