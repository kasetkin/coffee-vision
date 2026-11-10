"""OPS-22 P2, D4: the front's settings on each kind of source, against the float colour path at the same effort and
quality.

    python -I analysis/ops22/p2_sources.py REPO [--work DIR]

For one each of an 8-bit JPEG (12 MP and 50 MP), an 8-bit HEIC, a 10-bit HEIC, a DNG, a 16-bit PNG, a PNG with
transparency and the grinder example's JPEG XL: the front's time (canonical.FRONT_SETTINGS, held to 2 CPUs by taskset,
as the front's 2 vCPU, in a fresh process per photo, with its peak resident memory, VmHWM; this machine cannot hold a
process to the front's MemoryMax=600M: no systemd user bus, a read-only cgroup tree), the canonical photo's size and
pixels, and its 8-bit sRGB read-back (load_canonical) against the same photo made on the float path with the front's
effort, quality and thresholds: the mean, 99th percentile and largest absolute difference over every sample, in 8-bit
levels. The 10-bit HEIC and the 16-bit PNG are made here from the DNG's 16-bit development (no such photo is in the
dataset); the PNG with transparency is the 12 MP JPEG with a radial alpha. Derived photos go to --work, outside git.
Prints a Markdown table and writes the raw lines to --work/p2_sources.jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

SOURCES = {   # name: path under the repo, or None for one made here
    "JPEG, 8-bit (Pixel, 12 MP)": "dataset/segmenter_positives/PXL_20260911_061147113.jpg",
    "JPEG, 8-bit (Pixel 9 Pro, 50 MP)": "dataset/segmenter_positives/20261028_112247_Pixel_9_Pro_Brazil.jpg",
    "HEIC, 8-bit (iPhone)": "dataset/classes_labels_only/IMG_6256D.HEIC",
    "HEIC, 10-bit (from the DNG)": None,
    "DNG (Pixel 9 Pro, 9.9 MP)": "dataset/segmenter_positives/20261027_135613_Pixel_9_Pro_Guatemala.dng",
    "PNG, 16-bit (from the DNG)": None,
    "PNG, 8-bit with transparency": None,
    "JPEG XL, the grinder example (store's canonical, 50 MP)": "webapp/static/examples/brazil-grinder.jxl",
}


def make_derived(repo: Path, work: Path) -> dict[str, Path]:
    import imagecodecs as ic
    import pillow_heif
    import rawpy
    from PIL import Image

    paths = {}
    dng = repo / SOURCES["DNG (Pixel 9 Pro, 9.9 MP)"]
    with rawpy.imread(str(dng)) as raw:
        rgb16 = raw.postprocess(output_bps=16, use_camera_wb=True)   # sRGB, its own curve
    paths["PNG, 16-bit (from the DNG)"] = work / "dng16.png"
    paths["PNG, 16-bit (from the DNG)"].write_bytes(ic.png_encode(rgb16))
    heif = pillow_heif.from_bytes(mode="RGB;16", size=(rgb16.shape[1], rgb16.shape[0]), data=rgb16.tobytes())
    paths["HEIC, 10-bit (from the DNG)"] = work / "dng10.heic"
    heif.save(str(paths["HEIC, 10-bit (from the DNG)"]), quality=90)
    jpeg = np.asarray(Image.open(repo / SOURCES["JPEG, 8-bit (Pixel, 12 MP)"]).convert("RGB"))
    h, w = jpeg.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    alpha = np.clip(255 * (1.3 - r), 0, 255).astype(np.uint8)       # opaque centre, transparent corners
    paths["PNG, 8-bit with transparency"] = work / "alpha.png"
    paths["PNG, 8-bit with transparency"].write_bytes(ic.png_encode(np.dstack([jpeg, alpha])))
    return paths


def one(repo: Path, photo: Path, out: Path) -> dict:
    """The front's canonical photo of `photo`, in this process: what the check service does with it."""
    sys.path.insert(0, str(repo))
    from coffeecv import canonical as cn
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None   # the dataset's tools raise it; the front's 60 MP ceiling is not under test here
    data = photo.read_bytes()
    t = time.perf_counter()
    src = cn.decode(data, photo.name, settings=cn.FRONT_SETTINGS)
    bits = src.pixels.dtype.itemsize * 8
    canon = cn.encode(src, settings=cn.FRONT_SETTINGS)
    seconds = time.perf_counter() - t
    out.write_bytes(canon)
    return {"bytes_in": len(data), "bits": bits, "seconds": round(seconds, 2), "canonical_bytes": len(canon),
            "peak_mb": peak_mb()}


def peak_mb() -> int:
    """This process's peak resident memory (VmHWM), as the check service logs it. Not ru_maxrss, which after a fork
    and exec still holds the parent's peak."""
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return round(int(status["VmHWM"].split()[0]) / 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--work", type=Path, default=Path("/tmp/ops22/sources"))
    ap.add_argument("--one", nargs=2, type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print(json.dumps(one(args.repo, *args.one)))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    from coffeecv import canonical as cn
    from coffeecv.jxl_header import read_header
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    float_settings = replace(cn.FRONT_SETTINGS, name="front, float colour", colour="float")
    derived = make_derived(args.repo, args.work)
    rows = []
    for i, (name, rel) in enumerate(SOURCES.items()):
        photo = derived[name] if rel is None else args.repo / rel
        front_out = args.work / f"{i}.front.jxl"
        res = json.loads(subprocess.run(
            ["taskset", "-c", "0,1", sys.executable, "-I", __file__, str(args.repo), "--one", str(photo),
             str(front_out)], check=True, capture_output=True, text=True).stdout)
        front = front_out.read_bytes()
        data = photo.read_bytes()
        flt = cn.encode(cn.decode(data, photo.name, settings=float_settings), settings=float_settings)
        a, b = cn.load_canonical(front).astype(np.int16), cn.load_canonical(flt).astype(np.int16)
        d = np.abs(a - b)
        h = read_header(front)
        res.update(source=name, size=f"{h.width} x {h.height}", float_bytes=len(flt), mean=round(float(d.mean()), 3),
                   p99=int(np.percentile(d, 99)), max=int(d.max()), canonical=cn.is_canonical(front))
        rows.append(res)
        print(json.dumps(res), file=sys.stderr)
    (args.work / "p2_sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("| Source | Bits | Upload | Front's photo | Size | Seconds (2 CPUs) | Peak MB | vs float: mean | p99 | max |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['source']} | {r['bits']} | {r['bytes_in'] / 1e6:.2f} MB | {r['size']} | "
              f"{r['canonical_bytes'] / 1e6:.2f} MB | {r['seconds']} | {r['peak_mb']} | {r['mean']} | {r['p99']} | "
              f"{r['max']} |")


if __name__ == "__main__":
    main()
