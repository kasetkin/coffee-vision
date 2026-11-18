"""ML-29 P11 (R6): what applying gain maps costs the front, against its budget (ML-19 D22: 40 s for a 10 MB photo;
D38: the service's MemoryMax=600M, a 60 MP photo's peak at most 400 MB).

    python -I analysis/ml29/p11_front_cost.py REPO [--work DIR]

Each photo canonicalized with canonical.FRONT_SETTINGS twice, gain maps on (as adopted) and off (as before), each in
a fresh process held to 2 CPUs by `taskset -c 0,1` (the front's 2 vCPU), with its peak resident memory, VmHWM, as
analysis/ops22/p2_sources.py measures the front's settings (this machine cannot hold a process to MemoryMax). The
photos: the dataset's largest gain-map photo of each form within the front's 60 MP ceiling, a 12 MP Pixel Ultra HDR
JPEG, and a 60 MP Ultra HDR JPEG made here at the ceiling (into --work, outside git). Prints a Markdown table.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

MAX_PIXELS = 60_000_000   # webapp/front/check_service.py's ceiling for a photo that is not RAW
PHOTOS = {   # name: path under the repo, or None for the one made here
    "Ultra HDR JPEG (Pixel, 12 MP)":
        "dataset/2026-08-07__box_pictures_all_classes/class_001__Ethiopia_Sidamo/PXL_20260807_082019464.jpg",
    "Ultra HDR JPEG (Pixel 9 Pro, 50 MP)":
        "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/20261112_144002_Pixel_9_Pro_Brazil.jpg",
    "ISO 21496-1 JPEG (Galaxy Z Fold8 Ultra, 50 MP)":
        "dataset/segmenter_positives/20261103_150626_Galaxy_Z_Fold8_Ultra_Colombia.jpg",
    "ISO 21496-1 HEIC (iPhone, 48 MP)": "dataset/segmenter_positives/IMG_6762.HEIC",
    "Ultra HDR JPEG at the ceiling (made here, 60 MP)": None,
}


def make_ceiling(work: Path) -> Path:
    """A 60 MP Ultra HDR JPEG: a gradient's light, up to 4x white, through libultrahdr's encoder."""
    import imagecodecs as ic
    import numpy as np
    h, w = 7500, 8000                                  # 60 MP; libultrahdr encodes at most 8192 x 8192
    path = work / "ceiling_60mp.jpg"
    if not path.exists():
        ramp = np.linspace(0.05, 4.0, w, dtype=np.float16)
        hdr = np.empty((h, w, 4), np.float16)
        hdr[..., :3] = ramp[None, :, None]
        hdr[..., 3] = 1
        path.write_bytes(ic.ultrahdr_encode(hdr))
    return path


def one(repo: Path, photo: Path, gain_maps: bool) -> dict:
    """The front's canonical photo of `photo`, in this process, as the check service makes it."""
    sys.path.insert(0, str(repo))
    from coffeecv import canonical as cn
    settings = replace(cn.FRONT_SETTINGS, gain_maps=gain_maps)
    data = photo.read_bytes()
    t = time.perf_counter()
    made = cn.canonicalize_with(data, photo.name, settings, MAX_PIXELS)   # check_service.canonicalize's call
    seconds = time.perf_counter() - t
    return {"bytes_in": len(data), "size": f"{made.width} x {made.height}", "bits": made.bits, "seconds": round(seconds, 2),
            "canonical_bytes": len(made.canonical), "peak_mb": peak_mb()}


def peak_mb() -> int:
    """This process's peak resident memory (VmHWM), as the check service logs it."""
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return round(int(status["VmHWM"].split()[0]) / 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--work", type=Path, default=Path("/tmp/ml29_p11"))
    ap.add_argument("--one", nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print(json.dumps(one(args.repo, Path(args.one[0]), args.one[1] == "on")))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, rel in PHOTOS.items():
        photo = make_ceiling(args.work) if rel is None else args.repo / rel
        row = {"photo": name}
        for mode in ("off", "on"):
            res = json.loads(subprocess.run(
                ["taskset", "-c", "0,1", sys.executable, "-I", __file__, str(args.repo), "--one", str(photo), mode],
                check=True, capture_output=True, text=True).stdout)
            row[mode] = res
        rows.append(row)
        print(json.dumps(row), file=sys.stderr)
    print("| Photo | Upload | Front's photo | Seconds off -> on (2 CPUs) | Peak MB off -> on |")
    print("|---|---|---|---|---|")
    for r in rows:
        off, on = r["off"], r["on"]
        print(f"| {r['photo']} | {on['bytes_in'] / 1e6:.2f} MB | {on['size']}, {on['bits']}-bit | {off['seconds']} -> {on['seconds']} "
              f"| {off['peak_mb']} -> {on['peak_mb']} |")


if __name__ == "__main__":
    main()
