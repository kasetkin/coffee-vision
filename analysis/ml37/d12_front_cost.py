"""ML-37 D12 (R3): the front's canonicalization today against a linear Rec.2020 half-float canonical photo (D1), by
colour path and encoder, in time and peak memory, at FRONT_SETTINGS (quality 90, effort 3, kept up to 18 MP, over
that at most 15 MP), held to 2 CPUs as analysis/ml29/p11_front_cost.py holds them.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d12_front_cost.py [--cpus 0,1] [--repeats 2]

Each (photo, variant) runs in a fresh process under `taskset -c CPUS`, as p11 does: wall seconds of the decode (with
the gain map) and of the encode, CPU seconds of the process (user + system, less sensitive to the shared machine's
load than wall time), its peak resident memory (VmHWM, as the check service logs it) and the canonical photo's size.
The canonical photos are written to tmp/ml37/d12/out/ for d12_accuracy.py. The variants:

    today         canonical.canonicalize_with(FRONT_SETTINGS): integer path (float under a gain map), the peak pass
                  and the EETF, uint16 BT.709, chunked encoder. The baseline; its peak pass is timed apart.
    a_whole       D12 (a), the store's float path, the whole float16 picture handed to imagecodecs.jpegxl_encode.
    a_chunked     D12 (a), the float path per rectangle through a chunked float16 encoder (d12_proto).
    a_chunked32   a_chunked with the gain-map light held in float32 as canonical._with_gain_map holds it (gain-map
                  photos only): what float16 light saves.
    b_lcms16      LittleCMS's optimized transform into linear Rec.2020 uint16, then float16; chunked.
    b_lcmsf       LittleCMS's default transform from 8/16-bit samples straight to float32; chunked.
    b_lut         per-channel tables and a sum (exact for matrix-shaper profiles); chunked.
None of the new variants measures a peak: the tone map moves to the read-back (ML-37 R3).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "tmp" / "ml37" / "d12"
OUT = WORK / "out"

PHOTOS = {   # key: (name, path under the repo, or None for p11's 60 MP ceiling, made into WORK)
    "uhdr12": ("Ultra HDR JPEG (Pixel, 12 MP)",
               "dataset/2026-08-07__box_pictures_all_classes/class_001__Ethiopia_Sidamo/PXL_20260807_082019464.jpg"),
    "uhdr50": ("Ultra HDR JPEG (Pixel 9 Pro, 50 MP)",
               "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/20261112_144002_Pixel_9_Pro_Brazil.jpg"),
    "iso50": ("ISO 21496-1 JPEG (Galaxy Z Fold8 Ultra, 50 MP)",
              "dataset/segmenter_positives/20261103_150626_Galaxy_Z_Fold8_Ultra_Colombia.jpg"),
    "heic48": ("ISO 21496-1 HEIC (iPhone, 48 MP)", "dataset/segmenter_positives/IMG_6762.HEIC"),
    "ceil60": ("Ultra HDR JPEG at the ceiling (made here, 60 MP)", None),
    "plain48": ("Plain JPEG, no gain map (48 MP, the dataset's largest)",
                "dataset/tasty_coffee/class_001__Ethiopia_Sidamo/20261120_104254_KB2005_Ethiopia_Sidamo.jpg"),
    "dng12": ("DNG (Pixel 9 Pro, 12.5 MP, the largest under the front's 16 MP RAW ceiling)",
              "dataset/random_date_raccoon/class_013__Brazil_Decaff/20261109_130414_Pixel_9_Pro_Brazil_decaff.dng"),
}
VARIANTS = ("today", "a_whole", "a_chunked", "a_chunked32", "b_lcms16", "b_lcmsf", "b_lut")


def photo_path(key: str) -> Path:
    rel = PHOTOS[key][1]
    if rel is not None:
        return REPO / rel
    sys.path.insert(0, str(REPO / "analysis" / "ml29"))
    from p11_front_cost import make_ceiling
    WORK.mkdir(parents=True, exist_ok=True)
    return make_ceiling(WORK)


def vmhwm() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return round(int(line.split()[1]) / 1024)
    raise RuntimeError("no VmHWM")


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def measure(key: str, variant: str) -> dict:
    sys.path.insert(0, str(REPO / "webapp" / "front"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_service import MAX_PIXELS, RAW_MAX_PIXELS
    from coffeecv import canonical as cn
    import d12_proto as P
    import numpy as np
    photo = photo_path(key)
    data = photo.read_bytes()
    name = photo.name
    limit = RAW_MAX_PIXELS if cn.accepted_format(data, name).is_raw else MAX_PIXELS
    s = cn.FRONT_SETTINGS
    res: dict = {"bytes_in": len(data)}
    c0, t0 = cpu_seconds(), time.perf_counter()
    if variant == "today":
        src = cn.decode(data, name, limit, settings=s)
        t1 = time.perf_counter()
        res["mb_after_decode"] = vmhwm()
        src.peak()
        t2 = time.perf_counter()
        made = cn.encode(src, settings=s)
        t3 = time.perf_counter()
        h, w = src.pixels.shape[:2]
        res.update(decode_s=t1 - t0, peak_s=t2 - t1, encode_s=t3 - t2, gain_map="+" in src.decoder)
    else:
        path = P.light_float if variant.startswith("a_") else P.PATHS[variant.removeprefix("b_")]
        if variant == "a_chunked32":
            d = P.decode(data, name, limit, path, light_dtype=np.float32)
            if d.light is None:
                return {"skipped": "no gain map"}
        else:
            d = P.decode(data, name, limit, path)
        t1 = time.perf_counter()
        res["mb_after_decode"] = vmhwm()
        h, w = d.shape
        if variant == "a_whole":
            made = P.encode_whole(P.whole(d, path), s.quality, s.effort, cn.THREADS)
        else:
            made = P.encode_chunked(h, w, P.region_fn(d, path), level=s.quality, effort=s.effort,
                                    numthreads=cn.THREADS)
        t3 = time.perf_counter()
        res.update(decode_s=t1 - t0, peak_s=0.0, encode_s=t3 - t1, gain_map=d.light is not None)
    res.update(seconds=time.perf_counter() - t0, cpu_s=cpu_seconds() - c0, peak_mb=vmhwm(), size=f"{w} x {h}",
               canonical_bytes=len(made), load1=os.getloadavg()[0])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{key}_{variant}.jxl").write_bytes(made)
    return res


def free_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", default="0,1")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--photos", nargs="*", default=list(PHOTOS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--one", nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print(json.dumps(measure(*args.one)))
        return
    log = Path(__file__).with_name("d12_front_cost.jsonl")
    for key in args.photos:
        photo_path(key)                                    # the ceiling made before any timing
        for variant in args.variants:
            for rep in range(args.repeats):
                while free_mb() < 2000:                    # the shared machine: wait for room
                    print(f"waiting: {free_mb()} MB available", file=sys.stderr)
                    time.sleep(60)
                r = subprocess.run(["taskset", "-c", args.cpus, sys.executable, __file__, "--one", key, variant],
                                   capture_output=True, text=True)
                if r.returncode:
                    row = {"photo": key, "variant": variant, "rep": rep, "error": r.stderr[-2000:]}
                else:
                    row = {"photo": key, "variant": variant, "rep": rep, **json.loads(r.stdout)}
                print(json.dumps(row), file=sys.stderr)
                with log.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if "skipped" in row:
                    break


if __name__ == "__main__":
    main()
