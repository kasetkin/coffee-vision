"""ML-37 P3 (plan finding 7, R3): the front's time and peak memory with the linear canonical photo and the float colour
path (D1, D12 (a)), against its unit's MemoryMax=600M, on ML-29 R6's photos and the 40 photos of the front-forms study.

    python -I analysis/ml37/p3_front_cost.py --code CODE_ROOT --data DATA_ROOT --label NAME [--photos r6|forms|all]
        [--out FILE.jsonl] [--work DIR]

`--code` is the checkout whose coffeecv/ and webapp/front/ are measured: this ticket's branch for the new form, and a
`git archive` of the commit before it (4b0a9bc) for the form before, so both run on the same machine in one session.
`--data` holds dataset/ (the main checkout's, for a worktree). Each photo runs in a fresh process under
`taskset -c 0,1` (the front's 2 vCPU), as analysis/ml29/p11_front_cost.py runs it: canonical.canonicalize_with with
FRONT_SETTINGS and the check service's own ceiling (check_service.MAX_PIXELS, RAW_MAX_PIXELS for RAW), wall seconds,
CPU seconds (user + system), the peak resident memory (VmHWM, as the check service logs it) and the canonical photo's
size. This machine cannot hold a process to MemoryMax: the peak is what the cgroup would have to hold. Not on the front.

ML-29 R6's photos are p11_front_cost.py's five (the 60 MP ceiling made into --work) and D12's research's two more
(the dataset's largest plain JPEG, a 12.5 MP DNG); the forms study's are front_forms_sample.txt's 40.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R6 = {
    "Ultra HDR JPEG (Pixel, 12 MP)":
        "dataset/2026-08-07__box_pictures_all_classes/class_001__Ethiopia_Sidamo/PXL_20260807_082019464.jpg",
    "Ultra HDR JPEG (Pixel 9 Pro, 50 MP)":
        "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/20261112_144002_Pixel_9_Pro_Brazil.jpg",
    "ISO 21496-1 JPEG (Galaxy Z Fold8 Ultra, 50 MP)":
        "dataset/segmenter_positives/20261103_150626_Galaxy_Z_Fold8_Ultra_Colombia.jpg",
    "ISO 21496-1 HEIC (iPhone, 48 MP)": "dataset/segmenter_positives/IMG_6762.HEIC",
    "Ultra HDR JPEG at the ceiling (made here, 60 MP)": None,
    "Plain JPEG, no gain map (48 MP)":
        "dataset/tasty_coffee/class_001__Ethiopia_Sidamo/20261120_104254_KB2005_Ethiopia_Sidamo.jpg",
    "DNG (Pixel 9 Pro, 12.5 MP)":
        "dataset/random_date_raccoon/class_013__Brazil_Decaff/20261109_130414_Pixel_9_Pro_Brazil_decaff.dng",
}


def forms() -> dict[str, str]:
    rows = [line.split("\t") for line in (HERE / "front_forms_sample.txt").read_text().splitlines()
            if line and not line.startswith("#")]
    return {f"{i} {stratum}": path for i, stratum, path in rows}


def make_ceiling(work: Path) -> Path:
    """p11_front_cost.make_ceiling's 60 MP Ultra HDR JPEG: a gradient's light up to 4x white, by libultrahdr."""
    import imagecodecs as ic
    import numpy as np
    h, w = 7500, 8000
    path = work / "ceiling_60mp.jpg"
    if not path.exists():
        ramp = np.linspace(0.05, 4.0, w, dtype=np.float16)
        hdr = np.empty((h, w, 4), np.float16)
        hdr[..., :3] = ramp[None, :, None]
        hdr[..., 3] = 1
        path.write_bytes(ic.ultrahdr_encode(hdr))
    return path


def one(code: Path, photo: Path) -> dict:
    """The front's canonical photo of `photo`, in this process, as the check service makes it."""
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / "webapp" / "front"))
    from check_service import MAX_PIXELS, RAW_MAX_PIXELS
    from coffeecv import canonical as cn
    from coffeecv.jxl_header import read_header
    data = photo.read_bytes()
    raw = photo.suffix.lower() in cn.RAW_EXTENSIONS
    t, c = time.perf_counter(), time.process_time()
    made = cn.canonicalize_with(data, photo.name, cn.FRONT_SETTINGS, RAW_MAX_PIXELS if raw else MAX_PIXELS)
    seconds, cpu = time.perf_counter() - t, time.process_time() - c
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    h = read_header(made.canonical)
    return {"bytes_in": len(data), "size": f"{made.width} x {made.height}", "seconds": round(seconds, 2),
            "cpu_seconds": round(cpu, 2), "peak_mb": round(int(status["VmHWM"].split()[0]) / 1024),
            "canonical_bytes": len(made.canonical), "transfer": h.colour.transfer,
            "intensity_target": h.intensity_target}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--photos", choices=("r6", "forms", "all"), default="all")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--work", type=Path, default=Path("/tmp/ml37_p3"))
    ap.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print(json.dumps(one(args.code.resolve(), args.one)))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    photos = {**(R6 if args.photos in ("r6", "all") else {}), **(forms() if args.photos in ("forms", "all") else {})}
    for name, rel in photos.items():
        path = make_ceiling(args.work) if rel is None else args.data / rel
        proc = subprocess.run(["taskset", "-c", "0,1", sys.executable, "-I", __file__, "--code", str(args.code),
                               "--data", str(args.data), "--label", args.label, "--one", str(path)],
                              capture_output=True, text=True)
        row = {"label": args.label, "photo": name, "path": rel}
        row.update(json.loads(proc.stdout) if proc.returncode == 0 else {"error": proc.stderr.strip()[-300:]})
        line = json.dumps(row)
        print(line, flush=True)
        if args.out:
            with args.out.open("a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
