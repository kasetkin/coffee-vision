"""ML-7 P8 on yappy: the serving code of this branch, webapp.app, on LiteRT's GPU with torch unimportable, outside the
live service. Boots the app as a worker would (its warm-up included, ADR 0021), then sends each photo to /crop and
/classify through Flask's test client, timed. Records the boot, each graph's delegate report, the GPU proof from
the process's own maps (Intel's Vulkan driver and LiteRT's WebGPU library mapped, never Mesa's CPU renderer), and
each answer, for comparison with PyTorch's (p8_compare.py).

    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json COFFEE_CV_CHECKPOINT=models/<model>.pt \\
    COFFEE_CV_LITERT_DIR=<dir> COFFEE_CV_LITERT_RUNTIME=~/ml7/rt_vulkan COFFEE_CV_WARMUP_PHOTO=<photo> \\
    COFFEE_CV_LOG_DIR=<dir> <venv>/bin/python p8_yappy.py --out result.json PHOTO...

Run from the copy of the repo (cwd), with the venv P10 will lock: the webapp's pins without torch, plus
ai-edge-litert.
"""
from __future__ import annotations

import argparse
import importlib.abc
import io
import json
import sys
import time
from pathlib import Path


class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("torch", "torchvision", "timm", "segment_anything", "efficientvit"):
            raise ImportError(f"serving imported {name}")


def _maps() -> dict:
    libs = {Path(line.split()[-1]).name for line in Path("/proc/self/maps").read_text().splitlines()
            if len(line.split()) >= 6 and ".so" in line}
    return {"libvulkan_intel": any(n.startswith("libvulkan_intel") for n in libs),
            "libvulkan_lvp": any(n.startswith("libvulkan_lvp") for n in libs),
            "webgpu_accelerator": "libLiteRtWebGpuAccelerator.so" in libs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--passes", type=int, default=2, help="passes over the photos; the first is cold per photo")
    args = ap.parse_args()
    sys.meta_path.insert(0, _NoTorch())
    sys.path.insert(0, str(Path.cwd()))
    t0 = time.monotonic()
    import webapp.app as app
    boot = {"import_s": round(time.monotonic() - t0, 2), **app.WARM_UP}
    client = app.app.test_client()
    rows = []
    for k in range(args.passes):
        for photo in args.photos:
            data = photo.read_bytes()
            row = {"pass": k, "photo": photo.name}
            for endpoint in ("/crop", "/classify"):
                t = time.monotonic()
                r = client.post(endpoint, data={"photo": (io.BytesIO(data), photo.name)},
                                content_type="multipart/form-data")
                row[endpoint] = {"status": r.status_code, "s": round(time.monotonic() - t, 3)}
                if endpoint == "/classify" and r.is_json:
                    row["classify"] = r.get_json()
            rows.append(row)
            print(f"pass {k} {photo.name} crop {row['/crop']['s']:.1f} s classify {row['/classify']['s']:.1f} s",
                  flush=True)
    out = {"boot": boot, "build": app.BUILD, "reports": app.models.reports, "maps": _maps(),
           "manifest": {g: {k: s[k] for k in ("file", "accelerator", "precision", "batch")}
                        for g, s in app.models.manifest["graphs"].items()},
           "rows": rows}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"boot": boot, "maps": out["maps"],
                      "all_on_gpu": {g: r["all_on_gpu"] for g, r in app.models.reports.items()}}, indent=1))


if __name__ == "__main__":
    main()
