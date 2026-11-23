"""ML-37 D9: time and memory of the read-back's maths on a 15 MP and a 50 MP photo, 2 threads, in strips.

The picture is the 50 MP example webapp/static/examples/brazil-grinder.jxl (6144 x 8160, a canonical photo), and its
centre 3360 x 4480 (15.05 MP, the front's cap). Inputs are prepared once into tmp/ml37/d9/ (`prep`): its uint16
samples (today's input) and their linear Rec.2020 in float16 (D1's, emulated as in d9_realdata.py). Each method then
runs in its own process (`run`), which loads only its input, so its peak RSS is its own; best of three.

  today   rec2020_to_srgb, uint16 -> uint8 (LittleCMS NOOPTIMIZE, 128-row strips): ML-21's read-back
  lcms1   option (b), one transform: LINEAR_REC2020 -> SRGB, float16 in, float32 out, then clip (clip as the tone map)
  lcms2   option (b) with room for D6's tone map: LINEAR_REC2020 -> linear sRGB, clip, linear sRGB -> SRGB
  numpy   option (a): float32, nine multiply-adds, clip, IEC curve; strips on a thread pool

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d9_speed.py prep
    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d9_speed.py > analysis/ml37/d9_speed.txt
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagecodecs as ic
import numpy as np

from coffeecv import canonical as cn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d9_realdata import M32, numpy_readback  # noqa: E402

SRC = Path("/workspace/webapp/static/examples/brazil-grinder.jxl")
TMP = Path("/workspace/tmp/ml37/d9")
THREADS = 2
LIN_SRGB = ic.cms_profile("rgb", whitepoint=(0.3127, 0.3290), primaries=(0.64, 0.33, 0.30, 0.60, 0.15, 0.06),
                          gamma=1.0)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def prep() -> None:
    rec = ic.jpegxl_decode(SRC.read_bytes(), numthreads=2)
    h, w, _ = rec.shape
    y0, x0 = (h - 4480) // 2, (w - 3360) // 2
    for name, a in (("50", rec), ("15", rec[y0:y0 + 4480, x0:x0 + 3360])):
        a = np.ascontiguousarray(a)
        np.save(TMP / f"u16_{name}.npy", a)
        lin = np.empty(a.shape, np.float16)
        for y in range(0, a.shape[0], 256):
            lin[y:y + 256] = ic.cms_transform(a[y:y + 256], cn.REC2020, cn.LINEAR_REC2020, intent=cn.RELATIVE,
                                              outdtype=np.float32)
        np.save(TMP / f"lin16_{name}.npy", lin)
        print(name, a.shape, a.shape[0] * a.shape[1] / 1e6, "MP")


def strips(h: int, rows: int):
    return range(0, h, rows)


def run(size: str, method: str) -> None:
    src = np.load(TMP / (f"u16_{size}.npy" if method == "today" else f"lin16_{size}.npy"))
    base = rss_mb()
    h = src.shape[0]
    times = []
    for _ in range(3):
        t = time.perf_counter()
        if method == "today":
            out = cn.rec2020_to_srgb(src, strip=128, numthreads=THREADS)
        else:
            out = np.empty(src.shape, np.float32)
            rows = 128

            if method == "lcms1":
                def f(y):
                    o = out[y:y + rows]
                    ic.cms_transform(src[y:y + rows], cn.LINEAR_REC2020, cn.SRGB, intent=cn.RELATIVE,
                                     outdtype=np.float32, out=o)
                    np.clip(o, 0, 1, out=o)
            elif method == "lcms2":
                def f(y):
                    lin = ic.cms_transform(src[y:y + rows], cn.LINEAR_REC2020, LIN_SRGB, intent=cn.RELATIVE,
                                           outdtype=np.float32)   # a strip-sized temporary: no in-place transform
                    np.clip(lin, 0, 1, out=lin)      # D6's tone map would go here
                    ic.cms_transform(lin, LIN_SRGB, cn.SRGB, intent=cn.RELATIVE, outdtype=np.float32,
                                     out=out[y:y + rows])
            else:
                def f(y):
                    out[y:y + rows] = numpy_readback(src[y:y + rows], M32, np.float32)
            with ThreadPoolExecutor(THREADS) as pool:
                list(pool.map(f, strips(h, rows)))
        times.append(time.perf_counter() - t)
        out = None
    print(f"{size} MP {method:6s} best {min(times):6.2f} s (runs {', '.join(f'{x:.2f}' for x in times)}); "
          f"input {src.nbytes / 1e6:.0f} MB; peak RSS {rss_mb():.0f} MB, after loading the input {base:.0f} MB")


def main() -> None:
    if sys.argv[1:] == ["prep"]:
        prep()
    elif sys.argv[1:2] == ["run"]:
        run(sys.argv[2], sys.argv[3])
    else:
        print(f"{os.cpu_count()} CPUs here (Ryzen 5 5600U); {THREADS} threads per read-back; output kept whole "
              f"(uint8 today, float32 otherwise)")
        for size in ("15", "50"):
            for method in ("today", "lcms1", "lcms2", "numpy"):
                r = subprocess.run([sys.executable, __file__, "run", size, method], capture_output=True, text=True)
                print(r.stdout.strip() or r.stderr[-800:], flush=True)


if __name__ == "__main__":
    main()
