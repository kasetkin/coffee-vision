"""ML-19 P8: coffeecv.canonical itself on the front, one photo per process, for D23, D24 and D38.

    python front_measure.py REPO PHOTO [--max-pixels N]

Prints one JSON line: the photo, its size, the outcome (canonical, TooLarge or Unreadable), seconds and the process's
peak memory. Run each photo in its own `systemd-run --user --scope -p MemoryMax=500M -p MemorySwapMax=0` and read the
scope's memory.peak beside it (analysis/ml19/p1b/README.md).
"""
import json
import resource
import sys
import time
from pathlib import Path

repo, photo = Path(sys.argv[1]), Path(sys.argv[2])
max_pixels = int(sys.argv[sys.argv.index("--max-pixels") + 1]) if "--max-pixels" in sys.argv else None
sys.path.insert(0, str(repo))
from coffeecv import canonical as cn  # noqa: E402
from coffeecv.jxl_header import read_header  # noqa: E402 -- the result's size, without decoding it

t = time.perf_counter()
try:
    out = cn.canonicalize_file(photo, max_pixels=max_pixels)
    h = read_header(out)
    outcome, size, n = "canonical", [h.width, h.height], len(out)
except cn.TooLarge as exc:
    outcome, size, n = f"TooLarge: {exc}", None, 0
except cn.Unreadable as exc:
    outcome, size, n = f"Unreadable: {exc}", None, 0
print(json.dumps({"photo": photo.name, "bytes_in": photo.stat().st_size, "outcome": outcome, "size": size,
                  "bytes_out": n,
                  "seconds": round(time.perf_counter() - t, 2),
                  "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}), flush=True)
