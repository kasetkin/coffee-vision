"""ML-37 D12: the scale for d12_accuracy.txt's path differences: how far the lossy encode itself (FRONT_SETTINGS,
quality 90, effort 3) moves the float path's float16 samples, in read-back sRGB levels.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d12_encode_error.py [--photos ...]

Each photo decoded through d12_proto with the float path (D12 (a)), its float16 picture read back (d12_proto.
readback_levels) and compared with the read-back of tmp/ml37/d12/out/<photo>_a_chunked.jxl, which d12_front_cost.py
encoded from the same samples. Appends to d12_encode_error.txt next to this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imagecodecs as ic

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d12_accuracy as A      # noqa: E402
import d12_front_cost as FC   # noqa: E402
import d12_proto as P         # noqa: E402
from coffeecv import canonical as cn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", nargs="*", default=["uhdr12", "heic48", "iso50", "plain48", "dng12"])
    args = ap.parse_args()
    sys.path.insert(0, str(FC.REPO / "webapp" / "front"))
    from check_service import MAX_PIXELS, RAW_MAX_PIXELS
    lines = ["| Photo | encode vs its float16 input: max levels | mean | > 0.5 | > 1 |", "|---|---|---|---|---|"]
    for key in args.photos:
        photo = FC.photo_path(key)
        data, name = photo.read_bytes(), photo.name
        limit = RAW_MAX_PIXELS if cn.accepted_format(data, name).is_raw else MAX_PIXELS
        d = P.decode(data, name, limit, P.light_float)
        src = P.whole(d, P.light_float)
        del d
        enc = ic.jpegxl_decode((FC.OUT / f"{key}_a_chunked.jxl").read_bytes(), numthreads=2)
        diff = A.Diff()
        for y in range(0, src.shape[0], P.STRIP):
            diff.add(P.readback_levels(enc[y:y + P.STRIP]), P.readback_levels(src[y:y + P.STRIP]))
        lines.append(f"| {key} | {diff.row()} |")
        print(lines[-1], file=sys.stderr)
        del src, enc
    with Path(__file__).with_name("d12_encode_error.txt").open("a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
