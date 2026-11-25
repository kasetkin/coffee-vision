"""ML-37: where A's and B's 8-bit errors sit: the share of D27 blocks over 1 and 2 levels, and the blocks at sRGB's
gamut edge (a channel of the reference at 0 or 255 in at least 10% of the block's pixels).

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/front_forms_detail.py [--out tmp/ml37/front_forms]

Reads the files front_forms_measure.py wrote: the reference is C1 (lossless, the uncompressed 8-bit picture: its
decode equals it, front_forms_measure.txt), A is decoded and read back (matrix, clip, sRGB curve, rint to uint8), B is
decoded. Prints per-photo rows and totals (kept as analysis/ml37/front_forms_detail.txt).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import imagecodecs as ic
import numpy as np

from analysis.ml37 import d8_intensity_target as d8
from analysis.ml37.front_forms_measure import read_back8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/ml37/front_forms")
    out = Path(ap.parse_args().out)
    rows = list(csv.DictReader(open(out / "rows.csv")))
    tot = {k: 0 for k in ("blocks", "edge", "A1", "A2", "B1", "B2", "A2e", "B2e")}
    sums = {"A_edge": 0.0, "B_edge": 0.0, "A_in": 0.0, "B_in": 0.0}
    print(f"{'photo':36s} {'blocks':>7s} {'edge%':>6s} | {'A>1':>6s} {'A>2':>6s} {'B>1':>6s} {'B>2':>6s} | "
          f"block mean at edge A / B | elsewhere A / B")
    for r in rows:
        c1 = next((out / "C1").glob(f"{int(r['i']):02d}_*.jxl"))
        ref = ic.jpegxl_decode(c1.read_bytes(), numthreads=2)
        a = ic.jpegxl_decode((out / "A" / c1.name).read_bytes(), numthreads=2)
        got = {"A": np.empty(ref.shape, np.uint8), "B": ic.jpegxl_decode((out / "B" / c1.name).read_bytes(),
                                                                         numthreads=2)}
        for y in range(0, a.shape[0], 256):
            got["A"][y:y + 256] = read_back8(a[y:y + 256].astype(np.float32) / 65535)
        a = None
        rb = d8.blocks(ref.astype(np.float32))
        clipped = d8.blocks(((ref == 0) | (ref == 255)).astype(np.float32))      # share per block and channel
        edge = (clipped >= 0.1).any(axis=-1)                                     # per block
        ref = None
        d = {k: np.abs(d8.blocks(v.astype(np.float32)) - rb).max(axis=-1) for k, v in got.items()}  # worst channel
        got = None
        n, ne = edge.size, int(edge.sum())
        tot["blocks"] += n
        tot["edge"] += ne
        for k in ("A", "B"):
            tot[f"{k}1"] += int((d[k] > 1).sum())
            tot[f"{k}2"] += int((d[k] > 2).sum())
            tot[f"{k}2e"] += int((d[k][edge] > 2).sum())
            sums[f"{k}_edge"] += float(d[k][edge].sum())
            sums[f"{k}_in"] += float(d[k][~edge].sum())
        em = [f"{d[k][edge].mean():.3f}" if ne else "-" for k in ("A", "B")]
        im = [f"{d[k][~edge].mean():.3f}" for k in ("A", "B")]
        print(f"{r['i']:>2s} {r['stratum'][:33]:33s} {n:7d} {ne / n:6.1%} | {(d['A'] > 1).mean():6.2%} "
              f"{(d['A'] > 2).mean():6.2%} {(d['B'] > 1).mean():6.2%} {(d['B'] > 2).mean():6.2%} | "
              f"{em[0]} / {em[1]} | {im[0]} / {im[1]}", flush=True)
    b, e = tot["blocks"], tot["edge"]
    print(f"\nAll {len(rows)} photos: {b} blocks, {e} ({e / b:.1%}) at the gamut edge. Worst channel per block.")
    print(f"  blocks over 1 level: A {tot['A1'] / b:.3%}, B {tot['B1'] / b:.3%}; over 2 levels: A {tot['A2'] / b:.3%}, "
          f"B {tot['B2'] / b:.3%} (at the edge: A {tot['A2e']}, B {tot['B2e']} of {tot['A2']} and {tot['B2']})")
    print(f"  block mean (worst channel) at the edge: A {sums['A_edge'] / e:.3f}, B {sums['B_edge'] / e:.3f}; "
          f"elsewhere: A {sums['A_in'] / (b - e):.3f}, B {sums['B_in'] / (b - e):.3f}")


if __name__ == "__main__":
    sys.exit(main())
