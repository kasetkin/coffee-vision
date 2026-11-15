"""ML-29 P5's check of the wired tone map against P4's research (D33, D35).

    python analysis/ml29/hdr_pin.py

For each HDR file made here (tmp/ml29_made, D25), the canonical photo canonicalization makes now (store settings)
against two things, as the models read them (8-bit sRGB, 32 x 32 block means, tone_map_measure.compare):

- "vs research": the research's own rendering of the chosen operator, tone_map.eetf_bt2408 at the picture's measured
  peak (tone_map_measure.rendered, then models_read): production should be that operator, so the difference is the
  two code paths' rounding only.
- "vs reference": the research's reference, the darktable 8-bit sRGB export area-averaged to a quarter
  (tone_map_measure.darktable's). D35's regression pin: the highlights file within 0.5 level of P4's 6.99; 39.8.

Reads the files in tmp/ only; writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import tone_map_measure as m  # noqa: E402

from coffeecv import canonical as cn  # noqa: E402
from coffeecv import tone_map as tm  # noqa: E402

FILES = ("pq-p3-16bit.jxl", "pq-rec2020-10bit.avif", "pq-rec2020-12bit.heic", "hlg-p3-16bit.jxl",
         "hlg-rec2020-10bit.avif", "hlg-rec2020-12bit.heic", "pq-rec2020-highlights-to-812nits-16bit.jxl")
PIN = {"pq-rec2020-highlights-to-812nits-16bit.jxl": (6.99, 39.8)}   # P4's (b) EETF, measured peak


def main() -> None:
    ref_sig = m.Signal(m.DARKTABLE / f"{m.STEM}8bit-sRGB.jxl")
    ref = m.models_read(m.rendered(ref_sig, tm.clip_at_white, 1.0, scale=m.SCALE))
    del ref_sig
    failed = False
    for name in FILES:
        path = m.MADE / name
        made = cn.canonicalize_with(path.read_bytes(), name, cn.STORE_SETTINGS)
        got = cn.load_canonical(made.canonical, numthreads=m.THREADS)
        sig = m.Signal(path)
        peak = sig.peaks()["measured"]
        research = m.models_read(m.rendered(sig, tm.eetf_bt2408, peak))
        same = m.compare(got, research)
        rows = min(got.shape[0], ref.shape[0])       # a made file lost the export's bottom 4 rows
        r = m.compare(got[:rows], ref[:rows])
        line = (f"{name:46s} peak {peak:5.2f}  vs research {same['mean']:5.2f}; {same['worst']:4.1f}  "
                f"vs reference {r['mean']:5.2f}; {r['worst']:5.1f}")
        if name in PIN:
            pm, pw = PIN[name]
            ok = abs(r["mean"] - pm) <= 0.5 and abs(r["worst"] - pw) <= 0.5
            failed |= not ok
            line += f"  pin {pm}; {pw}: {'held' if ok else 'BROKEN'}"
        print(line, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
