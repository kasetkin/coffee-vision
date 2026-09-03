"""Sweep the FFT search band against the 30 hand-counted crops, via PRODUCTION code.

Why this exists rather than another entry in estimators.py: `m0_fft_radial` there
is a frozen private copy of the FFT method, kept as the incumbent for the 2026-08-11
five-method comparison. It hardcodes `lo, hi = 4, 80`, so editing
`coffeecv/bean_scale.py` and re-running `benchmark.py` changes nothing at all --
a band sweep driven through it would report a confident null no matter what the
band did. This calls `estimate_bean_pitch_k` directly, so what is measured is
what ships.

Two numbers per band, per the finding this was built to test:

- **MAPE** against `gt_spacing_px`, both at the shipped CALIBRATION_K and at a
  calibration refitted for that band. The refit matters: 1.18 was fitted at
  k_lo=4, so judging a wider band with it measures miscalibration, not the band.
  Widening the band and then refitting is the change you would actually make.
- **Pinned fraction** -- how often the argmax lands on the band's low edge,
  meaning the true period lies past it and is being clipped. Clipping k upward
  under-estimates pitch, and the rate is rig-dependent (21%-58% across rigs on a
  192-photo sample), which is what makes it a rig-invariance problem rather than
  a constant bias.

The README's ranking criterion is bias *consistency across rigs*, not raw
accuracy -- a constant multiplicative bias is one constant away from fixed, a
rig-dependent one is not fixable without a per-rig constant this project refuses
to introduce. So per-rig bias spread is reported alongside.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coffeecv.bean_scale import CALIBRATION_K, estimate_bean_pitch_k  # noqa: E402

HERE = Path(__file__).parent
SRC = Path(__file__).resolve().parents[2] / "data" / "cropped"


def measure(k_lo: int, k_hi: int, analysis_frac: float | None) -> list[dict]:
    gt = json.loads((HERE / "ground_truth.json").read_text())
    rows = []
    for m in gt:
        img = cv2.imread(str(SRC / m["photo"]))
        if img is None:
            raise SystemExit(f"missing photo: {SRC / m['photo']}")
        pitch, k = estimate_bean_pitch_k(img, k_lo=k_lo, k_hi=k_hi,
                                         analysis_frac=analysis_frac)
        rows.append({"rig": m["rig"], "name": m["name"], "gt": m["gt_spacing_px"],
                     "pitch": pitch, "k": k, "pinned": k == k_lo})
    return rows


def score(rows: list[dict], k_lo: int) -> dict:
    gt = np.array([r["gt"] for r in rows])
    pitch = np.array([r["pitch"] for r in rows])
    # Refit the calibration for this band: the shipped 1.18 was fitted at k_lo=4.
    raw = pitch / CALIBRATION_K
    refit_k = float(np.mean(gt / raw))
    mape = float(np.mean(np.abs(pitch - gt) / gt) * 100)
    mape_refit = float(np.mean(np.abs(raw * refit_k - gt) / gt) * 100)

    per_rig_bias = {}
    for rig in sorted({r["rig"] for r in rows}):
        sel = [r for r in rows if r["rig"] == rig]
        per_rig_bias[rig] = float(np.mean([r["pitch"] / r["gt"] for r in sel]))
    spread = max(per_rig_bias.values()) - min(per_rig_bias.values())

    return {
        "k_lo": k_lo,
        "mape": mape,
        "mape_refit": mape_refit,
        "refit_k": refit_k,
        "pinned": float(np.mean([r["pinned"] for r in rows]) * 100),
        "bias_spread": spread,
        "per_rig_bias": per_rig_bias,
        "per_rig_pinned": {rig: float(np.mean([r["pinned"] for r in rows if r["rig"] == rig]) * 100)
                           for rig in sorted({r["rig"] for r in rows})},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-lo", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    help="band low edges to sweep (shipped default is 4)")
    ap.add_argument("--k-hi", type=int, default=80)
    ap.add_argument("--analysis-frac", type=float, default=None,
                    help="override ANALYSIS_FRAC (default: the shipped 0.40)")
    ap.add_argument("--out", default=str(HERE / "band_sweep.json"))
    args = ap.parse_args()

    results = []
    for lo in args.k_lo:
        rows = measure(lo, args.k_hi, args.analysis_frac)
        s = score(rows, lo)
        results.append(s)
        print(f"k_lo={lo:<2} MAPE {s['mape']:5.1f}%  (refit {s['mape_refit']:5.1f}% "
              f"@ K={s['refit_k']:.3f})  pinned {s['pinned']:4.1f}%  "
              f"rig-bias spread {s['bias_spread']:.3f}", flush=True)

    print(f"\n{'k_lo':<5}" + "".join(f"{r:>12}" for r in results[0]["per_rig_pinned"]))
    print("pinned % per rig")
    for s in results:
        print(f"{s['k_lo']:<5}" + "".join(f"{v:>11.0f}%" for v in s["per_rig_pinned"].values()))

    best = min(results, key=lambda s: s["mape_refit"])
    print(f"\nlowest refitted MAPE: k_lo={best['k_lo']} at {best['mape_refit']:.1f}% "
          f"(shipped k_lo=4 is {next(s['mape_refit'] for s in results if s['k_lo'] == 4):.1f}%)")
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
