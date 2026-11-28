"""ML-37 D6: the tables of d6_readback_options.txt, from tmp/ml37/d6/readback_rows.json.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d6_summarize.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROWS = json.loads((REPO / "tmp" / "ml37" / "d6" / "readback_rows.json").read_text())
OPTS = ("a2020", "aSRGB", "aSRGBp", "b", "c2", "c4")


def med_max(rs, key):
    v = np.array([r[key] for r in rs if key in r], float)
    return (float(np.median(v)), float(v.max())) if v.size else (float("nan"), float("nan"))


def main() -> None:
    ok = [r for r in ROWS if not r.get("skipped")]
    print("skipped:", [(r["camera"], r["skipped"]) for r in ROWS if r.get("skipped")])
    gm = [r for r in ok if r["kind"] == "gainmap"]
    sdr = [r for r in ok if r["kind"] == "sdr"]
    print(f"gain-map photos: {len(gm)} ({sum(r['mask'] for r in gm)} with a bean mask); standard-range: {len(sdr)}")

    print("\n1. Against today's models' input T (load_canonical, 8-bit), ML-29 D27's 32x32 block means, levels of 255.")
    print("   median photo (worst photo): mean |d|, worst block, signed shift (+ brighter than T)")
    for region, tag in (("bean blocks", "bblk"), ("whole photo", "all")):
        print(f"   {region}, gain-map photos (n={len(gm)})")
        for k in OPTS:
            m, w, s = (med_max(gm, f"{k}_{tag}_{x}") for x in ("mean", "worst", "shift"))
            smin = min(r[f"{k}_{tag}_shift"] for r in gm if f"{k}_{tag}_shift" in r)
            print(f"     {k:7} mean {m[0]:6.3f} ({m[1]:6.3f})  worst {w[0]:6.2f} ({w[1]:6.2f})  "
                  f"shift {s[0]:+6.3f} (from {smin:+.3f} to {s[1]:+.3f})")
    print(f"   whole photo, standard-range photos (n={len(sdr)})")
    for k in OPTS:
        m, w, s = (med_max(sdr, f"{k}_all_{x}") for x in ("mean", "worst", "shift"))
        print(f"     {k:7} mean {m[0]:6.3f} ({m[1]:6.3f})  worst {w[0]:6.2f} ({w[1]:6.2f})  shift {s[0]:+6.3f}")

    print("\n2. Bean pixels, each option against a2020 (today's maths without its lossy integer photo); a2020_vs_T is that")
    print("   photo's own per-pixel coding noise. median photo (worst photo)")
    for k in ("a2020_vs_T",) + OPTS[1:]:
        cells = []
        for x, f in (("px_mean", "{:.3f}"), ("px_p99", "{:.1f}"), ("px_max", "{:.1f}"), ("px_over2", "{:.4%}")):
            m, w = med_max([r for r in gm if r["mask"]], f"{k}_{x}")
            cells.append(f"{x[3:]} {f.format(m)} ({f.format(w)})")
        print(f"     {k:11} " + "  ".join(cells))

    print("\n3. The ordering: max(R,G,B) of linear sRGB over that of linear Rec.2020, per bean pixel (5th/50th/95th "
          "percentile), the share of bean pixels a2020 leaves above 1.0 in sRGB (clipped per channel), and the peaks")
    for r in sorted(gm, key=lambda r: (r["camera"], r["peak2020"])):
        if not r["mask"]:
            continue
        print(f"     {r['camera'][:30]:30} peak 2020 {r['peak2020']:.3f} sRGB {r['peak_srgb']:.3f}  ratio "
              f"{r['srgb_over_2020_max_q5']:.2f}/{r['srgb_over_2020_max_q50']:.2f}/{r['srgb_over_2020_max_q95']:.2f}"
              f"  a2020 clipped {r['region_a2020_srgb_clipped']:.4%}  beans above white {r['region_above_white']:.4%}"
              f"  above c2's knee {r['region_above_c2']:.3%}, c4's {r['region_above_c4']:.3%}")
    pr = np.array([r["peak_srgb"] / r["peak2020"] for r in gm])
    print(f"   sRGB peak / Rec.2020 peak: median {np.median(pr):.3f}, range {pr.min():.3f}-{pr.max():.3f}")

    print("\n4. Standard-range photos: the region (bean mask, else the whole photo) above c2's and c4's knee")
    for r in sdr:
        print(f"     {r['camera'][:30]:30} peak {r['peak2020']:.3f}  above white {r['region_above_white']:.4%}  "
              f"above c2's knee {r['region_above_c2']:.3%}  c4's {r['region_above_c4']:.3%}  mask {r['mask']}")

    print("\n5. Time (this machine, shared; numpy, 2 threads at most)")
    t = np.array([r["t_peak_s"] / (r["h"] * r["w"] / 1e6) for r in ok])
    print(f"   the Rec.2020 peak pass (max over channels + 99.99th percentile): median {np.median(t) * 1000:.0f} ms per MP "
          f"(range {t.min() * 1000:.0f}-{t.max() * 1000:.0f})")


if __name__ == "__main__":
    main()
