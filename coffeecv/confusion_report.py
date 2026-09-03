"""Which class pairs does the model actually confuse? Aggregated over archived folds.

The plan's section 2d named two "confusable pairs" -- 006 Cerrado/007 MonteCristo
and 001 Sidamo/008 Kochere -- carried forward from the experiment log and the
iPhone/OnePlus spot-checks, and its own verification record flagged them as the
one claim never recomputed from confusion matrices. This recomputes them.

Cross-rig folds only, and deliberately: an in-distribution confusion matrix is
close to diagonal and says little, whereas transfer to an unseen camera is where
class similarity actually bites. Rates are symmetrised -- (i->j + j->i) over the
combined support of both classes -- because "these two look alike" is a property
of the pair, not of a direction, though the per-direction counts are printed too
since they are often lopsided.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from coffeecv.config import REPO_ROOT

EXPS = REPO_ROOT / "experiments"


def aggregate(lo: int, hi: int) -> tuple[np.ndarray, list[str], dict[str, str], int]:
    index = {int(r["exp"]): r for r in csv.DictReader(open(EXPS / "index.csv"))
             if r["exp"].isdigit()}
    agg, labels, names, used = None, None, {}, 0
    for e in sorted(index):
        if not (lo <= e <= hi and index[e]["xrig_macro_f1"]):
            continue
        dirs = list(EXPS.glob(f"exp{e}__*"))
        if not dirs:
            continue
        mf = dirs[0] / "metrics.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        sp = m["splits"].get("test_xrig")
        if not sp or "confusion_matrix" not in sp:
            continue
        cm = np.array(sp["confusion_matrix"], dtype=float)
        order = sp.get("confusion_matrix_row_order") or m["class_ids"]
        if labels is None:
            labels, agg, names = order, np.zeros_like(cm), m.get("class_labels", {})
        if order != labels or cm.shape != agg.shape:
            continue  # a different head width (e.g. post-class_010) is not summable with these
        agg += cm
        used += 1
    if agg is None:
        raise SystemExit(f"no cross-rig confusion matrices found in exp{lo}-{hi}")
    return agg, labels, names, used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-exp", type=int, default=106,
                    help="first exp id (default 106 = start of the current-config era)")
    ap.add_argument("--to-exp", type=int, default=10**9)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    agg, labels, names, used = aggregate(args.from_exp, args.to_exp)
    print(f"aggregated {used} cross-rig confusion matrices (exp{args.from_exp}-{args.to_exp})\n")

    row_tot = agg.sum(axis=1)
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            denom = row_tot[i] + row_tot[j]
            if denom:
                pairs.append(((agg[i, j] + agg[j, i]) / denom, labels[i], labels[j],
                              agg[i, j], agg[j, i]))
    pairs.sort(reverse=True)

    def nm(c): return f"{c} {names.get(c, c)[:16]}"
    print(f"{'rank':<6}{'pair':<44}{'rate':>8}   per-direction")
    for k, (r, a, b, ab, ba) in enumerate(pairs[:args.top], 1):
        print(f"{k:<6}{nm(a) + '  <->  ' + nm(b):<44}{r * 100:>7.1f}%   "
              f"{a}->{b} {ab:.0f}, {b}->{a} {ba:.0f}")

    print("\nper-class total off-diagonal rate (how confusable each class is overall):")
    for i in np.argsort(-(agg.sum(axis=1) - np.diag(agg)) / np.maximum(row_tot, 1)):
        rate = (row_tot[i] - agg[i, i]) / max(row_tot[i], 1)
        print(f"  {nm(labels[i]):<26}{rate * 100:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
