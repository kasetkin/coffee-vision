"""Compare archived experiments: aggregate deltas + the per-class breakdown.

Phase 7's most useful findings were per-class, not aggregate ("Kenya-AA collapsed
to 0.667 at 1000px", "Brazil-MonteCristo jumped to 0.929 at 900px") — the
aggregate macro-F1 moved for reasons that only the per-class view explained. This
prints both, against the patch_crop_size=900 noise band measured in Phase 7, so
a delta can be read against the bar it actually has to clear.

    python -m coffeecv.compare_experiments 36 --vs 37
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffeecv.config import REPO_ROOT

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# 3-seed spreads on the adopted patch_crop_size=900 config (EXPERIMENTS_LOG.md,
# "Noise band on patch_crop_size=900"). The pre-900 band is tighter and no longer
# applies. Anything inside these is indistinguishable from a reseed.
NOISE_BAND = {
    "val_macro_f1": 0.0144,
    "val_mcc": 0.0156,
    "test_macro_f1": 0.0479,
    "test_mcc": 0.0510,
}


def _find(exp_id: str) -> Path:
    matches = sorted(EXPERIMENTS_DIR.glob(f"exp{exp_id}__*"))
    if not matches:
        raise SystemExit(f"No archived experiment {exp_id!r} in {EXPERIMENTS_DIR}")
    return matches[0]


def _load(exp_id: str) -> tuple[dict, dict, dict]:
    d = _find(exp_id)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return json.loads((d / "metrics.json").read_text()), json.loads((d / "config.json").read_text()), meta


def _verdict(delta: float, band: float) -> str:
    if abs(delta) < band / 2:
        return "flat"
    if abs(delta) < band:
        return "inside noise"
    return "EXCEEDS BAND" + (" (better)" if delta > 0 else " (worse)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp", help="experiment id to evaluate, e.g. 36")
    p.add_argument("--vs", required=True, help="baseline experiment id to compare against")
    args = p.parse_args()

    m_new, c_new, meta_new = _load(args.exp)
    m_base, c_base, meta_base = _load(args.vs)

    changed = [
        f"{k}: {c_base[k]} -> {c_new[k]}"
        for k in sorted(c_new)
        if k != "env" and k in c_base and c_new[k] != c_base[k]
    ]
    print(f"exp {args.exp} ({meta_new.get('note','')})")
    print(f"  vs exp {args.vs} ({meta_base.get('note','')})")
    print(f"  changed: {'; '.join(changed) or '(nothing — identical config)'}\n")

    print(f"{'metric':<16} {'baseline':>9} {'new':>9} {'delta':>9} {'band':>8}  verdict")
    for split in ("val", "test"):
        for metric in ("macro_f1", "mcc"):
            key = f"{split}_{metric}"
            b = m_base["splits"][split][metric]
            n = m_new["splits"][split][metric]
            band = NOISE_BAND[key]
            print(f"{key:<16} {b:>9.4f} {n:>9.4f} {n - b:>+9.4f} {band:>8.4f}  {_verdict(n - b, band)}")
    print(f"{'best_epoch':<16} {m_base['best_epoch']:>9} {m_new['best_epoch']:>9}")
    print(f"{'epochs_trained':<16} {m_base['epochs_trained']:>9} {m_new['epochs_trained']:>9}")

    for split in ("val", "test"):
        print(f"\nper-class {split} f1:")
        base_pc, new_pc = m_base["splits"][split]["per_class"], m_new["splits"][split]["per_class"]
        rows = [
            (new_pc[cid]["label"], base_pc[cid]["f1"], new_pc[cid]["f1"], new_pc[cid]["f1"] - base_pc[cid]["f1"])
            for cid in new_pc if cid in base_pc
        ]
        for label, b, n, d in sorted(rows, key=lambda r: r[3]):
            bar = "#" * int(abs(d) * 100)
            print(f"  {label:<28} {b:.3f} -> {n:.3f}  {d:>+7.3f} {bar}")


if __name__ == "__main__":
    main()
