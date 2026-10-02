"""Compare archived experiments: aggregate deltas + the per-class breakdown.

Phase 7's most useful findings were per-class, not aggregate ("Kenya-AA collapsed
to 0.667 at 1000px", "Brazil-MonteCristo jumped to 0.929 at 900px") — the
aggregate macro-F1 moved for reasons that only the per-class view explained. This
prints both, against the patch_crop_size=900 noise band measured in Phase 7, so
a delta can be read against the bar it actually has to clear.

    python -m coffeecv.compare_experiments 36 --vs 37

Several ids on each side compare seed against seed (ticket ML-2 D24): per seed, both arms' val and test
macro-F1 and the paired delta (new - baseline), the deltas' mean and range, and per-class F1 side by side.
There is no verdict; the owner reads the table and decides.

    python -m coffeecv.compare_experiments 258 259 260 --vs 255 256 257
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


def _changed(c_base: dict, c_new: dict, skip: tuple[str, ...] = ("env",)) -> list[str]:
    return [f"{k}: {c_base[k]} -> {c_new[k]}" for k in sorted(c_new)
            if k not in skip and k in c_base and c_new[k] != c_base[k]]


def paired(new_ids: list[str], base_ids: list[str]) -> None:
    """One row per seed, new arm against baseline arm (ticket ML-2 D24). Prints, never judges."""
    if len(new_ids) != len(base_ids):
        raise SystemExit(f"{len(new_ids)} new runs but {len(base_ids)} baselines: pair them one to one")
    runs = [(_load(n), _load(b), n, b) for n, b in zip(new_ids, base_ids)]
    for (_, c_new, _), (_, c_base, _), n, b in runs:
        if c_new["seed"] != c_base["seed"]:
            raise SystemExit(f"exp {n} (seed {c_new['seed']}) is paired with exp {b} (seed {c_base['seed']})")
    # What differs between the arms, other than the seed-free provenance (env, dino.run_dir, ...).
    (_, c_new0, meta_new0), (_, c_base0, meta_base0), _, _ = runs[0]
    print(f"new:      exp {' '.join(new_ids)} ({meta_new0.get('note', '')})")
    print(f"baseline: exp {' '.join(base_ids)} ({meta_base0.get('note', '')})")
    print(f"changed:  {'; '.join(_changed(c_base0, c_new0, skip=('env', 'dino'))) or '(nothing)'}\n")

    print(f"{'seed':>5} {'exp':>9}  {'val base':>8} {'val new':>8} {'val Δ':>8}  "
          f"{'test base':>9} {'test new':>9} {'test Δ':>8}")
    deltas = {"val": [], "test": []}
    for (m_new, c_new, _), (m_base, _, _), n, b in runs:
        row = f"{c_new['seed']:>5} {n + '/' + b:>9}"
        for split in ("val", "test"):
            bv, nv = m_base["splits"][split]["macro_f1"], m_new["splits"][split]["macro_f1"]
            deltas[split].append(nv - bv)
            w = 8 if split == "val" else 9
            row += f"  {bv:>{w}.4f} {nv:>{w}.4f} {nv - bv:>+8.4f}"
        print(row)
    for split in ("val", "test"):
        d = deltas[split]
        print(f"{split:>5} Δ: mean {sum(d) / len(d):+.4f}, range {min(d):+.4f} .. {max(d):+.4f} "
              f"(spread {max(d) - min(d):.4f})")

    seeds = [c_new["seed"] for (_, c_new, _), *_ in runs]
    for split in ("val", "test"):
        print(f"\nper-class {split} F1, baseline -> new per seed, sorted by mean Δ:")
        print(f"  {'class':<28}" + "".join(f" {'s' + str(s):>14}" for s in seeds) + f" {'mean Δ':>8}")
        rows = []
        for cid, info in runs[0][0][0]["splits"][split]["per_class"].items():
            pairs = [(m_base["splits"][split]["per_class"][cid]["f1"], m_new["splits"][split]["per_class"][cid]["f1"])
                     for (m_new, _, _), (m_base, _, _), _, _ in runs]
            mean_d = sum(nv - bv for bv, nv in pairs) / len(pairs)
            rows.append((info["label"], pairs, mean_d))
        for label, pairs, mean_d in sorted(rows, key=lambda r: r[2]):
            print(f"  {label:<28}" + "".join(f" {bv:.3f} -> {nv:.3f}" for bv, nv in pairs) + f" {mean_d:>+8.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp", nargs="+", help="experiment id(s) to evaluate, e.g. 36")
    p.add_argument("--vs", nargs="+", required=True, help="baseline experiment id(s), one per id above")
    args = p.parse_args()
    if len(args.exp) > 1 or len(args.vs) > 1:
        paired(args.exp, args.vs)
        return
    args.exp, args.vs = args.exp[0], args.vs[0]

    m_new, c_new, meta_new = _load(args.exp)
    m_base, c_base, meta_base = _load(args.vs)

    changed = _changed(c_base, c_new)
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
