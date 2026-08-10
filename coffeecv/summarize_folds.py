"""Summarise a leave-one-rig-out sweep: per-fold table, arm comparison, per-class detail.

The unit of evidence is the set of folds, not any single one. A change is only
interesting if it moves the mean cross-rig score across all three held-out rigs
without costing in-distribution accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

from coffeecv.config import REPO_ROOT as REPO
EXPS = REPO / "experiments"

CHANCE = 1 / 9  # 9 balanced classes


def load(exp_dir: Path) -> dict | None:
    m = exp_dir / "metrics.json"
    if not m.exists():
        return None
    metrics = json.loads(m.read_text())
    meta = json.loads((exp_dir / "meta.json").read_text()) if (exp_dir / "meta.json").exists() else {}
    if "test_xrig" not in metrics.get("splits", {}):
        return None
    cfg = json.loads((exp_dir / "config.json").read_text())
    return {
        "dir": exp_dir.name,
        "exp": meta.get("exp", ""),
        "note": meta.get("note", ""),
        "heldout": (metrics.get("rigs") or {}).get("heldout", "?"),
        "val": metrics["splits"]["val"]["macro_f1"],
        "test": metrics["splits"]["test"]["macro_f1"],
        "xrig": metrics["splits"]["test_xrig"]["macro_f1"],
        "xrig_mcc": metrics["splits"]["test_xrig"]["mcc"],
        "scale": (cfg.get("patch_scale_frac_min", 0), cfg.get("patch_scale_frac_max", 0)),
        "best_epoch": metrics.get("best_epoch"),
        "metrics": metrics,
    }


def short(rig: str) -> str:
    return rig.replace("2026-08-07__box_pictures_all_classes", "old_box") \
              .replace("2026-08-09__", "")


def main() -> None:
    rows = [r for r in (load(d) for d in sorted(EXPS.iterdir()) if d.is_dir()) if r]
    if not rows:
        print("no leave-one-rig-out runs archived yet")
        return

    arms: dict[str, list[dict]] = {}
    for r in rows:
        arm = "scale" if r["scale"][1] > 0 else "baseline"
        arms.setdefault(arm, []).append(r)

    print(f"{'arm':<9} {'held-out rig':<12} {'val':>7} {'test':>7} {'XRIG':>7} {'xrig_mcc':>9} {'ep':>4}")
    print("-" * 60)
    for arm in sorted(arms):
        for r in sorted(arms[arm], key=lambda r: r["heldout"]):
            print(f"{arm:<9} {short(r['heldout']):<12} {r['val']:7.4f} {r['test']:7.4f} "
                  f"{r['xrig']:7.4f} {r['xrig_mcc']:9.4f} {r['best_epoch']:4}")
        xs = [r["xrig"] for r in arms[arm]]
        ts = [r["test"] for r in arms[arm]]
        print(f"{'':<9} {'MEAN':<12} {sum(ts)/len(ts):15.4f} {sum(xs)/len(xs):7.4f}   "
              f"(chance {CHANCE:.3f})")
        print()

    if len(arms) == 2 and all(len(v) == 3 for v in arms.values()):
        print("PAIRED per-fold deltas (scale - baseline):")
        b = {r["heldout"]: r for r in arms["baseline"]}
        s = {r["heldout"]: r for r in arms["scale"]}
        dx, dt = [], []
        for rig in sorted(b):
            if rig not in s:
                continue
            d_x = s[rig]["xrig"] - b[rig]["xrig"]
            d_t = s[rig]["test"] - b[rig]["test"]
            dx.append(d_x); dt.append(d_t)
            print(f"  {short(rig):<12} xrig {d_x:+.4f}   in-dist test {d_t:+.4f}")
        if dx:
            print(f"  {'MEAN':<12} xrig {sum(dx)/len(dx):+.4f}   in-dist test {sum(dt)/len(dt):+.4f}")
            print(f"  all folds improved cross-rig: {all(d > 0 for d in dx)}")

    # Per-class cross-rig detail for the most recent run of each arm
    print("\nper-class cross-rig F1 (last fold of each arm):")
    for arm in sorted(arms):
        r = arms[arm][-1]
        per = r["metrics"]["splits"]["test_xrig"].get("per_class", {})
        labels = r["metrics"]["class_labels"]
        if not per:
            continue
        print(f"  {arm} (held out {short(r['heldout'])}):")
        for cid, st in sorted(per.items()):
            f1 = st.get("f1", st) if isinstance(st, dict) else st
            print(f"    {labels.get(cid, cid):<28} {f1:.3f}")


if __name__ == "__main__":
    main()
