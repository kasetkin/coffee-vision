"""ML-37 bits for the classifier, level 4 (heads): refit the head per arm and seed, judged by D13's proposed bar.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/bits_cls_refit_eval.py [--ref f16]

Reads bits_cls_refit.py's features (every tmp/ml37/bits_cls/refit/<arms>/ folder; an arm is used if every photo has
it). Per seed and arm, the head is fitted as fit_frozen_head fits its shipping head, at the fixed C = 0.1
(fit_frozen_head.DEFAULT_C) by linear_head.fit_head_at (StandardScaler, L2 LogisticRegression, lbfgs), on the
unaugmented training patches (the augmented views are not drawn: see the note's §5), and scored on the seed's test
patches (patch macro-F1, metrics.compute_split_metrics' F1). The arms share the photos, split and boxes at a seed, so
each arm is compared with the reference arm (--ref, default f16) seed by seed, and the paired deltas are judged with
research_d13_adoption_bar.md's proposal: the mean paired change in test macro-F1 over the seeds, its 95% lower bound
from a photo bootstrap (test photos resampled within class, the same draw for both arms, 4000 draws a seed, as
d13_seed_range.py section C) above -0.010; no class's mean paired F1 change below -0.02. Also: each arm's macro-F1 and
per-class F1 over the seeds (range), the test patches and photos (mean of patch probabilities) whose answer differs
between the arms, and the heads' weight difference. Writes analysis/ml37/bits_cls_refit_eval.txt (not for a --root other than the default).
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.ml37 import bits_cls_common as bc
from coffeecv.class_list import load_classes
from coffeecv.config import REPO_ROOT
from coffeecv.fit_frozen_head import DEFAULT_C
from coffeecv.linear_head import fit_head_at, predict

OUT = Path(__file__).with_suffix(".txt")
B_BOOT = 4000
RNG = np.random.default_rng(20261123)
ORDER = ("f32", "f16", "u12", "u10", "u8", "today")


def f1s(y: np.ndarray, p: np.ndarray, k: int, w: np.ndarray | None = None) -> np.ndarray:
    """Per-class F1, rows weighted by w (d13_seed_range.py's f1s)."""
    if w is None:
        w = np.ones_like(y, dtype=float)
    tp = np.bincount(y[y == p], weights=w[y == p], minlength=k)
    sup = np.bincount(y, weights=w, minlength=k)
    pred = np.bincount(p, weights=w, minlength=k)
    denom = sup + pred
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, 0.0)


def load(root: Path) -> tuple[dict, list[str], int]:
    """{arm: {seed: {split: (X, y, photo)}}} over the photos every arm folder has."""
    dirs = [d for d in sorted(root.glob("*")) if d.is_dir()]
    names = None
    for d in dirs:
        s = {f.name for f in d.glob("*.npz")}
        names = s if names is None else names & s
    names = sorted(names or [])
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], [], []))))
    arms = []
    rows0 = {}
    for d in dirs:
        for name in names:
            z = np.load(d / name)
            rows = z["rows"]
            if name in rows0:
                assert np.array_equal(rows0[name], rows), f"{d.name}/{name}: boxes differ between arm folders"
            rows0[name] = rows
            for arm in z.files:
                if arm in ("rows", "pitch"):
                    continue
                if arm not in arms:
                    arms.append(arm)
                for seed in np.unique(rows[:, 0]) if len(rows) else []:
                    for si, split in enumerate(("train", "test")):
                        m = (rows[:, 0] == seed) & (rows[:, 1] == si)
                        if m.any():
                            X, Y, P = acc[arm][int(seed)][split]
                            X.append(z[arm][m])
                            Y.append(rows[m, 2])
                            P += [name] * int(m.sum())
    out = {arm: {seed: {split: (np.concatenate(v[0]), np.concatenate(v[1]), np.array(v[2]))
                        for split, v in sd.items()} for seed, sd in ad.items()} for arm, ad in acc.items()}
    return out, [a for a in ORDER if a in arms], len(names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="f16")
    ap.add_argument("--root", type=Path, default=bc.SCRATCH / "refit")
    a = ap.parse_args()
    data, arms, n_photos = load(a.root)
    keys = list(load_classes(REPO_ROOT / "dataset" / "classes.txt").keys)
    K = len(keys)
    seeds = sorted(data[arms[0]], key=lambda s: [42, 123, 7].index(s))
    L = [f"{n_photos} pool photos read; arms {arms}; seeds {seeds}; reference arm {a.ref}; heads at C = {DEFAULT_C}, "
         "unaugmented, fit_head_at"]
    for s in seeds:
        tr, te = data[arms[0]][s]["train"], data[arms[0]][s]["test"]
        L.append(f"  seed {s}: {len(tr[1])} training patches of {len(set(tr[2]))} photos, {len(te[1])} test patches "
                 f"of {len(set(te[2]))} photos")
    res = defaultdict(dict)
    for s in seeds:
        for arm in arms:
            t0 = time.time()
            Xtr, ytr, _ = data[arm][s]["train"]
            Xte, yte, pte = data[arm][s]["test"]
            lin, conv = fit_head_at(Xtr, ytr, DEFAULT_C, K)
            pred, prob, _ = predict(lin, Xte)
            res[s][arm] = dict(pred=pred, prob=prob, y=yte, photo=pte, conv=conv,
                               W=lin.weight.detach().numpy().copy(), f1=f1s(yte, pred, K))
            print(f"seed {s} arm {arm}: macro-F1 {res[s][arm]['f1'].mean():.4f}, converged {conv}, "
                  f"{time.time() - t0:.0f} s", flush=True)
    L += ["", "Test macro-F1 (patch) per arm and seed, and the range over seeds:"]
    for arm in arms:
        v = [res[s][arm]["f1"].mean() for s in seeds]
        L.append(f"  {arm:6s} " + " ".join(f"{x:.4f}" for x in v) + f"   range {min(v):.4f}-{max(v):.4f} "
                 f"({max(v) - min(v):.4f}), mean {np.mean(v):.4f}"
                 + ("" if all(res[s][arm]["conv"] for s in seeds) else "  (a fit did not converge)"))
    L += ["", "Per-class test F1, min-max over the seeds:"]
    L.append("  " + " " * 10 + " ".join(f"{arm:>13s}" for arm in arms))
    for ci, k in enumerate(keys):
        L.append(f"  {k:10s}" + " ".join(f"  {min(res[s][arm]['f1'][ci] for s in seeds):.3f}-"
                                         f"{max(res[s][arm]['f1'][ci] for s in seeds):.3f}" for arm in arms))
    ref = a.ref
    for arm in arms:
        if arm == ref:
            continue
        L += ["", f"== {arm} against {ref}, paired (same photos, split and boxes at each seed) =="]
        obs_pc, boot_macro, boot_pc = [], np.zeros((B_BOOT, len(seeds))), np.zeros((B_BOOT, len(seeds), K))
        for si, s in enumerate(seeds):
            r0, r1 = res[s][ref], res[s][arm]
            assert np.array_equal(r0["y"], r1["y"]) and np.array_equal(r0["photo"], r1["photo"])
            y = r0["y"]
            names = list(dict.fromkeys(r0["photo"]))
            pid = {n: i for i, n in enumerate(names)}
            ph = np.array([pid[n] for n in r0["photo"]])
            ph_class = np.zeros(len(names), int)
            ph_class[ph] = y
            by_class = [np.flatnonzero(ph_class == c) for c in range(K)]
            for bi in range(B_BOOT):
                mult = np.zeros(len(names))
                for idx in by_class:
                    np.add.at(mult, RNG.choice(idx, size=len(idx), replace=True), 1)
                w = mult[ph]
                d = f1s(y, r1["pred"], K, w) - f1s(y, r0["pred"], K, w)
                boot_pc[bi, si], boot_macro[bi, si] = d, d.mean()
            d = r1["f1"] - r0["f1"]
            obs_pc.append(d)
            dis = int((r1["pred"] != r0["pred"]).sum())
            pa0 = np.array([r0["prob"][ph == i].mean(0).argmax() for i in range(len(names))])
            pa1 = np.array([r1["prob"][ph == i].mean(0).argmax() for i in range(len(names))])
            ok0 = int((pa0 == ph_class).sum())
            ok1 = int((pa1 == ph_class).sum())
            dp = np.abs(r1["prob"] - r0["prob"]).max(1)
            wrel = np.linalg.norm(r1["W"] - r0["W"]) / np.linalg.norm(r0["W"])
            L.append(f"  seed {s:>3}: macro-F1 {r0['f1'].mean():.4f} -> {r1['f1'].mean():.4f} ({d.mean():+.4f}); "
                     f"patches with another answer {dis}/{len(y)}; photos with another answer "
                     f"{int((pa0 != pa1).sum())}/{len(names)} (right {ok0} -> {ok1}); max |dp| median "
                     f"{np.median(dp):.1e}, max {dp.max():.3f}; head weights rel. diff {wrel:.2e}")
        obs_pc = np.array(obs_pc)
        mean_boot = boot_macro.mean(1)
        lo, hi = np.quantile(mean_boot, [0.025, 0.975])
        verdict = "MET" if lo > -0.010 else "NOT MET"
        L.append(f"  mean paired change in test macro-F1 {obs_pc.mean(1).mean():+.4f}, photo-bootstrap 95% CI "
                 f"[{lo:+.4f}, {hi:+.4f}], SE {mean_boot.std():.4f}: lower bound above -0.010: {verdict}")
        pcm = boot_pc.mean(1)
        worst = []
        L.append("  per class, mean paired F1 change [bootstrap 95% CI] (flag below -0.02):")
        for ci, k in enumerate(keys):
            m = obs_pc[:, ci].mean()
            lo_c, hi_c = np.quantile(pcm[:, ci], [0.025, 0.975])
            worst.append(m)
            L.append(f"    {k:10s} {m:+.4f} [{lo_c:+.4f}, {hi_c:+.4f}]" + ("  FLAG" if m < -0.02 else ""))
        L.append(f"  lowest class change {min(worst):+.4f} ({keys[int(np.argmin(worst))]}); per-class bar: "
                 f"{'MET' if min(worst) >= -0.02 else 'NOT MET'}")
    L.append(f"peak RSS {bc.rss_mb():.0f} MB")
    if a.root == bc.SCRATCH / "refit":
        OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
