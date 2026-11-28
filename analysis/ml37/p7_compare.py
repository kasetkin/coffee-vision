"""ML-37 P7: the refitted head against the live family, judged by D13's bar (analysis/ml37/p7_bar.md).

Reads archived runs in experiments/ only; trains nothing. Pairs the candidate runs with the baseline runs by seed
(ADR 0002) and writes analysis/ml37/p7_compare_<label>.txt:

  1. ADR 0002's printout (coffeecv.compare_experiments' paired table: both arms' val and test macro-F1 per seed,
     the deltas, their mean and range, per-class F1).
  2. The test photos at each seed: shared by both arms, or one arm's only (the pool grew after exp289-291).
  3. D13 part 1: the mean paired change in test patch macro-F1, with a photo-level bootstrap. Test photos (a
     photo is its (true_label, folder_id, photo) triple in predictions_test.csv) are resampled with replacement
     within each class; a photo both arms have gets one draw for both, each arm's own photos are drawn apart;
     each seed independently. A drawn photo's patches enter with its multiplicity as their weight, and every
     class's F1 and the macro-F1 are recomputed. The bound is the 2.5th percentile of the mean change.
  4. D13 part 2: each class's mean paired test F1 change, its bootstrap interval beside it.
  5. D13 part 3: the OOD probe refitted on the seed-123 candidate, read from ood_eval's --out JSON
     (--ood-eval); without it the part is printed as pending, with the commands that produce it.
  6. Beside the bar, not deciding it: val's same statistic, photo-level answers on shared photos, the share of
     patches whose prediction changed.
  7. The verdict lines, and D21's branch on a miss.

    cd /workspace && uv run python analysis/ml37/p7_compare.py --base 289 290 291 --new 289 290 291 --label baseline_self
    cd /workspace && uv run python analysis/ml37/p7_compare.py --base 289 290 291 --new 292 293 294 --label exp292_294 \\
        --ood-eval analysis/ml37/p7_ood_eval.json

Tests: tests/test_p7_compare.py.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coffeecv import compare_experiments  # noqa: E402

EXP = ROOT / "experiments"
OUT_DIR = Path(__file__).resolve().parent

# D13 (ML-37 ticket, answered (a) 2026-11-26), fixed before the runs: analysis/ml37/p7_bar.md.
OVERALL_MARGIN = -0.010      # the 95% photo-bootstrap lower bound must be above this
CLASS_MARGIN = -0.02         # no class's mean paired F1 change below this
OOD_MAX_REFUSED = 4          # bean photos refused, at most
OOD_N_BEANS = 295            # the segmenter test split's bean photos the bar is worded on
OOD_SEED = 123               # the probe is refitted on the seed-123 candidate

N_BOOT = 2000
RNG_SEED = 20261127

D21 = ("D21 (a), on a miss: the head is refitted with the 8-bit resizes (D11 (a)), the rest of the change kept, same "
       "seeds, and judged by this same bar; passing adopts that, with D11 recorded as reversed. A second miss "
       "closes the ticket unadopted (P8's second branch): the branch is never merged, and the reasons go into "
       ".out-of-scope/.")


@dataclass
class Preds:
    """One run's patch predictions on one split."""
    classes: list[str]
    y: np.ndarray            # true class index per patch
    p: np.ndarray            # predicted class index per patch
    prob: np.ndarray         # (patches, classes) probabilities
    photos: list[tuple]      # photo key per patch


@dataclass
class Pair:
    """One seed's two arms on one split, photos indexed over their union."""
    seed: int
    k: int
    yb: np.ndarray
    pb: np.ndarray
    phb: np.ndarray          # union photo index of each baseline patch
    yn: np.ndarray
    pn: np.ndarray
    phn: np.ndarray
    strata: list[tuple[np.ndarray, np.ndarray, np.ndarray]]   # per class: (shared, base only, new only) photo ids
    n_photos: int


def photo_key(row: dict) -> tuple:
    """A patch's photo: (true_label, folder_id, photo), as analysis/ml37/d13_seed_range.py keys it."""
    return (row["true_label"], row["folder_id"], row["photo"])


def exp_dir(e: int) -> Path:
    matches = sorted(EXP.glob(f"exp{e}__*"))
    if not matches:
        raise SystemExit(f"no archived experiment exp{e} in {EXP}")
    return matches[0]


def load_preds(e: int, split: str) -> Preds:
    d = exp_dir(e)
    classes = json.loads((d / "metrics.json").read_text())["class_ids"]
    ci = {c: i for i, c in enumerate(classes)}
    with open(d / f"predictions_{split}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return Preds(classes=classes,
                 y=np.array([ci[r["true_label"]] for r in rows]),
                 p=np.array([ci[r["pred_label"]] for r in rows]),
                 prob=np.array([[float(r[f"p_{c}"]) for c in classes] for r in rows]),
                 photos=[photo_key(r) for r in rows])


def make_pair(seed: int, base: Preds, new: Preds) -> Pair:
    if base.classes != new.classes:
        raise SystemExit(f"seed {seed}: the arms' class lists differ: {base.classes} vs {new.classes}")
    k = len(base.classes)
    union = {key: i for i, key in enumerate(dict.fromkeys(base.photos + new.photos))}
    phb = np.array([union[key] for key in base.photos])
    phn = np.array([union[key] for key in new.photos])
    cls = np.full(len(union), -1)
    cls[phb], cls[phn] = base.y, new.y
    in_b = np.zeros(len(union), bool)
    in_n = np.zeros(len(union), bool)
    in_b[phb], in_n[phn] = True, True
    strata = [(np.flatnonzero((cls == c) & in_b & in_n), np.flatnonzero((cls == c) & in_b & ~in_n),
               np.flatnonzero((cls == c) & ~in_b & in_n)) for c in range(k)]
    return Pair(seed, k, base.y, base.p, phb, new.y, new.p, phn, strata, len(union))


def class_f1(y: np.ndarray, p: np.ndarray, k: int, w: np.ndarray | None = None) -> np.ndarray:
    """Per-class F1 under patch weights w, (patches,) or (draws, patches); F1 is 0 where a class has no support and
    no predictions."""
    if w is None:
        w = np.ones(len(y))
    eye = np.eye(k)
    hit = eye[y] * (y == p)[:, None]
    tp, sup, pred = w @ hit, w @ eye[y], w @ eye[p]
    denom = sup + pred
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


def _draw(rng: np.random.Generator, ids: np.ndarray, n_boot: int, n_photos: int) -> np.ndarray:
    """Multiplicities (n_boot, n_photos) of a with-replacement draw of len(ids) photos from ids."""
    m = np.zeros((n_boot, n_photos))
    if len(ids):
        picks = ids[rng.integers(0, len(ids), size=(n_boot, len(ids)))]
        np.add.at(m, (np.repeat(np.arange(n_boot), len(ids)), picks.ravel()), 1)
    return m


def bootstrap(pairs: list[Pair], n_boot: int = N_BOOT, rng_seed: int = RNG_SEED) -> np.ndarray:
    """Per-class F1 changes (new - base) under photo resampling: (n_boot, seeds, classes)."""
    rng = np.random.default_rng(rng_seed)
    out = np.zeros((n_boot, len(pairs), pairs[0].k))
    for s, pr in enumerate(pairs):
        mb = np.zeros((n_boot, pr.n_photos))
        mn = np.zeros((n_boot, pr.n_photos))
        for shared, base_only, new_only in pr.strata:
            ms = _draw(rng, shared, n_boot, pr.n_photos)
            mb += ms + _draw(rng, base_only, n_boot, pr.n_photos)
            mn += ms + _draw(rng, new_only, n_boot, pr.n_photos)
        out[:, s] = class_f1(pr.yn, pr.pn, pr.k, mn[:, pr.phn]) - class_f1(pr.yb, pr.pb, pr.k, mb[:, pr.phb])
    return out


def observed(pairs: list[Pair]) -> np.ndarray:
    """Per-class F1 changes (new - base): (seeds, classes)."""
    return np.array([class_f1(pr.yn, pr.pn, pr.k) - class_f1(pr.yb, pr.pb, pr.k) for pr in pairs])


@dataclass
class Summary:
    mean: float              # mean over seeds of the macro-F1 change
    per_seed: np.ndarray
    lo: float                # 2.5th percentile of the bootstrapped mean change
    hi: float
    se: float
    class_mean: np.ndarray   # per class, mean over seeds of the F1 change
    class_lo: np.ndarray
    class_hi: np.ndarray


def summarize(pairs: list[Pair], n_boot: int = N_BOOT, rng_seed: int = RNG_SEED) -> Summary:
    obs = observed(pairs)
    boot = bootstrap(pairs, n_boot, rng_seed)
    mean_boot = boot.mean(axis=2).mean(axis=1)          # macro over classes, then mean over seeds
    class_boot = boot.mean(axis=1)
    lo, hi = np.quantile(mean_boot, [0.025, 0.975])
    return Summary(mean=float(obs.mean(axis=1).mean()), per_seed=obs.mean(axis=1), lo=float(lo), hi=float(hi),
                   se=float(mean_boot.std()), class_mean=obs.mean(axis=0),
                   class_lo=np.quantile(class_boot, 0.025, axis=0), class_hi=np.quantile(class_boot, 0.975, axis=0))


def ood_counts(report: dict) -> dict:
    """ood_eval's --out JSON -> negatives caught and bean photos refused (unmeasurable counted, as ood_eval does)."""
    neg = report["negatives"]["all"]
    pos = report["positives"]["all"]
    return {"neg_n": neg["n"], "neg_caught": neg["caught"],
            "bean_n": pos["trained"]["n"] + pos["not_trained"]["n"],
            "bean_refused": pos["trained"]["refused"] + pos["not_trained"]["refused"],
            "checkpoint": report.get("checkpoint"), "probe": report.get("ood_probe")}


def ood_met(c: dict) -> bool:
    return c["neg_caught"] == c["neg_n"] and c["bean_refused"] <= OOD_MAX_REFUSED


def photo_answers(pr_base: Preds, pr_new: Preds) -> tuple[int, int, int]:
    """On photos both arms have, the photo's answer (argmax of its patches' mean probability, as serving averages
    its patches): (shared photos, right -> wrong, wrong -> right)."""
    def answers(pr: Preds) -> dict:
        rows: dict = {}
        for i, key in enumerate(pr.photos):
            rows.setdefault(key, []).append(i)
        return {key: (int(pr.prob[ix].mean(axis=0).argmax()), int(pr.y[ix[0]])) for key, ix in rows.items()}
    a, b = answers(pr_base), answers(pr_new)
    shared = a.keys() & b.keys()
    rw = sum(a[key][0] == a[key][1] and b[key][0] != b[key][1] for key in shared)
    wr = sum(a[key][0] != a[key][1] and b[key][0] == b[key][1] for key in shared)
    return len(shared), rw, wr


def patch_changes(pr_base: Preds, pr_new: Preds) -> tuple[int, int]:
    """Patches whose prediction changed, over the classes whose test patches are the same photos in the same order
    in both arms (there the boxes are the same draw): (patches compared, changed)."""
    n = changed = 0
    for c in range(len(pr_base.classes)):
        ib, inew = np.flatnonzero(pr_base.y == c), np.flatnonzero(pr_new.y == c)
        if [pr_base.photos[i] for i in ib] != [pr_new.photos[i] for i in inew]:
            continue
        n += len(ib)
        changed += int((pr_base.p[ib] != pr_new.p[inew]).sum())
    return n, changed


def seed_of(e: int) -> int:
    return json.loads((exp_dir(e) / "config.json").read_text())["seed"]


def check_against_metrics(e: int, split: str, pr: Preds, out) -> None:
    """The recomputed F1 must be the archived metrics.json's, or the statistic is not the recorded one."""
    m = json.loads((exp_dir(e) / "metrics.json").read_text())["splits"][split]
    f = class_f1(pr.y, pr.p, len(pr.classes))
    rec = np.array([m["per_class"][c]["f1"] for c in pr.classes])
    if abs(f.mean() - m["macro_f1"]) > 1e-6 or np.abs(f - rec).max() > 1e-6:
        raise SystemExit(f"exp{e} {split}: F1 recomputed from predictions ({f.mean():.6f}) is not metrics.json's "
                         f"({m['macro_f1']:.6f})")
    out(f"  exp{e} {split}: macro-F1 recomputed from predictions_{split}.csv {f.mean():.4f} = metrics.json's")


def run(base: list[int], new: list[int], label: str, ood_eval: Path | None, n_boot: int, rng_seed: int) -> str:
    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)

    if len(base) != len(new):
        raise SystemExit("pair the runs one to one: as many --new as --base")
    seeds = [seed_of(b) for b in base]
    if seeds != [seed_of(n) for n in new]:
        raise SystemExit(f"seeds differ: base {seeds}, new {[seed_of(n) for n in new]}")

    out(f"ML-37 P7 ({label}): exp {' '.join(map(str, new))} against exp {' '.join(map(str, base))}, paired by seed "
        f"({', '.join(map(str, seeds))}), judged by D13's bar (analysis/ml37/p7_bar.md).")
    out("Generated by analysis/ml37/p7_compare.py from experiments/ (read only). Metrics are patch macro-F1, as "
        "recorded.")
    out()
    out("1. ADR 0002's printout (coffeecv.compare_experiments, paired):")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compare_experiments.paired([str(n) for n in new], [str(b) for b in base])
    lines.extend("  " + s if s else "" for s in buf.getvalue().rstrip("\n").split("\n"))
    out()

    preds = {split: [(load_preds(b, split), load_preds(n, split)) for b, n in zip(base, new)]
             for split in ("test", "val")}
    out("2. The recomputed statistic against the record, and the test photos at each seed")
    for split in ("test", "val"):
        for (pb, pn), b, n in zip(preds[split], base, new):
            check_against_metrics(b, split, pb, out)
            if n != b:
                check_against_metrics(n, split, pn, out)
    pairs = {split: [make_pair(s, pb, pn) for s, (pb, pn) in zip(seeds, preds[split])] for split in ("test", "val")}
    classes = preds["test"][0][0].classes
    unshared_any = False
    for pr in pairs["test"]:
        sh = sum(len(s) for s, _, _ in pr.strata)
        bo = sum(len(x) for _, x, _ in pr.strata)
        no = sum(len(x) for _, _, x in pr.strata)
        out(f"  seed {pr.seed:>3}: test photos, baseline {sh + bo}, candidate {sh + no}; shared {sh}, baseline's only "
            f"{bo}, candidate's only {no}")
        diff = [f"{classes[c]} {len(s)} shared / {len(x)} / {len(y)}" for c, (s, x, y) in enumerate(pr.strata)
                if len(x) or len(y)]
        if diff:
            unshared_any = True
            out("    classes whose test photos differ (shared / baseline's only / candidate's only): " + ", ".join(diff))
    if unshared_any:
        out("  NOTE: the arms do not see the same test photos at every seed (the pool grew after the baseline ran;")
        out("  research_d13_adoption_bar.md §2, §10). Shared photos are drawn once for both arms, each arm's own")
        out("  photos apart, so the interval is wider than a fully paired one, and in the classes listed the change")
        out("  mixes the input with a different test set. The bar is read on it as printed; the owner decides.")
    out()

    t = summarize(pairs["test"], n_boot, rng_seed)
    v = summarize(pairs["val"], n_boot, rng_seed + 1)
    out(f"3. D13 part 1: mean paired change in test macro-F1, photo bootstrap ({n_boot} resamples, RNG seed {rng_seed})")
    for pr, d in zip(pairs["test"], t.per_seed):
        out(f"  seed {pr.seed:>3}: {d:+.4f}")
    out(f"  mean {t.mean:+.4f}; bootstrap SE {t.se:.4f}; 95% interval [{t.lo:+.4f}, {t.hi:+.4f}]")
    part1 = t.lo > OVERALL_MARGIN
    out(f"  lower bound {t.lo:+.4f} {'>' if part1 else '<='} {OVERALL_MARGIN:+.3f}: {'MET' if part1 else 'MISSED'}")
    out()

    out("4. D13 part 2: each class's mean paired test F1 change over the seeds, worst first (bootstrap 95% interval "
        "beside it, not deciding)")
    for c in np.argsort(t.class_mean, kind="stable"):
        flag = "  BELOW the margin" if t.class_mean[c] < CLASS_MARGIN else ""
        out(f"  {classes[c]:12s} {t.class_mean[c]:+.4f}  [{t.class_lo[c]:+.4f}, {t.class_hi[c]:+.4f}]{flag}")
    worst = int(np.argmin(t.class_mean))
    part2 = bool(t.class_mean[worst] >= CLASS_MARGIN)
    out(f"  worst class {classes[worst]} {t.class_mean[worst]:+.4f} {'>=' if part2 else '<'} {CLASS_MARGIN:+.2f}: "
        f"{'MET' if part2 else 'MISSED'}")
    out()

    out(f"5. D13 part 3: the OOD probe refitted on the seed-{OOD_SEED} candidate")
    part3: bool | None = None
    if ood_eval is None:
        cand = new[seeds.index(OOD_SEED)] if OOD_SEED in seeds else new[0]
        cfg = exp_dir(cand) / "config.json"
        ckpt = f"{json.loads(cfg.read_text()).get('dino', {}).get('run_dir', '<run_dir>')}/model.pt"
        cfg = cfg.relative_to(ROOT)
        out("  PENDING: no --ood-eval given. Fill it in by refitting the probe on the candidate and measuring it, e.g.")
        out(f"  (exp{cand}'s head, while its run dir still holds it; or ship it and pass models/<name>.pt, as ML-29 P11)")
        out(f"    python -m coffeecv.fit_ood_probe --checkpoint {ckpt} --config {cfg} --out <probe.json>")
        out(f"    python -m coffeecv.ood_eval --checkpoint {ckpt} --config {cfg} --probe <probe.json> "
            "--out analysis/ml37/p7_ood_eval.json")
        out("  then rerun this script with --ood-eval analysis/ml37/p7_ood_eval.json. The check: negatives caught ==")
        out(f"  negatives (every one), and bean photos refused <= {OOD_MAX_REFUSED} of {OOD_N_BEANS}, at threshold 0.5.")
    else:
        c = ood_counts(json.loads(ood_eval.read_text()))
        part3 = ood_met(c)
        out(f"  read from {ood_eval}: checkpoint {c['checkpoint']}, probe {c['probe']}")
        out(f"  negatives caught {c['neg_caught']}/{c['neg_n']} (every one: "
            f"{'yes' if c['neg_caught'] == c['neg_n'] else 'NO'})")
        out(f"  bean photos refused {c['bean_refused']}/{c['bean_n']} (at most {OOD_MAX_REFUSED}: "
            f"{'yes' if c['bean_refused'] <= OOD_MAX_REFUSED else 'NO'})")
        if c["bean_n"] != OOD_N_BEANS:
            out(f"  NOTE: the bar is worded on {OOD_N_BEANS} bean photos; this split has {c['bean_n']}.")
        out(f"  {'MET' if part3 else 'MISSED'}")
    out()

    out("6. Beside the bar, not deciding it")
    out(f"  val, the same statistic: mean {v.mean:+.4f} (per seed " + ", ".join(f"{d:+.4f}" for d in v.per_seed)
        + f"), 95% interval [{v.lo:+.4f}, {v.hi:+.4f}]")
    for (pb, pn), s in zip(preds["test"], seeds):
        n_sh, rw, wr = photo_answers(pb, pn)
        n_cmp, ch = patch_changes(pb, pn)
        share = f"{ch}/{n_cmp} ({ch / n_cmp:.1%})" if n_cmp else "none comparable"
        out(f"  seed {s:>3} test: photo answers on {n_sh} shared photos, right->wrong {rw}, wrong->right {wr}; "
            f"patch predictions changed {share} (classes with the same photos in the same order)")
    out()

    out("7. Verdict (D13; the owner decides, ADR 0002)")
    out(f"  part 1, overall:   {'MET' if part1 else 'MISSED'} (lower bound {t.lo:+.4f}, margin {OVERALL_MARGIN:+.3f})")
    out(f"  part 2, per class: {'MET' if part2 else 'MISSED'} (worst {classes[worst]} {t.class_mean[worst]:+.4f}, "
        f"margin {CLASS_MARGIN:+.2f})")
    out(f"  part 3, OOD probe: {'PENDING' if part3 is None else 'MET' if part3 else 'MISSED'}")
    if not (part1 and part2) or part3 is False:
        out("  D13's bar: MISSED. " + D21)
    elif part3 is None:
        out("  D13's bar: the in-distribution parts MET; the OOD probe's part pending.")
    else:
        out("  D13's bar: MET.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=int, nargs="+", required=True, help="baseline experiment numbers")
    ap.add_argument("--new", type=int, nargs="+", required=True, help="candidate experiment numbers, same seeds")
    ap.add_argument("--label", required=True, help="the output's name: p7_compare_<label>.txt")
    ap.add_argument("--ood-eval", type=Path, default=None, help="ood_eval's --out JSON for the seed-123 candidate")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED)
    args = ap.parse_args()
    text = run(args.base, args.new, args.label, args.ood_eval, args.boot, args.rng_seed)
    dst = OUT_DIR / f"p7_compare_{args.label}.txt"
    dst.write_text(text)
    print(text, end="")
    print(f"Wrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
