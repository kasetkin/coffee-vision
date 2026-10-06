"""Fit the shipped OOD probe and its conformal refusal threshold.

`ood_eval.py` answers "which metric separates best", and answers it with
photo-level cross-validation: every photo is scored by a probe fitted without
it. That is the right way to *compare* methods and the wrong way to *ship* one,
because it produces five different probes and no single set of coefficients.

This module produces the one artifact a deployment needs: coefficients fitted
once on the dev split, plus a refusal threshold calibrated by split-conformal
prediction, written to `<checkpoint>.ood_probe.json` beside the weights.

Two rules this file exists to enforce:

1. **Nothing here ever reads the holdout split.** The threshold is calibrated on
   dev positives. Holdout is scored exactly once, afterwards, by `--verify`, to
   estimate what was shipped -- never to choose it. Re-running `--verify` and
   then adjusting anything would silently turn the holdout into a second dev set,
   which is the one failure this split exists to prevent.
2. **The probe's negatives come from `ood_eval.CLEAN_NEGATIVE_TAGS`**, imported
   rather than restated, so the probe that ships is fitted from exactly the
   photos the probe that was measured was fitted from.

    python -m coffeecv.fit_ood_probe --checkpoint models/allrigs_dino3b16_seg_country_s123.pt \
        --negatives dataset/ood_negatives/2026-09__internet_proxy \
                    dataset/ood_negatives/2026-09__user_realworld \
        --positives dataset/ood_positives dataset/ood_positives_internet
    python -m coffeecv.fit_ood_probe ... --verify   # spends the holdout, once
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from coffeecv.config import CHECKPOINTS_DIR, REPO_ROOT
from coffeecv.class_list import load_classes
from coffeecv.infer import (_sha, config_for_checkpoint, inference_tta_for, load_model, load_ood_reference,
                            probe_path_for, probe_score, reference_path_for)
from coffeecv.ood_eval import (CLEAN_NEGATIVE_TAGS, Unmeasurable, _fit_logistic, auroc,
                               embed_photo, id_photos, negatives_from)

# P(not beans) > P(beans): the threshold the live DINO probe serves at (threshold_override, 2026-09-27).
DECISION_BOUNDARY = 0.5


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float | None:
    """Split-conformal upper threshold: refuse above it, and at most `alpha` of
    genuine photos are refused.

    Returns None when `alpha` is tighter than the calibration set can certify.
    With n photos the tightest honest claim is 1/(n+1) -- 16 photos cannot
    support a 1% promise no matter how they score, and returning a number
    anyway would be the single-photo calibration mistake this whole exercise
    exists to retire.
    """
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return None
    return float(s[k - 1])


def _score_with(probe: dict, embeds: np.ndarray) -> float:
    """Calibrate against the exact function production scores with."""
    return probe_score(embeds, probe)


def _collect(items, cfg, model, head, n_patches, tta, cache_dir, ckpt_sha, label):
    """Embed a set of photos, returning (rows, n_declined).

    Photos `classify_one` cannot measure are counted, not silently dropped: they
    are already refused in production for a different reason, and folding them
    into the score statistics would flatter whichever metric is being fitted.
    """
    rows, declined = [], 0
    for item in items:
        path = item["path"]
        try:
            _, embeds, _, _ = embed_photo(path, cfg, model, head, n_patches,
                                          tta, cache_dir, ckpt_sha)
        except Unmeasurable:
            declined += 1
            continue
        shown = path.resolve()
        shown = shown.relative_to(REPO_ROOT) if shown.is_relative_to(REPO_ROOT) else shown
        rows.append({"photo": str(shown), "tag": item.get("scenario_tag", "-"), "date": item.get("date", ""),
                     "embeds": embeds, "label": label})
    return rows, declined


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "best.pt"))
    p.add_argument("--config", default=None)
    p.add_argument("--negatives", nargs="+", required=True)
    p.add_argument("--positives", nargs="+", required=True,
                   help="unseen genuine bean photos; calibration only, never fitted on")
    p.add_argument("--id-split", default="test", choices=("val", "test"),
                   help="which of the checkpoint's own photos supply bean patches")
    p.add_argument("--n-patches", type=int, default=40)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--cache-dir", default=".ood_eval_cache")
    p.add_argument("--alpha", type=float, default=None,
                   help="target false-refusal rate for the shipped threshold; defaults to the "
                        "tightest the calibration set can certify, 1/(n+1)")
    p.add_argument("--calibrate-on", default="user_independent",
                   choices=("all", "user", "user_independent"),
                   help="which positives set the threshold. Conformal validity needs the "
                        "calibration photos to be exchangeable with what the service actually "
                        "meets, so the default is the narrowest and most deployment-like set: the "
                        "user's own photos from days the model never trained on. 'user' adds "
                        "same-day photos (easier, so the certificate flatters itself) and 'all' "
                        "adds internet photos (a different framing population entirely).")
    p.add_argument("--verify", action="store_true",
                   help="after fitting, score the HOLDOUT split once against the frozen probe. "
                        "Spends the holdout: do not re-run and then change anything.")
    p.add_argument("--training-days", nargs="+", default=[], metavar="YYYY-MM-DD",
                   help="days the head's training photos were shot. Positives from those days flatter the probe "
                        "(ticket ML-3 P8), so calibration also reports the other days' positives alone and "
                        "--verify prints both groups apart, from the manifests' date column. Reporting only: "
                        "the shipped threshold does not change.")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    training_days = set(args.training_days)

    checkpoint = Path(args.checkpoint)
    cfg, cfg_source = config_for_checkpoint(checkpoint, args.config)
    print(f"config: {cfg_source}")
    _, classes_file = cfg.resolve_paths()
    classes = load_classes(classes_file)
    ckpt_sha = _sha(checkpoint)

    ref_path = reference_path_for(checkpoint)
    model, head = load_model(checkpoint, cfg.model_name, len(classes), cfg.dropout)
    # The one checked loader (sha and embedding width), like every other caller (plan §9.2).
    ref = load_ood_reference(checkpoint, head)
    if ref is None:
        sys.exit(f"No OOD reference at {ref_path}; build one with coffeecv.build_ood_reference")
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    tta = False if args.no_tta else inference_tta_for(checkpoint, cfg.model_name)
    neg_dirs = [Path(d) for d in args.negatives]
    pos_dirs = [Path(d) for d in args.positives]

    # --- fit set -----------------------------------------------------------
    train_ids = id_photos(cfg, classes, args.id_split)
    bean_items = [{"path": q, "scenario_tag": "-"} for q in train_ids]
    dev_negs = [r for r in negatives_from(neg_dirs, "dev")
                if r["scenario_tag"] in CLEAN_NEGATIVE_TAGS]
    dropped = len(negatives_from(neg_dirs, "dev")) - len(dev_negs)

    print(f"\nfitting on {len(bean_items)} bean photos ({args.id_split} split) + "
          f"{len(dev_negs)} clean dev negatives ({dropped} dropped: ambiguous patch labels)")
    bean_rows, bean_declined = _collect(bean_items, cfg, model, head, args.n_patches,
                                        tta, cache_dir, ckpt_sha, 0.0)
    neg_rows, neg_declined = _collect(dev_negs, cfg, model, head, args.n_patches,
                                      tta, cache_dir, ckpt_sha, 1.0)
    fit_rows = bean_rows + neg_rows
    if len({r["label"] for r in fit_rows}) < 2:
        sys.exit("need both bean and negative photos to fit a probe")

    x = np.concatenate([r["embeds"] for r in fit_rows])
    y = np.concatenate([np.full(len(r["embeds"]), r["label"], dtype=float) for r in fit_rows])
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-6
    w = _fit_logistic((x - mu) / sd, y)
    probe = {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}
    print(f"  {len(x)} patches ({int((y == 0).sum())} bean / {int((y == 1).sum())} not-bean), "
          f"{bean_declined + neg_declined} photos declined as unmeasurable")

    # --- calibration -------------------------------------------------------
    dev_pos = negatives_from(pos_dirs, "dev")
    pos_rows, pos_declined = _collect(dev_pos, cfg, model, head, args.n_patches,
                                      tta, cache_dir, ckpt_sha, 0.0)
    # Which positives set the threshold is a real choice with a real cost, so it
    # is recorded in the artifact rather than left implicit. Narrower is more
    # honest here: a tight certificate over the wrong population is worth less
    # than a loose one over the right population.
    if args.calibrate_on == "all":
        cal_rows = pos_rows
    elif args.calibrate_on == "user":
        cal_rows = [r for r in pos_rows if r["tag"].startswith("user_beans")]
    else:
        cal_rows = [r for r in pos_rows if r["tag"] == "user_beans_independent"]
    if not cal_rows:
        sys.exit("no calibration positives found")
    for r in pos_rows:
        r["score"] = _score_with(probe, r["embeds"])
    cal = np.array([r["score"] for r in cal_rows])

    # What the two wider policies would have produced, recorded so the choice
    # above can be audited later without refitting -- and so the cost of the
    # narrow default (a looser certificate) is visible next to its benefit.
    alternatives = {}
    for name, rows in (("all", pos_rows),
                       ("user", [r for r in pos_rows if r["tag"].startswith("user_beans")]),
                       ("user_independent",
                        [r for r in pos_rows if r["tag"] == "user_beans_independent"])):
        if rows:
            s = np.array([r["score"] for r in rows])
            alternatives[name] = {"n": len(rows), "alpha_floor": 1.0 / (len(rows) + 1),
                                  "threshold_at_floor": float(s.max()),
                                  "max": float(s.max()), "median": float(np.median(s))}
    if training_days:
        rows = [r for r in pos_rows if r["tag"] == "user_beans_independent" and r["date"] not in training_days]
        if rows:
            s = np.array([r["score"] for r in rows])
            alternatives["user_independent_other_days"] = {
                "n": len(rows), "training_days": sorted(training_days), "alpha_floor": 1.0 / (len(rows) + 1),
                "threshold_at_floor": float(s.max()), "max": float(s.max()), "median": float(np.median(s))}
    floor = 1.0 / (len(cal) + 1)
    alpha = args.alpha if args.alpha is not None else floor
    if alpha < floor:
        sys.exit(f"alpha={alpha:.3f} is tighter than {len(cal)} calibration photos can certify "
                 f"(floor {floor:.3f}); collect more unseen genuine photos or raise --alpha")
    thr = conformal_threshold(cal, alpha)

    print(f"\ncalibrating on {len(cal_rows)} unseen genuine photos ({args.calibrate_on}), "
          f"{pos_declined} declined")
    print(f"  scores: min {cal.min():.4f}  median {np.median(cal):.4f}  max {cal.max():.4f}")
    table = {}
    for a in (0.20, 0.15, 0.10, 0.05, 0.02, 0.01):
        t = conformal_threshold(cal, a)
        table[f"{a:.2f}"] = t
        print(f"  alpha {a:>5.0%} -> " + (f"threshold {t:.4f}" if t is not None
                                          else f"not certifiable with n={len(cal)}"))
    print(f"\nSHIPPING threshold {thr:.4f} at certified alpha {alpha:.1%}")
    other = alternatives.get("user_independent_other_days")
    if other:
        print(f"  beside it, the {other['n']} user_independent dev positives shot off the training days "
              f"({', '.join(other['training_days'])}) alone would certify alpha {other['alpha_floor']:.1%} at "
              f"threshold {other['threshold_at_floor']:.4f} (reporting only)")

    dev_neg_scores = np.array([_score_with(probe, r["embeds"]) for r in neg_rows])
    caught = int((dev_neg_scores > thr).sum())
    print(f"  dev negatives caught at that threshold: {caught}/{len(dev_neg_scores)} "
          f"({caught / len(dev_neg_scores):.0%})")

    out = Path(args.out) if args.out else probe_path_for(checkpoint)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha": ckpt_sha,
        "method": "linear_probe",
        "score_is": "P(not beans), mean over patches; refuse when score > threshold",
        "embedding_dim": int(x.shape[1]),
        "threshold": thr,
        "alpha": alpha,
        "alpha_floor": floor,
        "thresholds_by_alpha": table,
        "calibrate_on": args.calibrate_on,
        "calibration": {
            "n": len(cal_rows),
            "photos": [r["photo"] for r in cal_rows],
            "scores": [round(float(s), 6) for s in cal],
            "declined": pos_declined,
        },
        "calibration_alternatives": alternatives,
        "all_positive_scores": {r["photo"]: round(r["score"], 6) for r in pos_rows},
        "fit": {
            "id_split": args.id_split,
            "n_bean_photos": len(bean_rows),
            "n_negative_photos": len(neg_rows),
            "n_patches_total": int(len(x)),
            "negative_batches": [str(d) for d in neg_dirs],
            "clean_tags": sorted(CLEAN_NEGATIVE_TAGS),
            "n_patches": args.n_patches,
            "tta": tta,
        },
        "dev_negatives_caught": {"k": caught, "n": len(dev_neg_scores)},
        **probe,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}  ({out.stat().st_size / 1000:.0f} KB)")

    # --- holdout, once -----------------------------------------------------
    if args.verify:
        print("\n=== HOLDOUT (spent once; do not tune against this) ===")
        h_neg = negatives_from(neg_dirs, "holdout")
        h_pos = negatives_from(pos_dirs, "holdout")
        hn, hn_dec = _collect(h_neg, cfg, model, head, args.n_patches, tta, cache_dir, ckpt_sha, 1.0)
        hp, hp_dec = _collect(h_pos, cfg, model, head, args.n_patches, tta, cache_dir, ckpt_sha, 0.0)
        ns = np.array([_score_with(probe, r["embeds"]) for r in hn])
        ps = np.array([_score_with(probe, r["embeds"]) for r in hp])
        print(f"positives n={len(ps)} (declined {hp_dec})  median {np.median(ps):.4f}  max {ps.max():.4f}")
        print(f"negatives n={len(ns)} (declined {hn_dec})  median {np.median(ns):.4f}  min {ns.min():.4f}")
        print(f"AUROC {auroc(ps, ns):.4f}")
        fr = int((ps > thr).sum())
        ct = int((ns > thr).sum())
        print(f"at shipped threshold {thr:.4f}: refused {fr}/{len(ps)} genuine "
              f"({fr / len(ps):.0%}, certified <= {alpha:.0%}), caught {ct}/{len(ns)} negatives "
              f"({ct / len(ns):.0%})")
        # The live probe serves at its decision boundary, not the calibrated value (owner override,
        # 2026-09-27), and today's holdout bar was read there. Printed in the same single pass.
        fr = int((ps > DECISION_BOUNDARY).sum())
        ct = int((ns > DECISION_BOUNDARY).sum())
        print(f"at the decision boundary {DECISION_BOUNDARY:g} (the live override): refused {fr}/{len(ps)} "
              f"genuine, caught {ct}/{len(ns)} negatives")
        if training_days:
            days = ", ".join(sorted(training_days))
            for name, on in ((f"on a training day ({days})", True), (f"off the training days", False)):
                g = [s for r, s in zip(hp, ps) if r["tag"] == "user_beans_independent" and (r["date"] in training_days) == on]
                if g:
                    g = np.array(g)
                    print(f"  user_independent genuine {name}: refused "
                          f"{int((g > thr).sum())}/{len(g)} at {thr:.4f}, {int((g > DECISION_BOUNDARY).sum())}/{len(g)} "
                          f"at {DECISION_BOUNDARY:g}")
        for r, s in sorted(zip(hp, ps), key=lambda t: -t[1])[:3]:
            print(f"  highest genuine: {s:.4f}  {r['photo']}")
        for r, s in sorted(zip(hn, ns), key=lambda t: t[1])[:3]:
            print(f"  lowest negative: {s:.4f}  {r['photo']}  [{r['tag']}]")


if __name__ == "__main__":
    main()
