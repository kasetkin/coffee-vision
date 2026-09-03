"""Photo-level pooling vs patch-level, measured at full-rig n. Scoring only, no retraining.

Phase 14 measured what pooling a photo's patches into one prediction buys, and got
box -0.035 / pixel +0.079 / sony +0.015, mean +0.020 -- then said so honestly:
"n=27 photos/fold, so +/-3.7 points per photo; directional, not precise." At that
n the box fold's regression is a single photo, which is not enough to justify
engineering a confidence-weighted or outlier-rejecting aggregator against it.

This raises n instead of getting cleverer: it scores **every photo in the
held-out rig** (180 for box/pixel/sony against Phase 14's 27, so ~6.7x, which
shrinks the per-fold error bar by ~2.6x) using the fold's own archived
checkpoint, fetched out of DVC history exactly as xrig_eval does.

Pooling here is the mean of a photo's per-patch probability vectors -- the same
thing infer.py does at inference. Patch accuracy and photo accuracy are computed
over identical patches, so the delta isolates pooling and nothing else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from coffeecv.config import REPO_ROOT
from coffeecv.dataset import resolve_rigs
from coffeecv.infer import config_for_checkpoint, load_model
from coffeecv.xrig_eval import (
    DEFAULT_SCRATCH,
    fetch_fold_checkpoint,
    resolve_class_ids,
    resolve_exp,
    run_photowise,
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct near 0 and 1, where normal-approx isn't."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(exp_id: int, n_patches: int, seed: int, dihedral: bool, scratch: Path) -> dict:
    config_path, sha = resolve_exp(exp_id)
    ckpt = fetch_fold_checkpoint(sha, exp_id, scratch)
    cfg, _ = config_for_checkpoint(ckpt, str(config_path))
    _, heldout_rig_dir, classes_file = cfg.resolve_paths()
    if heldout_rig_dir is None:
        raise SystemExit(f"exp{exp_id} has no heldout_rig -- an all-rigs run has no cross-rig split")
    rig = resolve_rigs([heldout_rig_dir])[0]
    class_ids, _ = resolve_class_ids(classes_file, config_path)
    model, head = load_model(ckpt, cfg.model_name, len(class_ids), cfg.dropout)

    probs, labels = run_photowise(model, head, cfg, rig, class_ids, n_patches, seed, dihedral)
    n_photos = len(probs) // n_patches
    probs = probs[: n_photos * n_patches].reshape(n_photos, n_patches, -1)
    photo_labels = labels[: n_photos * n_patches].reshape(n_photos, n_patches)[:, 0]

    patch_correct = int((probs.argmax(axis=2) == photo_labels[:, None]).sum())
    patch_total = n_photos * n_patches
    photo_correct = int((probs.mean(axis=1).argmax(axis=1) == photo_labels).sum())

    patch_acc = patch_correct / patch_total
    photo_acc = photo_correct / n_photos
    lo, hi = wilson(photo_correct, n_photos)
    return {
        "exp": exp_id, "rig": rig.name, "n_photos": n_photos, "n_patches_per_photo": n_patches,
        "patch_acc": patch_acc, "photo_acc": photo_acc, "delta": photo_acc - patch_acc,
        "photo_acc_ci95": [lo, hi], "photo_correct": photo_correct,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-id", type=int, nargs="+", required=True)
    ap.add_argument("--n-patches", type=int, default=16, help="patches per photo (Phase 14 used 40)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dihedral-tta", action="store_true")
    ap.add_argument("--scratch-dir", default=str(DEFAULT_SCRATCH))
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "photo_pooling.json"))
    args = ap.parse_args()

    rows = []
    for e in args.exp_id:
        r = evaluate(e, args.n_patches, args.seed, not args.no_dihedral_tta, Path(args.scratch_dir))
        rows.append(r)
        print(f"  exp{r['exp']:<4} {r['rig']:<38} n={r['n_photos']:<4} "
              f"patch {r['patch_acc']:.3f}  photo {r['photo_acc']:.3f}  "
              f"delta {r['delta']:+.3f}  [95% CI {r['photo_acc_ci95'][0]:.3f}-{r['photo_acc_ci95'][1]:.3f}]",
              flush=True)

    print(f"\n{'exp':<6}{'rig':<38}{'n':>5}{'patch':>8}{'photo':>8}{'delta':>9}")
    for r in rows:
        print(f"{r['exp']:<6}{r['rig']:<38}{r['n_photos']:>5}{r['patch_acc']:>8.3f}"
              f"{r['photo_acc']:>8.3f}{r['delta']:>+9.3f}")
    d = [r["delta"] for r in rows]
    print(f"{'MEAN':<49}{np.mean([r['patch_acc'] for r in rows]):>8.3f}"
          f"{np.mean([r['photo_acc'] for r in rows]):>8.3f}{np.mean(d):>+9.3f}"
          f"   ({sum(1 for v in d if v>0)}/{len(d)} folds positive)")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
