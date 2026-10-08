"""Fit the shipped OOD probe (ADR 0016, ticket ML-5 D9, D16).

Coefficients fitted once, written to `<checkpoint>.ood_probe.json` beside the weights. The probe refuses at
the fixed `infer.PROBE_THRESHOLD` (0.5, its own decision boundary); nothing here calibrates a threshold, and
the file's `threshold` field only records the fixed value serving uses.

The fit set:

- **Bean photos**: the checkpoint's own held-out photos (`--id-split`, its test split by default), minus any
  photo that would share a D13 group with a photo of the segmenter dataset's test split, which `ood_eval`
  measures the probe on (D4): the same bytes, or a burst shot within seg_dataset.GROUP_SECONDS of one on the
  same clock (`probe_beans`).
- **Negatives**: the segmenter dataset's training and validation negatives whose tag is in
  `ood_eval.CLEAN_NEGATIVE_TAGS`, imported rather than restated. Test negatives are never read here.

    python -m coffeecv.fit_ood_probe --checkpoint models/allrigs_dino3b16_seg_country_s123.pt
    python -m coffeecv.ood_eval --checkpoint models/allrigs_dino3b16_seg_country_s123.pt    # then measure
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from coffeecv.class_list import load_classes
from coffeecv.config import CHECKPOINTS_DIR, REPO_ROOT
from coffeecv.infer import PROBE_THRESHOLD, _sha, inference_tta_for, load_model, probe_path_for, probe_score
from coffeecv.ood_eval import (CLEAN_NEGATIVE_TAGS, Unmeasurable, _fit_logistic, embed_photo, id_photos,
                               split_config)
from coffeecv.repo_files import rel
from coffeecv.seg_dataset import SEG_DATASET_FILE, groups, load_seg_dataset, scan

FIT_SPLITS = ("train", "validation")


def probe_negatives(photos: list[dict]) -> list[dict]:
    """The segmenter dataset's training and validation negatives with a clean tag, in file order."""
    return [e for e in photos
            if e["source"] == "negative" and e["split"] in FIT_SPLITS and e["tag"] in CLEAN_NEGATIVE_TAGS]


def probe_beans(beans: list[Path], photos: list[dict],
                describe: Callable[[set[str]], list[dict]] = scan) -> tuple[list[Path], int]:
    """(`beans` without any photo in a group with a test photo of `photos`, how many were dropped).

    The groups are seg_dataset's (D13), over the test photos and `beans` together: byte-identical files, one
    source URL, and shots of one subject chained within GROUP_SECONDS on one clock. So a bean photo goes if it
    is a test photo, a copy of one, or in its burst (a pool photo the dataset did not draw, shot seconds
    before or after a test one). `describe` is seg_dataset.scan(only=...): their records, shot times
    included."""
    test = [e["path"] for e in photos if e["split"] == "test"]
    group = groups(describe({rel(q) for q in beans} | set(test)))
    in_test = {group[p] for p in test}
    kept = [q for q in beans if group[rel(q)] not in in_test]
    return kept, len(beans) - len(kept)


def _collect(paths: list[Path], cfg, model, head, n_patches, tta, cache_dir, ckpt_sha, label):
    """Embed a set of photos, returning (rows, n_declined).

    Photos `classify_one` cannot measure are counted, not silently dropped: they
    are already refused in production for a different reason.
    """
    rows, declined = [], 0
    for path in paths:
        try:
            _, embeds, _, _ = embed_photo(path, cfg, model, head, n_patches, tta, cache_dir, ckpt_sha)
        except Unmeasurable:
            declined += 1
            continue
        shown = path.resolve()
        shown = shown.relative_to(REPO_ROOT) if shown.is_relative_to(REPO_ROOT) else shown
        rows.append({"photo": str(shown), "embeds": embeds, "label": label})
    return rows, declined


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "best.pt"))
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default=str(SEG_DATASET_FILE),
                   help="the segmenter dataset file; its training and validation negatives are fitted on")
    p.add_argument("--id-split", default="test", choices=("val", "test"),
                   help="which of the checkpoint's own photos supply bean patches")
    p.add_argument("--n-patches", type=int, default=40)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--cache-dir", default=".ood_eval_cache")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    photos = load_seg_dataset(Path(args.dataset))
    checkpoint = Path(args.checkpoint)
    try:
        cfg, cfg_source = split_config(checkpoint, args.config)
    except ValueError as e:
        sys.exit(f"refusing: {e}")
    print(f"config: {cfg_source}")
    _, classes_file = cfg.resolve_paths()
    classes = load_classes(classes_file)
    ckpt_sha = _sha(checkpoint)
    model, head = load_model(checkpoint, cfg.model_name, len(classes), cfg.dropout)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    tta = False if args.no_tta else inference_tta_for(checkpoint, cfg.model_name)

    beans, beans_in_test = probe_beans(id_photos(cfg, classes, args.id_split), photos)
    negatives = probe_negatives(photos)
    unclean = sum(e["source"] == "negative" and e["split"] in FIT_SPLITS for e in photos) - len(negatives)
    print(f"\nfitting on {len(beans)} bean photos ({args.id_split} split; {beans_in_test} dropped: in the "
          f"segmenter test split's groups) + {len(negatives)} training and validation negatives ({unclean} dropped: "
          f"ambiguous patch labels)")
    bean_rows, bean_declined = _collect(beans, cfg, model, head, args.n_patches, tta, cache_dir, ckpt_sha, 0.0)
    neg_rows, neg_declined = _collect([REPO_ROOT / e["path"] for e in negatives], cfg, model, head,
                                      args.n_patches, tta, cache_dir, ckpt_sha, 1.0)
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

    # In-sample, so a sanity check of the fit and not a measurement: ood_eval measures on the test split.
    caught = sum(probe_score(r["embeds"], probe) > PROBE_THRESHOLD for r in neg_rows)
    print(f"  fitted negatives above {PROBE_THRESHOLD:g} (in-sample): {caught}/{len(neg_rows)}")

    out = Path(args.out) if args.out else probe_path_for(checkpoint)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha": ckpt_sha,
        "method": "linear_probe",
        "score_is": "P(not beans), mean over patches; refuse when score > threshold",
        "embedding_dim": int(x.shape[1]),
        "threshold": PROBE_THRESHOLD,
        "threshold_is": "fixed (ADR 0016); serving uses infer.PROBE_THRESHOLD, this field records it",
        "fit": {
            "dataset": args.dataset,
            "negative_splits": list(FIT_SPLITS),
            "id_split": args.id_split,
            "n_bean_photos": len(bean_rows),
            "n_bean_photos_dropped_in_test": beans_in_test,
            "n_negative_photos": len(neg_rows),
            "n_patches_total": int(len(x)),
            "clean_tags": sorted(CLEAN_NEGATIVE_TAGS),
            "n_patches": args.n_patches,
            "tta": tta,
        },
        "fit_negatives_above_threshold": {"k": int(caught), "n": len(neg_rows)},
        **probe,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}  ({out.stat().st_size / 1000:.0f} KB)")


if __name__ == "__main__":
    main()
