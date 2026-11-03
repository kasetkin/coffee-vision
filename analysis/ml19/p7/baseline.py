"""ML-19 P7 (D14, D48): the live classifier on the canonical test crops, beside its own numbers on today's crops.

    python analysis/ml19/p7/baseline.py [--checkpoint models/dino3b16_segxl0v6_country_aug8_s42.pt]

Builds the checkpoint's own test split (fold_data.build_fold_datasets, its card's config: the same seed, photo split
and patch budget, 160 patches a class) over data/segcropped/pool, which since ML-19 P6 holds the crops of canonical
photos, made by today's segmenter (xl0_v6_s42); scores every patch with the checkpoint in PyTorch, as training
scored it (metrics.compute_split_metrics); and writes analysis/ml19/p7/baseline.json and prints the comparison with
the checkpoint's source experiment's test metrics (its crops of the originals, JPEG q95). The two crop sets hold the
same photos in the same split; their patches differ (another decode, another mask), so the comparison is per class,
not per patch. This is the baseline ML-16's retrain is paired against (ADR 0002): the input's change alone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from coffeecv import infer  # noqa: E402
from coffeecv.fold_data import build_fold_datasets  # noqa: E402
from coffeecv.metrics import compute_split_metrics  # noqa: E402
from coffeecv.transforms import build_eval_transform  # noqa: E402

OUT = Path(__file__).resolve().parent / "baseline.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=REPO / "models/dino3b16_segxl0v6_country_aug8_s42.pt")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    card = json.loads(args.checkpoint.with_suffix(".json").read_text())
    cfg, source = infer.config_for_checkpoint(args.checkpoint, None)
    t0 = time.perf_counter()
    tf = build_eval_transform(cfg.patch_resize)
    fold = build_fold_datasets(cfg, tf, tf, only=("test",))
    ds = fold.test
    print(f"test split: {len(ds)} patches from {cfg.train_capture_dirs} ({source}), built in "
          f"{time.perf_counter() - t0:.0f} s; fallback photos {sorted(ds.fallback_photos)}", flush=True)
    model, head = infer.load_model(args.checkpoint, cfg.model_name, len(fold.class_ids))
    tensors = torch.stack([ds[i][0] for i in range(len(ds))])
    y = np.array([ds[i][1] for i in range(len(ds))])
    probs, _ = infer.forward_with_embeddings(model, head, tensors, tta=False)
    pred = probs.argmax(1)
    losses = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))
    metrics = compute_split_metrics(y, pred, losses, fold.class_ids, fold.class_labels)
    exp = card["source_experiment"]
    before = json.loads((REPO / "experiments" / exp / "metrics.json").read_text())["splits"]["test"]
    out = {"checkpoint": str(args.checkpoint.relative_to(REPO)), "source_experiment": exp,
           "crops": cfg.train_capture_dirs, "patches": len(ds), "fallback_photos": sorted(ds.fallback_photos),
           "patches_below_share": ds.n_below_share, "canonical": metrics,
           "today": {k: before[k] for k in ("n_samples", "accuracy", "macro_f1", "mcc", "per_class")}}
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\n{'class':12} {'F1 today':>9} {'F1 canonical':>13}   recall today -> canonical")
    for k, c in metrics["per_class"].items():
        b = before["per_class"][k]
        print(f"{k:12} {b['f1']:9.4f} {c['f1']:13.4f}   {b['recall']:.4f} -> {c['recall']:.4f}")
    print(f"{'all':12} {before['macro_f1']:9.4f} {metrics['macro_f1']:13.4f}   accuracy {before['accuracy']:.4f} -> "
          f"{metrics['accuracy']:.4f}, mcc {before['mcc']:.4f} -> {metrics['mcc']:.4f}")
    print(f"wrote {OUT.relative_to(REPO)} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
