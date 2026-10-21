"""Ticket ML-5 P8 (D22): the `seg_ml5_predict@<model>` stage. A named segmenter's masks of one split of the
segmenter dataset (labels/ml5/seg_dataset.yaml), every source, negatives included, computed once and stored,
because mask bits differ by a few pixels across CPUs and thread counts (ML-2 P0). The blind paired session
(seg_review paired) reads two of these and never recomputes a mask.

    python -m coffeecv.seg_predict xl0_v3 --split test --threads 4    # seg_ml5_predict@xl0_v3
    python -m coffeecv.seg_predict WEIGHTS:DECODER --split test --threads 4 --out DIR

The segmenter is a name in segment_beans.NAMED_SEGMENTERS, or (ticket ML-9 P2) WEIGHTS:DECODER for a candidate
not registered yet, as named_params takes it: registering one edits segment_beans.py, a dep of every segcrop
stage. Either runs at the serving box prompt and params.yaml's seg_mask_select. Out: data/seg_masks_ml5/<name>/
or --out (required for WEIGHTS:DECODER), <id>.png (1-bit) + index.csv, one row per photo: its source and split,
the photo and mask sha256, area fraction, the selected output's predicted IoU, and the D4 crop box (empty if the
mask is empty).

ML-2's mode, masks of labels/ml2's eval lists (seg_predict@<model>, seg_predict_ft), was removed with its stages
on 2026-10-21; it is in git history.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image

from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.repo_files import sha256_file
from coffeecv.seg_dataset import load_seg_dataset, photo_id
from coffeecv.segment_beans import NAMED_SEGMENTERS, BeanSegmenter, d4_box, mask_sha256, named_params

ML5_MASK_ROOT = REPO_ROOT / "data" / "seg_masks_ml5"


def split_entries(split: str) -> list[dict]:
    """[{id, source, split, path, sha256}] of one split of the segmenter dataset, in file order."""
    return [{"id": photo_id(e["path"]), "source": e["source"], "split": e["split"], "path": e["path"],
             "sha256": e["sha256"]} for e in load_seg_dataset() if e["split"] == split]


def run_split(model: str, split: str, out_dir: Path) -> None:
    cfg = RunConfig.from_params_yaml()
    if cfg.seg_prompt != "box":
        raise ValueError(f"seg_prompt {cfg.seg_prompt!r}: named segmenters take the serving box prompt")
    seg = BeanSegmenter(named_params(model, cfg.seg_mask_select))
    predict(split_entries(split), seg, out_dir, cfg)


def predict(entries: list[dict], seg: BeanSegmenter, out_dir, cfg: RunConfig) -> None:
    """Each entry's mask and index row; an entry carries its own columns (source and split) besides id, path
    and sha256."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for n, e in enumerate(entries, 1):
        path = REPO_ROOT / e["path"]
        if sha256_file(path) != e["sha256"]:
            raise ValueError(f"{e['path']}: sha256 differs from the list it came from")
        mask = seg.predict_mask(load_rgb_image(path))
        Image.fromarray(mask).convert("1").save(out_dir / f"{e['id']}.png", optimize=True)
        box = d4_box(mask) if mask.any() else None
        own = {k: v for k, v in e.items() if k not in ("id", "path", "sha256")}
        rows.append({"id": e["id"], **own, "path": e["path"], "photo_sha256": e["sha256"],
                     "mask_sha256": mask_sha256(mask), "height": mask.shape[0], "width": mask.shape[1],
                     "area_frac": f"{mask.mean():.4f}", "pred_iou": f"{seg.pred_iou:.4f}",
                     "box": " ".join(map(str, box)) if box else "",
                     "prompt": cfg.seg_prompt, "mask_select": cfg.seg_mask_select,
                     "threads": torch.get_num_threads(), "weights_sha256": seg.weights_sha256,
                     **({"decoder_sha256": seg.decoder_sha256} if seg.decoder_sha256 else {})})
        if n % 25 == 0 or n == len(entries):
            print(f"{n}/{len(entries)}  {time.perf_counter() - t0:.0f}s", flush=True)
    with open(out_dir / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help=f"one of {list(NAMED_SEGMENTERS)}, or WEIGHTS:DECODER")
    ap.add_argument("--split", choices=["train", "validation", "test"], required=True,
                    help="the split of labels/ml5/seg_dataset.yaml to predict")
    ap.add_argument("--threads", type=int, required=True,
                    help="torch.set_num_threads; pinned in dvc.yaml, because it changes mask bits")
    ap.add_argument("--out", type=Path, help="the masks' folder (default: data/seg_masks_ml5/<model> for a name)")
    args = ap.parse_args(argv)
    if args.model not in NAMED_SEGMENTERS:
        if ":" not in args.model:
            ap.error(f"model {args.model!r}: not one of {list(NAMED_SEGMENTERS)} and not WEIGHTS:DECODER")
        try:
            named_params(args.model)
        except ValueError as e:
            ap.error(str(e))
        if args.out is None:
            ap.error("--out is required for WEIGHTS:DECODER")
    torch.set_num_threads(args.threads)
    run_split(args.model, args.split, args.out or ML5_MASK_ROOT / args.model)


if __name__ == "__main__":
    main()
