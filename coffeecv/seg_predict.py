"""Ticket ML-2 P1: the `seg_predict@<model>` stage (plan §2.4, §4.2). Masks for every photo in the eval
lists, computed once and stored, because mask bits differ by a few pixels across CPUs and thread counts (P0).
Verdicts, overlays and the review tool read these stored masks and never recompute them.

    python -m coffeecv.seg_predict pretrained --threads 4
    python -m coffeecv.seg_predict ft_s42 --threads 4      # P5: L0 with the seed-42 fine-tuned decoder

A model is pretrained weights (any variant, ticket ML-5 P4) and optionally a fine-tuned decoder over them:
"pretrained" is the fine-tune's base, params.yaml seg_ft.weights, and a decoder runs over the base its card
records (`model_params`), so moving the fine-tune to another variant is one edit of params.yaml.

Lists: seg_eval + neg_seg_eval + pos_seg_eval. The plan names the first two; pos_seg_eval joined with D26, and
it holds the 10 OOD-positive base candidates, so with it every base candidate has a stored mask.
Prompt and output come from params.yaml (seg_prompt, seg_mask_select; D5).

Out: data/seg_masks/<model>/<id>.png (1-bit) + index.csv, one row per photo: the list, the photo and mask
sha256, area fraction, the selected output's predicted IoU, and the D4 crop box (empty if the mask is empty).
"""
from __future__ import annotations

import argparse
import csv
import time
from typing import NamedTuple

import torch
from PIL import Image

from coffeecv import seg_lists
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.repo_files import read_csv, sha256_file
from coffeecv.seg_base_masks import item_id
from coffeecv.seg_finetune import FtParams
from coffeecv.segment_beans import BeanSegmenter, SegParams, d4_box, decoder_base, mask_sha256

MASK_ROOT = REPO_ROOT / "data" / "seg_masks"
LISTS = ("seg_eval", "neg_seg_eval", "pos_seg_eval")
FT_SEEDS = (42, 123, 7)


class Model(NamedTuple):
    decoder: str | None = None    # a fine-tuned mask decoder (seg_finetune.py), or None: the pretrained base


MODELS = {"pretrained": Model(), **{f"ft_s{s}": Model(f"models/seg/ft_s{s}.pt") for s in FT_SEEDS}}


def model_params(model: str, cfg: RunConfig) -> SegParams:
    """The segmenter MODELS[model] names, at the serving prompt and output (params.yaml seg_prompt,
    seg_mask_select): the fine-tune's base (seg_ft.weights) for "pretrained", else the decoder over the base
    its card records."""
    decoder = MODELS[model].decoder
    weights = decoder_base(decoder) if decoder else FtParams.from_config(cfg).weights
    return SegParams(mask_select=cfg.seg_mask_select, prompt=cfg.seg_prompt, weights=weights, decoder=decoder)


def photo_entries(lists: dict[str, list[dict]]) -> list[dict]:
    """[{id, list, path, sha256}] over LISTS, in list order. Raises if an id repeats."""
    out, seen = [], set()
    for name in LISTS:
        for e in lists[name]:
            i = item_id(e)
            if i in seen:
                raise ValueError(f"item id {i} repeats")
            seen.add(i)
            out.append({"id": i, "list": name, "path": e["path"], "sha256": e["sha256"]})
    return out


def mask_items(model: str) -> list[dict]:
    """The stage's index.csv as judge/review items: {item, list, path, photo_sha256, mask, mask_sha256, ...},
    `mask` repo-relative. Read by seg_judge --masks, seg_eval and review_masks."""
    out_dir = MASK_ROOT / model
    rows = read_csv(out_dir / "index.csv")
    return [{"item": r["id"], "mask": str((out_dir / f"{r['id']}.png").relative_to(REPO_ROOT)), **r} for r in rows]


def run(model: str) -> None:
    cfg = RunConfig.from_params_yaml()
    _, lists = seg_lists.load_lists()
    entries = photo_entries(lists)
    out_dir = MASK_ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = BeanSegmenter(model_params(model, cfg))
    rows = []
    t0 = time.perf_counter()
    for n, e in enumerate(entries, 1):
        path = REPO_ROOT / e["path"]
        if sha256_file(path) != e["sha256"]:
            raise ValueError(f"{e['path']}: sha256 differs from photo_lists.yaml")
        mask = seg.predict_mask(load_rgb_image(path))
        Image.fromarray(mask).convert("1").save(out_dir / f"{e['id']}.png", optimize=True)
        box = d4_box(mask) if mask.any() else None
        rows.append({"id": e["id"], "list": e["list"], "path": e["path"], "photo_sha256": e["sha256"],
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
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--threads", type=int, required=True,
                    help="torch.set_num_threads; pinned in dvc.yaml, because it changes mask bits")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    run(args.model)


if __name__ == "__main__":
    main()
