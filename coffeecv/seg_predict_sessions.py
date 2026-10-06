"""Ticket ML-3 P5: the full-frame masks of whole sessions, for the owner's review of the new photos (D8).

segcrop keeps only crop-aligned masks; the review page shows a full-frame one. This writes them the way
`seg_predict` does for the eval lists (same segmenter construction, same index.csv), but for every photo segcrop
reads in the given sessions, in its order, with the session as the `list` column:

    python -m coffeecv.seg_predict_sessions ft_s123 --threads 4 --out ml3_new_ft_s123 --sessions 2026-09-11__pixel ...

Out: data/seg_masks/<out>/<id>.png (1-bit) + index.csv. Each mask's sha256 must equal the mask_sha256 in that
photo's segcrop_report.json, or this stops: the owner judges exactly the masks training crops with. A separate
module, not a `seg_predict` flag, so the four ML-2 mask stages that depend on seg_predict.py stay up to date.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import torch
from PIL import Image

from coffeecv import seg_lists
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.crop_tray import _find_images
from coffeecv.dataset import load_rgb_image
from coffeecv.sam_loader import L0_WEIGHTS
from coffeecv.seg_base_masks import item_id
from coffeecv.seg_predict import MASK_ROOT, MODELS
from coffeecv.segment_beans import BeanSegmenter, SegParams, d4_box, mask_sha256

SEGCROPPED_ROOT = REPO_ROOT / "data" / "segcropped"


def session_entries(sessions: list[str]) -> list[dict]:
    """[{id, list, path, sha256, segcrop_mask_sha256}] for every photo segcrop reads in `sessions`, in its order;
    `list` is the session. Photo hashes are taken here: no list pins these photos, their .dvc pointers do."""
    out, seen = [], set()
    for s in sessions:
        class_dirs = sorted(d for d in (REPO_ROOT / "dataset" / s).iterdir()
                            if d.is_dir() and d.name.startswith("class_"))
        if not class_dirs:
            raise FileNotFoundError(f"no class_* directories under dataset/{s}")
        for class_dir in class_dirs:
            report = SEGCROPPED_ROOT / s / class_dir.name / "segcrop_report.json"
            segcrop = {r["file"]: r["mask_sha256"] for r in json.loads(report.read_text())}
            for photo in _find_images(class_dir):
                rel = str(photo.relative_to(REPO_ROOT))
                i = item_id({"session": s, "class": class_dir.name.split("__")[0], "path": rel})
                if i in seen:
                    raise ValueError(f"item id {i} repeats")
                seen.add(i)
                out.append({"id": i, "list": s, "path": rel, "sha256": seg_lists.sha256_file(photo),
                            "segcrop_mask_sha256": segcrop[photo.name]})
    return out


def run(model: str, sessions: list[str], out: str) -> None:
    cfg = RunConfig.from_params_yaml()
    entries = session_entries(sessions)
    out_dir = MASK_ROOT / out
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = BeanSegmenter(SegParams(mask_select=cfg.seg_mask_select, prompt=cfg.seg_prompt, weights=L0_WEIGHTS,
                                  decoder=MODELS[model]))
    rows = []
    t0 = time.perf_counter()
    for n, e in enumerate(entries, 1):
        mask = seg.predict_mask(load_rgb_image(REPO_ROOT / e["path"]))
        if mask_sha256(mask) != e["segcrop_mask_sha256"]:
            raise ValueError(f"{e['path']}: the mask differs from the one segcrop cropped with "
                             "(segcrop_report.json); the owner would review a mask training does not use")
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
    print(f"{len(rows)} masks -> {out_dir.relative_to(REPO_ROOT)}; each equals its segcrop_report.json mask_sha256")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=[m for m in MODELS if MODELS[m]])
    ap.add_argument("--threads", type=int, required=True,
                    help="torch.set_num_threads; pinned in dvc.yaml, because it changes mask bits")
    ap.add_argument("--sessions", nargs="+", required=True, help="dataset/ session folders")
    ap.add_argument("--out", required=True, help="output name under data/seg_masks/")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    run(args.model, args.sessions, args.out)


if __name__ == "__main__":
    main()
