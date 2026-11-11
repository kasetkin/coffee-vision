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

Since ticket ML-26 a stage's run predicts only the photos new or changed since its last run, when nothing else
changed (coffeecv.incremental reads dvc.lock: the code, the model, the params and the command; the canonical photos'
code is among the deps, so a photo's canonical photo is a function of its original): a photo whose path, sha256,
source and split are those of its row in the recorded index.csv keeps that row and its mask, if the mask is on disk as
recorded; a photo no longer in the split loses its mask. Anything else, or --full, predicts every photo. The output is
the same bytes either way. A folder no dvc.lock stage writes is predicted in full and never cleaned.

ML-2's mode, masks of labels/ml2's eval lists (seg_predict@<model>, seg_predict_ft), was removed with its stages
on 2026-10-21; it is in git history.
"""
from __future__ import annotations

import argparse
import csv
import io
import time
from pathlib import Path

import torch
from PIL import Image

from coffeecv import incremental as inc
from coffeecv.canonical_store import load_for_record
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.seg_dataset import load_seg_dataset, photo_id
from coffeecv.segment_beans import NAMED_SEGMENTERS, BeanSegmenter, d4_box, mask_sha256, named_params

ML5_MASK_ROOT = REPO_ROOT / "data" / "seg_masks_ml5"


def split_entries(split: str) -> list[dict]:
    """[{id, source, split, path, sha256}] of one split of the segmenter dataset, in file order."""
    return [{"id": photo_id(e["path"]), "source": e["source"], "split": e["split"], "path": e["path"],
             "sha256": e["sha256"]} for e in load_seg_dataset() if e["split"] == split]


CHANGING_INPUTS = ("labels/ml5/seg_dataset.yaml", "data/canonical")


def recorded_rows(out_dir: Path, full: bool,
                  root: Path = REPO_ROOT) -> tuple[inc.Record | None, dict[str, dict], str, bool]:
    """(the last run's Record, its index.csv rows by id, how, whether the folder is a stage's) when the masks of
    `out_dir`'s stage may be kept (ML-26); (None, {}, why, ...) for a full run. Only a stage's folder is cleaned."""
    try:
        rel = out_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None, {}, "full (not a repository's masks)", False
    stage = inc.stage_of_output(rel, root) if (root / "dvc.lock").exists() else None
    if stage is None:
        return None, {}, f"full (no dvc.lock stage writes {rel})", False
    if full:
        return None, {}, "full (--full)", True
    outs, _, why = inc.recorded_outputs(stage, CHANGING_INPUTS, rel, root)
    if outs is None:
        return None, {}, f"full ({why})", True
    data = inc.recorded_output(root, outs["index.csv"]) if "index.csv" in outs else None
    if data is None:
        return None, {}, "full (the recorded index.csv is not in the cache)", True
    return (inc.Record({}, outs), {r["id"]: r for r in csv.DictReader(io.StringIO(data.decode()))}, "incremental",
            True)


def run_split(model: str, split: str, out_dir: Path, full: bool = False) -> None:
    cfg = RunConfig.from_params_yaml()
    if cfg.seg_prompt != "box":
        raise ValueError(f"seg_prompt {cfg.seg_prompt!r}: named segmenters take the serving box prompt")
    record, rows, how, clean = recorded_rows(out_dir, full)
    predict(split_entries(split), lambda: BeanSegmenter(named_params(model, cfg.seg_mask_select)), out_dir, cfg,
            record, rows, how, clean)


def kept_row(e: dict, record: inc.Record | None, rows: dict[str, dict], out_dir: Path) -> dict | None:
    """`e`'s recorded index row when its photo and its mask stand as recorded; else None (predict it)."""
    old = rows.get(e["id"]) if record is not None else None
    own = {k: str(v) for k, v in e.items() if k not in ("id", "path", "sha256")}
    if (old is None or old["path"] != e["path"] or old["photo_sha256"] != e["sha256"]
            or any(old.get(k) != v for k, v in own.items())):
        return None
    return old if inc.keeps(record, out_dir, f"{e['id']}.png") else None


def predict(entries: list[dict], make_seg, out_dir, cfg: RunConfig, record: inc.Record | None = None,
            recorded: dict[str, dict] | None = None, how: str = "full", clean: bool = False) -> None:
    """Each entry's mask and index row; an entry carries its own columns (source and split) besides id, path
    and sha256. `make_seg` builds the segmenter, only if a photo needs it; a segmenter is taken as it is. With a
    Record (ML-26), a photo's recorded row and mask are kept when they stand; with `clean`, a stage's folder, the
    masks of photos not in `entries` are deleted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = make_seg if hasattr(make_seg, "predict_mask") else None
    rows, kept = [], 0
    t0 = time.perf_counter()
    for n, e in enumerate(entries, 1):
        old = kept_row(e, record, recorded or {}, out_dir)
        if old is not None:
            rows.append(old)
            kept += 1
            continue
        if seg is None:
            seg = make_seg()
        mask = seg.predict_mask(load_for_record(e["path"], e["sha256"]))    # its canonical photo (ML-19 P6)
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
    removed = 0
    if clean:
        want = {f"{e['id']}.png" for e in entries} | {"index.csv"}
        for f in sorted(out_dir.iterdir()):
            if f.is_file() and f.name not in want:
                f.unlink()
                removed += 1
    print(f"{how}: {len(entries) - kept} photos predicted, {kept} kept, {removed} stale removed")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help=f"one of {list(NAMED_SEGMENTERS)}, or WEIGHTS:DECODER")
    ap.add_argument("--split", choices=["train", "validation", "test"], required=True,
                    help="the split of labels/ml5/seg_dataset.yaml to predict")
    ap.add_argument("--threads", type=int, required=True,
                    help="torch.set_num_threads; pinned in dvc.yaml, because it changes mask bits")
    ap.add_argument("--out", type=Path, help="the masks' folder (default: data/seg_masks_ml5/<model> for a name)")
    ap.add_argument("--full", action="store_true", help="predict every photo (ticket ML-26)")
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
    run_split(args.model, args.split, args.out or ML5_MASK_ROOT / args.model, args.full)


if __name__ == "__main__":
    main()
