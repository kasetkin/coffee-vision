"""ML-37, bits for the segmenter, levels 1-3 measured per photo: the encoder's embedding, the mask and its crop box,
and (on labelled photos) the mask against the owner's label, for each bit-depth arm and yardstick.

Research only: nothing imports this, no DVC stage runs it, nothing under data/, dataset/, labels/ or models/ is
written. The live segmenter (params.yaml seg_*: XL0 encoder, decoder xl0_v7r_s123, box prompt, multi3), torch on CPU
at 8 threads, float32. The arms and yardsticks are bits_seg_common's.

  --split validation   the segmenter dataset's validation photos the owner labelled in pass17 (accepted positives
                       and negatives): the photos seg_finetune.evaluate scores a decoder on (mean IoU on positives,
                       empty-or-tiny share on negatives). xl0_v7r_s123 never trained on them (its card: pass16's
                       train and neg_train roles); pass16's 120 validation positives chose its checkpoint.
  --split test         the segmenter dataset's test photos (no labels: D4); the stored masks of
                       seg_ml19_predict@xl0_v7r_s123 (4 threads, today's path) are read as a thread-count yardstick.

Per photo and form: the embedding's 1 - cosine and relative L2 against f32's (whole 256 x 64 x 64, and per location);
the full-resolution mask (BeanSegmenter.predict_mask's) against f32's mask (IoU, changed pixels, empty and D18
fallback flips, the D4 crop box's largest edge move); the predicted IoU; with a label, seg_finetune.evaluate's IoU
at the encoder's frame (labels by label_in_frame) and the full-resolution IoU. The photos are taken in a seeded
random order, so a run stopped early is a random sample. Rows go to tmp/ml37/bits_seg/<split>.jsonl (resumed), the
low-resolution logits to tmp/ml37/bits_seg/logits/<split>/<id>.npz. Run:

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/bits_seg_measure.py --split validation
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import bits_seg_common as sc
from coffeecv.seg_dataset import photo_id
from coffeecv.seg_predict import split_entries

SEED = 37015


def photos(split: str) -> list[dict]:
    ents = split_entries(split)
    if split == "validation":
        lab = sc.read_labels()
        ents = [e | {"label": lab[e["id"]]} for e in ents
                if e["id"] in lab and lab[e["id"]]["status"] in ("accepted", "negative")]
        for e in ents:
            if e["label"]["photo_sha256"] != e["sha256"]:
                raise ValueError(f"{e['id']}: label for another file")
    order = np.random.default_rng(SEED).permutation(len(ents))
    return [ents[i] for i in order]


def one(run: sc.Run, e: dict, yards: tuple[str, ...], logit_dir: Path) -> dict:
    t0 = time.perf_counter()
    rec, _ = sc.decode_rec(e["path"], e["sha256"])
    x = sc.bc.readback(rec, sc.THREADS)
    today = sc.cn.rec2020_to_srgb(rec, numthreads=sc.THREADS)
    h, w = x.shape[:2]
    feats, lows, piou = {}, {}, {}
    for arm in sc.ARMS:
        feats[arm] = run.embed(sc.arm_input(x, arm, run.side, today))
    xin = None
    if "fp16in" in yards:
        xin = sc.float_sam_input(x, run.side)
        feats["fp16in"] = run.embed(xin.astype(np.float16).astype(np.float32))
    xin = None
    if "shift1" in yards:
        feats["shift1"] = run.embed(sc.float_sam_input(sc.shift1(x), run.side))
    if "gen2" in yards:
        g = sc.gen2_rec(rec)
        rec = None
        feats["gen2"] = run.embed(sc.float_sam_input(sc.bc.readback(g, sc.THREADS), run.side))
        g = None
    rec = None
    x = None
    t_enc = time.perf_counter() - t0
    forms = list(feats)
    masks = {}
    for f in forms:
        lows[f], piou[f] = run.decode(feats[f], h, w)
        masks[f] = run.full_mask(lows[f])
    row = {"id": e["id"], "path": e["path"], "source": e["source"], "split": e["split"],
           "camera": sc.camera(e["path"], e["source"]), "shape": [h, w], "pred_iou": piou,
           "feat": {f: sc.feat_diff(feats[f], feats["f32"]) for f in forms if f != "f32"},
           "mask": {f: sc.mask_diff(masks[f], masks["f32"], run.min_area) for f in forms}}
    if e["split"] == "test":
        stored = np.array(Image.open(sc.STORED / f"{photo_id(e['path'])}.png")) > 0
        row["stored_vs_today"] = sc.mask_diff(stored, masks["today"], run.min_area)
        row["stored_vs_f32"] = sc.mask_diff(stored, masks["f32"], run.min_area)
    if "label" in e:
        lab = e["label"]
        row["status"] = lab["status"]
        full = np.array(Image.open(sc.REPO_ROOT / lab["mask"])) > 0
        if full.shape != (h, w):
            raise ValueError(f"{e['id']}: label {full.shape} does not match photo {(h, w)}")
        fl = sc.label_in_frame(full, run.side)
        flt = torch.from_numpy(fl)
        row["label"] = {f: sc.label_scores(run.frame_logits(lows[f]), flt) | {"iou_full": sc.iou(masks[f], full)}
                        for f in forms}
        row["label_area"] = float(full.mean())
    np.savez_compressed(logit_dir / f"{e['id']}.npz", **{f: lows[f] for f in forms})
    row["seconds"] = round(time.perf_counter() - t0, 1)
    row["encode_seconds"] = round(t_enc, 1)
    row["rss_mb"] = round(sc.rss_mb())
    feats = lows = masks = None
    gc.collect()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "test"), required=True)
    ap.add_argument("--yards", default="fp16in,shift1,gen2", help="comma list of yardsticks ('' for none)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deadline", default="", help="UTC YYYY-MM-DDTHH:MM: start no photo after it")
    a = ap.parse_args()
    yards = tuple(y for y in a.yards.split(",") if y)
    sc.SCRATCH.mkdir(parents=True, exist_ok=True)
    logit_dir = sc.SCRATCH / "logits" / a.split
    logit_dir.mkdir(parents=True, exist_ok=True)
    out = sc.SCRATCH / f"{a.split}.jsonl"
    done = {json.loads(r)["id"] for r in out.read_text().splitlines()} if out.exists() else set()
    todo = photos(a.split)
    if a.limit:
        todo = todo[:a.limit]
    run = sc.Run()
    print(f"{a.split}: {len(todo)} photos, {len(done)} done; yards {yards}; rss {sc.rss_mb():.0f} MB", flush=True)
    for i, e in enumerate(todo, 1):
        if e["id"] in done:
            continue
        if a.deadline and time.strftime("%Y-%m-%dT%H:%M", time.gmtime()) >= a.deadline:
            print("stop time reached", flush=True)
            break
        r = one(run, e, yards, logit_dir)
        with out.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{i}/{len(todo)} {time.strftime('%H:%M:%S', time.gmtime())} {e['id'][-50:]} {r['shape']} "
              f"{r['seconds']} s rss {r['rss_mb']} MB u8 IoU {r['mask']['u8']['iou']:.6f} "
              f"today {r['mask']['today']['iou']:.6f}", flush=True)


if __name__ == "__main__":
    main()
