"""ML-37 D14: how far xl0_v7r_s123's masks move when its input changes as ML-37 changes it.

Research only: nothing imports this, no DVC stage runs it, nothing under data/, dataset/, labels/ or models/ is
written. Reads the canonical store and the stored test masks of seg_ml19_predict@xl0_v7r_s123.

20 photos of the segmenter dataset's test split (fixed draw, below): 12 positives drawn across folders, the 4
positives with the lowest predicted IoU in the stored masks (the ones nearest a decision), 4 negatives. Each photo's
canonical photo is read five ways and masked by the live segmenter (named_params("xl0_v7r_s123"), the serving box
prompt, multi3), torch at 2 threads:

    S   the stored mask (seg_ml19_predict, 4 threads): the noise floor of a thread count (experiments.md)
    A   today: load_canonical's uint8 sRGB through BeanSegmenter.predict_mask (sam_resize on uint8)
    B   A's uint8 values as float, resized by PIL's bilinear in mode "F" channel by channel (ML-37 D11 (a)):
        the resizer alone
    C   the same canonical photo read back to float32 sRGB by LittleCMS (no rounding, clipped to [0, 1]), resized as
        B: today's photo without the 8-bit step (D3)
    D   C's photo re-encoded as ML-37 D1 stores it: linear Rec.2020 (LittleCMS to coffeecv.canonical.LINEAR_REC2020),
        clipped at 0 (D7 (a)), float16, lossy JPEG XL (imagecodecs.jpegxl_encode, level 95, effort 7,
        primaries BT2100: linear transfer), decoded, read back to float32 sRGB by LittleCMS, clipped, resized as B.
        A second lossy generation, so it overstates D1's loss somewhat; today's canonical photo has nothing above
        white, so D6's choice does not enter.

Every mask is compared with A's by IoU (both empty counts as 1); the input pictures by their difference in 8-bit
levels, per pixel and on 32 x 32 block means (ML-29 D27's measure). Run:

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d14_mask_sensitivity.py

Out: analysis/ml37/d14_mask_sensitivity.txt; per-photo rows in tmp/ml37/d14/rows.jsonl.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import imagecodecs as ic
import numpy as np
import torch
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import model_io
from coffeecv.canonical_store import canonical_for
from coffeecv.config import REPO_ROOT
from coffeecv.seg_dataset import load_seg_dataset, photo_id
from coffeecv.segment_beans import BeanSegmenter, d4_box, named_params

THREADS = 2
STORED = REPO_ROOT / "data" / "seg_masks_ml19" / "xl0_v7r_s123"
OUT_TXT = Path(__file__).with_suffix(".txt")
SCRATCH = REPO_ROOT / "tmp" / "ml37" / "d14"
SEED = 37014


def sample() -> list[dict]:
    import csv
    stored = {r["id"]: r for r in csv.DictReader(open(STORED / "index.csv"))}
    test = [e | {"id": photo_id(e["path"])} for e in load_seg_dataset() if e["split"] == "test"]
    test = [e for e in test if e["id"] in stored]
    for e in test:
        e["stored"] = stored[e["id"]]
    rng = np.random.default_rng(SEED)
    pos = [e for e in test if e["source"] != "negative"]
    neg = [e for e in test if e["source"] == "negative"]
    hard = sorted((e for e in pos if float(e["stored"]["pred_iou"] or 0) > 0),
                  key=lambda e: float(e["stored"]["pred_iou"]))[:4]
    for e in hard:
        e["why"] = "hard"
    rest = [e for e in pos if e not in hard]
    folders = sorted({e["path"].split("/")[1] for e in rest})
    rng.shuffle(folders)
    picked = []
    for f in folders:                       # one per folder first, in a shuffled folder order
        if len(picked) == 12:
            break
        cand = [e for e in rest if e["path"].split("/")[1] == f]
        picked.append(cand[rng.integers(len(cand))])
    while len(picked) < 12:
        e = rest[rng.integers(len(rest))]
        if e not in picked:
            picked.append(e)
    for e in picked:
        e["why"] = "drawn"
    # negatives: any the stored masks left non-empty first, then drawn
    nonempty = [e for e in neg if float(e["stored"]["area_frac"] or 0) > 0]
    negs = nonempty[:2]
    while len(negs) < 4:
        e = neg[rng.integers(len(neg))]
        if e not in negs:
            negs.append(e)
    for e in negs:
        e["why"] = "negative" + (" (stored mask non-empty)" if e in nonempty else "")
    return picked + hard + negs


def float_resize(rgb: np.ndarray, side: int) -> np.ndarray:
    """sam_resize in float: PIL's bilinear in mode F, one channel at a time (ML-37 D11 (a))."""
    h, w = rgb.shape[:2]
    if max(h, w) == side:
        return rgb
    nh, nw = model_io.sam_preprocess_shape(h, w, side)
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c]), mode="F")
                                .resize((nw, nh), Image.BILINEAR)) for c in range(3)], axis=-1)


def float_sam_input(rgb01: np.ndarray, side: int) -> np.ndarray:
    small = float_resize(rgb01.astype(np.float32), side)
    ih, iw = small.shape[:2]
    m = np.asarray(model_io.SAM_MEAN, np.float32)[:, None, None]
    s = np.asarray(model_io.SAM_STD, np.float32)[:, None, None]
    x = np.zeros((1, 3, side, side), np.float32)
    x[0, :, :ih, :iw] = (small.transpose(2, 0, 1) - m) / s
    return x


def mask_from_float(seg: BeanSegmenter, rgb01: np.ndarray) -> np.ndarray:
    """BeanSegmenter.predict_mask with the encoder's input made by float_sam_input instead of model_io.sam_input."""
    h, w = rgb01.shape[:2]
    x = float_sam_input(rgb01, seg.engine.image_side)
    seg.original_size = (h, w)
    seg.input_size = model_io.sam_preprocess_shape(h, w, seg.engine.frame)
    seg.engine.set_image(x)
    sel = seg.p.mask_select
    low, iou = seg._decode(h, w, multimask=sel != "single")
    k = 0 if sel == "single" else int(iou.argmax()) if sel == "best_iou" else int(sel[-1]) - 1
    seg.pred_iou = float(iou[k])
    return seg._upsample(low[k])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def block_means(x: np.ndarray, n: int = 32) -> np.ndarray:
    h, w = (x.shape[0] // n) * n, (x.shape[1] // n) * n
    return x[:h, :w].reshape(h // n, n, w // n, n, -1).mean(axis=(1, 3))


def level_diff(a01: np.ndarray, b01: np.ndarray) -> dict:
    d = (a01.astype(np.float32) - b01.astype(np.float32)) * 255
    bd = np.abs(block_means(a01.astype(np.float32) * 255) - block_means(b01.astype(np.float32) * 255))
    ad = np.abs(d)
    return {"px_mean": float(ad.mean()), "px_p99": float(np.percentile(ad, 99)), "px_max": float(ad.max()),
            "blk_mean": float(bd.mean()), "blk_max": float(bd.max())}


def srgb_float(rec_or_lin: np.ndarray, src_profile: bytes) -> np.ndarray:
    out = np.empty(rec_or_lin.shape, np.float32)
    for y in range(0, rec_or_lin.shape[0], 256):
        ic.cms_transform(rec_or_lin[y:y + 256], src_profile, cn.SRGB, intent=cn.RELATIVE, flags=cn.NOOPTIMIZE,
                         outdtype=np.float32, out=out[y:y + 256])
    return np.clip(out, 0.0, 1.0, out=out)


def one(seg: BeanSegmenter, e: dict) -> dict:
    t0 = time.perf_counter()
    data = canonical_for(e["path"], e["sha256"]).read_bytes()
    rec = cn.decode_canonical(data, THREADS)
    a8 = cn.rec2020_to_srgb(rec, numthreads=THREADS)
    stored = np.array(Image.open(STORED / f"{e['id']}.png")) > 0
    m = {"A": seg.predict_mask(a8)}
    iou_pred = {"A": seg.pred_iou}
    a01 = a8.astype(np.float32) / 255
    m["B"] = mask_from_float(seg, a01)
    iou_pred["B"] = seg.pred_iou
    c01 = srgb_float(rec, cn.REC2020)
    diffs = {"C_vs_A": level_diff(c01, a01)}
    round_trip = int((np.rint(c01 * 255).astype(np.int16) != a8).any(axis=-1).sum())
    m["C"] = mask_from_float(seg, c01)
    iou_pred["C"] = seg.pred_iou
    lin = np.empty(rec.shape, np.float32)
    for y in range(0, rec.shape[0], 256):
        ic.cms_transform(rec[y:y + 256], cn.REC2020, cn.LINEAR_REC2020, intent=cn.RELATIVE, flags=cn.NOOPTIMIZE,
                         outdtype=np.float32, out=lin[y:y + 256])
    del rec
    lin16 = np.clip(lin, 0, None).astype(np.float16)
    del lin
    t1 = time.perf_counter()
    jxl = ic.jpegxl_encode(lin16, level=95, effort=7, primaries="BT2100", numthreads=THREADS)
    t_enc = time.perf_counter() - t1
    del lin16
    dec = ic.jpegxl_decode(jxl, numthreads=THREADS).astype(np.float32)
    d01 = srgb_float(dec, cn.LINEAR_REC2020)
    del dec
    diffs["D_vs_A"] = level_diff(d01, a01)
    diffs["D_vs_C"] = level_diff(d01, c01)
    m["D"] = mask_from_float(seg, d01)
    iou_pred["D"] = seg.pred_iou
    h, w = a8.shape[:2]
    boxes = {k: (d4_box(v) if v.any() else None) for k, v in m.items()}
    row = {"id": e["id"], "source": e["source"], "why": e["why"], "shape": [h, w],
           "canonical_bytes": len(data), "linear_f16_bytes": len(jxl), "jxl_encode_s": round(t_enc, 1),
           "area": {k: float(v.mean()) for k, v in m.items()} | {"S": float(stored.mean())},
           "iou_vs_A": {"S": iou(stored, m["A"]), **{k: iou(m[k], m["A"]) for k in "BCD"}},
           "iou_D_vs_C": iou(m["D"], m["C"]), "pred_iou": iou_pred,
           "box": boxes, "box_shift_px_max": {k: (None if boxes[k] is None or boxes["A"] is None else
                                                  int(np.abs(np.subtract(boxes[k], boxes["A"])).max())) for k in "BCD"},
           "C_rounds_to_A_except_px": round_trip, "diff": diffs, "seconds": round(time.perf_counter() - t0, 1)}
    del a8, a01, c01, d01, m, stored
    gc.collect()
    return row


def report(rows: list[dict]) -> str:
    out = ["ML-37 D14: xl0_v7r_s123's masks under ML-37's input changes. analysis/ml37/d14_mask_sensitivity.py,",
           f"{len(rows)} test photos of the segmenter dataset, torch at {THREADS} threads, box prompt, multi3.",
           "IoU against A (today: load_canonical uint8 -> predict_mask). S: the stored mask (4 threads).",
           "B: A's values, float resize. C: float read-back, no rounding. D: linear half-float lossy JXL re-encode.",
           "Level differences in 8-bit levels (x255): per pixel mean / p99 / max; 32x32 block means mean / max.", ""]
    head = (f"{'photo':<62}{'why':<10}{'area A':>8}{'S':>8}{'B':>8}{'C':>8}{'D':>8}{'D/C':>8}"
            f"{'boxΔD':>6}{'C-A px':>8}{'D-A px':>8}{'D-A p99':>8}{'D-A blk':>8}{'D-A bmax':>9}{'size D/now':>11}")
    out.append(head)
    for r in rows:
        i = r["iou_vs_A"]
        dd = r["diff"]["D_vs_A"]
        out.append(f"{r['id'][-60:]:<62}{r['why'][:9]:<10}{r['area']['A']:>8.4f}{i['S']:>8.4f}{i['B']:>8.4f}"
                   f"{i['C']:>8.4f}{i['D']:>8.4f}{r['iou_D_vs_C']:>8.4f}{str(r['box_shift_px_max']['D']):>6}"
                   f"{r['diff']['C_vs_A']['px_mean']:>8.3f}{dd['px_mean']:>8.3f}{dd['px_p99']:>8.2f}"
                   f"{dd['blk_mean']:>8.3f}{dd['blk_max']:>9.3f}{r['linear_f16_bytes'] / r['canonical_bytes']:>11.2f}")
    out.append("")
    pos = [r for r in rows if r["source"] != "negative"]
    neg = [r for r in rows if r["source"] == "negative"]
    for name, rs in (("positives", pos), ("negatives", neg)):
        if not rs:
            continue
        for k in ("S", "B", "C", "D"):
            v = np.array([r["iou_vs_A"][k] for r in rs])
            out.append(f"{name:<10} IoU {k} vs A: median {np.median(v):.4f}, min {v.min():.4f}, "
                       f"< 0.99: {int((v < 0.99).sum())}, < 0.9: {int((v < 0.9).sum())}")
        emp = Counter((r["area"]["A"] == 0, r["area"]["D"] == 0) for r in rs)
        out.append(f"{name:<10} empty (A, D): {dict(emp)}")
    for k in ("C_vs_A", "D_vs_A", "D_vs_C"):
        pm = np.array([r["diff"][k]["px_mean"] for r in rows])
        bm = np.array([r["diff"][k]["blk_mean"] for r in rows])
        bx = np.array([r["diff"][k]["blk_max"] for r in rows])
        out.append(f"levels {k}: per-pixel mean median {np.median(pm):.3f} (max {pm.max():.3f}); block-mean mean "
                   f"median {np.median(bm):.3f} (max {bm.max():.3f}); largest block {bx.max():.3f}")
    sz = np.array([r["linear_f16_bytes"] / r["canonical_bytes"] for r in rows])
    out.append(f"file size, linear half-float re-encode / today's canonical: median {np.median(sz):.2f} "
               f"({sz.min():.2f} to {sz.max():.2f}); encode at {THREADS} threads median "
               f"{np.median([r['jxl_encode_s'] for r in rows]):.1f} s")
    out.append(f"C rounded to 8 bits differs from A in at most {max(r['C_rounds_to_A_except_px'] for r in rows)} pixels "
               "of a photo (LittleCMS float vs 8-bit output)")
    return "\n".join(out) + "\n"


def main() -> None:
    torch.set_num_threads(THREADS)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows_path = SCRATCH / "rows.jsonl"
    done = {json.loads(r)["id"]: json.loads(r) for r in rows_path.read_text().splitlines()} if rows_path.exists() else {}
    photos = sample()
    seg = BeanSegmenter(named_params("xl0_v7r_s123"))
    rows = []
    for i, e in enumerate(photos, 1):
        if e["id"] in done:
            rows.append(done[e["id"]])
            continue
        r = one(seg, e)
        rows.append(r)
        with rows_path.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{i}/{len(photos)} {e['id']} IoU {r['iou_vs_A']} {r['seconds']} s", flush=True)
    OUT_TXT.write_text(report(rows))
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    sys.exit(main())
