"""ML-37: does the choice between linear uint16 (I) and linear half-float (H) reach DINOv3's feature?

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/canon_int_dino.py [--n 12] [--threads 8]

The photos are bits_cls_noise.py's: --n photos of exp290's test split, one per session in turn, crop at most 16 MP.
For each, `pre` is the uint16 Rec.2020 (BT.709 curve) picture the store's encode hands libjxl today
(bits_cls_noise.pre_encode: the original decoded and tone-mapped at canonicalization); `lin` is the same picture in
linear Rec.2020 (LittleCMS, d8.u16_to_linear). The forms, all at the store's q95/e7:
  today     pre through jxl_encode.encode_rec2020 unchanged (intensity target 255)
  today203  pre, intensity target 203
  I203      rint(lin * 65535) as uint16, linear transfer, intensity target 203
  H203      lin as float16, linear transfer, intensity target 203 (the SDR content; light above white is not in
            these pictures, since `pre` is already tone-mapped)
  I255, H255  the same at 255
Each decoded file, and `lin` itself (the reference), is read back the one way (linear Rec.2020 -> matrix -> clip ->
sRGB curve, d8.read_back at peak 1, / 255), the crop's box cut, the bean mask filled as bits_cls_common.float_crop
does, and each test box made into the model's input with the 8-bit step once (bits_cls_common.model_input "u8":
rounded once at the crop, resizes in float). The reference also gives its float input ("ref f32"). DINOv3 ViT-B/16
cls_mean in float32 on CPU. Reported: 1 - cosine of each form's feature against the reference's u8 feature (the
same 8-bit step, no coding) and against the reference's f32 feature, median, p95 and max over the boxes; and I
against H directly. Writes analysis/ml37/canon_int_dino.txt.
"""
from __future__ import annotations

import argparse
import sys
import time

import imagecodecs as ic
import numpy as np
import torch

from analysis.ml37 import bits_cls_common as bc
from analysis.ml37 import canon_int_measure as cm
from analysis.ml37 import d8_intensity_target as d8
from analysis.ml37.bits_cls_check_boxes import exp290_test
from analysis.ml37.bits_cls_embed import embed
from analysis.ml37.bits_cls_noise import original_of, pick, pre_encode
from coffeecv import jxl_encode as je
from coffeecv.backbones import build_backbone
from coffeecv.dataset import load_bean_mask

FORMS = ("today", "today203", "I203", "H203", "I255", "H255")
Q, E = 95, 7


def encode(form: str, pre: np.ndarray, lin: np.ndarray, threads: int) -> bytes:
    if form == "today":
        return je.encode_rec2020(pre.shape[0], pre.shape[1], lambda y, x, ys, xs: pre[y:y + ys, x:x + xs],
                                 level=Q, effort=E, numthreads=threads, transfer="BT709", intensity_target=255)
    it = float(form[-3:])
    if form.startswith("today"):
        return d8.encode(pre, level=Q, effort=E, intensity_target=it, half=False)
    if form.startswith("I"):
        return cm.encode_linear_u16(np.rint(lin * 65535).astype(np.uint16), level=Q, effort=E, intensity_target=it)
    return d8.encode(lin.astype(np.float16), level=Q, effort=E, intensity_target=it, half=True)


def to_linear(form: str, got: np.ndarray) -> np.ndarray:
    if form.startswith("today"):
        return d8.u16_to_linear(got)
    if form.startswith("I"):
        return got.astype(np.float32) / 65535
    return got.astype(np.float32)


def crop_of(lin: np.ndarray, e: dict) -> tuple[np.ndarray, np.ndarray | None]:
    x = d8.read_back(bc.box_of(lin, e), 1.0) / 255
    mask = load_bean_mask(e["pool_path"], x.shape[:2])
    if mask is not None:
        x[~mask] = bc.FILL
    return x, mask


def stats(v: np.ndarray) -> str:
    return f"median {np.median(v):.2e}  p95 {np.percentile(v, 95):.2e}  max {v.max():.2e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    geo = bc.Geometry.from_params()
    entries = bc.pool_entries()
    chosen = pick(exp290_test(), entries, a.n, 16.0)
    bb = build_backbone("dinov3_vitb16")
    feats: dict[str, list[np.ndarray]] = {k: [] for k in ("ref_f32", "ref_u8", *FORMS)}
    t0 = time.time()
    for i, (_, t) in enumerate(chosen):
        e = entries[t["photo"]]
        pre = pre_encode(original_of(e))
        lin = d8.u16_to_linear(pre)
        x, mask = crop_of(lin, e)
        u8 = np.rint(x * 255).astype(np.uint8)
        boxes, _ = geo.boxes(u8, mask, 123, t["class_idx"], t["photo_idx"], "test", t["n"])
        u8 = None
        feats["ref_f32"].append(embed(bb, [bc.model_input(bc.cut(x, b), "f32") for b, _, _ in boxes]))
        feats["ref_u8"].append(embed(bb, [bc.model_input(bc.cut(x, b), "u8") for b, _, _ in boxes]))
        x = None
        sizes = []
        for form in FORMS:
            data = encode(form, pre, lin, a.threads)
            sizes.append(len(data))
            got = to_linear(form, ic.jpegxl_decode(data, numthreads=a.threads))
            data = None
            xf, _ = crop_of(got, e)
            got = None
            feats[form].append(embed(bb, [bc.model_input(bc.cut(xf, b), "u8") for b, _, _ in boxes]))
            xf = None
        pre = lin = None
        print(f"{i + 1}/{len(chosen)} {e['session']} {len(boxes)} boxes, bytes "
              + " ".join(f"{f}={s}" for f, s in zip(FORMS, sizes))
              + f"  {time.time() - t0:.0f}s rss {bc.rss_mb():.0f} MB", flush=True)
    f = {k: np.concatenate(v) for k, v in feats.items()}
    n = len(f["ref_f32"])
    print(f"\n{len(chosen)} photos, {n} test boxes; DINOv3 cls_mean, float32 on CPU, {a.threads} threads")
    print("\n1 - cosine against the uncoded picture's feature (u8: the same single 8-bit step; f32: no rounding)")
    print(f"  {'ref u8':9s} vs ref f32: {stats(bc.cos_l2(f['ref_u8'], f['ref_f32'])[0])}")
    for form in FORMS:
        print(f"  {form:9s} vs ref u8:  {stats(bc.cos_l2(f[form], f['ref_u8'])[0])}   "
              f"vs ref f32: {stats(bc.cos_l2(f[form], f['ref_f32'])[0])}")
    print("\n1 - cosine between forms (both coded, both rounded once)")
    for x, y in (("I203", "H203"), ("I255", "H255"), ("I203", "today203"), ("H203", "today203"),
                 ("I255", "today"), ("today203", "today")):
        print(f"  {x:9s} vs {y:9s}: {stats(bc.cos_l2(f[x], f[y])[0])}")
    print(f"\npeak RSS {bc.rss_mb():.0f} MB; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
