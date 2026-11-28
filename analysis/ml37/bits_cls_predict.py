"""ML-37 bits for the classifier, levels 2 and 3: how far each arm moves DINOv3's features and the live head's answers.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/bits_cls_predict.py

Reads bits_cls_embed.py's features of exp290's 1,600 test patches (266 photos) and bits_cls_noise.py's yardsticks.
Level 2: per arm and yardstick, 1 - cosine and relative L2 of each patch's cls_mean feature against the f32 arm's, and
the model input's RMS difference in 8-bit levels. Level 3: the live head (models/dino3b16_segxl0v7r_country_aug8_ml29gm_
s123.pt, exp290's; webapp/app.py's default checkpoint) applied to each arm's features as fit_frozen_head scores a split
(linear_head.predict): patches and photos (the mean of a photo's patch probabilities, as serving averages its
patches) whose answer differs from the f32 arm's, the change of the top-1 probability, and test macro-F1 and per-class
F1 per arm. The `today` arm is checked against exp290's own recorded test predictions (predictions_test.csv), which it
should reproduce. Writes analysis/ml37/bits_cls_predict.txt.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analysis.ml37 import bits_cls_common as bc
from analysis.ml37.bits_cls_check_boxes import EXP290
from coffeecv.class_list import load_classes
from coffeecv.config import REPO_ROOT
from coffeecv.linear_head import predict
from coffeecv.metrics import compute_split_metrics

OUT = Path(__file__).with_suffix(".txt")
CKPT = REPO_ROOT / "models" / "dino3b16_segxl0v7r_country_aug8_ml29gm_s123.pt"
SHOW = ("f16", "u12", "u10", "u8", "today", "fp16in", "shift1")


def camera(session: str, photo: str) -> str:
    s, p = session.lower(), photo.lower()
    for k in ("pixel", "sony", "iphone", "oneplus"):
        if k in s:
            return k
    for k, tag in (("pixel", "pxl_"), ("pixel", "pixel"), ("oneplus", "oneplus"), ("canon", "canon"),
                   ("iphone", "iphone"), ("iphone", "img_"), ("sony", "dsc"), ("sony", "pic_")):
        if tag in p:
            return k
    return "other"


def load_embed() -> dict:
    files = sorted((bc.SCRATCH / "embed").glob("*.npz"))
    acc = defaultdict(list)
    for f in files:
        z = np.load(f)
        n = len(z["f32"])
        for k in z["forms"]:
            acc[str(k)].append(z[str(k)])
        acc["photo"] += [str(z["photo"])] * n
        acc["cls"] += [str(z["cls"])] * n
        acc["cam"] += [camera(str(z["session"]), str(z["photo"]))] * n
        acc["rank"] += [int(f.stem)] * n
        acc["side"] += list(z["boxes"][:, 1] - z["boxes"][:, 0])
        acc["input_rms"].append(dict(zip([str(k) for k in z["forms"]], z["input_rms"])))
    out = {k: (np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)) for k, v in acc.items()
           if k != "input_rms"}
    out["input_rms"] = acc["input_rms"]
    out["n_photos"] = len(files)
    return out


def head():
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    w = ck["head"]["weight"]
    lin = torch.nn.Linear(w.shape[1], w.shape[0])
    lin.load_state_dict(ck["head"])
    return lin, list(ck["class_ids"])


def main() -> None:
    E = load_embed()
    classes = load_classes(REPO_ROOT / "dataset" / "classes.txt")
    keys = list(classes.keys)
    y = np.array([keys.index(c) for c in E["cls"]])
    n = len(y)
    L = [f"{n} test patches of {E['n_photos']} photos (exp290's test split, seed 123), "
         f"{len(set(E['cam']))} camera groups {sorted(set(E['cam']))}, 10 classes; dinov3_vitb16 cls_mean, float32",
         "", "== Level 2: features against the f32 arm ==",
         "form     1-cos median   p95          max          rel.L2 median  max        input RMS (8-bit levels, "
         "median over photos)"]
    ref = E["f32"]
    cosd = {}
    for k in SHOW:
        c, rl = bc.cos_l2(E[k], ref)
        cosd[k] = c
        rms = np.median([d[k] for d in E["input_rms"]])
        L.append(f"{k:8s} {np.median(c):12.3e} {np.percentile(c, 95):12.3e} {c.max():12.3e} {np.median(rl):12.3e} "
                 f"{rl.max():10.3e}  {rms:.4f}")
    # two different patches of the same photo
    pairs = [(i, i + 1) for i in range(n - 1) if E["photo"][i] == E["photo"][i + 1]]
    pc, _ = bc.cos_l2(ref[[i for i, _ in pairs]], ref[[j for _, j in pairs]])
    L.append(f"yardstick: two patches of the same photo (f32), 1-cos median {np.median(pc):.3e}, "
             f"min {pc.min():.3e} ({len(pairs)} pairs)")
    # noise-script yardsticks, on its photos
    nz = sorted((bc.SCRATCH / "noise").glob("*.npz"))
    if nz:
        rk = E["rank"]
        sub = defaultdict(list)
        for f in nz:
            z = np.load(f)
            idx = np.where(rk == int(z["rank"]))[0]
            for k in ("pre", "gen2", "front"):
                if k in z.files:
                    sub[k].append(bc.cos_l2(z[k], ref[idx])[0])
            for k in ("u8", "today", "f16", "shift1"):
                sub[k].append(cosd[k][idx])
            sub["rms_pre"].append(float(z["input_rms"][0]))
            sub["rms_gen2"].append(float(z["input_rms"][1]))
            if "front" in z.files:
                sub["rms_front"].append(float(z["input_rms"][3]))
        L.append(f"on bits_cls_noise.py's {len(nz)} photos ({sum(len(v) for v in sub['pre'])} patches), "
                 "1-cos against the stored photo's f32 arm, median / max:")
        for k, label in (("pre", "no lossy coding (the original's pre-encode picture)"),
                         ("gen2", "a second q95 e7 generation"), ("front", "the front's canonical (q90 e3)"),
                         ("u8", "u8 arm"), ("today", "today arm"),
                         ("f16", "f16 arm"), ("shift1", "1-pixel shift")):
            if sub[k]:
                v = np.concatenate(sub[k])
                L.append(f"  {label:52s} {np.median(v):.3e} / {v.max():.3e} ({len(v)} patches)")
        L.append(f"  input RMS, median over photos: no lossy coding {np.median(sub['rms_pre']):.3f} levels, second "
                 f"generation {np.median(sub['rms_gen2']):.3f} levels, front {np.median(sub['rms_front'] or [np.nan]):.3f} "
                 f"levels")
    L.append("")
    L.append("1-cos median per class (u8 / today / f16 / shift1):")
    for ci, k in enumerate(keys):
        m = y == ci
        L.append(f"  {k:10s} " + " ".join(f"{np.median(cosd[f][m]):.2e}" for f in ("u8", "today", "f16", "shift1")))
    L.append("by the box's side in crop pixels (the 448 px store upsamples a box under 448, so its rounding is not "
             "averaged away): 1-cos median / max of u8, u10, f16; |dp top1| max of u8 under the live head")
    side = E["side"]
    p_u8 = predict(head()[0], E["u8"])[1]
    p_32 = predict(head()[0], ref)[1]
    a32 = p_32.argmax(1)
    dtop = np.abs(p_u8[np.arange(n), a32] - p_32[np.arange(n), a32])
    for lo_s, hi_s in ((0, 300), (300, 448), (448, 700), (700, 1000), (1000, 10 ** 6)):
        m = (side >= lo_s) & (side < hi_s)
        if m.any():
            L.append(f"  side {lo_s:>4}-{min(hi_s, 9999):<4} n={m.sum():4d} " + " ".join(
                f"{f}: {np.median(cosd[f][m]):.2e} / {cosd[f][m].max():.2e}" for f in ("u8", "u10", "f16"))
                + f" | {dtop[m].max():.2e}")
    L.append("1-cos median per camera group (u8 / today / f16 / shift1):")
    for cam in sorted(set(E["cam"])):
        m = E["cam"] == cam
        L.append(f"  {cam:8s} n={m.sum():4d} " + " ".join(f"{np.median(cosd[f][m]):.2e}"
                                                         for f in ("u8", "today", "f16", "shift1")))

    # Level 3
    lin, ck_ids = head()
    assert ck_ids == keys, (ck_ids, keys)
    P = {k: predict(lin, E[k])[1] for k in ("f32", *SHOW)}
    exp = list(csv.DictReader(open(EXP290 / "predictions_test.csv")))
    pcols = [f"p_{k}" for k in keys]
    Pexp = np.array([[float(r[c]) for c in pcols] for r in exp])[:n]     # a prefix while the run is partial
    same_rows = [r["photo"] for r in exp][:n] == list(E["photo"])
    L += ["", "== Level 3: the live head (exp290, seed 123) on each arm ==",
          f"check: rows align with exp290's predictions_test.csv: {same_rows}; today arm vs exp290's recorded "
          f"probabilities: max |dp| {np.abs(P['today'] - Pexp).max():.2e}, answers equal "
          f"{int((P['today'].argmax(1) == Pexp.argmax(1)).sum())}/{n}; f32 arm vs recorded: answers equal "
          f"{int((P['f32'].argmax(1) == Pexp.argmax(1)).sum())}/{n}",
          "form     patches changed  |dp top1| median  p99       max     | photos changed (mean of patch probs) "
          "| test macro-F1 (patch)"]
    photos = sorted(set(E["photo"]))
    pidx = {p: np.where(E["photo"] == p)[0] for p in photos}

    def photo_ans(Pk):
        return np.array([Pk[pidx[p]].mean(0).argmax() for p in photos])

    ref_ans, ref_ph = P["f32"].argmax(1), photo_ans(P["f32"])
    top = P["f32"][np.arange(n), ref_ans]
    dummy = np.zeros(n)
    f1 = {}
    for k in ("f32", *SHOW):
        m = compute_split_metrics(y, P[k].argmax(1), dummy, keys, dict(classes.labels))
        f1[k] = m
        dp = np.abs(P[k][np.arange(n), ref_ans] - top)
        L.append(f"{k:8s} {int((P[k].argmax(1) != ref_ans).sum()):5d}/{n}     {np.median(dp):10.2e} "
                 f"{np.percentile(dp, 99):9.2e} {dp.max():9.2e} | {int((photo_ans(P[k]) != ref_ph).sum()):3d}/"
                 f"{len(photos)} | {m['macro_f1']:.4f}")
    m0 = compute_split_metrics(y, Pexp.argmax(1), dummy, keys, dict(classes.labels))
    L.append(f"(exp290 recorded: test macro-F1 {m0['macro_f1']:.4f})")
    L.append("per-class test F1 (f32 / f16 / u12 / u10 / u8 / today):")
    for k in keys:
        L.append(f"  {k:10s} " + " ".join(f"{f1[a]['per_class'][k]['f1']:.4f}"
                                           for a in ("f32", "f16", "u12", "u10", "u8", "today")))
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
