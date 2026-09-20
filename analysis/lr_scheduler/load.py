import json, glob, os, re
import numpy as np
EXP = "/workspace/experiments"
def exp_dir(n):
    m = glob.glob(f"{EXP}/exp{n}__*")
    assert len(m) == 1, (n, m)
    return m[0]
# last 30 valid experiments: exp155-173, 175, 200-209 (exp174 is VOID)
LAST30 = [n for n in list(range(155, 176)) + list(range(200, 210)) if n != 174]
REF = [146, 147, 148, 149, 150, 151, 152, 153]   # eta_min=0 references (e100p20)
def load(n):
    d = exp_dir(n)
    cfg = json.load(open(f"{d}/config.json"))
    hist = json.load(open(f"{d}/history.json"))
    met = json.load(open(f"{d}/metrics.json"))
    meta = json.load(open(f"{d}/meta.json"))
    return dict(n=n, dir=os.path.basename(d), cfg=cfg, hist=hist, met=met, meta=meta)
def arr(h, k):
    return np.array([r[k] for r in h if k in r], dtype=float)

WINDOW = list(range(154, 174)) + list(range(200, 210))          # 30 real runs
DUP_OF = {160: 157, 161: 158, 162: 159}                          # same training run, different extra-heldout eval
UNIQUE = [n for n in WINDOW if n not in DUP_OF]                  # 27 unique train/val curves
def kind(e):
    return "fold" if e["cfg"].get("heldout_rig") else "allrigs"
def summarize(e):
    h = e["hist"]
    tl, vl, vf = arr(h, "train_loss"), arr(h, "val_loss"), arr(h, "val_macro_f1")
    lr = arr(h, "lr")
    n = len(h)
    bf = int(np.argmax(vf)); bl = int(np.argmin(vl))
    return dict(n=n, best_f1_ep=bf+1, best_f1=vf.max(), minvl_ep=bl+1, minvl=vl.min(),
                vl_end=vl[-1], tl_end=tl[-1], tl_min=tl.min(), tl0=tl[0], vl0=vl[0],
                lr_at_bestf1=lr[bf], lr_at_minvl=lr[bl], lr_end=lr[-1],
                vl_last10=vl[-10:].mean(), tl_last10=tl[-10:].mean(),
                vf_last10_std=vf[-10:].std(), vf_last10_mean=vf[-10:].mean())
