"""Stage 1 of docs/lr_scheduler_plan.md: plateau arm A3 (exp210-213) vs cosine A0 (exp200-203), seed 42.

Prints the four pre-declared acceptance checks (plan section 4), the paired deltas with their SE, and how
closely the two arms track each other epoch by epoch; writes stage1_a3_vs_a0.png next to this file.
Run from anywhere: python analysis/lr_scheduler/stage1.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load, arr

PAIRS = [(210, 200), (211, 201), (212, 202), (213, 203)]   # (A3 plateau, A0 cosine): pixel, sony, oneplus, iphone
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1_a3_vs_a0.png")


def summarize(e):
    h, m = e["hist"], e["met"]
    x = arr(h, "xrig_macro_f1")
    events = [(r["epoch"], r["lr_event"]) for r in h if r.get("lr_event")]
    return dict(epochs=len(h), best_epoch=m["best_epoch"], x=x, lr_head=np.array([r.get("lr_head", r["lr"]) for r in h]),
                peak=m["splits"]["test_xrig"]["macro_f1"], last10=float(x[-10:].mean()), last=float(x[-1]),
                first_drop=next((ep for ep, ev in events if ev in ("drop", "floor")), None),
                floor=next((ep for ep, ev in events if ev == "floor"), None))


rows = []
print(f"{'fold':12s} {'arm':3s} {'epochs':>6s} {'best':>4s} {'1st drop':>8s} {'floor':>5s} | "
      f"{'xrig@peak':>9s} {'last10':>7s} {'last':>7s}")
for a3, a0 in PAIRS:
    e3, e0 = load(a3), load(a0)
    assert e3["cfg"]["scheduler"] == "plateau" and e0["cfg"].get("scheduler", "cosine") == "cosine", (a3, a0)
    assert e3["cfg"]["seed"] == e0["cfg"]["seed"] and e3["cfg"]["heldout_rig"] == e0["cfg"]["heldout_rig"], (a3, a0)
    s3, s0 = summarize(e3), summarize(e0)
    rig = os.path.basename(e3["cfg"]["heldout_rig"])
    for tag, s in (("A0", s0), ("A3", s3)):
        print(f"{rig:12s} {tag:3s} {s['epochs']:6d} {s['best_epoch']:4d} {str(s['first_drop']):>8s} {str(s['floor']):>5s} | "
              f"{s['peak']:9.4f} {s['last10']:7.4f} {s['last']:7.4f}")
    # Lockstep: same seed => same init, batch order and augmentation draws; only the head LR path and the stop differ.
    n = min(s3["epochs"], s0["epochs"])
    d = s3["x"][:n] - s0["x"][:n]
    r = dict(rig=rig, s0=s0, s3=s3, n=n, peak=s3["peak"] - s0["peak"], last10=s3["last10"] - s0["last10"],
             last=s3["last"] - s0["last"],
             move_corr=np.corrcoef(np.diff(s3["x"][10:n]), np.diff(s0["x"][10:n]))[0, 1],
             gap=np.abs(d[10:]).mean(), jitter=np.std(np.diff(s0["x"][30:])) / np.sqrt(2),
             lr_ratio=(s3["lr_head"][:n] / s0["lr_head"][:n]))
    rows.append(r)
    print(f"{'':12s} {'Δ':3s} {'':>6s} {'':>4s} {'':>8s} {'':>5s} | {r['peak']:+9.4f} {r['last10']:+7.4f} {r['last']:+7.4f}")

print()
for k, name in (("peak", "val-peak pick (deployed rule)"), ("last10", "mean of last 10 epochs"), ("last", "last epoch")):
    v = np.array([r[k] for r in rows])
    print(f"mean Δ {name:30s} {v.mean():+.4f}  SE {v.std(ddof=1) / np.sqrt(len(v)):.4f}  ({(v > 0).sum()}/{len(v)} positive)")

print("\nlockstep (epochs both arms ran, from epoch 11):")
for r in rows:
    print(f"  {r['rig']:12s} corr of epoch-to-epoch xrig moves {r['move_corr']:.2f}; mean |A3-A0| {r['gap']:.4f} "
          f"vs one run's own epoch jitter sd {r['jitter']:.4f}; head-LR ratio A3/A0 {r['lr_ratio'].min():.2f}-{r['lr_ratio'].max():.2f}")

ok = [(r["s3"]["first_drop"] is not None and 20 <= r["s3"]["first_drop"] <= 80,
       r["s3"]["floor"] is not None,
       60 <= r["s3"]["epochs"] <= 150,
       r["peak"] >= -0.06) for r in rows]
ok = list(zip(*ok))
print("\npre-declared checks (plan section 4):")
print(f"  1. first LR drop within epochs 20-80 in every fold ......... {'PASS' if all(ok[0]) else 'FAIL'}")
print(f"  2. floor reached before the cap in >=3 of 4 folds .......... {'PASS' if sum(ok[1]) >= 3 else 'FAIL'}")
print(f"  3. epochs used within 60-150 (so no stop before epoch 50) .. {'PASS' if all(ok[2]) else 'FAIL'}")
print(f"  4. no fold more than 0.06 below its A0 (val-peak pick) ..... {'PASS' if all(ok[3]) else 'FAIL'}")

# ---- figure: small multiples, one column per held-out rig; F1 and LR on separate axes (never a dual axis)
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a8983"
C0, C3 = "#2a78d6", "#eb6834"   # categorical slots 1-2 of the dataviz reference palette
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "axes.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), sharex="col", gridspec_kw=dict(height_ratios=[3, 1.6]))
lo = min(min(r["s0"]["x"][14:].min(), r["s3"]["x"][14:].min()) for r in rows) - 0.01
hi = max(max(r["s0"]["x"].max(), r["s3"]["x"].max()) for r in rows) + 0.01
for j, r in enumerate(rows):
    top, bot = axes[0, j], axes[1, j]
    for s, c, lab in ((r["s0"], C0, "A0 cosine"), (r["s3"], C3, "A3 plateau")):
        ep = np.arange(1, s["epochs"] + 1)
        top.plot(ep, s["x"], color=c, lw=1.3, label=lab, zorder=2)
        b = s["best_epoch"]
        top.scatter([b], [s["x"][b - 1]], s=46, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        bot.plot(ep, s["lr_head"], color=c, lw=1.3, drawstyle="steps-post" if lab.startswith("A3") else "default", zorder=2)
    ep = np.arange(1, max(r["s0"]["epochs"], r["s3"]["epochs"]) + 1)
    bot.plot(ep, np.full(len(ep), 1e-5), color=MUTED, lw=1.0, zorder=1)
    bot.text(3, 1.35e-5, "backbone LR 1e-5, both arms", color=INK2, fontsize=8)
    top.set_ylim(lo, hi)
    top.grid(True, color=GRID, lw=0.6)
    top.set_axisbelow(True)
    bot.set_yscale("log")
    bot.set_ylim(5e-6, 2e-3)
    bot.grid(True, color=GRID, lw=0.6)
    bot.set_axisbelow(True)
    bot.set_xlabel("epoch")
    top.set_title(f"held out {r['rig']}\n", fontsize=10, color=INK, loc="left", fontweight="bold")
    top.text(0, 1.015, f"Δ val-peak {r['peak']:+.3f}   Δ last-10 {r['last10']:+.3f}   move corr {r['move_corr']:.2f}",
             transform=top.transAxes, fontsize=8, color=INK2)
    if j == 0:
        top.set_ylabel("cross-rig macro-F1 (per epoch)")
        bot.set_ylabel("head LR")
h, labels = axes[0, 0].get_legend_handles_labels()
h.append(plt.Line2D([], [], marker="o", ls="", color=INK2, markeredgecolor=SURFACE, markersize=7))
labels.append("checkpoint the val-peak rule picks")
fig.legend(h, labels, loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.985))
fig.suptitle("LR scheduler Stage 1, seed 42: plateau (A3, exp210-213) vs cosine (A0, exp200-203)",
             x=0.01, y=0.985, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.935, "Same seed, so same init, batches and augmentation; only the head's LR path and the stop rule differ. "
         "The head is 5k of 11.2M parameters, the backbone LR is 1e-5 throughout in both arms, and the curves move together.",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.925))
fig.savefig(OUT, dpi=130)
print(f"\nwrote {OUT}")
