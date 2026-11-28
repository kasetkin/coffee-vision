"""ML-37, bits for the segmenter: the tables of levels 1-3 from bits_seg_measure.py's rows.

Reads tmp/ml37/bits_seg/{validation,test}.jsonl and prints the embedding change (level 1), the mask and crop-box
change against the f32 arm (level 2) and, on the labelled validation photos, seg_finetune.evaluate's measures against
the owner's pass17 labels with paired photo-bootstrap 95% CIs (level 3). Run:

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/bits_seg_report.py

Out: analysis/ml37/bits_seg_report.txt.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import bits_seg_common as sc

OUT = Path(__file__).with_suffix(".txt")
FORMS = ("f16", "u12", "u10", "u8", "today", "fp16in", "shift1", "gen2")
ARMS = ("f32", "f16", "u12", "u10", "u8", "today")
B = 10000
L: list[str] = []


def say(s: str = "") -> None:
    L.append(s)


def load(split: str) -> list[dict]:
    p = sc.SCRATCH / f"{split}.jsonl"
    rows = [json.loads(r) for r in p.read_text().splitlines()] if p.exists() else []
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def g(x: float) -> str:
    return f"{x:.1e}" if x != 0 else "0"


def pos(r: dict) -> bool:
    return r["source"] != "negative"


def served(r: dict) -> bool:
    """The f32 arm's mask is used for the crop: non-empty and at least seg_min_area_frac (no D18 fallback)."""
    a = r["mask"]["f32"]["area"]
    return a > 0 and a >= 0.028521


def describe(name: str, rows: list[dict]) -> None:
    say(f"{name}: {len(rows)} photos, {sum(pos(r) for r in rows)} positives, "
        f"{sum(not pos(r) for r in rows)} negatives; camera groups {dict(Counter(r['camera'] for r in rows))}")
    mp = np.array([r["shape"][0] * r["shape"][1] / 1e6 for r in rows])
    if len(mp):
        say(f"  megapixels median {np.median(mp):.1f} ({mp.min():.2f}-{mp.max():.1f}); seconds a photo median "
            f"{np.median([r['seconds'] for r in rows]):.1f}; peak rss {max(r['rss_mb'] for r in rows)} MB")


def level1(rows: list[dict], title: str) -> None:
    say(f"Level 1, {title}: the encoder's embedding (1 x 256 x 64 x 64) against the f32 arm's")
    say(f"{'form':<8}{'n':>5}{'1-cos med':>11}{'p95':>9}{'max':>9}{'relL2 med':>11}{'max':>9}"
        f"{'loc 1-cos max: med':>20}{'max':>9}")
    for f in FORMS:
        v = [r["feat"][f] for r in rows if f in r["feat"]]
        if not v:
            continue
        c = np.array([x["cos"] for x in v])
        rl = np.array([x["rel"] for x in v])
        lm = np.array([x["loc_max"] for x in v])
        say(f"{f:<8}{len(v):>5}{g(np.median(c)):>11}{g(np.percentile(c, 95)):>9}{g(c.max()):>9}"
            f"{g(np.median(rl)):>11}{g(rl.max()):>9}{g(np.median(lm)):>20}{g(lm.max()):>9}")
    both = [r for r in rows if "fp16in" in r["feat"]]
    if both:
        for a, b in (("u8", "f16"), ("u8", "fp16in"), ("u10", "fp16in"), ("u8", "gen2"), ("u8", "shift1"),
                     ("today", "u8")):
            if all(b in r["feat"] for r in both):
                ratio = np.array([r["feat"][a]["cos"] / max(r["feat"][b]["cos"], 1e-300) for r in both])
                say(f"  per-photo ratio of 1-cos, {a} / {b}: median {np.median(ratio):.3g} "
                    f"(min {ratio.min():.3g}, max {ratio.max():.3g})")
    say("  1-cos median per camera group (u10 / u8 / today / fp16in / shift1):")
    for cam, n in sorted(Counter(r["camera"] for r in rows).items()):
        rs = [r for r in rows if r["camera"] == cam]
        vals = [g(float(np.median([r["feat"][f]["cos"] for r in rs if f in r["feat"]])))
                if any(f in r["feat"] for r in rs) else "-" for f in ("u10", "u8", "today", "fp16in", "shift1")]
        say(f"    {cam:<9}{n:>4}  " + " / ".join(vals))
    say()


def level2(rows: list[dict], title: str) -> None:
    say(f"Level 2, {title}: the full-resolution mask and the D4 crop box against the f32 arm's. Groups: pos and neg,"
        " positives and negatives whose f32 mask is served (non-empty, >= seg_min_area_frac); fb, photos whose f32"
        " mask falls back (empty or tiny: the mask is not used, only a flip matters)")
    say(f"{'form':<8}{'group':<6}{'n':>5}{'IoU med':>9}{'mean':>9}{'min':>9}{'<0.999':>7}{'<0.99':>6}"
        f"{'chg px med':>11}{'p95':>8}{'max':>9}{'chg/area med':>13}{'empty flip':>11}{'fallback flip':>14}"
        f"{'box moved':>10}{'box move max':>13}")
    for f in FORMS:
        for grp, keep in (("pos", lambda r: pos(r) and served(r)), ("neg", lambda r: not pos(r) and served(r)),
                          ("fb", lambda r: not served(r))):
            rs = [r for r in rows if keep(r) and f in r["mask"]]
            if not rs:
                continue
            m = [r["mask"][f] for r in rs]
            i = np.array([x["iou"] for x in m])
            ch = np.array([x["changed"] for x in m])
            area = np.array([r["mask"]["f32"]["area"] * r["shape"][0] * r["shape"][1] for r in rs])
            rel = np.array([c / a for c, a in zip(ch, area) if a > 0])
            mv = [x["box_move"] for x in m if x["box_move"] is not None]
            say(f"{f:<8}{grp:<6}{len(rs):>5}{np.median(i):>9.5f}{i.mean():>9.5f}{i.min():>9.5f}"
                f"{int((i < 0.999).sum()):>7}{int((i < 0.99).sum()):>6}{np.median(ch):>11.0f}"
                f"{np.percentile(ch, 95):>8.0f}{ch.max():>9.0f}{(g(float(np.median(rel))) if len(rel) else '-'):>13}"
                f"{sum(x['empty_flip'] for x in m):>11}{sum(x['fallback_flip'] for x in m):>14}"
                f"{sum(1 for v in mv if v > 0):>6}/{len(mv):<3}{(max(mv) if mv else 0):>13}")
    say("  IoU against f32 per camera group, served masks, median / min (u10, u8, today, shift1):")
    for cam, n in sorted(Counter(r["camera"] for r in rows if served(r)).items()):
        rs = [r for r in rows if r["camera"] == cam and served(r)]
        parts = []
        for f in ("u10", "u8", "today", "shift1"):
            v = np.array([r["mask"][f]["iou"] for r in rs if f in r["mask"]])
            parts.append(f"{np.median(v):.5f}/{v.min():.4f}" if len(v) else "-")
        say(f"    {cam:<9}{n:>4}  " + "  ".join(parts))
    st = [r for r in rows if "stored_vs_today" in r]
    if st:
        i = np.array([r["stored_vs_today"]["iou"] for r in st])
        ch = np.array([r["stored_vs_today"]["changed"] for r in st])
        mv = [r["stored_vs_today"]["box_move"] for r in st if r["stored_vs_today"]["box_move"] is not None]
        say(f"  yardstick, thread count: the stored mask (seg_ml19_predict, 4 threads) against the today arm (8 threads),"
            f" {len(st)} photos: IoU median {np.median(i):.5f}, min {i.min():.5f}; changed px median {np.median(ch):.0f}, "
            f"max {ch.max()}; empty flips {sum(r['stored_vs_today']['empty_flip'] for r in st)}; box moved "
            f"{sum(1 for v in mv if v > 0)}/{len(mv)}, max {max(mv) if mv else 0} px")
    worst = sorted((r for r in rows if served(r)), key=lambda r: r["mask"]["u8"]["iou"])[:5]
    say("  lowest u8 IoU against f32 among served masks (photo, camera, f32 area, predicted IoU f32, IoU u10 / u8 / today / shift1):")
    for r in worst:
        say(f"    {r['id'][-60:]:<62}{r['camera']:<9}{r['mask']['f32']['area']:.4f}  {r['pred_iou']['f32']:.3f}  "
            + " / ".join(f"{r['mask'][f]['iou']:.5f}" for f in ("u10", "u8", "today", "shift1") if f in r["mask"]))
    say()


def boot_ci(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    idx = rng.integers(len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def level3(rows: list[dict]) -> None:
    rows = [r for r in rows if "label" in r]
    P = [r for r in rows if r["status"] == "accepted"]
    N = [r for r in rows if r["status"] == "negative"]
    say(f"Level 3, validation (pass17 labels): {len(P)} positives, {len(N)} negatives. seg_finetune.evaluate's measures:")
    say("  val_miou: mean hard IoU at the 1024 frame on positives; neg_empty: share of negatives empty or tiny "
        f"(< {sc.SELECT_MIN_AREA}); score = (val_miou + neg_empty) / 2. iou_full: mean IoU of the full-resolution "
        "serving mask against the full-resolution label.")
    forms = [f for f in FORMS if all(f in r["label"] for r in rows)]
    say(f"{'form':<8}{'val_miou':>10}{'iou_full':>10}{'neg_empty':>10}{'score':>9}{'median IoU':>11}"
        f"{'IoU<0.9':>8}{'neg serving fallback':>21}")
    for f in ("f32", *forms):
        vi = np.array([r["label"][f]["iou"] for r in P])
        vf = np.array([r["label"][f]["iou_full"] for r in P])
        ne = np.array([r["label"][f]["tiny"] for r in N], float)
        fb = sum((r["mask"][f]["area"] == 0 or r["mask"][f]["area"] < 0.028521) for r in N)
        say(f"{f:<8}{vi.mean():>10.5f}{vf.mean():>10.5f}{ne.mean():>10.4f}{(vi.mean() + ne.mean()) / 2:>9.5f}"
            f"{np.median(vi):>11.5f}{int((vi < 0.9).sum()):>8}{fb:>14}/{len(N)}")
    rng = np.random.default_rng(37016)
    say("  paired differences, photo bootstrap (positives resampled; 10,000 draws), mean [95% CI]:")
    for ref in ("f32", "f16"):
        for f in forms:
            if f == ref:
                continue
            d = np.array([r["label"][f]["iou"] - r["label"][ref]["iou"] for r in P])
            df = np.array([r["label"][f]["iou_full"] - r["label"][ref]["iou_full"] for r in P])
            m, lo, hi = boot_ci(d, rng)
            mf, lof, hif = boot_ci(df, rng)
            nd = np.array([float(r["label"][f]["tiny"]) - float(r["label"][ref]["tiny"]) for r in N])
            moved = int((np.abs(d) > 1e-3).sum())
            say(f"    {f:>6} - {ref:<4} val_miou {m:+.2e} [{lo:+.2e}, {hi:+.2e}]  iou_full {mf:+.2e} "
                f"[{lof:+.2e}, {hif:+.2e}]  |dIoU| max {np.abs(d).max():.1e}, > 1e-3 on {moved}/{len(P)}  "
                f"neg_empty change {nd.mean():+.4f} ({int(np.abs(nd).sum())} flips)")
    say("  per camera group (positives): n, val_miou f32, u8 - f32 and f16 - f32 mean [95% CI], today - f32 mean:")
    for cam, n in sorted(Counter(r["camera"] for r in P).items()):
        rs = [r for r in P if r["camera"] == cam]
        base = np.array([r["label"]["f32"]["iou"] for r in rs])
        parts = []
        for f in ("u8", "f16"):
            d = np.array([r["label"][f]["iou"] - r["label"]["f32"]["iou"] for r in rs])
            m, lo, hi = boot_ci(d, rng)
            parts.append(f"{f} {m:+.1e} [{lo:+.1e}, {hi:+.1e}]")
        dt = np.mean([r["label"]["today"]["iou"] - r["label"]["f32"]["iou"] for r in rs])
        say(f"    {cam:<9}{n:>4}  {base.mean():.4f}  " + "  ".join(parts) + f"  today {dt:+.1e}")
    say("  per source (positives): n, val_miou f32, u8 - f32 mean [95% CI]:")
    for s, n in sorted(Counter(r["source"] for r in P).items()):
        rs = [r for r in P if r["source"] == s]
        d = np.array([r["label"]["u8"]["iou"] - r["label"]["f32"]["iou"] for r in rs])
        m, lo, hi = boot_ci(d, rng)
        say(f"    {s:<20}{n:>4}  {np.mean([r['label']['f32']['iou'] for r in rs]):.4f}  {m:+.1e} [{lo:+.1e}, {hi:+.1e}]")
    say("  the label against itself as a scale: f32's IoU against the label, quartiles "
        + ", ".join(f"{q:.4f}" for q in np.percentile([r["label"]["f32"]["iou"] for r in P], [25, 50, 75])))
    say()


def main() -> None:
    val, test = load("validation"), load("test")
    say("ML-37, bits for the segmenter (bits_seg_measure.py rows; bits_seg_report.py). Live segmenter: XL0 encoder,")
    say("decoder xl0_v7r_s123, box prompt, multi3; torch CPU float32, 8 threads. Arms quantize the whole float photo at")
    say("load_canonical's 8-bit step; the float resize (PIL F) and the rest stay float32. today = load_canonical uint8 ->")
    say("model_io.sam_input. Yardsticks: fp16in (input cast to float16), shift1 (photo moved one pixel), gen2 (second")
    say("q95 e7 JPEG XL generation).")
    say()
    describe("validation", val)
    describe("test", test)
    say()
    for name, rows in (("validation", val), ("test", test), ("both sets", val + test)):
        if rows:
            level1(rows, name)
    for name, rows in (("validation", val), ("test", test)):
        if rows:
            level2(rows, name)
    if val:
        level3(val)
    OUT.write_text("\n".join(L) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
