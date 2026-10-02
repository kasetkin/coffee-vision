"""Ticket ML-2 P2/P5: the `seg_eval@<model>` stage (plan §2.4, §5). Scores a seg_predict stage's stored masks
against the judge's stored verdicts. It never calls the judge and never recomputes a mask: a mask with no verdict
for the pinned judge (D15: Opus 5.5, the current prompt) stops the stage, naming every missing key (D16).

    python -m coffeecv.seg_eval pretrained

Reports, per list:
  seg_eval       judge pass rate with a photo-level bootstrap 95% CI, split green / roasted (not gated); the
                 distributions of mask_area_frac, bean_frac_in_crop and retained_frac; the declines' failed rules
  pos_seg_eval   the OOD positives (D26), on their own line
  neg_seg_eval   mask area per tag, pile-like pooled, the user_samerig batch on its own line; the share of
                 empty-or-tiny masks under the D18 threshold, with Wilson intervals (the D11 test's form)
  heuristic      judge-free (D22): on each judge-accepted seg_eval mask, the share of the bean region today's
                 heuristic rectangle discards and that rectangle's IoU with the D4 crop
  d18            the tiny-mask threshold in force (params.yaml seg_min_area_frac, owner 2026-10-01) and how many
                 photos it triggers on; the plan's rule (half the smallest judge-accepted genuine mask) beside it
  d19            beans across (the FFT estimator, unchanged) on the old crop (the cam_* pool JPEG training reads)
                 vs the new filled D4 crop, on judge-accepted seg_eval masks. Gated on beans_across, not pitch
                 in px, which moves with the crop's size (owner 2026-10-01). Pass: median |rel change| <= 5%.
                 The tail is not gated: it is reported next to a trim-only control, the old crop with the D4
                 crop's per-side cut removed and no mask or fill, which shows the estimator's own bin jitter.

Out: outputs/seg_eval_<model>.json (DVC metric) and outputs/ml2_p2/<model>/per_photo.csv, plus review item
lists for review_masks --items: declines.csv (every non-accepted genuine mask) and audit.csv (audit_sample).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv import seg_lists
from coffeecv.bean_scale import estimate_bean_pitch, pitch_kwargs
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.geometry import compute_valid_region_rect
from coffeecv.infer import grayscale_like_training
from coffeecv.seg_defects import box_iou
from coffeecv.seg_judge import JUDGE_MODEL, build_prompt, prompt_sha256, require_verdicts, wilson
from coffeecv.seg_predict import mask_items
from coffeecv.segment_beans import SegParams, mask_and_crop

OUT_DIR = REPO_ROOT / "outputs"
CROPPED_ROOT = REPO_ROOT / "data" / "cropped"
SAMERIG_BATCH = "2026-09-11__user_samerig"
N_BOOT = 10_000
BOOT_SEED = 239
D19_MEDIAN_MAX = 0.05
NEG_TEST_MIN = 0.90               # D11: share of pile-like neg_seg_eval empty or tiny, pooled
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


# ---------------------------------------------------------------- small statistics

def quantiles(values: list[float]) -> dict | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    q = np.quantile(np.asarray(v, dtype=float), QUANTILES)
    return {"n": len(v), **{f"q{int(p * 100):02d}": round(float(x), 4) for p, x in zip(QUANTILES, q)}}


def bootstrap_rate(passed: list[bool], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Pass rate with a photo-level percentile bootstrap 95% CI (plan §5)."""
    a = np.asarray(passed, dtype=float)
    if a.size == 0:
        return {"k": 0, "n": 0, "rate": None, "boot95": None}
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"k": int(a.sum()), "n": int(a.size), "rate": round(float(a.mean()), 4),
            "boot95": [round(float(lo), 4), round(float(hi), 4)]}


def wilson_rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)] if n else None}


# ---------------------------------------------------------------- inputs

def heuristic_box(entry: dict, h: int, w: int) -> list[int]:
    """Today's crop rectangle [x, y, w, h] for a pool photo, from its session's crop_report.json; the whole
    frame for a passthrough session. It is in the raw photo's pixel frame (no EXIF rotation), the frame
    load_rgb_image and the masks use: checked against every seg_eval crop JPEG on 2026-10-01."""
    raw = Path(entry["path"])
    class_key = raw.parent.name.split("__")[0]
    reports = list((CROPPED_ROOT / entry["session"]).glob(f"{class_key}__*/crop_report.json"))
    if len(reports) != 1:
        raise FileNotFoundError(f"expected one crop_report.json for {entry['session']}/{class_key}, got {reports}")
    row = next((r for r in json.loads(reports[0].read_text()) if r["file"] == raw.name), None)
    if row is None:
        raise KeyError(f"{raw.name} is not in {reports[0]}")
    box = row.get("box") or [0, 0, w, h]
    x, y, bw, bh = box
    if x < 0 or y < 0 or x + bw > w or y + bh > h:
        raise ValueError(f"{entry['path']}: heuristic box {box} outside the {w}x{h} frame")
    return list(box)


def load_mask(item: dict) -> np.ndarray:
    return np.array(Image.open(REPO_ROOT / item["mask"])) > 0


def pitch_and_across(rgb: np.ndarray, cfg: RunConfig) -> tuple[float, float]:
    """The estimator exactly as infer.patches_for_photo runs it: luma, then pitch; beans_across over the
    safety-margin region's short side."""
    pitch = estimate_bean_pitch(grayscale_like_training(rgb), **pitch_kwargs(cfg))
    region = compute_valid_region_rect(*rgb.shape[:2], cfg.safety_margin)
    return pitch, min(region.width, region.height) / pitch


# ---------------------------------------------------------------- the stage

def evaluate(model: str, skip_pitch: bool = False) -> dict:
    cfg = RunConfig.from_params_yaml()
    _, lists = seg_lists.load_lists()
    entries = {e["path"]: (name, e) for name in ("seg_eval", "neg_seg_eval", "pos_seg_eval") for e in lists[name]}
    items = mask_items(model)
    psha = prompt_sha256(build_prompt())
    verdicts = require_verdicts([(it["photo_sha256"], it["mask_sha256"]) for it in items], JUDGE_MODEL, psha)
    audit = {e["path"] for e in lists["audit_sample"]}
    p = SegParams(mask_select=cfg.seg_mask_select, prompt=cfg.seg_prompt, min_area_frac=cfg.seg_min_area_frac)

    rows = []
    for it in items:
        name, e = entries[it["path"]]
        if name != it["list"] or e["sha256"] != it["photo_sha256"]:
            raise ValueError(f"{it['item']}: index.csv disagrees with photo_lists.yaml")
        v = verdicts[(it["photo_sha256"], it["mask_sha256"])]
        mask = load_mask(it)
        blank = np.broadcast_to(np.zeros(3, np.uint8), (*mask.shape, 3))          # geometry only: no photo decode
        info = mask_and_crop(blank, mask, p).info
        row = {"item": it["item"], "list": name, "path": it["path"], "photo_sha256": it["photo_sha256"],
               "mask": it["mask"], "mask_sha256": it["mask_sha256"], "verdict": v["verdict"],
               "failed_rules": " ".join(map(str, v.get("failed_rules") or [])),
               "roast": e.get("roast", ""), "tag": e.get("tag", ""), "batch": e.get("batch", ""),
               "pile_like": e.get("pile_like", ""), "audit": it["path"] in audit,
               "mask_area_frac": info["mask_area_frac"], "bean_frac_in_crop": info["bean_frac_in_crop"],
               "retained_frac": info["retained_frac"], "d4_box": info["box"]}
        if name == "seg_eval":
            hb = heuristic_box(e, *mask.shape)
            row["heuristic_box"] = hb
            if v["verdict"] == "accept" and mask.any():
                x, y, bw, bh = hb
                row["heuristic_discard"] = 1 - int(np.count_nonzero(mask[y:y + bh, x:x + bw])) / int(mask.sum())
                row["heuristic_iou"] = box_iou(hb, info["box"])
        rows.append(row)

    by = defaultdict(list)
    for r in rows:
        by[r["list"]].append(r)
    acc = lambda rs: [r["verdict"] == "accept" for r in rs]           # noqa: E731  unjudged counts as a fail
    se = by["seg_eval"]
    report = {"model": model, "judge": {"model": JUDGE_MODEL, "prompt_sha256": psha},
              "seg_eval": {"pass": bootstrap_rate(acc(se)),
                           "by_roast": {k: bootstrap_rate(acc([r for r in se if r["roast"] == k]))
                                        for k in sorted({r["roast"] for r in se})},
                           "by_session": {k: bootstrap_rate(acc([r for r in se if r["path"].split("/")[1] == k]))
                                          for k in sorted({r["path"].split("/")[1] for r in se})},
                           "declined_rules": dict(Counter(x for r in se if r["verdict"] != "accept"
                                                          for x in r["failed_rules"].split())),
                           "unjudged": sum(r["verdict"] == "unjudged" for r in se)},
              "pos_seg_eval": {"pass": bootstrap_rate(acc(by["pos_seg_eval"]))}}
    for key, rs in (("seg_eval", se), ("seg_eval_accepted", [r for r in se if r["verdict"] == "accept"]),
                    ("pos_seg_eval", by["pos_seg_eval"])):
        report.setdefault("distributions", {})[key] = {
            f: quantiles([r[f] for r in rs]) for f in ("mask_area_frac", "bean_frac_in_crop", "retained_frac")}

    # D18: the threshold in force; the plan's rule (half the smallest judge-accepted genuine mask) is reported only.
    genuine_acc = [r["mask_area_frac"] for r in se + by["pos_seg_eval"] if r["verdict"] == "accept"]
    thr = p.min_area_frac
    tiny = lambda r: r["mask_area_frac"] < thr                                    # noqa: E731
    report["d18"] = {"min_area_frac": thr,
                     "triggers": {k: sum(tiny(r) for r in by[k]) for k in ("seg_eval", "pos_seg_eval", "neg_seg_eval")},
                     "plan_rule": round(0.5 * min(genuine_acc), 6) if genuine_acc else None,
                     "plan_rule_from_n_accepted": len(genuine_acc)}

    neg = by["neg_seg_eval"]
    pile = [r for r in neg if r["pile_like"] is True]
    report["neg_seg_eval"] = {
        "judge_pass": bootstrap_rate(acc(neg)),
        "empty_or_tiny_pile_like": wilson_rate(sum(tiny(r) for r in pile), len(pile)),
        "empty_or_tiny_samerig": wilson_rate(sum(tiny(r) for r in pile if r["batch"] == SAMERIG_BATCH),
                                             sum(r["batch"] == SAMERIG_BATCH for r in pile)),
        "area_by_tag": {t: quantiles([r["mask_area_frac"] for r in neg if r["tag"] == t])
                        for t in sorted({r["tag"] for r in neg})},
        "area_samerig_pile_like": quantiles([r["mask_area_frac"] for r in pile if r["batch"] == SAMERIG_BATCH])}

    hr = [r for r in se if "heuristic_iou" in r]
    report["heuristic"] = {"n": len(hr), "discard": quantiles([r["heuristic_discard"] for r in hr]),
                           "iou_with_d4": quantiles([r["heuristic_iou"] for r in hr])}

    if model != "pretrained":
        report["p5_gate"] = p5_gate(report, rows)
    if not skip_pitch:
        report["d19"] = pitch_check([r for r in se if r["verdict"] == "accept"], entries, p, cfg)
    write_outputs(model, report, rows)
    return report


def p5_gate(report: dict, rows: list[dict]) -> dict:
    """The P5 gate (plan §8): the seg_eval pass rate above the pretrained one beyond its bootstrap interval (the
    upper end of P2's 95% CI), and D11's negative test (>= 90% of pile-like neg_seg_eval empty or tiny). Also the
    paired bootstrap of the difference on the same photos, reported, not gated. Read next to P1's catch rate."""
    pre = json.loads((OUT_DIR / "seg_eval_pretrained.json").read_text())["seg_eval"]["pass"]
    pre_rows = {r["path"]: r["verdict"] == "accept"
                for r in csv.DictReader((OUT_DIR / "ml2_p2" / "pretrained" / "per_photo.csv").read_text().splitlines())
                if r["list"] == "seg_eval"}
    se = [r for r in rows if r["list"] == "seg_eval"]
    if {r["path"] for r in se} != set(pre_rows):
        raise ValueError("seg_eval photos differ from the pretrained run's; the comparison is not paired")
    diff = np.array([(r["verdict"] == "accept") - pre_rows[r["path"]] for r in se], dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    boots = diff[rng.integers(0, diff.size, size=(N_BOOT, diff.size))].mean(axis=1)
    ft = report["seg_eval"]["pass"]
    neg = report["neg_seg_eval"]["empty_or_tiny_pile_like"]
    pass_gate = ft["rate"] > pre["boot95"][1]
    neg_gate = neg["n"] > 0 and neg["rate"] >= NEG_TEST_MIN
    return {"pretrained_pass": pre, "pass_rate": {"pass": bool(pass_gate), "rate": ft["rate"],
                                                  "must_exceed": pre["boot95"][1]},
            "paired_diff": {"mean": round(float(diff.mean()), 4),
                            "boot95": [round(float(x), 4) for x in np.quantile(boots, [0.025, 0.975])],
                            "gained": int((diff > 0).sum()), "lost": int((diff < 0).sum())},
            "negative_test": {"pass": bool(neg_gate), **neg, "min": NEG_TEST_MIN},
            "pass": bool(pass_gate and neg_gate)}


def pitch_check(rows: list[dict], entries: dict, p: SegParams, cfg: RunConfig) -> dict:
    """D19 on judge-accepted masks: the old crop is the pool JPEG training reads; the new one is the filled D4
    crop of the raw photo (mask_and_crop, the function P3 calls at both ends). The control trims the old crop
    by the D4 crop's per-side cut, (1 - keep_frac) / 4 of each side, with no mask or fill."""
    trim = (1.0 - p.keep_frac) / 4
    rel, rel_across, rel_control = [], [], []
    for n, r in enumerate(rows, 1):
        _, e = entries[r["path"]]
        old = np.array(Image.open(REPO_ROOT / e["crop"]).convert("RGB"))
        new = mask_and_crop(load_rgb_image(REPO_ROOT / r["path"]), load_mask(r), p).rgb
        h, w = old.shape[:2]
        dy, dx = round(h * trim), round(w * trim)
        (po, ao), (pn, an) = pitch_and_across(old, cfg), pitch_and_across(new, cfg)
        _, ac = pitch_and_across(old[dy:h - dy, dx:w - dx], cfg)
        r.update(pitch_old=po, pitch_new=pn, beans_across_old=ao, beans_across_new=an, beans_across_control=ac)
        rel.append(abs(pn - po) / po)
        rel_across.append(abs(an - ao) / ao)
        rel_control.append(abs(ac - ao) / ao)
        if n % 25 == 0 or n == len(rows):
            print(f"  d19 {n}/{len(rows)}", flush=True)
    if not rel:
        return {"n": 0}
    return {"n": len(rel), "beans_across_abs_rel_change": quantiles(rel_across),
            "control_beans_across_abs_rel_change": quantiles(rel_control),
            "pitch_abs_rel_change": quantiles(rel),
            "signed_pitch_rel_change_median": round(float(np.median(
                [(r["pitch_new"] - r["pitch_old"]) / r["pitch_old"] for r in rows])), 4),
            "control_trim_per_side": trim,
            "pass": bool(np.median(rel_across) <= D19_MEDIAN_MAX),
            "gate": {"measure": "beans_across", "median_max": D19_MEDIAN_MAX}}


def write_outputs(model: str, report: dict, rows: list[dict]) -> None:
    (OUT_DIR / f"seg_eval_{model}.json").write_text(json.dumps(report, indent=2) + "\n")
    out = OUT_DIR / "ml2_p2" / model
    out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "per_photo.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    review_fields = ("item", "path", "photo_sha256", "mask", "mask_sha256")
    for name, keep in (("declines", lambda r: r["list"] != "neg_seg_eval" and r["verdict"] != "accept"),
                       ("audit", lambda r: r["audit"])):
        with open(out / f"{name}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=review_fields)
            wr.writeheader()
            wr.writerows({k: r[k] for k in review_fields} for r in rows if keep(r))


def summary(rep: dict) -> str:
    def rate(d):
        if not d.get("n"):
            return "n/a"
        ci = d.get("boot95") or d.get("wilson95")
        return f"{d['k']}/{d['n']} = {100 * d['rate']:.1f}% [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]"
    se = rep["seg_eval"]
    lines = [f"seg_eval@{rep['model']} (judge {rep['judge']['model']}, prompt {rep['judge']['prompt_sha256'][:12]})",
             f"  seg_eval pass       {rate(se['pass'])}",
             *(f"    {k:16s}  {rate(v)}" for k, v in se["by_roast"].items()),
             f"  pos_seg_eval pass   {rate(rep['pos_seg_eval']['pass'])}",
             f"  declined rules      {se['declined_rules']}",
             f"  D18 min_area_frac   {rep['d18']['min_area_frac']}  triggers {rep['d18']['triggers']}",
             f"  neg empty/tiny      pile-like {rate(rep['neg_seg_eval']['empty_or_tiny_pile_like'])}; "
             f"samerig {rate(rep['neg_seg_eval']['empty_or_tiny_samerig'])}",
             f"  heuristic (n={rep['heuristic']['n']})   discard median "
             f"{(rep['heuristic']['discard'] or {}).get('q50')}, IoU with D4 median "
             f"{(rep['heuristic']['iou_with_d4'] or {}).get('q50')}"]
    if "p5_gate" in rep:
        g = rep["p5_gate"]
        lines.append(f"  P5 gate             {'PASS' if g['pass'] else 'FAIL'}: pass rate {g['pass_rate']['rate']} vs "
                     f"> {g['pass_rate']['must_exceed']} ({'ok' if g['pass_rate']['pass'] else 'no'}); negative test "
                     f"{g['negative_test']['rate']} vs >= {g['negative_test']['min']} "
                     f"({'ok' if g['negative_test']['pass'] else 'no'}); paired diff {g['paired_diff']['mean']} "
                     f"{g['paired_diff']['boot95']} (+{g['paired_diff']['gained']} / -{g['paired_diff']['lost']})")
    if "d19" in rep and rep["d19"].get("n"):
        d = rep["d19"]
        a, c = d["beans_across_abs_rel_change"], d["control_beans_across_abs_rel_change"]
        lines.append(f"  D19 across (n={d['n']})  |rel| median {a['q50']} -> {'pass' if d['pass'] else 'FAIL'}; "
                     f"p95 {a['q95']} (trim-only control {c['q95']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--skip-pitch", action="store_true", help="leave out D19 (it decodes every accepted photo)")
    args = ap.parse_args(argv)
    print(summary(evaluate(args.model, args.skip_pitch)))


if __name__ == "__main__":
    main()
