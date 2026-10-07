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
  d18            the tiny-mask threshold in force (params.yaml seg_min_area_frac, owner 2026-10-01) and how many
                 photos it triggers on; the plan's rule (half the smallest judge-accepted genuine mask) beside it

Until ticket ML-3 P2b it also compared each mask with the tray heuristic's crops: the share of the bean region
the heuristic rectangle discards (D22) and beans across on the old vs the new crop (D19). Both read the retired
data/cropped pools; their results are ML-2's record, in outputs/seg_eval_{pretrained,ft_s123}.json as of
commit 8eef7f5 ("heuristic" and "d19").

Out: outputs/seg_eval_<model>.json (DVC metric) and outputs/ml2_p2/<model>/per_photo.csv, plus review item
lists for review_masks --items: declines.csv (every non-accepted genuine mask) and audit.csv (audit_sample).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from coffeecv import seg_lists
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.seg_judge import JUDGE_MODEL, build_prompt, prompt_sha256, require_verdicts, wilson
from coffeecv.seg_predict import mask_items
from coffeecv.segment_beans import SegParams, mask_and_crop

OUT_DIR = REPO_ROOT / "outputs"
SAMERIG_BATCH = "2026-09-11__user_samerig"
N_BOOT = 10_000
BOOT_SEED = 239
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

def load_mask(item: dict) -> np.ndarray:
    return np.array(Image.open(REPO_ROOT / item["mask"])) > 0


# ---------------------------------------------------------------- the stage

def evaluate(model: str) -> dict:
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

    if model != "pretrained":
        report["p5_gate"] = p5_gate(report, rows)
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
             f"samerig {rate(rep['neg_seg_eval']['empty_or_tiny_samerig'])}"]
    if "p5_gate" in rep:
        g = rep["p5_gate"]
        lines.append(f"  P5 gate             {'PASS' if g['pass'] else 'FAIL'}: pass rate {g['pass_rate']['rate']} vs "
                     f"> {g['pass_rate']['must_exceed']} ({'ok' if g['pass_rate']['pass'] else 'no'}); negative test "
                     f"{g['negative_test']['rate']} vs >= {g['negative_test']['min']} "
                     f"({'ok' if g['negative_test']['pass'] else 'no'}); paired diff {g['paired_diff']['mean']} "
                     f"{g['paired_diff']['boot95']} (+{g['paired_diff']['gained']} / -{g['paired_diff']['lost']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    args = ap.parse_args(argv)
    print(summary(evaluate(args.model)))


if __name__ == "__main__":
    main()
