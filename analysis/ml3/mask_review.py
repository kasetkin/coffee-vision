"""Ticket ML-3 P5 (D8): the owner's review of ft_s123's masks on the 288 new photos.

    python analysis/ml3/mask_review.py items      # writes labels/ml3/mask_review_items.csv
    python -m coffeecv.review_masks --items labels/ml3/mask_review_items.csv --name ml3_new_ft_s123
    python analysis/ml3/mask_review.py summary    # after (or during) the review
    python analysis/ml3/mask_review.py exclude    # writes labels/ml3/pool_exclude.csv (owner, 2026-10-06)

`items` orders the masks so that stopping early (D8: "as many as needed") never skips the risky ones: first all
of random_date_raccoon (the hopper photo and the only other settings, R4), then every other D18 fallback, then
the rest in a shuffle seeded with params.yaml's seg_split_seed and stratified by session, so any prefix covers
every session in proportion. The masks are data/seg_masks/ml3_new_ft_s123 (stage seg_predict_ml3_new), each
equal to the one segcrop cropped with.

A D18 fallback trains on its whole photo, not on its mask (which is empty or tiny). So for a fallback the
decision is about the frame: accept = beans fill it and every patch may train; decline = background would
train. `summary` counts fallbacks apart from the masks.

`exclude` turns the review into the photos the camera pools leave out (merge_rig --exclude), by the owner's
decision of 2026-10-06 (option b): every declined mask, and every D18 fallback that shows background. The
fallbacks whose beans fill the frame stay, though the review declined them (it judged their empty masks); the
background ones are BACKGROUND_FALLBACKS, read from a contact sheet of all 24 on 2026-10-05.

`summary` prints the accept rate overall and per session, each with its denominators (masks judged, masks in the
session), the declines with their reasons, and the fallbacks' decisions. Decisions are the last row per item
made for the item's current mask (review_masks.current_decisions).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from coffeecv.config import RunConfig  # noqa: E402

NAME = "ml3_new_ft_s123"
MASK_DIR = REPO_ROOT / "data" / "seg_masks" / NAME
ITEMS = REPO_ROOT / "labels" / "ml3" / "mask_review_items.csv"
DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / f"{NAME}.decisions.jsonl"
RISKY_SESSION = "random_date_raccoon"
POOL_EXCLUDE = REPO_ROOT / "labels" / "ml3" / "pool_exclude.csv"
BACKGROUND_FALLBACKS = {
    "dataset/2026-09-24__oneplus/class_011__Peru_Minca/PXL_20260924_174814080.jpg",       # white surface
    "dataset/2026-09-24__oneplus/class_012__Rwanda_Gisuma/PXL_20260924_103614873.jpg",    # metal bowl
    "dataset/2026-09-24__oneplus/class_012__Rwanda_Gisuma/PXL_20260924_103948127.jpg",    # plate
    "dataset/2026-09-24__oneplus/class_012__Rwanda_Gisuma/PXL_20260924_104055590.jpg",    # table
    "dataset/2026-09-24__pixel/class_012__Rwanda_Gisuma/PXL_20260924_055543146.jpg",      # blue surface
    "dataset/2026-09-24__sony/class_012__Rwanda_Gisuma/PIC_20260924_150406.JPG",          # plate on a table
    "dataset/random_date_raccoon/class_001__Ethiopia_Sidamo/PXL_20260927_103241427.jpg",  # the hopper ring
    "dataset/random_date_raccoon/class_014__Colombia_Excelso/PXL_20261003_061204827.jpg", # grinder hopper
}
FIELDS = ["item", "path", "mask", "mask_sha256", "photo_sha256", "session", "fallback", "area_frac", "why_here"]


def fallbacks() -> set[str]:
    """Repo-relative paths of the D18 fallbacks, from each session's segcrop_report.json."""
    out = set()
    for report in (REPO_ROOT / "data" / "segcropped").glob("*/class_*/segcrop_report.json"):
        session, class_dir = report.parts[-3], report.parts[-2]
        out |= {f"dataset/{session}/{class_dir}/{r['file']}" for r in json.loads(report.read_text()) if r["fallback"]}
    return out


def stratified(rows: list[dict], seed: int) -> list[dict]:
    """Each session shuffled with `seed`, then interleaved by position within its session ((k + 0.5) / n), so a
    prefix holds every session in proportion to its size."""
    rng = np.random.default_rng(seed)
    keyed = []
    for session in sorted({r["session"] for r in rows}):
        mine = [r for r in rows if r["session"] == session]
        order = rng.permutation(len(mine))
        keyed += [((k + 0.5) / len(mine), session, mine[j]) for k, j in enumerate(order)]
    return [r for _, _, r in sorted(keyed, key=lambda x: (x[0], x[1]))]


def items() -> int:
    seed = RunConfig.from_params_yaml().seg_split_seed
    fb = fallbacks()
    rows = []
    for r in csv.DictReader((MASK_DIR / "index.csv").read_text().splitlines()):
        rows.append({"item": r["id"], "path": r["path"], "mask": str((MASK_DIR / f"{r['id']}.png").relative_to(REPO_ROOT)),
                     "mask_sha256": r["mask_sha256"], "photo_sha256": r["photo_sha256"], "session": r["list"],
                     "fallback": int(r["path"] in fb), "area_frac": r["area_frac"]})
    first = [dict(r, why_here=RISKY_SESSION) for r in rows if r["session"] == RISKY_SESSION]
    second = [dict(r, why_here="D18 fallback") for r in rows if r["session"] != RISKY_SESSION and r["fallback"]]
    rest = [dict(r, why_here="stratified shuffle") for r in rows if r["session"] != RISKY_SESSION and not r["fallback"]]
    ordered = first + second + stratified(rest, seed)
    if len({r["item"] for r in ordered}) != len(rows):
        raise ValueError("items lost or repeated while ordering")
    with open(ITEMS, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(ordered)
    print(f"{len(ordered)} items -> {ITEMS.relative_to(REPO_ROOT)}: {len(first)} {RISKY_SESSION}, "
          f"{len(second)} other fallbacks, {len(rest)} stratified")
    return 0


def read_items() -> list[dict]:
    return list(csv.DictReader(ITEMS.read_text().splitlines()))     # as review_masks reads it: no comment lines


def summary() -> int:
    from coffeecv.review_masks import current_decisions
    its = read_items()
    dec = current_decisions(DECISIONS, its)
    by_session = Counter(it["session"] for it in its)

    def rate(subset: list[dict], label: str) -> None:
        judged = [it for it in subset if it["item"] in dec]
        acc = sum(dec[it["item"]]["decision"] == "accept" for it in judged)
        r = f"{acc / len(judged):.1%}" if judged else "-"
        print(f"  {label:<24} accepted {acc:>3} / judged {len(judged):>3} ({r:>6})   of {len(subset):>3} in it")

    masks = [it for it in its if it["fallback"] == "0"]
    print(f"{NAME}: {len(dec)} of {len(its)} judged ({DECISIONS.relative_to(REPO_ROOT)})")
    print("\nmasks (D18 fallbacks apart):")
    rate(masks, "all sessions")
    for s in sorted(by_session):
        rate([it for it in masks if it["session"] == s], s)
    fb = [it for it in its if it["fallback"] == "1"]
    print(f"\nD18 fallbacks (whole frame trains; accept = beans fill it): {len(fb)}")
    rate(fb, "all fallbacks")
    for it in fb:
        d = dec.get(it["item"])
        print(f"    {it['path']:<78} {d['decision'] if d else 'unjudged'}{' (' + d['reason'] + ')' if d and d.get('reason') else ''}")
    declines = [(it, dec[it["item"]]) for it in masks if it["item"] in dec and dec[it["item"]]["decision"] == "decline"]
    print(f"\nmask declines: {len(declines)}")
    for reason, n in Counter(d.get("reason") or "no reason" for _, d in declines).most_common():
        print(f"  {n:>3}  {reason}")
    for it, d in declines:
        print(f"    {it['path']:<78} {d.get('reason') or ''}")
    return 0


def exclusions() -> list[dict]:
    """The pool exclusion rows, from the committed review (see the module docstring)."""
    from coffeecv.review_masks import current_decisions
    its = read_items()
    dec = current_decisions(DECISIONS, its)
    if len(dec) != len(its):
        raise ValueError(f"{len(its) - len(dec)} items are not judged yet")
    fallback = {it["path"] for it in its if it["fallback"] == "1"}
    if not BACKGROUND_FALLBACKS <= fallback:
        raise ValueError(f"not D18 fallbacks: {sorted(BACKGROUND_FALLBACKS - fallback)}")
    rows = []
    for it in its:
        declined = dec[it["item"]]["decision"] == "decline"
        if it["path"] in BACKGROUND_FALLBACKS:
            reason = "D18 fallback with background"
        elif declined and it["path"] not in fallback:
            reason = "mask declined"
        else:
            continue
        _, session, class_folder, file = it["path"].split("/")
        rows.append({"session": session, "class_folder": class_folder, "file": file, "reason": reason})
    return sorted(rows, key=lambda r: (r["session"], r["class_folder"], r["file"]))


def exclude() -> int:
    rows = exclusions()
    with open(POOL_EXCLUDE, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["session", "class_folder", "file", "reason"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} photos -> {POOL_EXCLUDE.relative_to(REPO_ROOT)}: "
          + ", ".join(f"{n} {r}" for r, n in Counter(r["reason"] for r in rows).most_common()))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("items", "summary", "exclude"):
        sys.exit(__doc__)
    sys.exit({"items": items, "summary": summary, "exclude": exclude}[cmd]())
