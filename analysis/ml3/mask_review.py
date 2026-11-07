"""Ticket ML-3 P5 (D8): the owner's review of ft_s123's masks on the 288 new photos.

    python -m coffeecv.review_masks --items labels/ml3/mask_review_items.csv --name ml3_new_ft_s123
    python analysis/ml3/mask_review.py summary    # after (or during) the review
    python analysis/ml3/mask_review.py exclude    # writes labels/ml3/pool_exclude.csv (owner, 2026-10-06)

labels/ml3/mask_review_items.csv orders the masks so that stopping early (D8: "as many as needed") never skips
the risky ones: first all of random_date_raccoon (the hopper photo and the only other settings, R4), then every
other D18 fallback, then the rest stratified by session. The masks were data/seg_masks/ml3_new_ft_s123 (stage
seg_predict_ml3_new), each equal to the one segcrop cropped with; the `items` command that wrote the file went
with them on 2026-10-21 and is in git history.

A D18 fallback trains on its whole photo, not on its mask (which is empty or tiny). So for a fallback the
decision is about the frame: accept = beans fill it and every patch may train; decline = background would
train. `summary` counts fallbacks apart from the masks.

`exclude` turns the review into the photos the camera pools leave out (merge_rig --exclude), by the owner's
decision of 2026-10-06 (option b): every declined mask, and every D18 fallback that shows background. The
fallbacks whose beans fill the frame stay, though the review declined them (it judged their empty masks); the
background ones are BACKGROUND_FALLBACKS, read from a contact sheet of all 24 on 2026-10-05.

Since ticket ML-8 P4 the file also holds the owner's verdict on the 20 photos new to its re-crop
(labels/ml2/review/ml8_p4_new_crops.decisions.jsonl, items from analysis/ml8/p4_review_items.py): there a
D18 fallback was judged on its frame, so every decline is excluded, a fallback's as "D18 fallback with
background" (ML8_FALLBACKS: the declines that were fallbacks then). Its last decision per photo counts, made on the
XL0-v3 crop the photo had then: like ML-3's, a decision holds for its photo across re-crops (ML-9 P5 re-cropped
every session with XL0-v4 and changed nearly every mask).

Since ticket ML-9 P5 it also holds the owner's declines among the crops of XL0-v4 (2026-10-23, from the sheets
of the new Brazil photos and of the masks the re-crop shrank; labels/ml2/review/ml9_p5_crops.decisions.jsonl):
each row names its reason, a fallback's beginning "D18 fallback". An accept there overrides an earlier
review's decline of the same photo: two Brazil photos ML-8 P4 declined as XL0-v3 fallbacks train again on
their XL0-v4 masks (owner, 2026-10-23).

Since 2026-10-24 it also holds the owner's re-review of the 27 photos the file then excluded, each on its XL0-v4 mask
(labels/ml2/review/pool_exclude_rereview.decisions.jsonl, items from analysis/ml3/pool_exclude_rereview_items.py):
an accept there readmits its photo, like ML-9 P5's; a decline keeps the photo out under its earlier reason.

Since 2026-10-25 it also holds the owner's review of the 18 Brazil photos new to random_date_raccoon's crops that day
(labels/ml2/review/new_crops_2026-10-25.decisions.jsonl, items from analysis/ml3/new_crops_items.py): none is a
fallback, and every decline is "mask declined". The owner's words for the 6: their masks are not good enough, but the
photos are all good ones; the owner moved them to dataset/segmenter_positives the same day (MOVED_OUT).

Since 2026-11-04 it also holds the owner's review of the 54 photos new to the crops of canonical photos (ticket ML-19
P6), the D18 fallbacks and P7's two Ethiopia photos wrong on every patch (labels/ml2/review/ml19_p6_new_crops.decisions.jsonl,
items from analysis/ml19/p6_review_items.py, 55): as in ML-8 P4, a fallback was judged on its frame, so every decline
is excluded, a fallback's as "D18 fallback with background" (ML19_FALLBACKS). The two KB2005 fallbacks were accepted
(no beans, an empty mask) and moved by the owner to dataset/ood_negatives: not coffee. The owner's 10 new Brazil
photos of the same day were judged after (ML19_NEW_DECISIONS, none a fallback): every decline is "mask declined".
Since 2026-11-07 it also holds the owner's review for ticket ML-16 of the re-crop with xl0_v7r_s123 (D8): the 87
photos added to random_date_raccoon on 2026-11-06, with that session's new fallbacks and P7's Ethiopia photo
(labels/ml2/review/ml16_new_raccoon_2026-11-06.decisions.jsonl, 88 items, from p6_review_items.py --session
random_date_raccoon), and the 6 fallbacks the re-crop made in 2026-10-21__new_green
(labels/ml2/review/ml16_new_green_fallbacks.decisions.jsonl): as in ML-19 P6, each fallback judged on its frame, every
decline excluded, a fallback's as "D18 fallback with background" (ML16_FALLBACKS).
A photo the owner moved out of its session leaves the file with it (MOVED_OUT): there is no crop left for merge_rig
to exclude, and it refuses a row that matches none. The decisions stay in the review files.

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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

NAME = "ml3_new_ft_s123"
ITEMS = REPO_ROOT / "labels" / "ml3" / "mask_review_items.csv"
DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / f"{NAME}.decisions.jsonl"
RISKY_SESSION = "random_date_raccoon"
POOL_EXCLUDE = REPO_ROOT / "labels" / "ml3" / "pool_exclude.csv"
STRIP_MANIFEST = REPO_ROOT / "labels" / "ml3" / "strip_manifest.csv"
ML8_DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / "ml8_p4_new_crops.decisions.jsonl"
ML8_ITEMS = 20
ML8_FALLBACKS = {
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261016_081726673.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261016_081741399.jpg",
    "dataset/random_date_raccoon/class_014__Colombia_Excelso/IMG_20261016_120539.jpg",
}
ML9_DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / "ml9_p5_crops.decisions.jsonl"
REREVIEW_DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / "pool_exclude_rereview.decisions.jsonl"
NEW_CROPS_DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / "new_crops_2026-10-25.decisions.jsonl"
NEW_CROPS_ITEMS = 18
ML19_DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / "ml19_p6_new_crops.decisions.jsonl"
ML19_ITEMS = 55
# the owner's 10 new Brazil photos of 2026-11-04, judged after the first round (p6_review_items.py --new-only)
ML19_NEW_DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / "ml19_p6_new_crops_2026-11-04.decisions.jsonl"
ML19_NEW_ITEMS = 10
ML16_RACCOON_DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / "ml16_new_raccoon_2026-11-06.decisions.jsonl"
ML16_RACCOON_ITEMS = 88
ML16_GREEN_DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / "ml16_new_green_fallbacks.decisions.jsonl"
ML16_GREEN_ITEMS = 6
ML16_FALLBACKS = {   # the D18 fallbacks among ML-16's items, as segcrop recorded them with xl0_v7r_s123
    "dataset/random_date_raccoon/class_001__Ethiopia_Sidamo/20261105_121749_Pixel_9_Pro_Ethiopia.jpg",
    "dataset/random_date_raccoon/class_001__Ethiopia_Sidamo/20261105_122859_OnePlus_Ethiopia.jpg",
    "dataset/random_date_raccoon/class_001__Ethiopia_Sidamo/20261105_122929_OnePlus_Ethiopia.jpg",
    "dataset/random_date_raccoon/class_015__Indonesia/20261105_114349_OnePlus_Indonesia.jpg",
    "dataset/2026-10-21__new_green/class_011__Peru_Minca/PXL_20261021_090349561.jpg",
    "dataset/2026-10-21__new_green/class_011__Peru_Minca/PXL_20261021_090529833.jpg",
    "dataset/2026-10-21__new_green/class_011__Peru_Minca/PXL_20261021_090549814.jpg",
    "dataset/2026-10-21__new_green/class_011__Peru_Minca/PXL_20261021_090618448.jpg",
    "dataset/2026-10-21__new_green/class_011__Peru_Minca/PXL_20261021_090648983.jpg",
    "dataset/2026-10-21__new_green/class_015__Indonesia/PXL_20261021_085525122.jpg",
}
ML19_FALLBACKS = {   # the D18 fallbacks among ML-19 P6's items, as segcrop recorded them on the canonical photos
    "dataset/2026-08-30__pixel/class_010__Indonesia_Java/PXL_20260830_124324645.jpg",
    "dataset/random_date_raccoon/class_001__Ethiopia_Sidamo/20261103_110905_iPhone_13_Ethiopia.HEIC",
    "dataset/random_date_raccoon/class_014__Colombia_Excelso/20261030_113737_KB2005_Colombia.jpg",
    "dataset/random_date_raccoon/class_014__Colombia_Excelso/20261030_113926_KB2005_Colombia.jpg",
    "dataset/random_date_raccoon/class_014__Colombia_Excelso/20261103_123806_OnePlus_Colombia.jpg",
    "dataset/random_date_raccoon/class_014__Colombia_Excelso/20261103_150520_Galaxy_Z_Fold8_Ultra_Colombia.jpg",
}
# Moved to dataset/segmenter_positives by the owner on 2026-10-24: the empty or tiny XL0-v4 masks
# (coffeecv.find_tiny_masks); the two other photos moved had never been reviewed.
MOVED_OUT = {
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/20261023_125403.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/IMG_20261022_110423.jpg",
    "dataset/random_date_raccoon/class_014__Colombia_Excelso/IMG_20261016_120539.jpg",
    "dataset/random_date_raccoon/class_014__Colombia_Excelso/PXL_20261018_041127921.jpg",
    # The 6 the 2026-10-25 review declined, moved by the owner the same day: good photos, masks not good enough.
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/20261025_124114.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261025_074233345.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261025_074242228.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261025_074245705.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261025_112002683.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261025_112059290.jpg",
    # The 6 the file excluded that the segmenter dataset did not hold, moved by the owner the same day for labelling.
    "dataset/2026-09-24__oneplus/class_012__Rwanda_Gisuma/PXL_20260924_103114778.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/20261023_125142.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/20261023_125206.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/DSC_0001.JPG",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/IMG_9391.HEIC",
    "dataset/random_date_raccoon/class_014__Colombia_Excelso/PXL_20261018_041117629.jpg",
}
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
    rows = ml8_exclusions() + ml9_exclusions() + new_crops_exclusions() + ml19_exclusions() + ml16_exclusions()
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
    readmitted = {r["path"] for f in (ML9_DECISIONS, REREVIEW_DECISIONS)
                  for r in last_decisions(f).values() if r["decision"] == "accept"}
    rows = [r for r in rows if f"dataset/{r['session']}/{r['class_folder']}/{r['file']}" not in readmitted | MOVED_OUT]
    return sorted(present_names(rows), key=lambda r: (r["session"], r["class_folder"], r["file"]))


def present_names(rows: list[dict]) -> list[dict]:
    """`rows` with each photo under its present name: the reviews name a photo by the path it had then, and a
    photo the owner renamed since (bytes unchanged) is found by its sha256 (the owner, 2026-11-03: "use hash for
    images, not path"). labels/ml3/strip_manifest.csv gives the hash each path was stripped to, and the present
    path is the newest row with that hash in the photo's own session folder (a byte copy in another folder, such
    as dataset/segmenter_positives, is never taken for it). No photo is read, so this runs without the data."""
    lines = [line for line in STRIP_MANIFEST.read_text().splitlines() if not line.startswith("#")]
    sha_of, newest = {}, {}
    for r in csv.DictReader(lines):
        sha_of[r["path"]] = r["sha256_after"]
        parts = r["path"].split("/")
        if len(parts) > 2 and parts[0] == "dataset":
            newest[(parts[1], r["sha256_after"])] = r["path"]
    out = []
    for r in rows:
        path = f"dataset/{r['session']}/{r['class_folder']}/{r['file']}"
        now = newest.get((r["session"], sha_of.get(path)), path)
        _, session, class_folder, file = now.split("/")
        out.append({**r, "session": session, "class_folder": class_folder, "file": file})
    return out


def last_decisions(path: Path) -> dict[str, dict]:
    """{item: its last decision row} of a decisions file, whatever mask the photo has now."""
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["item"]] = r
    return out


def declined_rows(decisions: list[dict], is_fallback) -> list[dict]:
    rows = []
    for r in decisions:
        if r["decision"] != "decline":
            continue
        _, session, class_folder, file = r["path"].split("/")
        rows.append({"session": session, "class_folder": class_folder, "file": file,
                     "reason": "D18 fallback with background" if is_fallback(r) else "mask declined"})
    return rows


def ml8_exclusions() -> list[dict]:
    """Ticket ML-8 P4's declines (see the module docstring): every declined photo of the 20 new to its re-crop."""
    dec = last_decisions(ML8_DECISIONS)
    if len(dec) != ML8_ITEMS:
        raise ValueError(f"ML-8 P4: {len(dec)} items judged, not all {ML8_ITEMS}")
    declined = {r["path"] for r in dec.values() if r["decision"] == "decline"}
    if not ML8_FALLBACKS <= declined:
        raise ValueError(f"ML8_FALLBACKS not declined: {sorted(ML8_FALLBACKS - declined)}")
    return declined_rows(list(dec.values()), lambda r: r["path"] in ML8_FALLBACKS)


def ml9_exclusions() -> list[dict]:
    """Ticket ML-9 P5's declines (see the module docstring)."""
    return declined_rows(list(last_decisions(ML9_DECISIONS).values()), lambda r: r["reason"].startswith("D18 fallback"))


def new_crops_exclusions() -> list[dict]:
    """The declines of the 2026-10-25 review (see the module docstring): every item judged, no fallback among them."""
    dec = last_decisions(NEW_CROPS_DECISIONS)
    if len(dec) != NEW_CROPS_ITEMS:
        raise ValueError(f"2026-10-25: {len(dec)} items judged, not all {NEW_CROPS_ITEMS}")
    return declined_rows(list(dec.values()), lambda r: False)


def ml19_exclusions() -> list[dict]:
    """Ticket ML-19 P6's declines (see the module docstring): every item judged, each declined photo excluded."""
    dec = last_decisions(ML19_DECISIONS)
    if len(dec) != ML19_ITEMS:
        raise ValueError(f"ML-19 P6: {len(dec)} items judged, not all {ML19_ITEMS}")
    rows = declined_rows(list(dec.values()), lambda r: r["path"] in ML19_FALLBACKS)
    new = last_decisions(ML19_NEW_DECISIONS)
    if len(new) != ML19_NEW_ITEMS:
        raise ValueError(f"ML-19 P6, 2026-11-04: {len(new)} items judged, not all {ML19_NEW_ITEMS}")
    return rows + declined_rows(list(new.values()), lambda r: False)


def ml16_exclusions() -> list[dict]:
    """Ticket ML-16's declines (see the module docstring): every item of both reviews judged."""
    rows = []
    for path, n, what in ((ML16_RACCOON_DECISIONS, ML16_RACCOON_ITEMS, "the 87 new raccoon photos"),
                          (ML16_GREEN_DECISIONS, ML16_GREEN_ITEMS, "the new_green fallbacks")):
        dec = last_decisions(path)
        if len(dec) != n:
            raise ValueError(f"ML-16, {what}: {len(dec)} items judged, not all {n}")
        rows += declined_rows(list(dec.values()), lambda r: r["path"] in ML16_FALLBACKS)
    return rows


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
    if cmd not in ("summary", "exclude"):
        sys.exit(__doc__)
    sys.exit({"summary": summary, "exclude": exclude}[cmd]())
