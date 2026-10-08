"""Ticket ML-2 P5: the segmenter's fine-tuning labels, built with the judge loop (D6, plan §8). An annotation
step, outside `dvc repro` (D16): it calls the judge.

Photos: seg_train + seg_val + pos_seg_train (D26). Each goes through at most three judged rounds:

    round 1   the pretrained mask: whole-image box, seg_mask_select output (D5, the serving prompt)
    round 2   on a decline, SAM is rerun with the whole-image box plus the judge's corrective points, single
              output (as for the D25 base masks), and judged again
    round 3   the same, with the points of rounds 1 and 2 together

A mask accepted in any round is the photo's label. One still declined after round 3, one the judge could not
answer (unjudged), or a decline that adds no new point is dropped, with the failed rules logged. Seg-train
negatives (neg_seg_train) get an empty label and are never judged (D11).

    python -m coffeecv.seg_labels run --workers 4            # resumable: each step skips what is done
    python -m coffeecv.seg_labels run --only ID [ID ...]     # a pilot on some photos
    python -m coffeecv.seg_labels status                     # yield by round, drop reasons, pending

A call that fails (plan limit, network) stores no verdict, so the photo stays pending and the next `run`
picks it up. Masks are computed once here and stored (mask bits differ across CPUs, P0); verdicts are keyed
by the mask's hash, so a recomputed mask would need judging again.

Out (dvc add data/seg_labels when complete):
    data/seg_labels/r<k>/<id>.png + index.csv    every round's masks (1-bit) and the points that made them
    data/seg_labels/neg/<id>.png                 the empty labels of the negatives
    data/seg_labels/labels.csv                   one row per photo: accepted / negative / dropped / pending
Verdicts and their points: labels/ml2/verdicts.jsonl (git, D16), `round` 1-3.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from coffeecv import seg_judge, seg_lists
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.seg_base_masks import item_id
from coffeecv.segment_beans import THREADS, BeanSegmenter, SegParams, mask_sha256

LABEL_ROOT = REPO_ROOT / "data" / "seg_labels"
LABELS_CSV = LABEL_ROOT / "labels.csv"
POSITIVE_LISTS = ("seg_train", "seg_val", "pos_seg_train")
NEGATIVE_LISTS = ("neg_seg_train",)
ROUNDS = (1, 2, 3)
CORRECTION_OUTPUT = "single"          # decoder output for box + points, as for the D25 base masks
INDEX_FIELDS = ["id", "list", "path", "photo_sha256", "mask_sha256", "height", "width", "area_frac", "pred_iou",
                "include", "exclude", "output", "threads", "weights_sha256"]
LABEL_FIELDS = ["id", "list", "path", "photo_sha256", "status", "round", "mask", "mask_sha256", "area_frac",
                "failed_rules", "reason"]


# ---------------------------------------------------------------- photos and stored rounds

def entries(lists: dict[str, list[dict]], names: tuple[str, ...]) -> list[dict]:
    """[{id, list, path, sha256}] over `names`, in list order."""
    out, seen = [], set()
    for name in names:
        for e in lists[name]:
            i = item_id(e)
            if i in seen:
                raise ValueError(f"item id {i} repeats")
            seen.add(i)
            out.append({"id": i, "list": name, "path": e["path"], "sha256": e["sha256"]})
    return out


def round_dir(r: int) -> Path:
    return LABEL_ROOT / f"r{r}"


def read_index(r: int) -> dict[str, dict]:
    f = round_dir(r) / "index.csv"
    if not f.exists():
        return {}
    return {row["id"]: row for row in csv.DictReader(f.read_text().splitlines())}


def write_index(r: int, index: dict[str, dict]) -> None:
    with open(round_dir(r) / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        wr.writeheader()
        wr.writerows(index[k] for k in sorted(index))


def judge_item(row: dict, r: int) -> dict:
    """An index row as a seg_judge item."""
    return {"item": row["id"], "path": row["path"], "photo_sha256": row["photo_sha256"],
            "mask": str((round_dir(r) / f"{row['id']}.png").relative_to(REPO_ROOT)),
            "mask_sha256": row["mask_sha256"], "trial": 0, "round": r}


def stored_verdicts() -> dict[tuple[str, str], dict]:
    """The latest verdict of each (photo_sha256, mask_sha256) by the pinned judge and prompt; failed calls
    are not verdicts."""
    psha = seg_judge.prompt_sha256(seg_judge.build_prompt())
    out = {}
    for v in seg_judge.read_verdicts():
        if (v["model"] == seg_judge.JUDGE_MODEL and v["backend"] == seg_judge.JUDGE_BACKEND
                and v["prompt_sha256"] == psha and v["trial"] == 0 and not seg_judge._call_failed(v)):
            out[(v["photo_sha256"], v["mask_sha256"])] = v
    return out


# ---------------------------------------------------------------- the loop's decisions (pure)

def points_so_far(verdicts: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """(include, exclude) as photo fractions, from every verdict of the earlier rounds, in order."""
    inc, exc = [], []
    for v in verdicts:
        for p in v.get("points_photo") or []:
            (inc if p["label"] == "include" else exc).append((round(p["x"], 4), round(p["y"], 4)))
    return inc, exc


def next_step(verdicts: list[dict | None]) -> tuple[str, str]:
    """What the loop does with a photo, given the verdicts of its rounds so far (None = round not judged yet).

    ("pending", why)   waits for a verdict
    ("accepted", "")   the last verdict accepts
    ("correct", "")    decline with new points and a round left: build the next round's mask
    ("dropped", why)   unjudged, declined in the last round, or declined with no new point
    """
    if not verdicts or verdicts[-1] is None:
        return "pending", f"round {len(verdicts) or 1} not judged"
    v = verdicts[-1]
    if v["verdict"] == "accept":
        return "accepted", ""
    if v["verdict"] != "decline":
        return "dropped", "unjudged"
    if len(verdicts) == len(ROUNDS):
        return "dropped", f"declined in round {len(verdicts)}"
    if not v.get("points_photo"):
        return "dropped", "declined with no corrective point"
    return "correct", ""


def photo_rounds(e: dict, indexes: dict[int, dict], verdicts: dict) -> tuple[list[dict], list[dict | None]]:
    """The photo's stored round rows and their verdicts, up to the first round with no mask."""
    rows, vs = [], []
    for r in ROUNDS:
        row = indexes[r].get(e["id"])
        if row is None:
            break
        rows.append(row)
        vs.append(verdicts.get((row["photo_sha256"], row["mask_sha256"])))
    return rows, vs


# ---------------------------------------------------------------- steps

class _Segmenter:
    """L0 loaded on first use, so `run` on a finished set loads nothing."""

    def __init__(self):
        self.seg = None

    def __call__(self) -> BeanSegmenter:
        if self.seg is None:
            cfg = RunConfig.from_params_yaml()
            torch.set_num_threads(THREADS)
            self.seg = BeanSegmenter(SegParams(mask_select=cfg.seg_mask_select, prompt=cfg.seg_prompt))
        return self.seg


def _photo(e: dict) -> np.ndarray:
    path = REPO_ROOT / e["path"]
    if seg_lists.sha256_file(path) != e["sha256"]:
        raise ValueError(f"{e['path']}: sha256 differs from photo_lists.yaml")
    return load_rgb_image(path)


def _save(r: int, e: dict, mask: np.ndarray, iou: float, inc: list, exc: list, output: str,
          seg: BeanSegmenter) -> dict:
    Image.fromarray(mask).convert("1").save(round_dir(r) / f"{e['id']}.png", optimize=True)
    return {"id": e["id"], "list": e["list"], "path": e["path"], "photo_sha256": e["sha256"],
            "mask_sha256": mask_sha256(mask), "height": mask.shape[0], "width": mask.shape[1],
            "area_frac": f"{mask.mean():.4f}", "pred_iou": f"{iou:.4f}", "include": json.dumps(inc),
            "exclude": json.dumps(exc), "output": output, "threads": torch.get_num_threads(),
            "weights_sha256": seg.weights_sha256}


def make_masks(r: int, todo: list[tuple[dict, list[dict | None]]], seg: _Segmenter) -> None:
    """Round r's mask for each (entry, verdicts of rounds 1..r-1)."""
    if not todo:
        return
    round_dir(r).mkdir(parents=True, exist_ok=True)
    index = read_index(r)
    for n, (e, vs) in enumerate(todo, 1):
        rgb = _photo(e)
        if r == 1:
            mask = seg().predict_mask(rgb)
            index[e["id"]] = _save(r, e, mask, seg().pred_iou, [], [], seg().p.mask_select, seg())
        else:
            inc, exc = points_so_far(vs)
            mask, iou = seg().predict_with_points(rgb, inc, exc, CORRECTION_OUTPUT)
            index[e["id"]] = _save(r, e, mask, iou, inc, exc, CORRECTION_OUTPUT, seg())
        if n % 25 == 0 or n == len(todo):
            print(f"  round {r} masks: {n}/{len(todo)}", flush=True)
            write_index(r, index)


def write_negatives(negs: list[dict]) -> dict[str, dict]:
    """Empty labels for the seg-train negatives (D11): a 1-bit PNG of the photo's size, all zero."""
    out_dir = LABEL_ROOT / "neg"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for e in negs:
        f = out_dir / f"{e['id']}.png"
        if f.exists():
            mask = np.array(Image.open(f)) > 0
        else:
            rgb = _photo(e)
            mask = np.zeros(rgb.shape[:2], bool)
            Image.fromarray(mask).convert("1").save(f, optimize=True)
        rows[e["id"]] = {"id": e["id"], "list": e["list"], "path": e["path"], "photo_sha256": e["sha256"],
                         "status": "negative", "round": "", "mask": str(f.relative_to(REPO_ROOT)),
                         "mask_sha256": mask_sha256(mask), "area_frac": "0.0000", "failed_rules": "",
                         "reason": "D11: negative, empty by definition"}
    return rows


def label_rows(pos: list[dict]) -> list[dict]:
    """One labels.csv row per positive photo, from the stored rounds and verdicts."""
    indexes = {r: read_index(r) for r in ROUNDS}
    verdicts = stored_verdicts()
    out = []
    for e in pos:
        rows, vs = photo_rounds(e, indexes, verdicts)
        status, why = next_step(vs)
        if status == "correct":
            status, why = "pending", f"round {len(vs) + 1} mask not made"
        last = rows[-1] if rows else None
        v = vs[-1] if vs else None
        out.append({"id": e["id"], "list": e["list"], "path": e["path"], "photo_sha256": e["sha256"],
                    "status": status, "round": len(rows) if rows else "",
                    "mask": str((round_dir(len(rows)) / f"{e['id']}.png").relative_to(REPO_ROOT))
                    if status == "accepted" else "",
                    "mask_sha256": last["mask_sha256"] if status == "accepted" else "",
                    "area_frac": last["area_frac"] if last else "",
                    "failed_rules": " ".join(map(str, v.get("failed_rules", []))) if v else "",
                    "reason": why or (v or {}).get("reason", "")})
    return out


def write_labels(pos: list[dict], negs: list[dict]) -> list[dict]:
    rows = label_rows(pos) + list(write_negatives(negs).values())
    LABEL_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LABELS_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    return rows


def _session(row: dict) -> str:
    return row["id"].split("__")[0] + ("__" + row["id"].split("__")[1] if row["id"].count("__") >= 3 else "")


def report(rows: list[dict]) -> dict:
    """Yield by round and list, drop reasons, the corrected masks that entered the set (plan §8 leads with it)."""
    pos = [r for r in rows if r["status"] != "negative"]
    acc = [r for r in pos if r["status"] == "accepted"]
    by_round = Counter(int(r["round"]) for r in acc)
    rep = {"photos": len(pos), "negatives": len(rows) - len(pos),
           "status": dict(Counter(r["status"] for r in pos)),
           "accepted_by_round": {f"r{k}": by_round.get(k, 0) for k in ROUNDS},
           "corrected_accepted": sum(v for k, v in by_round.items() if k > 1),
           "by_list": {}, "by_session": {},
           "dropped_reasons": dict(Counter(r["reason"] for r in pos if r["status"] == "dropped")),
           "dropped_failed_rules": dict(Counter(r["failed_rules"] for r in pos if r["status"] == "dropped"))}
    for key, fn in (("by_list", lambda r: r["list"]), ("by_session", _session)):
        groups = {}
        for r in pos:
            g = groups.setdefault(fn(r), Counter())
            g[r["status"] if r["status"] != "accepted" else f"accepted_r{r['round']}"] += 1
        rep[key] = {k: dict(sorted(v.items())) for k, v in sorted(groups.items())}
    return rep


def print_report(rep: dict) -> None:
    print(f"{rep['photos']} photos + {rep['negatives']} negatives (empty labels)")
    print(f"  status: {rep['status']}")
    print(f"  accepted by round: {rep['accepted_by_round']}  (corrected masks in the set: {rep['corrected_accepted']})")
    if rep["dropped_reasons"]:
        print(f"  dropped: {rep['dropped_reasons']}  failed rules: {rep['dropped_failed_rules']}")
    for s, c in rep["by_session"].items():
        print(f"    {s:48s} {c}")


def run(only: list[str] | None, workers: int, claude_bin: str, judge: bool) -> None:
    _, lists = seg_lists.load_lists()
    pos = entries(lists, POSITIVE_LISTS)
    negs = entries(lists, NEGATIVE_LISTS)
    if only:
        unknown = set(only) - {e["id"] for e in pos}
        if unknown:
            raise SystemExit(f"not in {POSITIVE_LISTS}: {sorted(unknown)}")
        pos = [e for e in pos if e["id"] in set(only)]
    seg = _Segmenter()
    for r in ROUNDS:
        indexes = {k: read_index(k) for k in ROUNDS}
        verdicts = stored_verdicts()
        todo = []
        for e in pos:
            if e["id"] in indexes[r]:
                continue
            _, vs = photo_rounds(e, indexes, verdicts)
            if r == 1 or (len(vs) == r - 1 and next_step(vs)[0] == "correct"):
                todo.append((e, vs))
        print(f"round {r}: {len(todo)} masks to make")
        make_masks(r, todo, seg)
        index = read_index(r)
        items = [judge_item(index[e["id"]], r) for e in pos if e["id"] in index]
        if judge and items:
            seg_judge.judge(items, seg_judge.JUDGE_MODEL, seg_judge.JUDGE_BACKEND, batch=False, rejudge=False,
                            dry_run=False, workers=workers, claude_bin=claude_bin, limit=None)
    rep = report(write_labels(pos, negs))
    print_report(rep)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "status"])
    ap.add_argument("--only", nargs="+", metavar="ID", help="run: only these photo ids (a pilot)")
    ap.add_argument("--workers", type=int, default=1, help="parallel judge calls")
    ap.add_argument("--no-judge", action="store_true", help="run: make the masks a step needs, send nothing")
    ap.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    ap.add_argument("--out", type=Path, help="status: also write the report as JSON")
    args = ap.parse_args(argv)
    if args.command == "run":
        run(args.only, args.workers, args.claude_bin, judge=not args.no_judge)
        return
    _, lists = seg_lists.load_lists()
    rep = report(write_labels(entries(lists, POSITIVE_LISTS), entries(lists, NEGATIVE_LISTS)))
    print_report(rep)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
