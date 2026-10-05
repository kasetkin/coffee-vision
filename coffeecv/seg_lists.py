"""Ticket ML-2 P1: the segmenter's photo lists, one committed file (plan §4.1).

    python -m coffeecv.seg_lists build     # writes labels/ml2/photo_lists.yaml (refuses to move or drop entries)
    python -m coffeecv.seg_lists check     # structural checks + coverage against today's pools

Every list names raw photos, because the segmenter sees whole photos, never crops:

  seg_train / seg_val / seg_eval   the pooled 70/15/15 split at seg_split_seed (D21), 239, a seed no head
                                   fit uses. seg_eval is that split's val+test; seg_val is ~15% of its train
                                   split, stratified by session, and only selects fine-tune checkpoints (P5).
  base_candidates                  ~70 of seg_eval, stratified by (session, roast), >= 6 roasted (D12), then
                                   10 of pos_seg_eval, stratified by batch (D25). The owner accepts or
                                   declines their masks; 25 + 25 accepted ones become base_dev /
                                   base_heldout later, in the same file.
  audit_sample                     50 of seg_eval for the owner's blind P2 audit (D12), drawn now, before any
                                   P2 result. Disjoint from base_candidates: the owner has already seen those
                                   masks, so reviewing them again would not be blind.
  neg_seg_train / neg_seg_eval     OOD *dev* negatives only (D11), ~2/3 : 1/3 within each (batch, tag), so
                                   user_samerig keeps its own eval line. Near-duplicate shots form one group
                                   and never straddle the two sides. The holdout is never listed.
  pos_seg_train / pos_seg_eval     OOD *dev* positives (D25, owner 2026-09-30): other setups than the pools.
                                   ~2/3 : 1/3 within each (batch, tag), whole near-duplicate groups. A dev
                                   photo from a burst that also has a holdout photo is eval-only, never
                                   training. The holdout is never listed.

The file is append-only (D20): `build` rebuilds from the data and refuses to write if any committed entry
would change list, change hash or disappear. The eval lists only ever grow; no eval photo moves to training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from coffeecv.class_list import folder_classes
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import pooled_class_photos, resolve_captures, split_photos_by_class

LISTS_FILE = REPO_ROOT / "labels" / "ml2" / "photo_lists.yaml"
CROPPED_ROOT = REPO_ROOT / "data" / "cropped"
NEGATIVES_ROOT = REPO_ROOT / "dataset" / "ood_negatives"
POSITIVE_DIRS = (REPO_ROOT / "dataset" / "ood_positives", REPO_ROOT / "dataset" / "ood_positives_internet")

POSITIVE_LISTS = ("seg_train", "seg_val", "seg_eval")
NEGATIVE_LISTS = ("neg_seg_train", "neg_seg_eval")
OOD_POSITIVE_LISTS = ("pos_seg_train", "pos_seg_eval")
# The roasted photos in today's data: class_010 (Indonesia Java) in all three 2026-08-30 sessions, 126 photos,
# each session checked by eye. The ticket's D20 names only the pixel session (46).
ROASTED = {(s, "class_010") for s in ("2026-08-30__pixel", "2026-08-30__sony", "2026-08-30__oneplus")}
# D11 / plan F8: the tags whose photos show a pile, for the negative test. The rest (empty_tray,
# non_food_objects, real_world_negatives) are listed too, but are not in the gated pool.
PILE_LIKE_TAGS = frozenset({"confusable_grain", "other_nuts_seeds", "ground_coffee", "green_legume",
                            "wrong_bean_type"})

SEG_VAL_FRAC = 0.15
N_BASE_CANDIDATES = 70
MIN_ROASTED_CANDIDATES = 6
N_AUDIT = 50
NEG_EVAL_FRAC = 1 / 3
NEAR_DUP_SECONDS = 120
POS_EVAL_FRAC = 1 / 3
N_POS_BASE_CANDIDATES = 10
N_BASE_EACH = 25              # D14: base_dev (prompt development) and base_heldout (final check)
MIN_ROASTED_BASES = 6         # across both, so each side gets about 3
DECISIONS_FILE = REPO_ROOT / "labels" / "ml2" / "review" / "base_candidates.decisions.jsonl"
BASE_MASK_INDEX = REPO_ROOT / "data" / "seg_masks" / "base_points" / "index.csv"

# RNG streams under seg_split_seed; the split itself is split_photos_by_class's own stream.
_STREAM = {"seg_val": 1, "base_candidates": 2, "audit_sample": 3, "negatives": 4, "positives": 6,
           "pos_base_candidates": 7, "base_split": 8}      # 5 is seg_base_masks' pick
_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


# --- positives -------------------------------------------------------------------------------------

def crop_sessions() -> dict[tuple[str, str], str]:
    """(class_NNN, crop file name) -> session, from the per-session crop dirs the cam_* pools merge.
    Refuses a name two sessions share: the cam_* pool could not say which raw photo it came from."""
    index: dict[tuple[str, str], str] = {}
    for crop in sorted(CROPPED_ROOT.glob("2026-*/class_*/*__cropped.jpg")):
        key = (crop.parent.name.split("__")[0], crop.name)
        session = crop.parts[-3]
        if key in index:
            raise ValueError(f"{crop.name} of {key[0]} is in both {index[key]} and {session}")
        index[key] = session
    return index


def raw_photo(session: str, class_key: str, crop_name: str) -> Path:
    """The raw original of a crop. Matched on the class number: the label half of a class dir's name
    differs between a session's raw and cropped trees."""
    stem = crop_name.removesuffix("__cropped.jpg")
    hits = [p for p in (REPO_ROOT / "dataset" / session).glob(f"{class_key}__*/{stem}.*") if p.is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one raw photo for {session}/{class_key}__*/{stem}.*, found {hits}")
    return hits[0]


def split_basis(cfg: RunConfig) -> tuple[list[str], dict[str, float]]:
    """(capture dirs, photo fractions) ML-2's split is drawn over: the lists file's own meta once it exists,
    so a later change of params.yaml's pools (ticket ML-3) cannot move it; params.yaml only for a first build."""
    if LISTS_FILE.exists():
        meta, _ = load_lists()
        return list(meta["capture_dirs"]), dict(zip(("train", "val", "test"), meta["photo_frac"]))
    return list(cfg.train_capture_dirs), {"train": cfg.train_photo_frac, "val": cfg.val_photo_frac,
                                          "test": cfg.test_photo_frac}


def pooled_split(cfg: RunConfig) -> dict[str, list[dict]]:
    """{"train": [...], "eval": [...]} positive entries: the head fit's pooled split as ML-2 drew it, at
    seg_split_seed: one class per class folder (class_list.folder_classes), whatever classes.txt's format,
    so each folder keeps its class index. A folder with no photos in these pools is skipped (ML-3's new
    folders have no tray-heuristic crops); it sorts after every ML-2 folder, so no index moves."""
    capture_dirs, frac = split_basis(cfg)
    captures = resolve_captures([REPO_ROOT / d for d in capture_dirs])
    classes = folder_classes(REPO_ROOT / cfg.classes_file)
    sessions = crop_sessions()
    out: dict[str, list[dict]] = {"train": [], "eval": []}
    for class_idx, class_id in enumerate(classes.keys):
        pool, _ = pooled_class_photos(captures, classes.folders[class_id])
        if not pool:
            continue
        for split, photos in split_photos_by_class(pool, cfg.seg_split_seed, class_idx, frac).items():
            for ph in photos:
                class_key = ph.path.parent.name.split("__")[0]
                session = sessions[(class_key, ph.name)]
                raw = raw_photo(session, class_key, ph.name)
                out["train" if split == "train" else "eval"].append({
                    "path": rel(raw),
                    "sha256": sha256_file(raw),
                    "session": session,
                    "camera": ph.capture.removeprefix("cam_"),
                    "class": class_key,
                    "roast": "roasted" if (session, class_key) in ROASTED else "green",
                    "crop": rel(ph.path),
                })
    return out


def draw(entries: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    """k entries without replacement, returned in the input's (path) order."""
    idx = sorted(rng.choice(len(entries), size=k, replace=False)) if k else []
    return [entries[i] for i in idx]


def allocate(sizes: dict, total: int, floors: dict | None = None) -> dict:
    """Split `total` over strata in proportion to `sizes` (largest remainder), after giving each stratum in
    `floors` at least its floor. Deterministic: ties break on the stratum key."""
    floors = {k: min(v, sizes[k]) for k, v in (floors or {}).items()}
    n = sum(sizes.values())
    if total > n:
        raise ValueError(f"cannot draw {total} from {n}")
    exact = {k: total * s / n for k, s in sizes.items()}
    quota = {k: max(int(exact[k]), floors.get(k, 0)) for k in sizes}
    order = sorted(sizes, key=lambda k: (-(exact[k] - int(exact[k])), k))
    while sum(quota.values()) < total:
        k = next(k for k in order if quota[k] < sizes[k])
        quota[k] += 1
        order.remove(k)
        order.append(k)
    while sum(quota.values()) > total:        # floors pushed the sum over: take from the largest strata
        k = max((k for k in quota if quota[k] > floors.get(k, 0)), key=lambda k: (quota[k], k))
        quota[k] -= 1
    return quota


def stratified_draw(entries: list[dict], key, quota: dict, rng: np.random.Generator) -> list[dict]:
    strata: dict = defaultdict(list)
    for e in entries:
        strata[key(e)].append(e)
    picked = []
    for k in sorted(strata):
        picked += draw(strata[k], quota.get(k, 0), rng)
    return sorted(picked, key=lambda e: e["path"])


# --- negatives -------------------------------------------------------------------------------------

def dev_negatives() -> list[dict]:
    """Every dev-split row of every negative batch's manifest. The holdout rows are never read further."""
    rows = []
    for manifest in sorted(NEGATIVES_ROOT.glob("*.manifest.csv")):
        batch = manifest.name.removesuffix(".manifest.csv")
        for r in csv.DictReader(manifest.read_text().splitlines()):
            if r["split"] != "dev":
                continue
            path = NEGATIVES_ROOT / batch / r["filename"]
            if not path.is_file():
                raise FileNotFoundError(f"{manifest} lists {r['filename']}, which is missing")
            rows.append({"path": rel(path), "batch": batch, "tag": r["scenario_tag"], "camera": r["camera"],
                         "source_url": r["source_url"], "source_title": r["source_title"],
                         "filename": r["filename"]})
    return rows


def capture_time(row: dict) -> datetime | None:
    """A user capture's timestamp, from the camera's file name (kept in source_title when renamed)."""
    for text in (row["source_title"], row["filename"]):
        m = _TIMESTAMP_RE.search(text)
        if m:
            return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")
    return None


def near_duplicate_groups(rows: list[dict]) -> list[list[dict]]:
    """Internet photos group by source_url. User captures of the same tag on the same camera in the same
    batch chain into one group while consecutive shots are <= NEAR_DUP_SECONDS apart; a new tag is a new
    subject even seconds later. Each row lands in exactly one group."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    user: dict[tuple, list[tuple[datetime, dict]]] = defaultdict(list)
    for r in rows:
        if r["source_url"]:
            by_key[("url", r["source_url"])].append(r)
            continue
        t = capture_time(r)
        if t is None:
            raise ValueError(f"{r['path']}: no source_url and no timestamp to group near-duplicates by")
        user[(r["batch"], r["camera"], r["tag"])].append((t, r))
    groups = list(by_key.values())
    for shots in user.values():
        shots.sort(key=lambda s: (s[0], s[1]["path"]))
        current = [shots[0][1]]
        for (t_prev, _), (t, r) in zip(shots, shots[1:]):
            if (t - t_prev).total_seconds() <= NEAR_DUP_SECONDS:
                current.append(r)
            else:
                groups.append(current)
                current = [r]
        groups.append(current)
    return sorted((sorted(g, key=lambda r: r["path"]) for g in groups), key=lambda g: g[0]["path"])


def split_negatives(rows: list[dict], rng: np.random.Generator) -> dict[str, list[dict]]:
    """~NEG_EVAL_FRAC of each (batch, tag) to eval, whole groups at a time, closest to the target count."""
    strata: dict[tuple, list[list[dict]]] = defaultdict(list)
    for g in near_duplicate_groups(rows):
        tags = {(r["batch"], r["tag"]) for r in g}
        if len(tags) != 1:
            raise ValueError(f"near-duplicate group spans batches or tags: {[r['path'] for r in g]}")
        strata[tags.pop()].append(g)
    out: dict[str, list[dict]] = {"neg_seg_train": [], "neg_seg_eval": []}
    for key in sorted(strata):
        groups = strata[key]
        target = round(NEG_EVAL_FRAC * sum(len(g) for g in groups))
        n_eval = 0
        for i in rng.permutation(len(groups)):
            g = groups[i]
            to_eval = abs(n_eval + len(g) - target) < abs(n_eval - target)
            n_eval += len(g) if to_eval else 0
            gid = g[0]["path"]
            for r in g:
                out["neg_seg_eval" if to_eval else "neg_seg_train"].append({
                    "path": r["path"], "sha256": sha256_file(REPO_ROOT / r["path"]), "batch": r["batch"],
                    "tag": r["tag"], "pile_like": r["tag"] in PILE_LIKE_TAGS, "camera": r["camera"],
                    "group": gid})
    return {k: sorted(v, key=lambda e: e["path"]) for k, v in out.items()}


# --- OOD positives (D25) ---------------------------------------------------------------------------

def positive_rows() -> list[dict]:
    """Every row, dev and holdout, of the OOD positive manifests. Holdout rows are read only to find the
    bursts they share with dev photos; they are never listed."""
    rows = []
    for d in POSITIVE_DIRS:
        manifest = d.parent / f"{d.name}.manifest.csv"
        for r in csv.DictReader(manifest.read_text().splitlines()):
            path = d / r["filename"]
            if not path.is_file():
                raise FileNotFoundError(f"{manifest} lists {r['filename']}, which is missing")
            rows.append({"path": rel(path), "batch": r.get("batch") or d.name, "tag": r["scenario_tag"],
                         "camera": r["camera"], "source_url": r["source_url"], "source_title": r["source_title"],
                         "filename": r["filename"], "split": r["split"]})
    return rows


def split_positives(rows: list[dict], rng: np.random.Generator) -> dict[str, list[dict]]:
    """Dev positives, ~POS_EVAL_FRAC of each (batch, tag) to eval, whole near-duplicate groups at a time.
    A group that also holds a holdout photo sends its dev photos to eval, never training: the segmenter
    must not train on a near-copy of a photo the OOD holdout check will score. Those photos count toward
    the eval target, so the free groups fill the rest of it."""
    strata: dict[tuple, list[list[dict]]] = defaultdict(list)
    for g in near_duplicate_groups(rows):
        tags = {(r["batch"], r["tag"]) for r in g}
        if len(tags) != 1:
            raise ValueError(f"near-duplicate group spans batches or tags: {[r['path'] for r in g]}")
        strata[tags.pop()].append(g)
    out: dict[str, list[dict]] = {"pos_seg_train": [], "pos_seg_eval": []}
    for key in sorted(strata):
        groups = [g for g in strata[key] if any(r["split"] == "dev" for r in g)]
        near = [any(r["split"] != "dev" for r in g) for g in groups]
        dev = [[r for r in g if r["split"] == "dev"] for g in groups]
        target = round(POS_EVAL_FRAC * sum(len(g) for g in dev))
        n_eval = sum(len(g) for g, n in zip(dev, near) if n)
        for i in rng.permutation(len(groups)):
            if near[i]:
                to_eval = True
            else:
                to_eval = abs(n_eval + len(dev[i]) - target) < abs(n_eval - target)
                n_eval += len(dev[i]) if to_eval else 0
            gid = groups[i][0]["path"]
            for r in dev[i]:
                out["pos_seg_eval" if to_eval else "pos_seg_train"].append({
                    "path": r["path"], "sha256": sha256_file(REPO_ROOT / r["path"]), "batch": r["batch"],
                    "tag": r["tag"], "camera": r["camera"], "group": gid, "near_holdout": near[i]})
    return {k: sorted(v, key=lambda e: e["path"]) for k, v in out.items()}


# --- base masks: dev / held-out split (D14) --------------------------------------------------------

def accepted_bases(candidates: list[dict], decisions: list[dict], index: dict[str, str]) -> list[dict]:
    """The candidates whose latest decision is accept. Refuses one whose stored mask no longer matches the
    mask the owner accepted: the known-good label is that exact mask."""
    latest = {}
    for d in decisions:
        latest[d["path"]] = d                            # the last write wins (D12)
    out = []
    for e in candidates:
        d = latest.get(e["path"])
        if d is None or d["decision"] != "accept":
            continue
        if index.get(e["path"]) != d["mask_sha256"]:
            raise ValueError(f"{e['path']}: the stored mask is not the one the owner accepted")
        out.append({**e, "mask_sha256": d["mask_sha256"]})
    return out


def split_bases(accepted: list[dict], rng: np.random.Generator, n_each: int = N_BASE_EACH) -> dict[str, list[dict]]:
    """2 x n_each of the accepted bases, stratified by (session or batch, roast), with >= MIN_ROASTED_BASES
    roasted, dealt alternately to base_dev and base_heldout so both sides get the same mix."""
    stratum = lambda e: (e.get("session") or e["batch"], e.get("roast", "unknown"))   # noqa: E731
    sizes = defaultdict(int)
    for e in accepted:
        sizes[stratum(e)] += 1
    floors = {k: MIN_ROASTED_BASES for k in sizes if k[1] == "roasted"}
    if floors:                                           # spread the roasted floor over the roasted strata
        per = -(-MIN_ROASTED_BASES // len(floors))
        floors = {k: per for k in floors}
    quota = allocate(dict(sizes), 2 * n_each, floors)
    strata: dict = defaultdict(list)
    for e in accepted:
        strata[stratum(e)].append(e)
    out: dict[str, list[dict]] = {"base_dev": [], "base_heldout": []}
    side = 0
    for k in sorted(strata):
        for e in [strata[k][i] for i in rng.permutation(len(strata[k]))[:quota.get(k, 0)]]:
            out[("base_dev", "base_heldout")[side]].append(e)
            side ^= 1
    return {k: sorted(v, key=lambda e: e["path"]) for k, v in out.items()}


# --- build, merge, check ---------------------------------------------------------------------------

def build(cfg: RunConfig) -> dict[str, list[dict]]:
    seed = cfg.seg_split_seed
    split = pooled_split(cfg)
    train = sorted(split["train"], key=lambda e: e["path"])
    seg_eval = sorted(split["eval"], key=lambda e: e["path"])

    by_session = lambda e: e["session"]                               # noqa: E731
    sizes = defaultdict(int)
    for e in train:
        sizes[by_session(e)] += 1
    quota = {s: round(SEG_VAL_FRAC * n) for s, n in sizes.items()}
    seg_val = stratified_draw(train, by_session, quota, np.random.default_rng([seed, _STREAM["seg_val"]]))
    val_paths = {e["path"] for e in seg_val}
    seg_train = [e for e in train if e["path"] not in val_paths]

    stratum = lambda e: (e["session"], e["roast"])                    # noqa: E731
    sizes = defaultdict(int)
    for e in seg_eval:
        sizes[stratum(e)] += 1
    floors = {k: MIN_ROASTED_CANDIDATES for k in sizes if k[1] == "roasted"}
    quota = allocate(dict(sizes), N_BASE_CANDIDATES, floors)
    bases = stratified_draw(seg_eval, stratum, quota, np.random.default_rng([seed, _STREAM["base_candidates"]]))
    base_paths = {e["path"] for e in bases}
    audit = draw([e for e in seg_eval if e["path"] not in base_paths], N_AUDIT,
                 np.random.default_rng([seed, _STREAM["audit_sample"]]))

    negs = split_negatives(dev_negatives(), np.random.default_rng([seed, _STREAM["negatives"]]))
    pos = split_positives(positive_rows(), np.random.default_rng([seed, _STREAM["positives"]]))
    by_batch = lambda e: e["batch"]                                   # noqa: E731
    sizes = defaultdict(int)
    for e in pos["pos_seg_eval"]:
        sizes[by_batch(e)] += 1
    quota = allocate(dict(sizes), N_POS_BASE_CANDIDATES)
    bases += stratified_draw(pos["pos_seg_eval"], by_batch, quota,
                             np.random.default_rng([seed, _STREAM["pos_base_candidates"]]))
    return {"seg_train": seg_train, "seg_val": seg_val, "seg_eval": seg_eval,
            "base_candidates": bases, "audit_sample": audit, **negs, **pos}


def merge_append_only(old: dict[str, list[dict]], new: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """`old` with `new`'s extra entries appended. Refuses (D20) if an old entry would leave its list, change
    its hash, or a photo would join a second list it is not in today."""
    problems = []
    merged = {}
    for name in dict.fromkeys([*old, *new]):
        old_entries, new_entries = old.get(name, []), new.get(name, [])
        new_by_path = {e["path"]: e for e in new_entries}
        for e in old_entries:
            n = new_by_path.get(e["path"])
            if n is None:
                problems.append(f"{name}: {e['path']} would be dropped")
            elif n["sha256"] != e["sha256"]:
                problems.append(f"{name}: {e['path']} changed content ({e['sha256'][:12]} -> {n['sha256'][:12]})")
        old_paths = {e["path"] for e in old_entries}
        merged[name] = old_entries + [e for e in new_entries if e["path"] not in old_paths]
    if problems:
        raise ValueError("the photo lists are append-only (D20); a rebuild would change committed entries:\n  "
                         + "\n  ".join(problems[:20]) + (f"\n  ... {len(problems) - 20} more" if len(problems) > 20 else ""))
    return merged


def structural_problems(lists: dict[str, list[dict]]) -> list[str]:
    """Checks that need only the file and the git-tracked manifests, not the photos."""
    problems = []
    paths = {name: [e["path"] for e in entries] for name, entries in lists.items()}
    for name, ps in paths.items():
        if len(set(ps)) != len(ps):
            problems.append(f"{name} lists a photo twice")
    for group in (POSITIVE_LISTS, NEGATIVE_LISTS, OOD_POSITIVE_LISTS):
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                both = set(paths.get(a, [])) & set(paths.get(b, []))
                if both:
                    problems.append(f"{a} and {b} share {len(both)} photos, e.g. {sorted(both)[0]}")
    evals = set(paths.get("seg_eval", [])) | set(paths.get("pos_seg_eval", []))
    for name in ("base_candidates", "base_dev", "base_heldout"):
        outside = set(paths.get(name, [])) - evals
        if outside:
            problems.append(f"{name} has {len(outside)} photos outside seg_eval/pos_seg_eval, e.g. {sorted(outside)[0]}")
    outside = set(paths.get("audit_sample", [])) - set(paths.get("seg_eval", []))
    if outside:
        problems.append(f"audit_sample has {len(outside)} photos outside seg_eval, e.g. {sorted(outside)[0]}")
    near = [e["path"] for e in lists.get("pos_seg_train", []) if e.get("near_holdout")]
    if near:
        problems.append(f"pos_seg_train holds {len(near)} photos from bursts with a holdout photo, e.g. {near[0]}")
    if set(paths.get("base_dev", [])) & set(paths.get("base_heldout", [])):
        problems.append("base_dev and base_heldout share photos")
    outside = (set(paths.get("base_dev", [])) | set(paths.get("base_heldout", []))) - set(paths.get("base_candidates", []))
    if outside:
        problems.append(f"{len(outside)} base_dev/base_heldout photos are not base candidates, e.g. {sorted(outside)[0]}")
    if set(paths.get("base_candidates", [])) & set(paths.get("audit_sample", [])):
        problems.append("audit_sample overlaps base_candidates: the audit would not be blind")
    holdout = set()
    for manifest in sorted(NEGATIVES_ROOT.glob("*.manifest.csv")):
        batch = manifest.name.removesuffix(".manifest.csv")
        holdout |= {rel(NEGATIVES_ROOT / batch / r["filename"])
                    for r in csv.DictReader(manifest.read_text().splitlines()) if r["split"] != "dev"}
    for d in POSITIVE_DIRS:
        manifest = d.parent / f"{d.name}.manifest.csv"
        holdout |= {rel(d / r["filename"])
                    for r in csv.DictReader(manifest.read_text().splitlines()) if r["split"] != "dev"}
    for name, ps in paths.items():
        leaked = holdout & set(ps)
        if leaked:
            problems.append(f"{name} lists {len(leaked)} holdout negatives, e.g. {sorted(leaked)[0]}")
    sides = defaultdict(set)
    for name in (*NEGATIVE_LISTS, *OOD_POSITIVE_LISTS):
        for e in lists.get(name, []):
            sides[e["group"]].add(name)
    straddle = sorted(g for g, s in sides.items() if len(s) > 1)
    if straddle:
        problems.append(f"{len(straddle)} near-duplicate groups straddle a train/eval pair, e.g. {straddle[0]}")
    return problems


def load_lists(path: Path = LISTS_FILE) -> tuple[dict, dict[str, list[dict]]]:
    doc = yaml.safe_load(path.read_text())
    return doc["meta"], doc["lists"]


def write_lists(meta: dict, lists: dict[str, list[dict]], path: Path = LISTS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Ticket ML-2: the segmenter's photo lists. Written by `python -m coffeecv.seg_lists build`;\n"
              "# append-only (D20). Do not edit entries by hand. See coffeecv/seg_lists.py for each list.\n")
    body = yaml.safe_dump({"meta": meta, "lists": lists}, sort_keys=False, width=200)
    path.write_text(header + body)


def summary(lists: dict[str, list[dict]]) -> str:
    lines = []
    for name, entries in lists.items():
        extra = ""
        if name in ("seg_eval", "base_candidates", "audit_sample", "seg_train", "seg_val", "base_dev", "base_heldout"):
            extra = f"  roasted {sum(e.get('roast') == 'roasted' for e in entries)}"
        elif name in NEGATIVE_LISTS:
            extra = f"  pile-like {sum(e['pile_like'] for e in entries)}"
        elif name in OOD_POSITIVE_LISTS:
            extra = f"  near-holdout {sum(e['near_holdout'] for e in entries)}"
        lines.append(f"  {name:16s} {len(entries):4d}{extra}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "check", "bases"])
    args = ap.parse_args(argv)
    cfg = RunConfig.from_params_yaml()

    if args.command == "bases":
        meta, lists = load_lists()
        if "base_dev" in lists or "base_heldout" in lists:
            raise ValueError("base_dev / base_heldout are already assigned; the lists are append-only (D20)")
        decisions = [json.loads(l) for l in DECISIONS_FILE.read_text().splitlines() if l.strip()]
        index = {r["path"]: r["mask_sha256"] for r in csv.DictReader(BASE_MASK_INDEX.read_text().splitlines())}
        accepted = accepted_bases(lists["base_candidates"], decisions, index)
        if len(accepted) < 2 * N_BASE_EACH:
            raise ValueError(f"{len(accepted)} accepted base masks; D14 needs {2 * N_BASE_EACH}")
        lists.update(split_bases(accepted, np.random.default_rng([cfg.seg_split_seed, _STREAM["base_split"]])))
        problems = structural_problems(lists)
        if problems:
            raise ValueError("refusing to write inconsistent lists:\n  " + "\n  ".join(problems))
        write_lists(meta, lists)
        print(f"{len(accepted)} accepted; wrote base_dev / base_heldout\n{summary(lists)}")
        return 0

    if args.command == "build":
        capture_dirs, frac = split_basis(cfg)
        new = build(cfg)
        if LISTS_FILE.exists():
            meta, old = load_lists()
            if meta["seg_split_seed"] != cfg.seg_split_seed:
                raise ValueError(f"{rel(LISTS_FILE)} was built at seg_split_seed {meta['seg_split_seed']}, "
                                 f"params.yaml says {cfg.seg_split_seed}; the lists are append-only (D20)")
            new = merge_append_only(old, new)
        meta = {"seg_split_seed": cfg.seg_split_seed,
                "photo_frac": [frac["train"], frac["val"], frac["test"]],
                "capture_dirs": capture_dirs, "seg_val_frac": SEG_VAL_FRAC,
                "neg_eval_frac": round(NEG_EVAL_FRAC, 4), "near_dup_seconds": NEAR_DUP_SECONDS}
        problems = structural_problems(new)
        if problems:
            raise ValueError("refusing to write inconsistent lists:\n  " + "\n  ".join(problems))
        write_lists(meta, new)
        print(f"wrote {rel(LISTS_FILE)}\n{summary(new)}")
        return 0

    meta, lists = load_lists()
    problems = structural_problems(lists)
    fresh = build(cfg)
    for name in (*POSITIVE_LISTS, *NEGATIVE_LISTS, *OOD_POSITIVE_LISTS):
        have, want = {e["path"] for e in lists.get(name, [])}, {e["path"] for e in fresh[name]}
        if have != want:
            problems.append(f"{name}: {len(want - have)} photos in today's pools are unlisted, "
                            f"{len(have - want)} listed photos are not where today's split puts them")
    print(summary(lists))
    if problems:
        print("FAIL\n  " + "\n  ".join(problems))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
