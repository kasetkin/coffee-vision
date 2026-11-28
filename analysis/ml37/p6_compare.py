"""ML-37 P6's checks: the pool re-cropped from the store D1 re-canonicalized, against the crops before it
(docs/ml-0037_plan_linear_canonical_photo.html, P6; docs/agents/experiments.md, "Changing the segmenter").

    python analysis/ml37/p6_compare.py report [--jobs 4]  # -> analysis/ml37/p6_report.txt
    python analysis/ml37/p6_compare.py look                # -> tmp/ml37/p6/look/

Before: every segcrop@<session> and merge_segpool output as f0d36ab's dvc.lock records it (P5's commit, the last
before the re-crop), read from the DVC cache by object path: each .dir listing expanded from .dvc/cache/files/md5/,
each file read where the listing's md5 puts it. Never through dvc (no checkout; the re-crop holds the repo lock).
After: data/segcropped/ in the workspace, once `dvc repro -s segcrop@<session>` for every session and then
merge_segpool have run (--after-root for a copy). A session whose files are older than f0d36ab is refused as not
re-cropped.

report  1. Per session and for the pool: photos (the segcrop reports' rows), crops, masks, D18 fallbacks, and per
           class, before against after; a photo one side has and the other not, by name.
        2. D18 fallbacks (the reports' `fallback`): the photos that became one and those that stopped, with their
           mask_area_frac against seg_min_area_frac. The photos that enter or leave the pool (merge_rig's output,
           labels/ml3/pool_exclude.csv applied), and every exclusion row whose photo changed its fallback or its crop:
           listed for the owner to judge before training (ML-8 P4).
        3. Per crop, on the photos both sides have: the crop box and size, the full-frame mask's sha256, and where the
           box is unchanged (both fallbacks: the whole photo) the mean |pixel difference| over the crop and over the
           bean pixels both masks keep, the largest, and the crop mask's changed pixels. Every crop in
           tmp/ml37/p6/crops.csv; the pool's files checked identical to their session's.
        4. The look set `look` renders.
        Cached in tmp/ml37/p6/ (the workspace files' md5 by size and mtime, the pixel differences by the two md5s),
        so a second run reads only what changed.
look    Full size, each photo of the look set: every D18 fallback after the re-crop, every photo entering or leaving
        the pool, every P5 photo in the crops with a D13 mask IoU under 0.9 (tmp/ml37/p5/photos), and 5 ordinary
        crops (a fixed draw). Per photo <item>__photo.jpg: its canonical photo with the full-frame mask the stage
        made outlined in magenta (the segmenter re-run as segcrop runs it, its sha256 checked against the after
        report's, as analysis/ml8/p4_review_items.py does), the D4 box after dashed yellow and before dashed cyan;
        and <item>__crop.jpg: the crop training reads (after, else before) with its own mask outlined. JPEG quality
        95, not downscaled. items.csv beside them is coffeecv.review_masks' input for the owner's verdicts:
            python -m coffeecv.review_masks --items tmp/ml37/p6/look/items.csv --name ml37_p6 --canonical
        (declines to labels/ml3/pool_exclude.csv, then merge_segpool again).

Reads the DVC cache, data/segcropped, data/canonical, tmp/ml37/p5 and models_pretrained (not in git); writes
tmp/ml37/p6/ and the report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BEFORE_COMMIT = "f0d36ab"           # ML-37 P5's last commit: the store re-canonicalized, the crops not yet re-made
CACHE = REPO / ".dvc" / "cache" / "files" / "md5"
AFTER = REPO / "data" / "segcropped"
CANONICAL = REPO / "data" / "canonical"
EXCLUDE = REPO / "labels" / "ml3" / "pool_exclude.csv"
P5_PHOTOS = REPO / "tmp" / "ml37" / "p5" / "photos"
WORK = REPO / "tmp" / "ml37" / "p6"
LOOK = WORK / "look"
OUT = Path(__file__).with_name("p6_report.txt")
POOL = "pool"
CROP, MASK = "__cropped.jxl", "__beanmask.png"
LOW_IOU = 0.9
ORDINARY = 5
DRAW_SEED = 37006
DECODE_THREADS = 2
STRIP = 512


# ----- the two sides -----

@dataclass
class Side:
    """One side's files: {session or "pool": {relpath: (path to read, md5 or None until hashed)}}."""
    name: str
    files: dict[str, dict[str, Path]]
    md5s: dict[str, dict[str, str]] = field(default_factory=dict)

    def json(self, session: str, rel: str):
        p = self.files[session].get(rel)
        return None if p is None else json.loads(p.read_text())

    def reports(self, session: str) -> dict[tuple[str, str], dict]:
        """{(class dir, original's file name): its segcrop_report.json row}."""
        out = {}
        for rel in sorted(self.files[session]):
            if rel.endswith("/segcrop_report.json"):
                for r in self.json(session, rel):
                    out[(rel.split("/")[0], r["file"])] = r
        return out


def stem(file: str) -> str:
    return Path(file).stem


def crop_rel(class_dir: str, file: str) -> str:
    return f"{class_dir}/{stem(file)}{CROP}"


def mask_rel(class_dir: str, file: str) -> str:
    return f"{class_dir}/{stem(file)}{MASK}"


def before_lock() -> dict:
    out = subprocess.run(["git", "show", f"{BEFORE_COMMIT}:dvc.lock"], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout
    return yaml.safe_load(out)["stages"]


def sessions() -> list[str]:
    """merge_segpool's sessions, in its order, as dvc.yaml and f0d36ab's dvc.lock both name them."""
    from coffeecv.merge_rig import merge_cmd_args
    stages = yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"]
    name, ss = merge_cmd_args(stages["merge_segpool"]["cmd"])
    locked = {k.split("@", 1)[1] for k in before_lock() if k.startswith("segcrop@")}
    if name != POOL or set(ss) != locked:
        raise SystemExit(f"dvc.yaml's merge_segpool ({name}: {sorted(ss)}) is not f0d36ab's segcrop stages "
                         f"({sorted(locked)})")
    return ss


def before_side(ss: list[str]) -> Side:
    lock = before_lock()
    files, md5s = {}, {}
    for s in ss + [POOL]:
        (out,) = lock["merge_segpool" if s == POOL else f"segcrop@{s}"]["outs"]
        if out["path"] != f"data/segcropped/{s}":
            raise SystemExit(f"{BEFORE_COMMIT}'s dvc.lock: {s}'s out is {out['path']}")
        d = out["md5"]
        listing = json.loads((CACHE / d[:2] / d[2:]).read_text())
        files[s] = {e["relpath"]: CACHE / e["md5"][:2] / e["md5"][2:] for e in listing}
        md5s[s] = {e["relpath"]: e["md5"] for e in listing}
        missing = [r for r, p in files[s].items() if not p.exists()]
        if missing or len(listing) != out["nfiles"]:
            raise SystemExit(f"{s}: {len(missing)} of {len(listing)} cache objects missing (nfiles {out['nfiles']})")
    return Side("before", files, md5s)


def after_side(ss: list[str], root: Path, partial: bool) -> tuple[Side, list[str]]:
    """The workspace's crops; sessions not (yet) re-cropped are refused, or with --partial left out."""
    since = datetime.fromisoformat(subprocess.run(["git", "log", "-1", "--format=%cI", BEFORE_COMMIT], cwd=REPO,
                                                  capture_output=True, text=True, check=True).stdout.strip())
    files, ready, problems = {}, [], []
    for s in ss + [POOL]:
        d = root / s
        head = d / ("merge_manifest.json" if s == POOL else "segcrop_manifest.json")
        if not head.exists():
            problems.append(f"{s}: no {head.name} (not finished)")
            continue
        old = [p for p in d.rglob("*.json") if p.stat().st_mtime < since.timestamp()]
        if old:
            problems.append(f"{s}: {len(old)} json files older than {BEFORE_COMMIT} ({since:%Y-%m-%d %H:%M}), "
                            f"e.g. {old[0].relative_to(d)} (not re-cropped)")
            continue
        files[s] = {p.relative_to(d).as_posix(): p for p in sorted(d.rglob("*")) if p.is_file()}
        ready.append(s)
    if problems and not partial:
        raise SystemExit("the crops after are not ready:\n  " + "\n  ".join(problems))
    if POOL not in ready and partial:
        problems.append("the pool is left out")
    return Side("after", files), problems


# ----- md5 of the workspace files, cached -----

def md5_file(path: str) -> tuple[str, str]:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return path, h.hexdigest()


def hash_after(after: Side, jobs: int) -> None:
    cache_path = WORK / "after_md5.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = []
    for s, fs in after.files.items():
        for p in fs.values():
            st = p.stat()
            c = cache.get(str(p))
            if not (c and c[0] == st.st_size and c[1] == st.st_mtime_ns):
                todo.append(str(p))
    if todo:
        print(f"hashing {len(todo)} files after, {jobs} jobs", file=sys.stderr, flush=True)
        with ProcessPoolExecutor(jobs) as pool:
            for i, (p, h) in enumerate(pool.map(md5_file, todo, chunksize=4), 1):
                st = Path(p).stat()
                cache[p] = [st.st_size, st.st_mtime_ns, h]
                if i % 500 == 0:
                    print(f"  {i}/{len(todo)}", file=sys.stderr, flush=True)
        WORK.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    after.md5s = {s: {r: cache[str(p)][2] for r, p in fs.items()} for s, fs in after.files.items()}


# ----- pixel differences, cached -----

def decode(path: str) -> np.ndarray:
    import imagecodecs as ic
    return ic.jpegxl_decode(Path(path).read_bytes(), numthreads=DECODE_THREADS)


def read_mask(path: str | None, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    if path is None:
        return np.ones(shape, bool)
    m = np.array(Image.open(path).convert("1"), dtype=bool)
    if m.shape != shape:
        raise ValueError(f"{path}: mask {m.shape}, crop {shape}")
    return m


def pixel_diff(job: tuple[str, str, str | None, str | None]) -> dict:
    """Two crops of one box: mean and largest |difference| over every pixel and channel, the mean over the pixels both
    masks keep (bean pixels; every pixel of a fallback), the share of pixels changed, the crop mask's changed pixels."""
    b_crop, a_crop, b_mask, a_mask = job
    a, b = decode(a_crop), decode(b_crop)
    if a.shape != b.shape:
        return {"error": f"shapes {b.shape} -> {a.shape}"}
    mb, ma = read_mask(b_mask, b.shape[:2]), read_mask(a_mask, a.shape[:2])
    both = mb & ma
    total = bean = 0.0
    largest = changed = 0
    for y in range(0, a.shape[0], STRIP):
        d = np.abs(a[y:y + STRIP].astype(np.int16) - b[y:y + STRIP].astype(np.int16))
        total += float(d.sum(dtype=np.int64))
        bean += float(d[both[y:y + STRIP]].sum(dtype=np.int64))
        largest = max(largest, int(d.max()))
        changed += int(d.any(axis=2).sum())
    n = a.shape[0] * a.shape[1]
    return {"mean": total / (n * 3), "bean_mean": bean / max(int(both.sum()) * 3, 1), "max": largest,
            "changed_frac": changed / n, "mask_px": int((mb ^ ma).sum()) if b_mask and a_mask else None}


# ----- the comparison -----

def photo_key(session: str, class_dir: str, file: str) -> str:
    return f"{session}/{class_dir}/{file}"


def pool_members(side: Side) -> dict[str, tuple[str, str]]:
    """{pool crop relpath: (the session it was copied from, its relpath there)}. merge_rig files a crop under the
    pool's class folder of the same class number, whose name may differ from the session's (the pool's
    class_008__Ethiopia_Kochere holds tasty_coffee's class_008__Ethiopia_Yirgacheffe), so a crop is matched by its
    class number and file name."""
    index: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for s, files in side.files.items():
        if s == POOL:
            continue
        for r in files:
            if r.endswith(CROP) and "/" in r:
                folder, name = r.split("/", 1)
                index.setdefault((folder.split("__")[0], name), []).append((s, r))
    out = {}
    for rel in side.files.get(POOL, {}):
        if rel.endswith(CROP):
            folder, name = rel.split("/", 1)
            src = index.get((folder.split("__")[0], name), [])
            out[rel] = src[0] if len(src) == 1 else ("?", "")
    return out


def box_shift(b: list | None, a: list | None) -> int | None:
    """The largest move of any edge of the D4 box, in pixels; None if either is a fallback."""
    if b is None or a is None:
        return None
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[0] + a[2] - b[0] - b[2]), abs(a[1] + a[3] - b[1] - b[3]))


def compare(before: Side, after: Side, ss: list[str], jobs: int) -> dict:
    """Every photo both sides' reports have, its row on each side and the pixel differences."""
    rows = []
    cache_path = WORK / "crop_diff.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    jobs_todo = {}
    for s in ss:
        if s not in after.files:
            continue
        rb, ra = before.reports(s), after.reports(s)
        for k in sorted(rb.keys() & ra.keys()):
            b, a = rb[k], ra[k]
            cr, mr = crop_rel(*k), mask_rel(*k)
            bm, am = before.md5s[s].get(cr), after.md5s[s].get(cr)
            r = {"session": s, "class_dir": k[0], "file": k[1], "b": b, "a": a, "b_md5": bm, "a_md5": am,
                 "identical": bm is not None and bm == am,
                 "same_box": (b["box"] == a["box"]) if not (b["fallback"] or a["fallback"])
                 else (b["fallback"] and a["fallback"] and b["size"] == a["size"]),
                 "shift": box_shift(b["box"], a["box"]), "diff": None}
            if r["same_box"] and bm and am:
                key = f"{bm}:{am}"
                if r["identical"]:
                    r["diff"] = {"mean": 0.0, "bean_mean": 0.0, "max": 0, "changed_frac": 0.0,
                                 "mask_px": 0 if b["box"] else None}
                elif key in cache:
                    r["diff"] = cache[key]
                else:
                    jobs_todo[key] = (str(before.files[s][cr]), str(after.files[s][cr]),
                                      str(before.files[s][mr]) if mr in before.files[s] else None,
                                      str(after.files[s][mr]) if mr in after.files[s] else None)
            rows.append(r)
    if jobs_todo:
        print(f"decoding {len(jobs_todo)} crop pairs of an unchanged box, {jobs} jobs", file=sys.stderr, flush=True)
        keys = list(jobs_todo)
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, (k, d) in enumerate(zip(keys, pool.map(pixel_diff, [jobs_todo[k] for k in keys])), 1):
                cache[k] = d
                if i % 50 == 0 or i == len(keys):
                    print(f"  {i}/{len(keys)}", file=sys.stderr, flush=True)
                    WORK.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(cache))
        for r in rows:
            if r["diff"] is None and r["same_box"] and r["b_md5"] and r["a_md5"]:
                r["diff"] = cache[f"{r['b_md5']}:{r['a_md5']}"]
    return rows


def p5_ious(ss: list[str]) -> dict[str, float]:
    """{session/class/file: P5's D13 mask IoU} for the photos in the crops' sessions, from P5's records."""
    cache_path = WORK / "p5_iou.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from p5_compare import iou, unpacked
    out = {}
    for s in ss:
        for pb in sorted((P5_PHOTOS / s).rglob("*.jxl.before.npz")):
            pa = pb.with_name(pb.name.replace(".before.npz", ".after.npz"))
            if pa.exists():
                rel = pb.relative_to(P5_PHOTOS).as_posix()[:-len(".jxl.before.npz")]
                out[rel] = iou(unpacked(np.load(pb)), unpacked(np.load(pa)))
    WORK.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=0))
    return out


def look_set(before: Side, after: Side, ss: list[str], ious: dict[str, float]) -> dict[str, dict]:
    """{session/class/file: {"why": [...], "session", "class_dir", "file"}}, the photos `look` renders."""
    out: dict[str, dict] = {}

    def add(s: str, c: str, f: str, why: str) -> None:
        out.setdefault(photo_key(s, c, f), {"session": s, "class_dir": c, "file": f, "why": []})["why"].append(why)

    for s in ss:
        if s in after.files:
            for (c, f), r in after.reports(s).items():
                if r["fallback"]:
                    add(s, c, f, "fallback")
    for rel, why in pool_moves(before, after).items():
        c, name = rel.split("/")
        s = why[1]
        f = source_file(before, after, s, c, name)
        add(s, c, f, why[0])
    for key, v in sorted(ious.items()):
        if v < LOW_IOU:
            s, c, f = key.split("/")
            add(s, c, f, f"P5 IoU {v:.3f}")
    rng = np.random.default_rng(DRAW_SEED)
    pool = sorted((s, c, f) for s in ss if s in after.files for (c, f), r in after.reports(s).items()
                  if not r["fallback"] and photo_key(s, c, f) not in out)
    for i in rng.choice(len(pool), size=min(ORDINARY, len(pool)), replace=False):
        add(*pool[int(i)], "ordinary")
    return out


def source_file(before: Side, after: Side, s: str, c: str, crop_name: str) -> str:
    """The original's file name of a crop, from either side's report."""
    for side in (after, before):
        if s in side.files:
            for (cc, f) in side.reports(s):
                if cc == c and f"{stem(f)}{CROP}" == crop_name:
                    return f
    return crop_name[:-len(CROP)]


def pool_moves(before: Side, after: Side) -> dict[str, tuple[str, str]]:
    """{pool crop relpath: ("enters" or "leaves", session)}."""
    if POOL not in after.files:
        return {}
    pb, pa = pool_members(before), pool_members(after)
    return {**{r: ("enters", pa[r][0]) for r in pa.keys() - pb.keys()},
            **{r: ("leaves", pb[r][0]) for r in pb.keys() - pa.keys()}}


# ----- report -----

def counts(side: Side, s: str) -> dict:
    rep = side.reports(s)
    fs = side.files[s]
    return {"photos": len(rep), "crops": sum(r.endswith(CROP) for r in fs), "masks": sum(r.endswith(MASK) for r in fs),
            "fallbacks": sum(r["fallback"] for r in rep.values()),
            "classes": Counter(c for c, _ in rep)}


def run_report(a: argparse.Namespace) -> None:
    ss = sessions()
    before = before_side(ss)
    after, problems = after_side(ss, a.after_root, a.partial)
    hash_after(after, a.jobs)
    min_area = yaml.safe_load((REPO / "params.yaml").read_text())["seg_min_area_frac"]
    done = [s for s in ss if s in after.files]
    partial = bool(problems)
    L = ["ML-37 P6: the pool re-cropped from the store D1 re-canonicalized (P5), against the crops before it. "
         "analysis/ml37/p6_compare.py",
         f"Before: f0d36ab's dvc.lock (segcrop@<session>, merge_segpool), read from the DVC cache. After: "
         f"{a.after_root.relative_to(REPO) if a.after_root.is_relative_to(REPO) else a.after_root}."]
    if problems:
        L += ["PARTIAL (--partial): " + "; ".join(problems)]

    # 1. counts
    L += ["", "1. Photos, crops and classes. 'photos' the segcrop reports' rows, 'crops' __cropped.jxl, 'masks' "
          "__beanmask.png (none on a D18 fallback), 'fb' D18 fallbacks.",
          f"{'session':<38}{'photos':>14}{'crops':>14}{'masks':>14}{'fb':>10}  class counts"]
    tot = defaultdict(lambda: [0, 0])
    diffs = []
    for s in ss:
        if s not in after.files:
            L.append(f"{s:<38}  (not re-cropped)")
            continue
        cb, ca = counts(before, s), counts(after, s)
        for k in ("photos", "crops", "masks", "fallbacks"):
            tot[k][0] += cb[k]
            tot[k][1] += ca[k]
        cls = "same" if cb["classes"] == ca["classes"] else "DIFFER: " + ", ".join(
            f"{c.split('__')[0]} {cb['classes'][c]}->{ca['classes'][c]}"
            for c in sorted(cb["classes"].keys() | ca["classes"].keys()) if cb["classes"][c] != ca["classes"][c])
        L.append(f"{s:<38}" + "".join(f"{f'{cb[k]}->{ca[k]}':>14}" for k in ("photos", "crops", "masks"))
                 + f"{str(cb['fallbacks']) + '->' + str(ca['fallbacks']):>10}  {cls}")
        rb, ra = before.reports(s), after.reports(s)
        diffs += [f"  only before: {photo_key(s, *k)}" for k in sorted(rb.keys() - ra.keys())]
        diffs += [f"  only after:  {photo_key(s, *k)}" for k in sorted(ra.keys() - rb.keys())]
        for side, other, word in ((before, after, "before"), (after, before, "after")):
            extra = sorted(set(side.files[s]) - set(other.files[s]))
            extra = [r for r in extra if r.endswith((CROP, MASK))]
            diffs += [f"  file only {word}: {s}/{r}" for r in extra]
    L.append(f"{'every session':<38}" + "".join(f"{f'{tot[k][0]}->{tot[k][1]}':>14}"
                                                 for k in ("photos", "crops", "masks"))
             + f"{str(tot['fallbacks'][0]) + '->' + str(tot['fallbacks'][1]):>10}")
    L += ["Differences by photo:" if diffs else "No photo or crop file is on one side only."] + diffs

    if POOL in after.files:
        mb, ma = before.json(POOL, "merge_manifest.json"), after.json(POOL, "merge_manifest.json")
        L += ["", f"The pool (merge_segpool): {mb['images']} -> {ma['images']} photos, {mb['classes']} -> "
              f"{ma['classes']} classes, {len(mb.get('excluded', []))} -> {len(ma.get('excluded', []))} excluded.",
              f"{'class':<10}{'before':>8}{'after':>8}  per session where it differs"]
        for c in sorted(mb["per_class_counts"].keys() | ma["per_class_counts"].keys()):
            pb, pa = mb["per_class_counts"].get(c, {}), ma["per_class_counts"].get(c, {})
            nb, na = (sum(1 for r in pool_members(side) if r.startswith(f"class_{c}__")) for side in (before, after))
            d = ", ".join(f"{s} {pb.get(s, 0)}->{pa.get(s, 0)}" for s in ss if pb.get(s, 0) != pa.get(s, 0))
            L.append(f"class_{c:<4}{nb:>8}{na:>8}  {d}")
        moves = pool_moves(before, after)
        L += [f"Pool files only before or after: {len(moves)} (listed in 2)."]
        ex_b, ex_a = set(mb.get("excluded", [])), set(ma.get("excluded", []))
        L += [f"  excluded only before: {e}" for e in sorted(ex_b - ex_a)]
        L += [f"  excluded only after:  {e}" for e in sorted(ex_a - ex_b)]
        bad = []
        for side in (before, after):
            for rel, (s, srel) in pool_members(side).items():
                for r, sr in ((rel, srel), (rel[:-len(CROP)] + MASK, srel[:-len(CROP)] + MASK)):
                    if s == "?" and partial:
                        continue                  # its session not re-cropped yet
                    if r in side.files[POOL] and (s == "?" or side.md5s[POOL][r] != side.md5s[s].get(sr)):
                        bad.append(f"  {side.name}: pool/{r} is not its session's file ({s}/{sr})")
        L += [f"Every pool crop and mask is its session's file, byte for byte, both sides: "
              f"{'yes' if not bad else 'NO'}"] + bad

    # 2. fallbacks and the pool's moves
    rows = compare(before, after, ss, a.jobs)
    became = [r for r in rows if r["a"]["fallback"] and not r["b"]["fallback"]]
    stopped = [r for r in rows if r["b"]["fallback"] and not r["a"]["fallback"]]
    still = [r for r in rows if r["b"]["fallback"] and r["a"]["fallback"]]
    excl = {(r["session"], r["class_folder"], r["file"]): r["reason"] for r in csv.DictReader(EXCLUDE.open())}

    def fb_line(r: dict) -> str:
        k = (r["session"], r["class_dir"], r["file"])
        return (f"  {photo_key(*k)}: mask_area_frac {r['b']['mask_area_frac']:.4f} -> {r['a']['mask_area_frac']:.4f}"
                + (f"  [excluded: {excl[k]}]" if k in excl else ""))

    L += ["", f"2. D18 fallbacks (mask_area_frac under seg_min_area_frac {min_area}): {tot['fallbacks'][0]} before, "
          f"{tot['fallbacks'][1]} after; {len(still)} both, {len(became)} became one, {len(stopped)} stopped.",
          "Became a fallback:" if became else "None became a fallback."] + [fb_line(r) for r in became]
    L += ["Stopped being one:" if stopped else "None stopped being one."] + [fb_line(r) for r in stopped]
    L += ["Fallbacks both sides:"] + [fb_line(r) for r in still]
    moves = pool_moves(before, after)
    if POOL in after.files:
        L += ["", "Photos entering or leaving the pool, for the owner to judge before training (ML-8 P4; declines "
              "to labels/ml3/pool_exclude.csv):" if moves else "No photo enters or leaves the pool."]
        by_key = {(r["session"], r["class_dir"], f"{stem(r['file'])}{CROP}"): r for r in rows}
        for rel, (how, s) in sorted(moves.items()):
            c, name = rel.split("/")
            r = by_key.get((s, c, name))
            state = ("" if r is None else f" (fallback {r['b']['fallback']} -> {r['a']['fallback']}, "
                     f"mask_area_frac {r['b']['mask_area_frac']:.4f} -> {r['a']['mask_area_frac']:.4f})")
            L.append(f"  {how}: {s}/{rel}{state}")
    by_photo = {(r["session"], r["class_dir"], r["file"]): r for r in rows}
    ex_lines = []
    for k, reason in sorted(excl.items()):
        if k[0] not in done:
            continue
        r = by_photo.get(k)
        if r is None:
            ex_lines.append(f"  {photo_key(*k)} ({reason}): not on both sides")
        elif r["b"]["fallback"] != r["a"]["fallback"] or not r["same_box"] or r["b"]["mask_sha256"] != \
                r["a"]["mask_sha256"]:
            ex_lines.append(f"  {photo_key(*k)} ({reason}): fallback {r['b']['fallback']} -> {r['a']['fallback']}, "
                            f"box {r['b']['box']} -> {r['a']['box']}, mask "
                            f"{'same' if r['b']['mask_sha256'] == r['a']['mask_sha256'] else 'changed'}")
    L += ["", f"pool_exclude.csv rows ({sum(k[0] in done for k in excl)} on re-cropped sessions) whose photo's "
          f"fallback, box or mask changed: {len(ex_lines)}"] + ex_lines

    # 3. per crop
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "crops.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "class_dir", "file", "fallback_before", "fallback_after", "box_before", "box_after",
                    "size_before", "size_after", "edge_shift", "mask_sha_same", "crop_identical", "mean_abs_diff",
                    "bean_mean_abs_diff", "max_abs_diff", "changed_frac", "crop_mask_px_changed"])
        for r in rows:
            d = r["diff"] or {}
            w.writerow([r["session"], r["class_dir"], r["file"], r["b"]["fallback"], r["a"]["fallback"],
                        r["b"]["box"], r["a"]["box"], r["b"]["size"], r["a"]["size"], r["shift"],
                        r["b"]["mask_sha256"] == r["a"]["mask_sha256"], r["identical"], d.get("mean"),
                        d.get("bean_mean"), d.get("max"), d.get("changed_frac"), d.get("mask_px")])
    L += ["", "3. Per crop, the photos both sides have. 'mask=' the full-frame mask's sha256 unchanged, 'same' "
          "the crop file byte for byte, 'box=' the D4 box unchanged (both fallbacks: the whole photo, same size); "
          "'shift' the largest move of a box edge in px (median, largest) where the box moved; |diff| over the crops "
          "of an unchanged box, 8-bit levels: 'mean' the mean of the per-crop means over every pixel and channel, "
          "'bean' over the bean pixels both masks keep, '(max)' the largest per-crop mean, 'blk' the largest pixel, "
          "'mpx' the largest count of crop-mask pixels changed. Every crop: tmp/ml37/p6/crops.csv.",
          f"{'session':<38}{'n':>6}{'mask=':>7}{'same':>6}{'box=':>6}{'moved':>7}{'shift':>12}"
          f"{'size%':>8}{'mean':>8}{'bean':>8}{'(max)':>8}{'blk':>5}{'mpx':>8}"]

    def line(name: str, rs: list[dict]) -> str:
        moved = [r for r in rs if r["shift"] is not None and not r["same_box"]]
        ds = [r["diff"] for r in rs if r["diff"] and "error" not in r["diff"]]
        area = [(r["a"]["size"][0] * r["a"]["size"][1]) / (r["b"]["size"][0] * r["b"]["size"][1]) - 1 for r in rs]
        sh = [r["shift"] for r in moved]
        mpx = [d["mask_px"] for d in ds if d.get("mask_px") is not None]
        return (f"{name:<38}{len(rs):>6}{sum(r['b']['mask_sha256'] == r['a']['mask_sha256'] for r in rs):>7}"
                f"{sum(r['identical'] for r in rs):>6}{sum(r['same_box'] for r in rs):>6}{len(moved):>7}"
                + (f"{f'{np.median(sh):.0f}, {max(sh)}':>12}" if sh else f"{'-':>12}")
                + f"{(max(area, key=abs) * 100 if area else 0):>8.2f}"
                + (f"{np.mean([d['mean'] for d in ds]):>8.3f}{np.mean([d['bean_mean'] for d in ds]):>8.3f}"
                   f"{max(d['mean'] for d in ds):>8.3f}{max(d['max'] for d in ds):>5}" if ds else f"{'-':>29}")
                + (f"{max(mpx):>8}" if mpx else f"{'-':>8}"))

    for s in done:
        L.append(line(s, [r for r in rows if r["session"] == s]))
    L.append(line("every session", rows))
    L.append("'size%' the largest change of a crop's area, in percent.")
    errs = [r for r in rows if r["diff"] and "error" in r["diff"]]
    L += [f"  decode mismatch: {photo_key(r['session'], r['class_dir'], r['file'])}: {r['diff']['error']}"
          for r in errs]
    top = sorted((r for r in rows if r["shift"] is not None), key=lambda r: -r["shift"])[:15]
    L += ["The 15 largest box moves (box before -> after [x0, y0, w, h]):"]
    L += [f"  {r['shift']:>5}  {photo_key(r['session'], r['class_dir'], r['file'])}  {r['b']['box']} -> "
          f"{r['a']['box']}" for r in top if r["shift"]]
    top = sorted((r for r in rows if r["diff"] and "error" not in r["diff"] and r["diff"]["max"]),
                 key=lambda r: -r["diff"]["mean"])[:15]
    L += ["The 15 largest mean |diff| on an unchanged box (mean, bean mean, largest pixel, share of pixels changed):"]
    L += [f"  {r['diff']['mean']:.3f} {r['diff']['bean_mean']:.3f} {r['diff']['max']:>3} "
          f"{r['diff']['changed_frac']:.3f}  {photo_key(r['session'], r['class_dir'], r['file'])}" for r in top]

    # 4. the look set
    ious = p5_ious(ss)
    look = look_set(before, after, ss, ious)
    low = {k: v for k, v in ious.items() if v < LOW_IOU}
    L += ["", "4. The look set (`python analysis/ml37/p6_compare.py look` renders it, full size, to "
          f"tmp/ml37/p6/look/): {len(look)} photos. P5's photos in the crops with a D13 IoU under {LOW_IOU}: "
          f"{len(low)} (of {len(ious)} P5 records in these sessions)."]
    L += [f"  {k}: {', '.join(v['why'])}" for k, v in sorted(look.items())]
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ----- look -----

def draw(rgb: np.ndarray, mask: np.ndarray | None, boxes: list[tuple[list | None, tuple]]) -> np.ndarray:
    from PIL import Image, ImageDraw
    from coffeecv import seg_overlay
    px = max(2, max(rgb.shape[:2]) // 1500)
    out = seg_overlay.outline(rgb, mask, px) if mask is not None else rgb
    if any(b for b, _ in boxes):
        im = Image.fromarray(out)
        d = ImageDraw.Draw(im)
        for b, colour in boxes:
            if b:
                x0, y0, w, h = b
                dash = 12 * px
                for p, q in (((x0, y0), (x0 + w, y0)), ((x0 + w, y0), (x0 + w, y0 + h)),
                             ((x0 + w, y0 + h), (x0, y0 + h)), ((x0, y0 + h), (x0, y0))):
                    n = max(1, int(np.hypot(q[0] - p[0], q[1] - p[1]) / dash))
                    for k in range(0, n, 2):
                        t0, t1 = k / n, min(k + 1, n) / n
                        d.line([(p[0] + (q[0] - p[0]) * t0, p[1] + (q[1] - p[1]) * t0),
                                (p[0] + (q[0] - p[0]) * t1, p[1] + (q[1] - p[1]) * t1)], fill=colour, width=2 * px)
        out = np.asarray(im)
    return out


def save_jpeg(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(rgb).save(path, quality=95)


def run_look(a: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from coffeecv import canonical as cn
    from coffeecv.config import RunConfig
    from coffeecv.segcrop_session import crop_photo
    from coffeecv.segment_beans import BeanSegmenter, mask_sha256, seg_params
    ss = sessions()
    before = before_side(ss)
    after, problems = after_side(ss, a.after_root, a.partial)
    if problems:
        print("PARTIAL: " + "; ".join(problems), file=sys.stderr)
    look = look_set(before, after, ss, p5_ious(ss))
    threads = {after.json(s, "segcrop_manifest.json")["threads"] for s in ss if s in after.files}
    if len(threads) != 1:
        raise SystemExit(f"the sessions were cropped at {sorted(threads)} threads")
    torch.set_num_threads(threads.pop())
    seg = BeanSegmenter(seg_params(RunConfig.from_params_yaml(REPO / "params.yaml")))
    (LOOK / "masks").mkdir(parents=True, exist_ok=True)
    items = []
    chosen = sorted(look.items())[:a.limit or None]
    for n, (key, it) in enumerate(chosen, 1):
        s, c, f = it["session"], it["class_dir"], it["file"]
        item = f"{n:03d}__{'+'.join(w.split()[0] for w in it['why'])}__{s}__{c.split('__')[0]}__{stem(f)}"
        rb = before.reports(s).get((c, f)) if s in before.files else None
        ra = after.reports(s).get((c, f)) if s in after.files else None
        print(f"{n}/{len(chosen)} {key}: {', '.join(it['why'])}", flush=True)
        canonical = CANONICAL / s / c / f"{f}.jxl"
        note = ""
        if canonical.exists():
            rgb = cn.load_canonical(canonical.read_bytes())
            mask = crop_photo(canonical, seg).photo_mask.astype(bool)
            sha = mask_sha256(mask)
            if ra is not None and sha != ra["mask_sha256"]:
                note = "mask NOT the one segcrop recorded"
                print(f"  {note}", file=sys.stderr)
            save_jpeg(draw(rgb, mask, [((rb or {}).get("box"), (0, 255, 255)), ((ra or {}).get("box"),
                                                                                 (255, 255, 0))]),
                      LOOK / f"{item}__photo.jpg")
            mpath = LOOK / "masks" / f"{item}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mpath)
            items.append({"item": item, "path": f"dataset/{s}/{c}/{f}", "mask": str(mpath.relative_to(REPO)),
                          "mask_sha256": sha, "session": s, "why": "; ".join(it["why"]),
                          "fallback": (ra or {}).get("fallback"), "area_frac": (ra or {}).get("mask_area_frac"),
                          "note": note})
            del rgb, mask
        side = after if s in after.files and crop_rel(c, f) in after.files[s] else before
        cp = side.files[s].get(crop_rel(c, f)) if s in side.files else None
        if cp is not None:
            crop = decode(str(cp))
            mp = side.files[s].get(mask_rel(c, f))
            cm = np.array(Image.open(mp).convert("1"), dtype=bool) if mp else None
            save_jpeg(draw(crop, cm, []), LOOK / f"{item}__crop{'' if side is after else '_before'}.jpg")
    with open(LOOK / "items.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0]) if items else ["item"])
        w.writeheader()
        w.writerows(items)
    print(f"{len(chosen)} of {len(look)} photos -> {LOOK} ({len(items)} review items in items.csv)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["report", "look"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--after-root", type=Path, default=AFTER, help="the crops after (a copy, to test)")
    ap.add_argument("--partial", action="store_true", help="leave out the sessions not yet re-cropped")
    ap.add_argument("--limit", type=int, default=0, help="look: only the first N photos of the set (a toy run)")
    ap.add_argument("--out", type=Path, default=OUT, help="the report (a test writes elsewhere)")
    a = ap.parse_args()
    a.after_root = a.after_root.resolve()
    if a.step == "report":
        run_report(a)
    else:
        run_look(a)


if __name__ == "__main__":
    main()
