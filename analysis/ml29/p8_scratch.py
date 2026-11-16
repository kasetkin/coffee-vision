"""ML-29 P8 (D12 step 1): the gain-map photos canonicalized with the setting on into a scratch store, and how they
change from today's canonical photos, per camera (D13), for the owner.

    python analysis/ml29/p8_scratch.py canonicalize [--jobs 2] [--limit N]   # resumable; --limit for a toy run
    python analysis/ml29/p8_scratch.py report [--jobs 2]
    python analysis/ml29/p8_scratch.py page [--per-camera 4] [--largest 12]
    python analysis/ml29/p8_scratch.py check [--per-camera 2]

- canonicalize: every photo the scan (analysis/ml29/scan_dataset.py, --scan) finds a gain map in, under a folder the
  store canonicalizes (dvc.yaml's canonicalize stages), through coffeecv.canonical with the store's settings and
  Settings.gain_maps on (P7), into OUT/canonical/<its path under dataset/>.jxl: outside data/canonical and DVC, the
  real store untouched. One JSON line per photo in OUT/canonical.jsonl (its decoder names the map applied, e.g.
  "pillow8+Ultra HDR"); a photo already done is skipped, one that failed is tried again. Memory: a 50 MP photo takes
  about 3 GB on the gain-map path, so the default is 2 jobs.
- report: each scratch photo against today's canonical photo, both read as the models read them
  (canonical.load_canonical: 8-bit sRGB), in 8-bit levels: the mean change (signed: brighter is positive), its mean
  absolute value and 99th percentile, over the photo, in the bean region and outside it. The bean region is the
  photo's mask in --masks (data/seg_masks_ml19: the test split's hand-checked masks, ML-19 P6), where it has one.
  Writes OUT/report.csv and prints the summary per camera (the median photo): how many photos change by less than
  a level on average.
- page: OUT/review/index.html for the owner, side by side (today | with the gain map): the photos whose bean region
  (or, without a mask, whole photo) changes most, and a seeded sample per camera; each a full-resolution crop
  around the bean region (CODING_STANDARDS: CV output checked at full resolution) and the whole photo small.
- check: the plan's two checks: the scratch store holds every photo the scan counts (none failed, none missing,
  each with its map read), and a sample of photos without a map canonicalize with the setting on to today's bytes.
  Three iPhone 13 HEICs are counted apart: their Apple map's headroom was in the maker notes, which the strip
  removes, so no map is read and they stay as today. And (D38) every map applied must have one channel: the setting
  applies maps in linear Rec.2020, exact for a one-channel map, not for a three-channel one, which fails the check.

Everything written is under OUT (outputs/, gitignored: the owner's photos' pixels).
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import html
import json
import multiprocessing
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coffeecv import canonical as cn  # noqa: E402

DATASET = REPO / "dataset"
SCAN = REPO / "tmp" / "ml29_p8" / "scan" / "files.csv"
MASKS = REPO / "data" / "seg_masks_ml19" / "xl0_v7r_s123"
OUT = REPO / "outputs" / "ml29_p8"
GAIN_MAP_SETTINGS = dataclasses.replace(cn.STORE_SETTINGS, name="store, gain maps", gain_maps=True)
DATASET_MAX_PIXELS = 2**28    # as coffeecv.canonicalize_folder reads the owner's 200 MP photos
CROP = 1024                   # the review's full-resolution crop, pixels a side
SEED = 29
# The scan's name for an Apple map with no ISO 21496-1 item: the iPhone 13's older form, whose headroom is in Apple's
# maker notes, which the strip removes (ML-3 D13), so no reader can apply it and the photo stays as today.
APPLE_WITHOUT_HEADROOM = "Apple (auxiliary image)"


# ----- which photos -----

def store_folders(dvc_yaml: Path) -> list[str]:
    """The dataset folders the store canonicalizes: dvc.yaml's canonicalize stages."""
    return list(yaml.safe_load(dvc_yaml.read_text())["stages"]["canonicalize"]["foreach"])


def targets(rows: list[dict], folders: list[str], dataset: Path) -> tuple[list[dict], collections.Counter]:
    """(the scan's gain-map photos under a store folder; the other gain-map photos, counted by their folder)."""
    chosen, left_out = [], collections.Counter()
    for r in rows:
        if not r["gain_map"]:
            continue
        folder = Path(r["file"]).relative_to(dataset).parts[0]
        if folder in folders:
            chosen.append(r)
        else:
            left_out[folder] += 1
    return chosen, left_out


def camera(r: dict) -> str:
    return " ".join(x for x in (r.get("make", ""), r.get("model", "")) if x) or "unknown"


def scratch_path(original: Path, out: Path = OUT) -> Path:
    return cn.canonical_path(original, DATASET, out / "canonical")


def read_scan(scan: Path) -> list[dict]:
    return list(csv.DictReader(scan.open()))


# ----- canonicalize -----

def canonicalize_one(job: tuple[Path, Path, int]) -> dict:
    """One photo canonicalized with the gain-map setting into `target`: what the log records of it."""
    original, target, threads = job
    Image.MAX_IMAGE_PIXELS = DATASET_MAX_PIXELS
    t = time.perf_counter()
    data = original.read_bytes()
    try:
        source = cn.decode(data, original.name, settings=GAIN_MAP_SETTINGS)
        made = cn.encode(source, settings=GAIN_MAP_SETTINGS, numthreads=threads)
    except Exception as exc:                      # recorded and tried again next run; `check` fails on it
        return {"path": str(original), "decoder": "", "bytes": 0, "seconds": 0.0, "map_channels": 0,
                "error": f"{type(exc).__name__}: {exc}"}
    channels = 0
    if "+" in source.decoder:                     # D38: which maps the linear Rec.2020 maths is exact for
        from coffeecv import gain_map
        channels = gain_map.read(data).pixels.shape[2]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(made)
    tmp.replace(target)
    return {"path": str(original), "decoder": source.decoder, "bytes": len(made),
            "seconds": round(time.perf_counter() - t, 2), "map_channels": channels, "error": ""}


def read_log(log: Path) -> dict[str, dict]:
    """The log's last line per photo."""
    if not log.exists():
        return {}
    return {d["path"]: d for d in map(json.loads, log.read_text().splitlines())}


def done_paths(log: Path) -> set[str]:
    return {p for p, d in read_log(log).items() if not d["error"]}


def map_channel_problems(log: dict[str, dict]) -> dict[str, list[str]]:
    """ML-29 D38: the photos done whose map has more than one channel (applied in linear Rec.2020, not in the space
    ISO 21496-1 names for it), and those logged before the count was."""
    done = {p: d for p, d in log.items() if not d["error"] and "+" in d["decoder"]}
    return {"three channels": sorted(p for p, d in done.items() if d.get("map_channels", 1) > 1),
            "not counted": sorted(p for p, d in done.items() if "map_channels" not in d)}


def run_canonicalize(a: argparse.Namespace) -> None:
    chosen, left_out = targets(read_scan(a.scan), store_folders(REPO / "dvc.yaml"), DATASET)
    log = a.out / "canonical.jsonl"
    done = done_paths(log)
    todo = [Path(r["file"]) for r in chosen if r["file"] not in done][:a.limit]
    print(f"{len(chosen)} gain-map photos in the store's folders ({sum(left_out.values())} elsewhere: "
          f"{dict(left_out)}); {len(done)} done, {len(todo)} to do, {a.jobs} jobs")
    t0 = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("spawn")) as pool, log.open("a") as f:
        jobs = [(p, scratch_path(p, a.out), a.threads) for p in todo]
        for i, d in enumerate(pool.map(canonicalize_one, jobs), 1):
            f.write(json.dumps(d) + "\n")
            f.flush()
            if d["error"]:
                print(f"  FAILED {d['path']}: {d['error']}")
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} photos, {time.perf_counter() - t0:.0f} s", flush=True)


# ----- report -----

def fit_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A mask at the photo's size (a canonical photo over the store's cap is smaller than its original)."""
    if mask.shape == tuple(shape):
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(im) > 127


def _stats(d: np.ndarray, prefix: str) -> dict:
    if d.size == 0:
        return {f"{prefix}_mean": "", f"{prefix}_mean_abs": "", f"{prefix}_p99_abs": ""}
    a = np.abs(d)
    return {f"{prefix}_mean": float(d.mean()), f"{prefix}_mean_abs": float(a.mean()),
            f"{prefix}_p99_abs": float(np.percentile(a, 99))}


def change(today: np.ndarray, scratch: np.ndarray, mask: np.ndarray | None) -> dict:
    """The scratch photo's change from today's, in 8-bit levels per channel value: over the photo, in the bean region
    and outside it (empty without a mask); `under_a_level`: the photo's mean absolute change is below one level."""
    d = scratch.astype(np.int16) - today.astype(np.int16)
    out = {**_stats(d, "all")}
    if mask is None:
        out.update({**_stats(d[:0], "bean"), **_stats(d[:0], "rest"), "bean_frac": ""})
    else:
        out.update({**_stats(d[mask], "bean"), **_stats(d[~mask], "rest"), "bean_frac": float(mask.mean())})
    out["under_a_level"] = out["all_mean_abs"] < 1
    return out


def mask_index(masks: Path) -> dict[str, Path]:
    """An original's path under the repo -> its mask PNG."""
    index = masks / "index.csv"
    if not index.exists():
        return {}
    return {r["path"]: masks / f"{r['id']}.png" for r in csv.DictReader(index.open())}


def load_mask(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127


def measure_one(job: tuple[str, str, str, str | None, int, Path]) -> dict:
    original, cam, decoder, mask_path, threads, out = job
    p = Path(original)
    row = {"path": str(p.relative_to(REPO)), "camera": cam, "form": decoder.partition("+")[2]}
    try:
        today = cn.load_canonical(cn.canonical_path(p).read_bytes(), numthreads=threads)
        scratch = cn.load_canonical(scratch_path(p, out).read_bytes(), numthreads=threads)
    except (OSError, cn.NotCanonical, cn.Unreadable) as exc:
        return {**row, "error": f"{type(exc).__name__}: {exc}"}
    if today.shape != scratch.shape:
        return {**row, "error": f"sizes differ: {today.shape} today, {scratch.shape} scratch"}
    mask = load_mask(Path(mask_path) if mask_path else None)
    mask = None if mask is None else fit_mask(mask, today.shape[:2])
    return {**row, "mask": mask is not None, **change(today, scratch, mask), "error": ""}


REPORT_FIELDS = ["path", "camera", "form", "mask", "bean_frac", "all_mean", "all_mean_abs", "all_p99_abs",
                 "bean_mean", "bean_mean_abs", "bean_p99_abs", "rest_mean", "rest_mean_abs", "rest_p99_abs",
                 "under_a_level", "error"]


def summary(rows: list[dict]) -> list[dict]:
    """Per camera, the median photo's change, over the photo and (photos with a mask) in and outside the bean
    region; how many photos change by less than a level on average."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["camera"]].append(r)

    def median(rs: list[dict], key: str):
        values = [float(r[key]) for r in rs if r[key] != ""]
        return statistics.median(values) if values else ""

    out = []
    for cam in sorted(by):
        rs = by[cam]
        out.append({"camera": cam, "photos": len(rs), "with_mask": sum(r["bean_mean_abs"] != "" for r in rs),
                    "under_a_level": sum(str(r["under_a_level"]) == "True" for r in rs),
                    **{k: median(rs, k) for k in ("all_mean", "all_mean_abs", "bean_mean", "bean_mean_abs",
                                                   "rest_mean_abs")}})
    return out


def print_summary(rows: list[dict]) -> None:
    def f(x):
        return f"{x:+.2f}" if isinstance(x, float) else "-"
    print(f"{'camera':34s} {'photos':>6s} {'masks':>5s} {'<1 lvl':>6s}  {'all':>6s} {'|all|':>6s}  "
          f"{'bean':>6s} {'|bean|':>6s} {'|rest|':>6s}   (8-bit levels, the median photo)")
    for s in summary(rows):
        print(f"{s['camera'][:34]:34s} {s['photos']:6d} {s['with_mask']:5d} {s['under_a_level']:6d}  "
              f"{f(s['all_mean']):>6s} {f(s['all_mean_abs']):>6s}  {f(s['bean_mean']):>6s} "
              f"{f(s['bean_mean_abs']):>6s} {f(s['rest_mean_abs']):>6s}")


def run_report(a: argparse.Namespace) -> None:
    cams = {r["file"]: camera(r) for r in read_scan(a.scan)}
    masks = mask_index(a.masks)
    done = {p: d for p, d in read_log(a.out / "canonical.jsonl").items() if not d["error"]}
    jobs = [(p, cams.get(p, "unknown"), d["decoder"], str(masks.get(str(Path(p).relative_to(REPO)), "")) or None,
             a.threads, a.out) for p, d in sorted(done.items())]
    print(f"{len(jobs)} scratch photos, {sum(j[3] is not None for j in jobs)} with a mask; {a.jobs} jobs")
    rows = []
    with ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, r in enumerate(pool.map(measure_one, jobs), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    with (a.out / "report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, REPORT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["error"]]
    for r in bad:
        print(f"  NOT MEASURED {r['path']}: {r['error']}")
    print_summary([r for r in rows if not r["error"]])
    print(f"-> {(a.out / 'report.csv').relative_to(REPO)}")


# ----- the review page -----

def crop_box(mask: np.ndarray | None, shape: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1): a window of `size` (or the photo, if smaller) centred on the bean region's bounding box,
    or on the photo's centre without a mask, moved inside the photo."""
    h, w = shape
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
        cy, cx = (ys[0] + ys[-1] + 1) // 2, (xs[0] + xs[-1] + 1) // 2
    else:
        cy, cx = h // 2, w // 2
    sh, sw = min(size, h), min(size, w)
    y0 = int(min(max(cy - sh // 2, 0), h - sh))
    x0 = int(min(max(cx - sw // 2, 0), w - sw))
    return y0, y0 + sh, x0, x0 + sw


def chosen_for_page(rows: list[dict], per_camera: int, largest: int) -> tuple[list[dict], dict[str, list[dict]]]:
    """(the largest changes, by the bean region's mean absolute change or the photo's without a mask; a seeded
    sample per camera of the rest)."""
    def size(r):
        return float(r["bean_mean_abs"] if r["bean_mean_abs"] != "" else r["all_mean_abs"])
    top = sorted(rows, key=size, reverse=True)[:largest]
    taken = {r["path"] for r in top}
    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for r in rows:
        if r["path"] not in taken:
            by[r["camera"]].append(r)
    return top, {cam: rng.sample(rs, min(per_camera, len(rs))) for cam, rs in sorted(by.items())}


def _save(a: np.ndarray, path: Path, width: int | None = None) -> None:
    im = Image.fromarray(a)
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(path, quality=92)


def page_item(r: dict, masks: dict[str, Path], out: Path, threads: int) -> str:
    original = REPO / r["path"]
    today = cn.load_canonical(cn.canonical_path(original).read_bytes(), numthreads=threads)
    scratch = cn.load_canonical(scratch_path(original, out).read_bytes(), numthreads=threads)
    mask = load_mask(masks.get(r["path"]))
    mask = None if mask is None else fit_mask(mask, today.shape[:2])
    y0, y1, x0, x1 = crop_box(mask, today.shape[:2], CROP)
    stem, img = r["path"].replace("/", "__"), out / "review" / "img"
    for kind, a in (("today", today), ("scratch", scratch)):
        _save(a[y0:y1, x0:x1], img / f"{stem}.{kind}.crop.jpg")
        _save(a, img / f"{stem}.{kind}.whole.jpg", width=480)

    def n(key):
        return f"{float(r[key]):+.2f}" if r[key] != "" else "-"
    region = "bean region" if mask is not None else "photo centre (no mask)"
    pics = "".join(f'<figure><img src="img/{html.escape(stem)}.{k}.{part}.jpg"><figcaption>{label}</figcaption></figure>'
                   for part in ("crop", "whole") for k, label in (("today", "today"), ("scratch", "with the gain map")))
    return (f'<section><h3>{html.escape(Path(r["path"]).name)}</h3><p>{html.escape(r["camera"])}, '
            f'{html.escape(r["form"] or "no map read")}; the crop: {region}, full resolution. Mean change (levels): '
            f'photo {n("all_mean")}, bean region {n("bean_mean")} (|{n("bean_mean_abs")}|), outside '
            f'{n("rest_mean")}</p><div class="pair">{pics}</div></section>')


def run_page(a: argparse.Namespace) -> None:
    rows = [r for r in csv.DictReader((a.out / "report.csv").open()) if not r["error"]]
    masks = mask_index(a.masks)
    review = a.out / "review"
    (review / "img").mkdir(parents=True, exist_ok=True)
    top, sample = chosen_for_page(rows, a.per_camera, a.largest)
    parts = ["<!doctype html><meta charset='utf-8'><title>ML-29 P8: gain maps, side by side</title><style>"
             "body{font-family:sans-serif;margin:1em}.pair{display:grid;grid-template-columns:1fr 1fr;gap:6px}"
             "figure{margin:0}img{width:100%}figcaption{font-size:.8em;color:#555}</style>",
             f"<h1>ML-29 P8: today's canonical photo | with its gain map</h1><p>{len(rows)} photos measured; "
             "levels are 8-bit sRGB as the models read them; positive is brighter.</p>",
             f"<h2>The {len(top)} largest changes</h2>"]
    parts += [page_item(r, masks, a.out, a.threads) for r in top]
    for cam, rs in sample.items():
        parts.append(f"<h2>{html.escape(cam)}: {len(rs)} at random</h2>")
        parts += [page_item(r, masks, a.out, a.threads) for r in rs]
    (review / "index.html").write_text("\n".join(parts))
    print(f"-> {(review / 'index.html').relative_to(REPO)}: {len(top)} largest, "
          f"{sum(len(v) for v in sample.values())} sampled over {len(sample)} cameras")


# ----- the plan's checks -----

def run_check(a: argparse.Namespace) -> None:
    rows = read_scan(a.scan)
    folders = store_folders(REPO / "dvc.yaml")
    chosen, left_out = targets(rows, folders, DATASET)
    log = read_log(a.out / "canonical.jsonl")
    scanned = sum(1 for r in rows if r["gain_map"])
    ok = [log[r["file"]] for r in chosen if r["file"] in log and not log[r["file"]]["error"]]
    failed = [r["file"] for r in chosen if r["file"] in log and log[r["file"]]["error"]]
    missing = [r["file"] for r in chosen if r["file"] not in log]
    form = {r["file"]: r["gain_map"] for r in chosen}
    unread = [d["path"] for d in ok if "+" not in d["decoder"]]
    headless = [p for p in unread if form[p] == APPLE_WITHOUT_HEADROOM]
    unread = [p for p in unread if p not in headless]
    print(f"1. Counts: the scan finds {scanned} gain-map photos, {len(chosen)} in the store's folders and "
          f"{sum(left_out.values())} elsewhere ({dict(left_out)}); the scratch store holds {len(ok)}; "
          f"{len(failed)} failed, {len(missing)} not yet done, {len(unread)} with no map read; "
          f"{len(headless)} Apple maps without their headroom, left as today (expected: see the docstring)")
    for p in failed + unread[:20]:
        print(f"   {p}: {log[p]['error'] or log[p]['decoder']}")
    for p in headless:
        print(f"   no headroom: {p}")
    channels = map_channel_problems({p: log[p] for p in (r["file"] for r in chosen) if p in log})
    print(f"   D38: {len(channels['three channels'])} maps of three channels (linear Rec.2020 not their maths), "
          f"{len(channels['not counted'])} logged before the count")
    for kind, paths in channels.items():
        for p in paths[:20]:
            print(f"   {kind}: {p}")
    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for r in rows:
        if not r["gain_map"] and Path(r["file"]).relative_to(DATASET).parts[0] in folders:
            by[camera(r)].append(Path(r["file"]))
    sample = [p for cam in sorted(by) for p in rng.sample(by[cam], min(a.per_camera, len(by[cam])))]
    differ = []
    for p in sample:
        made = cn.encode(cn.decode(p.read_bytes(), p.name, settings=GAIN_MAP_SETTINGS), settings=GAIN_MAP_SETTINGS,
                         numthreads=a.threads)
        if made != cn.canonical_path(p).read_bytes():
            differ.append(str(p))
    print(f"2. Without a map: {len(sample)} photos ({a.per_camera} per camera, {len(by)} cameras) with the setting "
          f"on, {len(sample) - len(differ)} the same bytes as today's canonical photo")
    for p in differ:
        print(f"   DIFFERS {p}")
    good = not (failed or missing or unread or differ or channels["three channels"] or channels["not counted"]) \
        and len(ok) == len(chosen)
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("canonicalize", "report", "page", "check"))
    ap.add_argument("--scan", type=Path, default=SCAN)
    ap.add_argument("--masks", type=Path, default=MASKS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--jobs", type=int, default=2, help="photos at a time (default 2: memory, see above)")
    ap.add_argument("--threads", type=int, default=cn.THREADS, help="libjxl threads per photo (no effect on output)")
    ap.add_argument("--limit", type=int, default=None, help="canonicalize: at most this many photos (a toy run)")
    ap.add_argument("--per-camera", type=int, default=None, help="page: 4 per camera; check: 2 per camera")
    ap.add_argument("--largest", type=int, default=12, help="page: the largest changes shown")
    a = ap.parse_args()
    a.out = a.out.resolve()
    if a.command == "canonicalize":
        run_canonicalize(a)
    elif a.command == "report":
        run_report(a)
    elif a.command == "page":
        a.per_camera = 4 if a.per_camera is None else a.per_camera
        run_page(a)
    else:
        a.per_camera = 2 if a.per_camera is None else a.per_camera
        run_check(a)


if __name__ == "__main__":
    main()
