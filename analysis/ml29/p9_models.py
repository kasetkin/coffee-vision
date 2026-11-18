"""ML-29 P9 (D12 step 2): today's models on the scratch photos of P8, against today's canonical photos, for the owner.

    python analysis/ml29/p9_models.py crop      # resumable: the shipped segmenter on each scratch photo
    python analysis/ml29/p9_models.py answer    # resumable: the live classifier and OOD probe, today and scratch
    python analysis/ml29/p9_models.py report    # per camera and per class -> analysis/ml29/p9_report.txt

Every photo P8 applied a gain map to (its log, outputs/ml29_p8/canonical.jsonl). Each is answered twice, from
today's canonical photo and from its scratch one, on this machine at one thread count (docs/agents/experiments.md):

- crop: segcrop's own function (segcrop_session.crop_photo: the shipped segmenter, params.yaml's seg_*, torch at 4
  threads as dvc.yaml's segcrop stage) on the scratch photo, into OUT/crops/scratch/<session>/<class dir>/ as the
  stage writes them (<stem>__cropped.jxl, lossless, and <stem>__beanmask.png; none on a D18 fallback). Today's crop
  is the stage's own, data/segcropped/<session>/...; a folder with no segcrop stage (segmenter_positives,
  ood_negatives) has today's made here too, into OUT/crops/today/. One JSON line per crop in OUT/crops.jsonl (its
  box, fallback and mask hash). P10 trains on these same scratch crops.
- answer: what serving does with a crop and its mask (infer.classify_segmented on the live checkpoint, webapp/app.py's
  CHECKPOINT: 40 patches, seed key [42, 0], the checkpoint's TTA setting, the OOD probe on "auto"), torch at 4
  threads, both crops of a photo in one process. One JSON line per photo in OUT/answers.jsonl: each side's top-1
  country and its probability, verdict and probe score; the masks' overlap (IoU in the canonical photo's frame, each
  mask placed at its crop's box; masks are cropped to their D4 box, so this is the region patches are drawn from).
- report: per camera and per class, the answers that change and which way against the label, the masks' overlap,
  the OOD verdicts that flip. A photo's split is the live model's (exp283), rebuilt as analysis/ml27/folders_eval.py
  does from the pool it trained on and checked against its prediction files: a photo it trained on is reported apart
  from the val and test photos it never saw.

The crops and answers stay under OUT (outputs/, gitignored: the owner's photos' pixels); the report, counts and photo
names only, goes to analysis/ml29/.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "ml29"))
sys.path.insert(0, str(REPO / "analysis" / "ml27"))

import p8_scratch as p8  # noqa: E402

from coffeecv import canonical as cn  # noqa: E402
from coffeecv import crop_io  # noqa: E402

OUT = REPO / "outputs" / "ml29_p9"
REPORT = REPO / "analysis" / "ml29" / "p9_report.txt"
SEGCROPPED = REPO / "data" / "segcropped"
CHECKPOINT = REPO / "models" / "dino3b16_segxl0v7r_country_aug8_ml27_s42.pt"   # webapp/app.py's CHECKPOINT
LIVE_EXP, LIVE_COMMIT = 283, "80c44ac"   # its fit and the commit it trained on (its config.json's git_commit)
THREADS = 4                              # dvc.yaml's segcrop stage; the answers at the same
N_PATCHES, SEED_KEY = 40, [42, 0]        # webapp/app.py's N_PATCHES; classify_segmented's default seed key
NOT_BEANS = "ood_negatives"              # photos with no beans: the probe should refuse them


# ----- which photos, and where their crops are -----

def photos(log: Path) -> list[Path]:
    """The originals P8 applied a map to."""
    return sorted(Path(p) for p, d in p8.read_log(log).items() if not d["error"] and "+" in d["decoder"])


def segcrop_sessions(dvc_yaml: Path) -> set[str]:
    return set(yaml.safe_load(dvc_yaml.read_text())["stages"]["segcrop"]["foreach"])


def crop_file(original: Path, root: Path) -> Path:
    """<root>/<session>/<class dir>/<stem>__cropped.jxl, the segcrop stage's layout (no class dir: the folder)."""
    rel = original.relative_to(p8.DATASET)
    return root / rel.parent / crop_io.crop_name(original.stem)


def today_crop(original: Path, sessions: set[str], out: Path) -> Path:
    """The segcrop stage's crop of today's canonical photo, or the one made here for a folder it does not crop."""
    session = original.relative_to(p8.DATASET).parts[0]
    return crop_file(original, SEGCROPPED if session in sessions else out / "crops" / "today")


def scratch_crop(original: Path, out: Path) -> Path:
    return crop_file(original, out / "crops" / "scratch")


# ----- the masks' overlap -----

def frame_mask(mask: np.ndarray | None, box: list[int] | None, shape: tuple[int, int]) -> np.ndarray:
    """A crop's mask placed in the canonical photo's frame at its box [x0, y0, w, h]; a fallback (no mask): all True,
    as every patch is accepted there."""
    out = np.zeros(shape, bool)
    if mask is None:
        out[:] = True
        return out
    x0, y0, w, h = box
    out[y0:y0 + h, x0:x0 + w] = mask
    return out


def frame_shape(sides: list[tuple[np.ndarray | None, list[int] | None, tuple[int, int]]]) -> tuple[int, int]:
    """The canonical photo's (h, w) from its two crops, without decoding it: a fallback's crop is the whole photo;
    otherwise a frame both boxes fit in, which is all the overlap needs (outside them both masks are empty)."""
    for mask, _, shape in sides:
        if mask is None:
            return shape
    return (max(b[1] + b[3] for _, b, _ in sides), max(b[0] + b[2] for _, b, _ in sides))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# ----- crop -----

def run_crop(a: argparse.Namespace) -> None:
    import torch

    from coffeecv.config import RunConfig
    from coffeecv.segcrop_session import crop_photo, write_crop
    from coffeecv.segment_beans import BeanSegmenter, seg_params
    torch.set_num_threads(a.threads)
    sessions = segcrop_sessions(REPO / "dvc.yaml")
    log = a.out / "crops.jsonl"
    done = set()
    if log.exists():
        done = {(d["kind"], d["path"]) for d in map(json.loads, log.read_text().splitlines())}
    jobs = []
    for p in photos(a.p8_out / "canonical.jsonl")[:a.limit]:
        jobs.append(("scratch", p, p8.scratch_path(p, a.p8_out), scratch_crop(p, a.out)))
        if p.relative_to(p8.DATASET).parts[0] not in sessions:
            jobs.append(("today", p, cn.canonical_path(p), today_crop(p, sessions, a.out)))
    todo = [j for j in jobs if (j[0], str(j[1])) not in done]
    print(f"{len(jobs)} crops ({sum(j[0] == 'today' for j in jobs)} of today's canonical photos, for folders with "
          f"no segcrop stage); {len(jobs) - len(todo)} done, {len(todo)} to do; torch at {a.threads} threads")
    seg = BeanSegmenter(seg_params(RunConfig.from_params_yaml()))
    t0 = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        for i, (kind, original, canonical, out) in enumerate(todo, 1):
            crop = crop_photo(canonical, seg)
            out.parent.mkdir(parents=True, exist_ok=True)
            crop_io.mask_path(out).unlink(missing_ok=True)      # a fallback writes none
            write_crop(crop, out)
            f.write(json.dumps({"kind": kind, "path": str(original), "crop": str(out.relative_to(REPO)),
                                **{k: crop.info[k] for k in ("fallback", "box", "mask_area_frac", "mask_sha256")},
                                "size": [crop.rgb.shape[1], crop.rgb.shape[0]]}) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} crops, {time.perf_counter() - t0:.0f} s", flush=True)


# ----- answer -----

def load_crop(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    rgb = crop_io.load_crop(path)
    mask_file = crop_io.mask_path(path)
    if not mask_file.exists():
        return rgb, None
    with Image.open(mask_file) as im:
        return rgb, np.asarray(im.convert("L")) > 127


def segcrop_box(crop: Path) -> tuple[list[int] | None, bool]:
    """(box, fallback) of a crop the segcrop stage wrote, from its class's segcrop_report.json."""
    stem = crop.name[:-len(crop_io.CROP_SUFFIX)]
    for e in json.loads((crop.parent / "segcrop_report.json").read_text()):
        if Path(e["file"]).stem == stem:
            return e["box"], e["fallback"]
    raise KeyError(f"{crop} not in its segcrop_report.json")


def side(entry: dict, mask: np.ndarray | None) -> dict:
    """One answer, as the report reads it."""
    ranked = entry.get("ranked") or [[None, None, None]]
    ood = entry.get("ood") or {}
    return {"top1": ranked[0][0], "p": None if ranked[0][2] is None else round(ranked[0][2], 4),
            "verdict": entry["verdict"], "probe": ood.get("probe"), "warned": bool(ood.get("warned")),
            "fallback": mask is None}


def run_answer(a: argparse.Namespace) -> None:
    import torch

    from coffeecv.class_list import load_classes
    from coffeecv.infer import (SegmentedPhoto, TorchClassifier, classify_segmented, config_for_checkpoint,
                                inference_tta_for, load_model, load_ood_probe, load_ood_reference)
    torch.set_num_threads(a.threads)
    sessions = segcrop_sessions(REPO / "dvc.yaml")
    made = {(d["kind"], d["path"]): d for d in map(json.loads, (a.out / "crops.jsonl").read_text().splitlines())}
    log = a.out / "answers.jsonl"
    done = {json.loads(x)["path"] for x in log.read_text().splitlines()} if log.exists() else set()
    todo = [p for p in photos(a.p8_out / "canonical.jsonl") if ("scratch", str(p)) in made and str(p) not in done]
    print(f"{len(done)} photos answered, {len(todo)} to do; torch at {a.threads} threads")
    cfg, _ = config_for_checkpoint(CHECKPOINT, None)
    classes = load_classes(REPO / cfg.classes_file)
    keys, labels = list(classes.keys), dict(classes.labels)
    model, head = load_model(CHECKPOINT, cfg.model_name, len(keys))
    classifier = TorchClassifier(model, head)
    tta = inference_tta_for(CHECKPOINT, cfg.model_name)
    ref = load_ood_reference(CHECKPOINT, classifier.embedding_dim)
    probe = load_ood_probe(CHECKPOINT, None, classifier.embedding_dim)
    t0 = time.perf_counter()
    with log.open("a") as f:
        for i, p in enumerate(todo, 1):
            row, masks = {"path": str(p)}, {}
            for kind in ("today", "scratch"):
                crop = today_crop(p, sessions, a.out) if kind == "today" else scratch_crop(p, a.out)
                rgb, mask = load_crop(crop)
                info = made.get((kind, str(p)))
                box, fallback = (info["box"], info["fallback"]) if info else segcrop_box(crop)
                assert fallback == (mask is None), crop
                seg = SegmentedPhoto(rgb, rgb, mask, None, None if fallback else {"box": box},
                                     {"seg_fallback": fallback}, {})
                entry = classify_segmented(seg, cfg, keys, labels, classifier, ref, n_patches=N_PATCHES,
                                           seed_key=SEED_KEY, tta=tta, probe=probe)
                row[kind] = side(entry, mask)
                masks[kind] = (mask, box, rgb.shape[:2])
            shape = frame_shape(list(masks.values()))
            row["mask_iou"] = round(iou(*(frame_mask(m, b, shape) for m, b, _ in masks.values())), 4)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} photos, {time.perf_counter() - t0:.0f} s", flush=True)


# ----- report -----

def flip(today: dict, scratch: dict, truth: str | None) -> str:
    """How a photo's answer moved: same, or for a labelled photo right->wrong / wrong->right / wrong->wrong, else
    changed; a refusal on either side is its own kind."""
    t_ref, s_ref = today["verdict"].startswith("REFUSED"), scratch["verdict"].startswith("REFUSED")
    if t_ref or s_ref:
        return "refused both" if t_ref and s_ref else ("now refused" if s_ref else "no longer refused")
    if today["top1"] == scratch["top1"]:
        return "same"
    if truth is None:
        return "changed"
    if today["top1"] == truth:
        return "right->wrong"
    return "wrong->right" if scratch["top1"] == truth else "wrong->wrong"


def live_splits() -> dict[str, str]:
    """{crop name: train | val | test} of the live model's fit: its pool at LIVE_COMMIT's dvc.lock, split as it drew
    it, checked against its prediction files (folders_eval)."""
    import folders_eval as fe

    from coffeecv.class_list import load_classes
    from coffeecv.infer import config_for_checkpoint
    lock = yaml.safe_load(subprocess.check_output(["git", "show", f"{LIVE_COMMIT}:dvc.lock"], cwd=REPO))
    (pool,) = lock["stages"]["merge_segpool"]["outs"]
    cfg, _ = config_for_checkpoint(CHECKPOINT, None)
    split_map = fe.splits(load_classes(REPO / "dataset" / "classes.txt"), fe.pool_listing(pool["md5"]), cfg)
    fe.check_against_record(LIVE_EXP, split_map)
    return {name: split for split, names in split_map.items() for name in names}


def truth_of(original: Path, classes) -> str | None:
    """The photo's country (its class folder), or None: no class folder, or a folder of photos with no beans."""
    parts = original.relative_to(p8.DATASET).parts
    if len(parts) < 3 or not parts[1].startswith("class_"):
        return None
    import folders_eval as fe
    return classes.key_of(fe.folder_id(parts[1]))


def group_lines(title: str, rows: list[dict], key) -> list[str]:
    kinds = ("same", "right->wrong", "wrong->right", "wrong->wrong", "changed", "now refused", "no longer refused",
             "refused both")
    out = [title, f"  {'':34s} {'photos':>6s} " + " ".join(f"{k:>12s}" for k in kinds)
           + f" {'IoU med':>8s} {'IoU<0.9':>8s}"]
    by = collections.defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    for k in sorted(by, key=str):
        rs = by[k]
        c = collections.Counter(r["flip"] for r in rs)
        ious = sorted(r["mask_iou"] for r in rs)
        out.append(f"  {str(k)[:34]:34s} {len(rs):6d} " + " ".join(f"{c[x]:12d}" for x in kinds)
                   + f" {ious[len(ious) // 2]:8.3f} {sum(x < 0.9 for x in ious):8d}")
    return out


def run_report(a: argparse.Namespace) -> None:
    from coffeecv.class_list import load_classes
    classes = load_classes(REPO / "dataset" / "classes.txt")
    cams = {r["file"]: p8.camera(r) for r in p8.read_scan(a.scan)}
    split = live_splits()
    rows = []
    for d in map(json.loads, (a.out / "answers.jsonl").read_text().splitlines()):
        p = Path(d["path"])
        folder = p.relative_to(p8.DATASET).parts[0]
        truth = truth_of(p, classes)
        crop_name = crop_io.crop_name(p.stem)
        where = ("no beans" if folder == NOT_BEANS else "segmenter set" if truth is None
                 else split.get(crop_name, "not in its pool"))
        rows.append({**d, "camera": cams.get(d["path"], "unknown"), "truth": truth, "where": where,
                     "flip": flip(d["today"], d["scratch"], truth)})
    unseen = [r for r in rows if r["where"] in ("val", "test", "not in its pool")]
    lines = [f"ML-29 P9: the live model ({CHECKPOINT.stem}, exp{LIVE_EXP}) on {len(rows)} photos whose gain map P8 "
             "applied, today's canonical photo against the scratch one.",
             f"Serving's path from the segcrop crop: {N_PATCHES} patches, seed key {SEED_KEY}, the OOD probe; torch "
             f"at {a.threads} threads, one machine. Generated by analysis/ml29/p9_models.py report.", "",
             "Where the photos are: " + ", ".join(f"{k} {v}" for k, v in
                                                  sorted(collections.Counter(r['where'] for r in rows).items())),
             "(train/val/test: the live model's own split; 'not in its pool': labelled but not in the pool it trained "
             "on, so unseen; 'segmenter set': beans with no class; 'no beans': ood_negatives, which it should refuse)",
             ""]
    lines += group_lines("Every photo, by camera:", rows, lambda r: r["camera"])
    lines += [""] + group_lines("Every photo, by where it is:", rows, lambda r: r["where"])
    lines += [""] + group_lines("Every labelled photo, by true class (trained on or not):",
                                [r for r in rows if r["truth"] is not None], lambda r: r["truth"])
    lines += [""] + group_lines("Photos the live model never trained on (val, test, not in its pool), by camera:",
                                unseen, lambda r: r["camera"])
    lines += [""] + group_lines("The same, by true class:", unseen, lambda r: r["truth"])
    labelled = [r for r in unseen if r["truth"] is not None
                and not r["today"]["verdict"].startswith("REFUSED") and not r["scratch"]["verdict"].startswith("REFUSED")]
    for kind in ("today", "scratch"):
        right = sum(r[kind]["top1"] == r["truth"] for r in labelled)
        lines.append(f"Unseen labelled photos answered by both, {kind}: {right}/{len(labelled)} right")
    probe_shift = [r["scratch"]["probe"] - r["today"]["probe"] for r in rows
                   if r["today"]["probe"] is not None and r["scratch"]["probe"] is not None]
    if probe_shift:
        q = np.percentile(probe_shift, [5, 50, 95])
        lines.append(f"The OOD probe's score, scratch minus today: median {q[1]:+.4f}, 5th to 95th percentile "
                     f"{q[0]:+.4f} to {q[2]:+.4f} (refusing above its threshold)")
    warned = collections.Counter((r["today"]["warned"], r["scratch"]["warned"]) for r in rows)
    lines.append(f"Warned (today, scratch): {dict(sorted(warned.items()))}")
    lines += ["", "Photos whose answer or verdict changes:"]
    for r in sorted(rows, key=lambda r: (r["flip"], r["camera"], r["path"])):
        if r["flip"] in ("same", "refused both"):
            continue
        t, s = r["today"], r["scratch"]
        lines.append(f"  {r['flip']:17s} {Path(r['path']).relative_to(p8.DATASET)} ({r['camera']}, {r['where']}, "
                     f"true {r['truth']}): today {t['top1']} {t['p']} {t['verdict']}, scratch {s['top1']} {s['p']} "
                     f"{s['verdict']}; mask IoU {r['mask_iou']}")
    text = "\n".join(lines) + "\n"
    a.report.write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("crop", "answer", "report"))
    ap.add_argument("--scan", type=Path, default=p8.SCAN)
    ap.add_argument("--p8-out", type=Path, default=p8.OUT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--threads", type=int, default=THREADS, help="torch threads (pinned: masks depend on it)")
    ap.add_argument("--limit", type=int, default=None, help="crop: the first N photos only (a toy run)")
    a = ap.parse_args()
    a.out = a.out.resolve()
    {"crop": run_crop, "answer": run_answer, "report": run_report}[a.command](a)


if __name__ == "__main__":
    main()
