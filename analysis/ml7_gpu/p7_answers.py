"""ML-7 P7: does DINOv3 in fp16 on LiteRT's GPU change an answer? (plan P7). The 13 fixtures cannot show a rare
flip, least of all a guard verdict on a true negative, so this runs serving's answers over the photos of
p7_answer_photos.txt (every true negative, neg_ood, and about 200 positives) with each readout set: the fp32 GPU
cell, the fp16 one(s) and, optionally, PyTorch. Three steps, each resumable (a photo already on disk is skipped),
one line per photo; a photo that fails is kept by name and the run goes on:

  patches  release venv. Serving's own preprocessing, as make_refs.make_refs does it: patches_for_photo(path,
           cfg, 40, [42, 0]) (the segmenter inside) and build_eval_transform, to float32 [40, 3, 224, 224] saved
           as OUT/patches/<sha256>.npy, so every cell reads the same bytes; OUT/patches.jsonl holds each photo's
           role, path, seg_fallback, mask_area_frac, time and the .npy's sha256. --torch adds PyTorch's readouts
           (forward_with_embeddings) as a readout set of its own, "torch" (about 22 s a photo on yappy).
  run      venv_litert. One cell's DINOv3 (dino_b8: 5 calls a photo) over each photo's patches, the readouts
           [40, 1536] to OUT/readouts/<cell>/<sha256>.npy; OUT/readouts/<cell>.jsonl holds each photo's time and
           the sha256 of the patches it read; OUT/readouts/<cell>.runs.jsonl one record per invocation: the
           model file's sha256, load and wall time, the delegate's report and the GPU proof (cellrun).
  compare  release venv. Each set's answers, by parity._answers (the head, the mean probabilities and the
           guard, as classify_one gives them), compared in pairs: each set against --base (the fp32 cell), and
           every set against torch where it is there. Per photo: top-1 and the guard's score and verdict per set;
           per pair the least patch cosine, the largest class-probability change, the score's change and whether
           top-1, the first two in order and the verdict held. The summary per pair: the flips by role, the
           least cosine's distribution, the photos nearest the guard's threshold and how far their score moved,
           the photos with the smallest top-1 margin and whether the order held. A JSON (paths under the repo
           and under ~/ml7 only) and a short summary on stdout.
  seeds    release venv. Patch sampling's spread against fp16's drift, over runs whose patches were cut with other
           seeds (patches --seed-key 42 k, each in its own OUT; p7_served_photos.txt): per readout set, every
           pair of seeds compared photo by photo (top-1, verdict, largest probability change, margin change);
           fp16 against fp32 per seed; then the two pooled side by side (flip counts with Wilson intervals) and,
           for each photo fp16 flipped, whether its top-1 also changes between seeds. Photos: those every run of
           a set answered. E.g. on yappy:

    ML7_REPO=/opt/coffee-cv/current /opt/coffee-cv/current/.venv/bin/python p7_answers.py seeds \\
        --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt --runs seed0=p7/answers seed1=p7/seed1 \\
        seed2=p7/seed2 seed3=p7/seed3 --base seed0 --sets litert_gpu litert_gpu_fp16 --fp32 litert_gpu \\
        --fp16 litert_gpu_fp16 --json p7/seeds.json

On yappy, from ~/ml7 (p7_answers.sh runs them in order); the photos under p7/photos/<path in the list>:

    ML7_REPO=/opt/coffee-cv/current /opt/coffee-cv/current/.venv/bin/python p7_answers.py patches \\
        --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt --root p7/photos --out p7/answers
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json venv_litert/bin/python p7_answers.py run \\
        --cell litert_gpu --model-dir p7/whole --runtime-dir rt_vulkan --out p7/answers
    VK_ICD_FILENAMES=... venv_litert/bin/python p7_answers.py run --cell litert_gpu_fp16 --model-dir p7/fp16 \\
        --runtime-dir rt_vulkan --out p7/answers
    ML7_REPO=/opt/coffee-cv/current /opt/coffee-cv/current/.venv/bin/python p7_answers.py compare \\
        --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt --out p7/answers --base litert_gpu \\
        --json p7/answers/p7_answers.json

The comparison's logic is plain functions over arrays and answers (tests/test_ml7_answers.py).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parity  # noqa: E402  (numpy only at import; the venv_litert step uses it too)

LIST = HERE / "p7_answer_photos.txt"
N_PATCHES, SEED_KEY = 40, [42, 0]            # classify_one's own (make_refs.make_refs)
TORCH = "torch"                              # PyTorch's readout set, made by `patches --torch`
NEAR = 15                                    # photos listed nearest the threshold, and with the smallest margin
PERCENTILES = (0.1, 1, 5, 50)


# ---------------------------------------------------------------------------------------------------- files
def read_list(path: Path) -> list[tuple[str, str, str]]:
    """(role, sha256, path under the repo) per line of a `<role> <sha256> <path>` list."""
    out = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            role, sha, rel = line.split(maxsplit=2)
            out.append((role, sha, rel))
    return out


def last_rows(path: Path) -> dict[str, dict]:
    """A jsonl's rows by sha256, the last one winning (a photo retried after an error); a line cut off by a kill
    is skipped. Insertion order is the first appearance, so the list's order is kept."""
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[r["sha256"]] = r
    return rows


def last_rows_all(path: Path) -> list[dict]:
    """Every row of a jsonl (a cell's run records), a cut-off line skipped."""
    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def save_array(path: Path, a: np.ndarray) -> str:
    """np.save to a temporary file, then renamed into place (a killed run leaves no half file that a resume
    would skip). Returns the sha256 of the bytes written."""
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    data = buf.getvalue()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def load_array(path: Path) -> tuple[np.ndarray, str]:
    """The array and the sha256 of its file's bytes: what a cell read, to check the cells read the same."""
    data = path.read_bytes()
    return np.load(io.BytesIO(data), allow_pickle=False), hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path | None) -> str | None:
    """A path for a committed record: as given if relative (to ~/ml7), else relative to the working directory,
    else its name alone (no host paths)."""
    if path is None or not path.is_absolute():
        return None if path is None else str(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return path.name


def _progress(k: int, n: int, role: str, rel: str, s: float, t0: float, error: str | None = None) -> None:
    print(f"photo {k}/{n} {role} {Path(rel).name} {s:.1f} s{' ERROR ' + error if error else ''} "
          f"(elapsed {time.perf_counter() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------------------------- the comparison (pure)
def answer_view(ans: dict) -> dict:
    """One set's answer for a photo, from parity._answers: top-1 and second (argmax's order on a tie), top-1's
    margin over the second in mean probability, and the guard's score, threshold and verdict if there is one."""
    probs = np.asarray(ans["probs"], dtype=np.float64)
    order = np.argsort(-probs, kind="stable")
    v = {"top1": int(order[0]), "top2": int(order[1]), "margin": float(probs[order[0]] - probs[order[1]])}
    if "score" in ans:
        v.update(score=float(ans["score"]), threshold=float(ans["threshold"]), refused=bool(ans["refused"]))
    return v


def answer_change(ref_ans: dict, got_ans: dict) -> dict:
    """One photo's answer under one set (got) against another (ref): whether top-1, the first two in order and the
    guard's verdict held, the largest class-probability change, both margins, the margin's change (the reference's
    top-1 followed: its lead over the best other class in got, less its lead in ref, so a flip is a change past
    -ref_margin), and the guard's score change beside the reference's distance from the threshold."""
    a, b = answer_view(ref_ans), answer_view(got_ans)
    ref_p, got_p = np.asarray(ref_ans["probs"], np.float64), np.asarray(got_ans["probs"], np.float64)
    lead = float(got_p[a["top1"]] - np.delete(got_p, a["top1"]).max())
    m = {"top1_same": a["top1"] == b["top1"], "top2_same": (a["top1"], a["top2"]) == (b["top1"], b["top2"]),
         "prob_max_change": float(np.abs(got_p - ref_p).max()),
         "ref_margin": a["margin"], "got_margin": b["margin"], "margin_change": lead - a["margin"]}
    if "score" in a and "score" in b:
        m.update(ref_score=a["score"], distance_from_threshold=a["score"] - a["threshold"],
                 score_change=b["score"] - a["score"], verdict_same=a["refused"] == b["refused"])
    return m


def pair_measures(ref_readouts: np.ndarray, got_readouts: np.ndarray, ref_ans: dict, got_ans: dict) -> dict:
    """One photo, one set (got) against another (ref) on the same patches: the least patch cosine, NaN or Inf in
    got, and the answer's change (answer_change)."""
    return {"least_cosine": parity.least_cosine(got_readouts, ref_readouts),
            "nonfinite": not bool(np.isfinite(got_readouts).all()), **answer_change(ref_ans, got_ans)}


def pair_key(ref: str, got: str) -> str:
    return f"{got}_vs_{ref}"


def pairs_for(sets: list[str], base: str) -> list[tuple[str, str]]:
    """(ref, got): each set against the base (the fp32 cell), then every set against PyTorch where it is there."""
    out = [(base, s) for s in sets if base in sets and s not in (base, TORCH)]
    if TORCH in sets:
        out += [(TORCH, s) for s in sets if s != TORCH]
    return out


def _dist(values: list[float], percentiles: tuple = PERCENTILES) -> dict | None:
    if not values:
        return None
    v = np.asarray(values, dtype=np.float64)
    return {"min": float(v.min()), **{f"p{p:g}": float(np.percentile(v, p)) for p in percentiles},
            "max": float(v.max())}


def summarise_pair(rows: list[dict], key: str, k: int = NEAR) -> dict:
    """One pair over the photos that have it: the top-1 and verdict flips, listed and counted by role (with the
    changes in the first two's order); the least cosine's distribution (NaN left out, the photos with a NaN or Inf
    listed); the largest probability and score changes; the k photos nearest the guard's threshold, with their
    score's move as a share of that distance; the k with the smallest top-1 margin, and whether the order held."""
    have = [(r, r["pairs"][key]) for r in rows if key in r["pairs"]]
    by_role: dict[str, dict] = {}
    for r, m in have:
        b = by_role.setdefault(r["role"], {"photos": 0, "top1_flips": 0, "verdict_flips": 0,
                                           "top2_order_changes": 0})
        b["photos"] += 1
        b["top1_flips"] += int(not m["top1_same"])
        b["verdict_flips"] += int(not m.get("verdict_same", True))
        b["top2_order_changes"] += int(not m["top2_same"])
    guarded = [(r, m) for r, m in have if "score_change" in m]
    probs = [m["prob_max_change"] for _, m in have if np.isfinite(m["prob_max_change"])]
    near = sorted(guarded, key=lambda rm: abs(rm[1]["distance_from_threshold"]))[:k]
    small = sorted(have, key=lambda rm: rm[1]["ref_margin"])[:k]
    return {
        "photos": len(have), "by_role": dict(sorted(by_role.items())),
        "top1_flips": [r["path"] for r, m in have if not m["top1_same"]],
        "verdict_flips": [r["path"] for r, m in have if not m.get("verdict_same", True)],
        "nonfinite": [r["path"] for r, m in have if m["nonfinite"]],
        "least_cosine": _dist([m["least_cosine"] for _, m in have if np.isfinite(m["least_cosine"])]),
        "prob_max_change": _dist(probs),
        "score_change_max_abs": max((abs(float(m["score_change"])) for _, m in guarded), default=None),
        "near_threshold": [
            {"photo": r["path"], "role": r["role"], "ref_score": float(m["ref_score"]),
             "distance": float(m["distance_from_threshold"]), "score_change": float(m["score_change"]),
             "moved_of_distance": (abs(float(m["score_change"])) / abs(float(m["distance_from_threshold"]))
                                   if m["distance_from_threshold"] else None),
             "verdict_same": bool(m["verdict_same"])} for r, m in near],
        "smallest_margin": [
            {"photo": r["path"], "role": r["role"], "ref_margin": float(m["ref_margin"]),
             "got_margin": float(m["got_margin"]), "top1_same": bool(m["top1_same"]),
             "top2_same": bool(m["top2_same"])} for r, m in small],
    }


PHOTO_PAIR_KEYS = ("least_cosine", "nonfinite", "top1_same", "top2_same", "verdict_same", "prob_max_change",
                   "score_change")


def photo_record(row: dict) -> dict:
    """A photo's row for the JSON: each pair's measures less what the sets' answers already give (margins, the
    reference's score), the probability and score changes and the margins to 6 significant digits; the cosine and
    the guard's score kept whole."""
    def g6(x):
        return float(f"{x:.6g}")
    sets = {s: {**v, "margin": g6(v["margin"])} for s, v in row["sets"].items()}
    pairs = {}
    for key, m in row["pairs"].items():
        kept = {k: m[k] for k in PHOTO_PAIR_KEYS if k in m}
        pairs[key] = {**kept, **{k: g6(kept[k]) for k in ("prob_max_change", "score_change") if k in kept}}
    return {**row, "sets": sets, "pairs": pairs}


def dumps(out: dict) -> str:
    """The JSON, indented, with one line per photo (about 800 photos)."""
    head = json.dumps({**out, "per_photo": "__PER_PHOTO__"}, indent=1, default=float)
    rows = ",\n".join("  " + json.dumps(r, default=float) for r in out["per_photo"])
    return head.replace('"__PER_PHOTO__"', "[\n" + rows + "\n ]" if rows else "[]") + "\n"


def summarise_sets(rows: list[dict]) -> dict:
    """Per set, per role: the photos it answered and how many the guard refused."""
    out: dict[str, dict] = {}
    for r in rows:
        for s, v in r["sets"].items():
            b = out.setdefault(s, {}).setdefault(r["role"], {"photos": 0, "refused": 0})
            b["photos"] += 1
            b["refused"] += int(bool(v.get("refused")))
    return {s: dict(sorted(v.items())) for s, v in sorted(out.items())}


# ------------------------------------------------------------------------------ patch sampling's spread (pure)
SEED_PERCENTILES = (50, 95)
WILSON_Z = 1.959964                                # 95 %


def seed_pairs(names: list[str]) -> list[tuple[str, str]]:
    """(ref, got): every pair of runs once, in the order given (no seed is the truth; base first)."""
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]


def wilson(k: int, n: int, z: float = WILSON_Z) -> list[float] | None:
    """Wilson's score interval for k events in n trials (95 % by default); None for no trials."""
    if n == 0:
        return None
    p, z2 = k / n, z * z
    centre, half = (p + z2 / (2 * n)) / (1 + z2 / n), z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [max(0.0, float(centre - half)), min(1.0, float(centre + half))]


def summarise_changes(ms: list[dict]) -> dict:
    """answer_change measures pooled: the top-1 flips (count, rate, Wilson 95 %), the verdict flips, and the
    distributions of the largest class-probability change and of the margin's change, in absolute value."""
    n, k = len(ms), sum(not m["top1_same"] for m in ms)
    return {"photos": n, "top1_flips": k, "top1_flip_rate": k / n if n else None,
            "top1_flip_wilson95": wilson(k, n), "verdict_flips": sum(not m.get("verdict_same", True) for m in ms),
            "prob_max_change": _dist([m["prob_max_change"] for m in ms], SEED_PERCENTILES),
            "margin_change_abs": _dist([abs(m["margin_change"]) for m in ms], SEED_PERCENTILES)}


def sampling_spread(answers: dict[str, dict[str, dict]], photos: list[str]) -> dict:
    """One readout set's answers under each seed's patches ({run: {photo: parity._answers}}), over the photos
    given: every pair of seeds compared photo by photo (answer_change), summarised per pair and pooled over pairs,
    and how many photos' top-1 is not the same under every seed."""
    pairs = seed_pairs(list(answers))
    per_photo = {p: {pair_key(a, b): answer_change(answers[a][p], answers[b][p]) for a, b in pairs} for p in photos}
    return {"pairs": {pair_key(a, b): summarise_changes([per_photo[p][pair_key(a, b)] for p in photos])
                      for a, b in pairs},
            "pooled": summarise_changes([m for by_pair in per_photo.values() for m in by_pair.values()]),
            "photos_top1_not_unanimous": sum(len({answer_view(answers[r][p])["top1"] for r in answers}) > 1
                                             for p in photos),
            "per_photo": per_photo}


def precision_drift(fp32: dict[str, dict[str, dict]], fp16: dict[str, dict[str, dict]], photos: list[str]) -> dict:
    """fp16 against fp32 on the same patches, per seed (the runs that have both sets) and pooled over seeds."""
    runs = [r for r in fp32 if r in fp16]
    per_photo = {p: {r: answer_change(fp32[r][p], fp16[r][p]) for r in runs} for p in photos}
    return {"per_seed": {r: summarise_changes([per_photo[p][r] for p in photos]) for r in runs},
            "pooled": summarise_changes([m for by_run in per_photo.values() for m in by_run.values()]),
            "per_photo": per_photo}


def comparison(spread: dict, drift: dict, fp32: dict[str, dict[str, dict]], fp16: dict[str, dict[str, dict]],
               photos: list[str], ids: list | None = None) -> dict:
    """Sampling's spread (fp32's, between seeds) beside fp16's drift (against fp32, per seed): the pooled flip
    counts and rates with their intervals, the probability changes side by side, and each photo fp16 flipped under
    any seed: fp32's top-1 and margin per seed, fp16's margin change, and whether its fp32 top-1 also changes
    between seeds ([pairs flipped, pairs]), i.e. whether fp16 only flips calls that sampling already flips."""
    def name(i):
        return ids[i] if ids else i
    keys = ("photos", "top1_flips", "top1_flip_rate", "top1_flip_wilson95", "verdict_flips")
    flipped = []
    for p in photos:
        in_runs = [r for r, m in drift["per_photo"][p].items() if not m["top1_same"]]
        if not in_runs:
            continue
        s = list(spread["per_photo"][p].values())
        k = sum(not m["top1_same"] for m in s)
        flipped.append({
            "photo": p, "fp16_flipped_in": in_runs,
            "fp32_top1": {r: name(answer_view(fp32[r][p])["top1"]) for r in drift["per_photo"][p]},
            "fp16_top1": {r: name(answer_view(fp16[r][p])["top1"]) for r in drift["per_photo"][p]},
            "fp32_margin": {r: m["ref_margin"] for r, m in drift["per_photo"][p].items()},
            "fp16_margin_change": {r: m["margin_change"] for r, m in drift["per_photo"][p].items()},
            "sampling_pairs_flipped": [k, len(s)], "sampling_flips": k > 0})
    return {"sampling": {k: spread["pooled"][k] for k in keys}, "fp16": {k: drift["pooled"][k] for k in keys},
            "prob_max_change": {"sampling": spread["pooled"]["prob_max_change"],
                                "fp16": drift["pooled"]["prob_max_change"]},
            "margin_change_abs": {"sampling": spread["pooled"]["margin_change_abs"],
                                  "fp16": drift["pooled"]["margin_change_abs"]},
            "fp16_flipped": flipped,
            "fp16_flipped_photos_sampling_also_flips": [sum(f["sampling_flips"] for f in flipped), len(flipped)]}


# ------------------------------------------------------------------------------------------------- patches
def patches(args) -> None:
    import torch

    import graphs  # noqa: F401  (puts the repo, or the release under ML7_REPO, on sys.path)
    from coffeecv.infer import forward_with_embeddings, inference_tta_for, patches_for_photo
    from coffeecv.transforms import build_eval_transform
    from make_refs import Context

    torch.set_num_threads(args.threads)
    photos = read_list(args.photos)[:args.limit]
    pdir, tdir = args.out / "patches", args.out / "readouts" / TORCH
    pdir.mkdir(parents=True, exist_ok=True)
    if args.torch:
        tdir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in photos if not (pdir / f"{p[1]}.npy").exists()
            or (args.torch and not (tdir / f"{p[1]}.npy").exists())]
    print(f"{len(photos)} photos, {len(photos) - len(todo)} done", flush=True)
    if not todo:
        return
    ctx = Context.load(args.checkpoint)
    if inference_tta_for(ctx.checkpoint, ctx.cfg.model_name):     # as make_refs: the graphs are one plain pass
        raise SystemExit("p7_answers: this checkpoint is served with dihedral TTA; the graphs do not cover it")
    transform = build_eval_transform(ctx.cfg.patch_resize)
    t0 = time.perf_counter()
    with (args.out / "patches.jsonl").open("a") as f, (tdir.with_suffix(".jsonl").open("a") if args.torch
                                                         else contextlib.nullcontext()) as ft:
        for k, (role, sha, rel) in enumerate(todo, 1):
            t = time.perf_counter()
            path, npy = args.root / rel, pdir / f"{sha}.npy"
            error = None
            if not npy.exists():
                row: dict = {"sha256": sha, "role": role, "path": rel, "seed_key": args.seed_key}
                try:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                        raise ValueError("the photo does not match its sha256 in the list")
                    with torch.no_grad():
                        crops, info = patches_for_photo(path, ctx.cfg, N_PATCHES, args.seed_key)
                        x = torch.stack([transform(c) for c in crops]).numpy().astype(np.float32, copy=False)
                    if x.shape != (N_PATCHES, 3, 224, 224):
                        raise ValueError(f"patches of shape {x.shape}")
                    row |= {"patch_sha256": save_array(npy, x), "seg_fallback": bool(info.get("seg_fallback")),
                            "mask_area_frac": info.get("mask_area_frac")}
                except Exception as e:                                  # noqa: BLE001  kept by name, the run goes on
                    row["error"] = error = f"{type(e).__name__}: {e}"
                row["s"] = round(time.perf_counter() - t, 2)
                f.write(json.dumps(row) + "\n")
                f.flush()
            if ft is not None and error is None and not (tdir / f"{sha}.npy").exists():
                tt = time.perf_counter()
                trow: dict = {"sha256": sha}
                try:
                    x, input_sha = load_array(npy)
                    with torch.no_grad():
                        _, y = forward_with_embeddings(ctx.model, ctx.head_module, torch.from_numpy(x))
                    y = np.asarray(y, dtype=np.float32)
                    save_array(tdir / f"{sha}.npy", y)
                    trow |= {"input_sha256": input_sha, "nonfinite": not bool(np.isfinite(y).all())}
                except Exception as e:                                  # noqa: BLE001  kept by name, the run goes on
                    trow["error"] = error = f"torch: {type(e).__name__}: {e}"
                trow["s"] = round(time.perf_counter() - tt, 2)
                ft.write(json.dumps(trow) + "\n")
                ft.flush()
            _progress(k, len(todo), role, rel, time.perf_counter() - t, t0, error)


# ----------------------------------------------------------------------------------------------------- run
def run(args) -> None:
    import backends
    import cellrun
    from run_litert import CapturedStderr, delegate_report

    if args.cell not in backends.CELLS:
        raise SystemExit(f"p7_answers: no cell {args.cell} in backends.CELLS")
    index = [r for r in last_rows(args.out / "patches.jsonl").values() if "error" not in r][:args.limit]
    rdir = args.out / "readouts"
    dest = rdir / args.cell
    dest.mkdir(parents=True, exist_ok=True)
    todo = [r for r in index if not (dest / f"{r['sha256']}.npy").exists()]
    print(f"{args.cell}: {len(index)} photos with patches, {len(index) - len(todo)} done", flush=True)
    if not todo:
        return
    cell = backends.CELLS[args.cell]
    model_file = args.model_dir / "tflite" / f"{args.graph}.tflite"
    record = {"cell": args.cell, "graph": args.graph, "model_dir": _rel(args.model_dir),
              "model_sha256": _file_sha256(model_file), "threads": args.threads,
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    sampler = cellrun.Sampler(period=1.0)                       # the GPU proof's counters, over hours: 1 Hz
    sampler.start()
    probe = cellrun.Probe()
    dino_s, errors = [], 0
    try:
        with CapturedStderr(rdir / f"{args.cell}.stderr") as err:
            t = time.perf_counter()
            model = backends.open_graph(args.cell, args.graph, args.model_dir, args.threads, args.runtime_dir)
            batch = int(model.info.get("batch") or args.graph.split("_b")[1])
            record |= {"load_s": round(time.perf_counter() - t, 3), "info": model.info}
            print(f"loaded {args.cell} {args.graph} (batch {batch}) in {record['load_s']:.1f} s", flush=True)
            w0, cpu0, rc6_0 = time.perf_counter(), cellrun.cpu_seconds(), probe.rc6_ms()
            with (rdir / f"{args.cell}.jsonl").open("a") as f:
                for k, r in enumerate(todo, 1):
                    t = time.perf_counter()
                    row: dict = {"sha256": r["sha256"]}
                    try:
                        x, input_sha = load_array(args.out / "patches" / f"{r['sha256']}.npy")
                        tc = time.perf_counter()
                        y = np.concatenate([model(x[i:i + batch]) for i in range(0, len(x), batch)])
                        row |= {"dino_s": round(time.perf_counter() - tc, 3)}
                        save_array(dest / f"{r['sha256']}.npy", np.asarray(y, dtype=np.float32))
                        row |= {"input_sha256": input_sha, "nonfinite": not bool(np.isfinite(y).all())}
                        dino_s.append(row["dino_s"])
                    except Exception as e:                              # noqa: BLE001  kept by name, the run goes on
                        row["error"] = f"{type(e).__name__}: {e}"
                        errors += 1
                    row["s"] = round(time.perf_counter() - t, 3)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    _progress(k, len(todo), r["role"], r["path"], row["s"], w0, row.get("error"))
            w1, cpu1, rc6_1 = time.perf_counter(), cellrun.cpu_seconds(), probe.rc6_ms()
    finally:
        sampler.stop()
    counters = sampler.between(w0, w1)
    if rc6_0 is not None and rc6_1 is not None:
        counters["rc6_ms_delta"] = rc6_1 - rc6_0
    record |= {"photos": len(todo), "errors": errors, "wall_s": round(w1 - w0, 1),
               "dino_s_per_photo_median": statistics.median(dino_s) if dino_s else None,
               "cpu_over_wall": round((cpu1 - cpu0) / (w1 - w0), 2), "peak_rss_mb": cellrun.peak_rss_mb(),
               "pl1_throttled": counters.get("pl1_throttled"), "gpu_mhz_median": counters.get("gpu_mhz_median")}
    if cell.path is not None:
        record["gpu_proof"] = cellrun.gpu_proof(cellrun.loaded(), counters, cell.path)
    if cell.runtime == "litert" and cell.device == "gpu":
        record["delegate"] = delegate_report(err.text)
    with (rdir / f"{args.cell}.runs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps({k: record.get(k) for k in ("cell", "photos", "errors", "wall_s", "dino_s_per_photo_median")}
                     | {"gpu_proof": record.get("gpu_proof", {}).get("ok"),
                        "all_on_gpu": record.get("delegate", {}).get("all_on_gpu")}), flush=True)


# ------------------------------------------------------------------------------------------------- compare
RUN_KEYS = ("cell", "graph", "model_dir", "model_sha256", "threads", "started", "load_s", "photos", "errors",
            "wall_s", "dino_s_per_photo_median", "cpu_over_wall", "peak_rss_mb", "pl1_throttled", "gpu_mhz_median")


def _run_record(r: dict) -> dict:
    """A run's record for the committed JSON: no host paths (the libraries cellrun saw are left out)."""
    out = {k: r.get(k) for k in RUN_KEYS}
    if "gpu_proof" in r:
        out["gpu_proof"] = {k: r["gpu_proof"].get(k) for k in ("ok", "why_not", "gpu_busy_frac", "path")}
    if "delegate" in r:
        out["delegate"] = r["delegate"]
    return out


def compare(args) -> None:
    import graphs  # noqa: F401  (the repo, or the release under ML7_REPO, on sys.path)
    from make_refs import Context

    rdir = args.out / "readouts"
    index = last_rows(args.out / "patches.jsonl")
    sets = args.sets or (sorted(p.name for p in rdir.iterdir() if p.is_dir()) if rdir.exists() else [])
    set_rows = {s: last_rows(rdir / f"{s}.jsonl") for s in sets}
    sets = [s for s in sets if set_rows[s]]                             # a set no photo reached is left out
    pairs = pairs_for(sets, args.base)
    ctx = Context.load(args.checkpoint)
    ids = ctx.class_ids
    print(f"{len(index)} photos; sets {', '.join(sets)}; pairs "
          f"{', '.join(pair_key(a, b) for a, b in pairs)}", flush=True)
    rows, mismatch, guard = [], [], None
    for sha, p in index.items():
        if "error" in p:
            continue
        readouts, answers = {}, {}
        for s in sets:
            sr, npy = set_rows[s].get(sha), rdir / s / f"{sha}.npy"
            if sr is None or "error" in sr or not npy.exists():
                continue
            if sr.get("input_sha256") != p["patch_sha256"]:              # the cells must have read the same bytes
                mismatch.append({"set": s, "photo": p["path"]})
                continue
            readouts[s] = np.load(npy, allow_pickle=False)
            answers[s] = parity._answers(readouts[s], ctx.head, ctx.ref, ctx.probe)
            if guard is None and "method" in answers[s]:
                guard = {"method": answers[s]["method"], "threshold": float(answers[s]["threshold"])}
        views = {}
        for s, a in answers.items():
            v = answer_view(a)
            views[s] = {"top1": ids[v["top1"]], "top2": ids[v["top2"]], "margin": v["margin"],
                        **({"score": v["score"], "refused": v["refused"]} if "score" in v else {})}
        rows.append({"path": p["path"], "role": p["role"], "seg_fallback": p.get("seg_fallback"), "sets": views,
                     "pairs": {pair_key(a, b): pair_measures(readouts[a], readouts[b], answers[a], answers[b])
                               for a, b in pairs if a in readouts and b in readouts}})
    summary = {pair_key(a, b): summarise_pair(rows, pair_key(a, b)) for a, b in pairs}
    runs = {s: [_run_record(r) for r in last_rows_all(rdir / f"{s}.runs.jsonl")] for s in sets}
    out = {
        "what": "ML-7 P7: answers under each readout set over p7_answer_photos.txt (p7_answers.py compare)",
        "checkpoint": _rel(args.checkpoint), "guard": guard, "base": args.base,
        "photos": len(index), "patch_errors": {p["path"]: p["error"] for p in index.values() if "error" in p},
        "sets": {s: {"photos": sum(r["sets"].get(s) is not None for r in rows),
                     "errors": {index[k]["path"] if k in index else k: v["error"]
                                for k, v in set_rows[s].items() if "error" in v},
                     "runs": runs[s]} for s in sets},
        "input_mismatch": mismatch, "refusals": summarise_sets(rows), "pairs": summary,
        "per_photo": [photo_record(r) for r in rows],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(dumps(out))
    print_summary(out)


def print_summary(out: dict) -> None:
    print(f"{out['photos']} photos, {len(out['patch_errors'])} patch errors, "
          f"{len(out['input_mismatch'])} input mismatches; guard {out['guard']}")
    for s, r in out["refusals"].items():
        print(f"  {s}: refused " + ", ".join(f"{role} {b['refused']}/{b['photos']}" for role, b in r.items()))
    for key, s in out["pairs"].items():
        c, roles = s["least_cosine"] or {}, s["by_role"]
        print(f"{key}: {s['photos']} photos; top-1 flips {len(s['top1_flips'])}, verdict flips "
              f"{len(s['verdict_flips'])}, NaN/Inf {len(s['nonfinite'])}")
        print("  by role: " + ", ".join(f"{role} {b['top1_flips']}/{b['verdict_flips']}/{b['photos']}"
                                        for role, b in roles.items()) + " (top-1 flips/verdict flips/photos)")
        if c:
            print(f"  least cosine: min {c['min']:.8f}, p1 {c['p1']:.8f}, p5 {c['p5']:.8f}, p50 {c['p50']:.8f}")
        if s["prob_max_change"]:
            print(f"  largest probability change {s['prob_max_change']['max']:.2e}"
                  + ("" if s["score_change_max_abs"] is None
                     else f"; largest |score change| {s['score_change_max_abs']:.2e}"))
        for n in s["near_threshold"][:3]:
            print(f"  near threshold: {Path(n['photo']).name} ({n['role']}) {n['distance']:+.2e} from it, moved "
                  f"{n['score_change']:+.2e}, verdict {'held' if n['verdict_same'] else 'FLIPPED'}")
        for n in s["smallest_margin"][:3]:
            print(f"  small margin: {Path(n['photo']).name} ({n['role']}) {n['ref_margin']:.4f} -> "
                  f"{n['got_margin']:.4f}, top-1 {'held' if n['top1_same'] else 'FLIPPED'}, first two "
                  f"{'held' if n['top2_same'] else 'reordered'}")


# --------------------------------------------------------------------------------------------------- seeds
def read_answers(out: Path, s: str, index: dict[str, dict], ctx) -> tuple[dict[str, dict], list[str]]:
    """One run's answers under one readout set, by photo sha256, read as compare reads them (a photo with an error,
    or no readouts, is left out); and the photos whose readouts were not made from this run's patches."""
    rdir, got, mismatch = out / "readouts", {}, []
    rows = last_rows(rdir / f"{s}.jsonl")
    for sha, p in index.items():
        sr, npy = rows.get(sha), rdir / s / f"{sha}.npy"
        if "error" in p or sr is None or "error" in sr or not npy.exists():
            continue
        if sr.get("input_sha256") != p["patch_sha256"]:
            mismatch.append(p["path"])
            continue
        got[sha] = parity._answers(np.load(npy, allow_pickle=False), ctx.head, ctx.ref, ctx.probe)
    return got, mismatch


def seeds(args) -> None:
    import graphs  # noqa: F401  (the repo, or the release under ML7_REPO, on sys.path)
    from make_refs import Context

    runs = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"p7_answers: --runs takes name=OUT, not {spec}")
        runs[name] = Path(path)
    if args.base not in runs:
        raise SystemExit(f"p7_answers: --base {args.base} is not among --runs")
    names = [args.base] + [n for n in runs if n != args.base]
    index = {n: last_rows(runs[n] / "patches.jsonl") for n in names}
    rdir = runs[args.base] / "readouts"
    sets = list(dict.fromkeys((args.sets or sorted(p.name for p in rdir.iterdir() if p.is_dir()))
                              + [s for s in (args.fp32, args.fp16) if s]))
    ctx = Context.load(args.checkpoint)
    answers: dict[str, dict[str, dict]] = {s: {} for s in sets}
    mismatch = []
    for n in names:
        for s in sets:
            got, bad = read_answers(runs[n], s, index[n], ctx)
            mismatch += [{"run": n, "set": s, "photo": p} for p in bad]
            if got:                                               # a run without the set is left out of it
                answers[s][n] = got
    meta = {sha: p for n in reversed(names) for sha, p in index[n].items()}     # the base's row wins
    order = list(dict.fromkeys(sha for n in names for sha in index[n]))
    photos = {s: [sha for sha in order if a and all(sha in by_photo for by_photo in a.values())]
              for s, a in answers.items()}
    guard = next(({"method": a["method"], "threshold": float(a["threshold"])} for by_run in answers.values()
                  for by_photo in by_run.values() for a in by_photo.values() if "method" in a), None)
    spread = {s: sampling_spread(answers[s], photos[s]) for s in sets if len(answers[s]) > 1}
    drift = cmp = None
    if args.fp32 in sets and args.fp16 in sets and answers[args.fp32] and answers[args.fp16]:
        common = [sha for sha in photos[args.fp32] if sha in set(photos[args.fp16])]
        drift = precision_drift(answers[args.fp32], answers[args.fp16], common)
        if len(answers[args.fp32]) > 1:
            cmp = comparison(sampling_spread(answers[args.fp32], common), drift, answers[args.fp32],
                             answers[args.fp16], common, ctx.class_ids)
            for f in cmp["fp16_flipped"]:
                f["photo"] = meta[f["photo"]]["path"]
    same = {pair_key(a, b): k for a, b in seed_pairs(names)
            if (k := sum(p.get("patch_sha256") is not None and index[b].get(sha, {}).get("patch_sha256")
                         == p["patch_sha256"] for sha, p in index[a].items()))}

    def g6(x):
        return float(f"{x:.6g}")

    def view(a):
        v = answer_view(a)
        return {"top1": ctx.class_ids[v["top1"]], "margin": g6(v["margin"]),
                **({"refused": v["refused"]} if "refused" in v else {})}

    def brief(m):
        return {"top1_same": m["top1_same"], **({"verdict_same": m["verdict_same"]} if "verdict_same" in m else {}),
                "prob_max_change": g6(m["prob_max_change"]), "margin_change": g6(m["margin_change"])}
    per_photo = []
    for sha in order:
        in_sets = [s for s in sets if sha in set(photos[s])]
        if not in_sets:
            continue
        rec = {"path": meta[sha]["path"], "role": meta[sha]["role"],
               "sets": {s: {n: view(a[sha]) for n, a in answers[s].items()} for s in in_sets},
               "spread": {s: {k: brief(m) for k, m in spread[s]["per_photo"][sha].items()}
                          for s in in_sets if s in spread}}
        if drift and sha in drift["per_photo"]:
            rec["drift"] = {n: brief(m) for n, m in drift["per_photo"][sha].items()}
        per_photo.append(rec)
    out = {
        "what": "ML-7 P7: patch sampling's spread (every pair of seeds, per readout set) against fp16's drift "
                "(fp16 against fp32 on the same patches, per seed) (p7_answers.py seeds). Pooled rates count "
                "photo-pairs, which share photos: their Wilson intervals are indicative only",
        "checkpoint": _rel(args.checkpoint), "guard": guard, "base": args.base, "fp32": args.fp32, "fp16": args.fp16,
        "runs": {n: {"out": _rel(runs[n]),
                     "seed_key": next((p.get("seed_key", SEED_KEY) for p in index[n].values()), None),
                     "photos": len(index[n]), "patch_errors": sum("error" in p for p in index[n].values()),
                     "answered": {s: len(answers[s].get(n, {})) for s in sets}} for n in names},
        "same_patches": same, "input_mismatch": mismatch,
        "sets": {s: {"runs": list(answers[s]), "photos": len(photos[s])} for s in sets},
        "spread": {s: {k: v for k, v in sp.items() if k != "per_photo"} for s, sp in spread.items()},
        "drift": drift and {k: v for k, v in drift.items() if k != "per_photo"},
        "comparison": cmp, "per_photo": per_photo,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(dumps(out))
    print_seeds(out)


def _changes_line(s: dict) -> str:
    p, m = s["prob_max_change"] or {}, s["margin_change_abs"] or {}
    ci = s["top1_flip_wilson95"]
    return (f"top-1 flips {s['top1_flips']}/{s['photos']}"
            + (f" ({s['top1_flip_rate']:.2%}, 95% {ci[0]:.2%}-{ci[1]:.2%})" if ci else "")
            + f", verdict flips {s['verdict_flips']}"
            + (f"; prob change p50 {p['p50']:.4f} p95 {p['p95']:.4f} max {p['max']:.4f}" if p else "")
            + (f"; |margin change| p50 {m['p50']:.4f} p95 {m['p95']:.4f} max {m['max']:.4f}" if m else ""))


def print_seeds(out: dict) -> None:
    print("runs: " + ", ".join(f"{n} {r['seed_key']} ({r['photos']} photos)" for n, r in out["runs"].items())
          + f"; {len(out['input_mismatch'])} input mismatches; same patches {out['same_patches'] or 'none'}")
    for s, sp in out["spread"].items():
        print(f"sampling, {s}: {out['sets'][s]['photos']} photos in {', '.join(out['sets'][s]['runs'])}; "
              f"top-1 not the same under every seed: {sp['photos_top1_not_unanimous']}")
        for k, v in sp["pairs"].items():
            print(f"  {k}: {_changes_line(v)}")
        print(f"  pooled: {_changes_line(sp['pooled'])}")
    if out["drift"]:
        print(f"precision, {out['fp16']} vs {out['fp32']}:")
        for n, v in out["drift"]["per_seed"].items():
            print(f"  {n}: {_changes_line(v)}")
        print(f"  pooled: {_changes_line(out['drift']['pooled'])}")
    c = out["comparison"]
    if c:
        for side, label in (("sampling", f"sampling ({out['fp32']}, seed pairs)"), ("fp16", "fp16 (per seed)")):
            dists = {k: c[k][side] for k in ("prob_max_change", "margin_change_abs")}
            print(f"comparison, {label}: {_changes_line(c[side] | dists)}")
        k, n = c["fp16_flipped_photos_sampling_also_flips"]
        print(f"  photos fp16 flipped: {n}, of which sampling also flips {k}")
        for f in c["fp16_flipped"]:
            print(f"  {Path(f['photo']).name}: fp16 flipped in {', '.join(f['fp16_flipped_in'])}; fp32 top-1 "
                  + ", ".join(f"{r} {t} by {f['fp32_margin'][r]:.3f}" for r, t in f["fp32_top1"].items())
                  + f"; seed pairs flipped {f['sampling_pairs_flipped'][0]}/{f['sampling_pairs_flipped'][1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("patches")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--photos", type=Path, default=LIST)
    p.add_argument("--root", type=Path, required=True, help="where the list's paths are (yappy: p7/photos)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--torch", action="store_true", help="PyTorch's readouts too (slow: ~22 s a photo on yappy)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="the first N photos only (a rehearsal)")
    p.add_argument("--seed-key", type=int, nargs=2, default=SEED_KEY,
                   help="patch sampling's seed (serving: 42 0); another one, in its own --out, for sampling's spread")
    r = sub.add_parser("run")
    r.add_argument("--cell", required=True, help="a backends.CELLS name: litert_gpu, litert_gpu_fp16, ...")
    r.add_argument("--graph", default="dino_b8")
    r.add_argument("--model-dir", type=Path, required=True, help="holds tflite/<graph>.tflite")
    r.add_argument("--runtime-dir", type=Path, default=None, help="LiteRT's GPU library (rt_vulkan)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--limit", type=int, default=None, help="the first N photos only (a rehearsal)")
    c = sub.add_parser("compare")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--sets", nargs="+", default=None, help="readout sets (default: every one under OUT/readouts)")
    c.add_argument("--base", default="litert_gpu", help="the fp32 set the others are compared against")
    c.add_argument("--json", type=Path, required=True)
    d = sub.add_parser("seeds")
    d.add_argument("--checkpoint", type=Path, required=True)
    d.add_argument("--runs", nargs="+", required=True, help="name=OUT per seed's run (patches --seed-key ...)")
    d.add_argument("--base", required=True, help="the run with serving's seed (its photo order and rows)")
    d.add_argument("--sets", nargs="+", default=None, help="readout sets (default: every one under base's readouts)")
    d.add_argument("--fp32", default=None, help="the fp32 set fp16 is compared against (litert_gpu)")
    d.add_argument("--fp16", default=None, help="the fp16 set (litert_gpu_fp16)")
    d.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    {"patches": patches, "run": run, "compare": compare, "seeds": seeds}[args.cmd](args)


if __name__ == "__main__":
    main()
