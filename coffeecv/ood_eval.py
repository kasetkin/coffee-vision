"""Measure the shipped OOD guard on the segmenter dataset's test split (ADR 0016, ticket ML-5 D16).

The guard is the checkpoint's linear probe at the fixed `infer.PROBE_THRESHOLD`, run on every photo after
classification. This module scores every **test** photo of `labels/ml5/seg_dataset.yaml` through the serving
path (`patches_for_photo`, so the segmenter's crop or its whole-frame fallback, then `probe_score`) and
reports:

- negatives caught: probe score above the threshold, overall and per scenario tag;
- positives refused (segmenter positives, pool photos, internet positives), per source and apart by whether
  the classifier trained on the photo: its train split holds a file with the same sha256. Segmenter
  positives are often byte copies of pool photos, so the bytes decide, not the path;
- beside the probe (D9, D22): the segmenter's empty-mask rate on the test negatives, and its empty-or-tiny
  rate (a mask under the checkpoint's seg_min_area_frac, the D18 fallback serving takes), overall and per tag.
  Not a guard: the probe refuses, these say how often the segmenter alone would have.

A photo `classify_one` cannot measure (no bean pitch) counts as caught or refused, because that is what the
user gets, and is counted apart. Training and validation photos are never scored here; `fit_ood_probe`
fits on the training and validation negatives, so the test split is the only unseen evidence. Nothing is
chosen from these numbers (the threshold is fixed), so running this again spends nothing.

The method comparison this file used to run (centroid, kNN, Mahalanobis, energy, ensembles; ADR 0005,
docs/ood_guard_eval.md) chose the probe and retired with the calibrated threshold; it is in git history.

    python -m coffeecv.ood_eval --checkpoint models/allrigs_dino3b16_seg_country_s123.pt --out ood.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from coffeecv.class_list import ClassList, load_classes
from coffeecv.config import CHECKPOINTS_DIR, REPO_ROOT
from coffeecv.dataset import pooled_class_photos, resolve_captures, split_photos_by_class
from coffeecv.infer import (PROBE_THRESHOLD, _sha, config_for_checkpoint, forward_with_embeddings,
                            inference_tta_for, load_model, load_ood_probe, patches_for_photo, probe_path_for,
                            probe_score)
from coffeecv.repo_files import sha256_file
from coffeecv.seg_dataset import SEG_DATASET_FILE, SOURCES, load_seg_dataset
from coffeecv.transforms import build_eval_transform

# Tags whose every patch is unambiguously not-beans, so a patch-level probe can
# be trained from the photo-level tag. An empty tray has no bean patches; a
# sparse scattering of beans does, so those tags are deliberately absent.
# Module-level because `fit_ood_probe` fits the shipped probe from exactly this set.
CLEAN_NEGATIVE_TAGS = frozenset({"empty_tray", "ground_coffee", "confusable_grain",
                                  "other_nuts_seeds", "non_food_objects",
                                  "real_world_negatives", "green_legume"})

class Measured(NamedTuple):
    """One photo through the serving path: the probe's score (None: unmeasurable, which classify_one refuses)
    and what the segmenter gave (None: not known, as on an unmeasurable photo or a checkpoint without one)."""
    score: float | None
    empty: bool | None = None               # the segmenter's mask is empty
    empty_or_tiny: bool | None = None       # empty or under seg_min_area_frac: serving's D18 fallback


def _counts(rows: list[dict], key: str) -> dict:
    return {"n": len(rows), key: sum(r["over"] for r in rows), "unmeasurable": sum(r["score"] is None for r in rows)}


def _seg_counts(rows: list[dict]) -> dict:
    return {"n": len(rows), "empty": sum(r["empty"] is True for r in rows),
            "empty_or_tiny": sum(r["empty_or_tiny"] is True for r in rows),
            "unknown": sum(r["empty_or_tiny"] is None for r in rows)}


def guard_report(photos: list[dict], measure: Callable[[str], Measured], trained_sha256: set[str],
                 threshold: float = PROBE_THRESHOLD, min_area_frac: float | None = None) -> dict:
    """Score the test split of `photos` (from `load_seg_dataset`) and count, at `threshold`, negatives
    caught and positives refused, and the segmenter's empty and empty-or-tiny masks on the negatives.
    `measure(path)` is one photo's `Measured`; an unmeasurable photo counts as over the threshold, as
    classify_one refuses it. `trained_sha256` holds the sha256 of every photo the classifier trained on.
    `min_area_frac` is the tiny-mask threshold behind `empty_or_tiny`, recorded for the printout."""
    rows = []
    for e in photos:
        if e["split"] != "test":
            continue
        m = measure(e["path"])
        rows.append({"path": e["path"], "source": e["source"], "tag": e.get("tag"), "score": m.score,
                     "over": m.score is None or m.score > threshold, "trained": e["sha256"] in trained_sha256,
                     "empty": m.empty, "empty_or_tiny": m.empty_or_tiny})
    neg = [r for r in rows if r["source"] == "negative"]
    tags = sorted({r["tag"] for r in neg})
    pos = [r for r in rows if r["source"] != "negative"]

    def by_training(group: list[dict]) -> dict:
        return {"trained": _counts([r for r in group if r["trained"]], "refused"),
                "not_trained": _counts([r for r in group if not r["trained"]], "refused")}

    return {
        "threshold": threshold,
        "min_area_frac": min_area_frac,
        "negatives": {"all": _counts(neg, "caught"),
                      "by_tag": {t: _counts([r for r in neg if r["tag"] == t], "caught") for t in tags},
                      "segmenter": {"all": _seg_counts(neg),
                                    "by_tag": {t: _seg_counts([r for r in neg if r["tag"] == t]) for t in tags}}},
        "positives": {**{s: by_training([r for r in pos if r["source"] == s])
                         for s in SOURCES if s != "negative"},
                      "all": by_training(pos)},
        "per_photo": rows,
    }


def format_report(report: dict) -> str:
    """The report as printed: counts with Wilson 95% intervals, unmeasurable photos named apart."""
    def line(label: str, c: dict, key: str) -> str:
        if not c["n"]:
            return f"  {label}: none"
        lo, hi = wilson(c[key], c["n"])
        un = f", {c['unmeasurable']} of them unmeasurable" if c["unmeasurable"] else ""
        return f"  {label}: {c[key]}/{c['n']} ({c[key] / c['n']:.0%}, 95% CI {lo:.0%}-{hi:.0%}{un})"

    thr = f"{report['threshold']:g}"
    neg = report["negatives"]
    out = [f"test split, probe threshold {thr} (fixed, ADR 0016)",
           line(f"negatives caught at {thr}", neg["all"], "caught").strip()]
    out += [line(f"  {tag}", c, "caught") for tag, c in neg["by_tag"].items()]
    tiny = f"area < {report['min_area_frac']:g}" if report.get("min_area_frac") is not None else "tiny"
    seg = neg["segmenter"]
    out.append("segmenter on the test negatives (beside the probe, not a guard; D9, D22):")
    for label, c in (("all", seg["all"]), *((f"  {t}", c) for t, c in seg["by_tag"].items())):
        un = f"; {c['unknown']} unknown (unmeasurable or no segmenter)" if c["unknown"] else ""
        out.append(f"  {label}: empty mask: {c['empty']}/{c['n']}, empty or tiny ({tiny}): "
                   f"{c['empty_or_tiny']}/{c['n']}{un}")
    out.append(f"positives refused at {thr}:")
    for source, c in report["positives"].items():
        out.append(line(f"{source}, trained on", c["trained"], "refused"))
        out.append(line(f"{source}, not trained on", c["not_trained"], "refused"))
    return "\n".join(out)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct near 0 and 1, where normal-approx isn't.

    Kept local: the photo-pooling evaluator it was once copied from was retired
    with the fold protocol (ticket ML-1).
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))



def raw_photo_index() -> dict[str, Path]:
    """Filename stem -> raw photo, across every session in `dataset/`.

    An index rather than a path transform because the rigs a checkpoint trains on
    are *merged* ones (`data/segcropped/cam_pixel` is three sessions stitched together
    by the merge_segcam_* stages), so a cropped photo's directory no longer names the
    session its raw original lives in. The stems survive both the crop and the
    merge unchanged and are unique across the whole tree, which makes them the one
    thing that still joins the two sides.
    """
    index: dict[str, Path] = {}
    for class_dir in (REPO_ROOT / "dataset").glob("*/class_*__*"):
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic"):
                index[p.stem] = p
    return index


def raw_photo_for(cropped: Path, index: dict[str, Path]) -> Path:
    stem = cropped.name.replace("__cropped.jpg", "")
    if stem not in index:
        raise FileNotFoundError(f"No raw photo under dataset/ for {cropped.name}")
    return index[stem]


def id_photos(cfg, classes: ClassList, split: str) -> list[Path]:
    """The checkpoint's own `split` photos (train, val or test), as *raw* paths.

    Reuses the training pipeline's own pooled split (`pooled_class_photos` +
    `split_photos_by_class`) with this checkpoint's seed, so for a checkpoint trained
    by the current code the photos returned are the ones it actually held out.

    NOT for a checkpoint trained before ticket ML-1 (2026-09-29) -- both deployed
    models included. Those were split per camera and class with a camera-position
    seed; the pooled split recomputed here is a different partition, so part of
    what it returns as "held out" was in that checkpoint's training set and scores
    optimistically in-distribution. Such a card still carries the legacy
    `train_rigs` key, which `infer.config_for_checkpoint` names in the config source
    it returns.

    There used to be a second list here, every photo of the checkpoint's held-out
    camera; with no camera held out since ML-1 there is none.

    Each class pools its folders at its own class index, as MultiPhotoPatchDataset does. A class with
    no photos in any pool raises (ticket ML-3): skipping it would drop that class from the probe's
    positives without a word.
    """
    train_dirs, _ = cfg.resolve_paths()
    frac = {"train": cfg.train_photo_frac, "val": cfg.val_photo_frac, "test": cfg.test_photo_frac}
    index = raw_photo_index()
    captures = resolve_captures(train_dirs)

    train_photos: list[Path] = []
    for class_idx, key in enumerate(classes.keys):
        pool, _absent = pooled_class_photos(captures, classes.folders[key])  # a dir need not carry every class
        if not pool:
            raise ValueError(f"class {key} (folders {', '.join(classes.folders[key])}) has no photos in "
                             f"{[c.name for c in captures]}")
        chosen = split_photos_by_class(pool, cfg.seed, class_idx, frac)[split]
        train_photos.extend(raw_photo_for(p.path, index) for p in chosen)

    return train_photos


class Unmeasurable(Exception):
    """Pitch estimation threw, so no patches exist to score.

    `classify_one` turns this into `REFUSED (unmeasurable)` -- a refusal, just not
    one the OOD metric produced. It has to stay distinguishable here: scoring it as
    a very high number would credit whichever metric is being tested for a refusal
    it had no part in, and dropping it silently would understate how often the
    pipeline declines to answer.
    """


def embed_photo(path: Path, cfg, model, head, n_patches: int, tta: bool,
                 cache_dir: Path | None, ckpt_sha: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """(probs, embeds, diagnostics) for one photo, cached on disk.

    The cache is keyed by the checkpoint hash as well as the photo, because
    embeddings only mean anything relative to the weights that produced them --
    the same reason `infer.py` refuses a reference built for a different
    checkpoint. Caching matters here because scoring is ~5s/photo under TTA and
    every additional candidate metric re-reads the same forward passes.
    """
    key = None
    if cache_dir is not None:
        # v3 in the key because entries written before logits (v2) or the segmenter's empty-mask flag
        # (v3, ML-5 D9) were captured cannot serve a request that needs them, and a stale hit would be silent.
        key = cache_dir / f"v3_{ckpt_sha[:12]}_{n_patches}_{int(tta)}_{path.stem}.npz"
        if key.exists():
            z = np.load(key, allow_pickle=True)
            diag = json.loads(str(z["diag"]))
            if diag.get("unmeasurable"):
                raise Unmeasurable(diag["error"])
            return z["probs"], z["embeds"], z["logits"], diag

    try:
        patches, diag = patches_for_photo(path, cfg, n_patches, [42, 0])
    except (ValueError, OSError) as exc:
        # Same catch classify_one uses, so the harness declines on exactly the
        # photos production declines on rather than on its own set.
        if key is not None:
            key.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(key, probs=np.zeros(0), embeds=np.zeros(0),
                                diag=json.dumps({"unmeasurable": True, "error": str(exc)}))
        raise Unmeasurable(str(exc)) from exc
    transform = build_eval_transform(cfg.patch_resize)
    probs, embeds, logits = forward_with_embeddings(
        model, head, torch.stack([transform(x) for x in patches]), tta=tta, return_logits=True)

    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(key, probs=probs, embeds=embeds, logits=logits,
                            diag=json.dumps(diag))
    return probs, embeds, logits, diag


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0,
                   iters: int = 400, lr: float = 0.5) -> np.ndarray:
    """Plain L2-regularised logistic regression, full-batch, standardised inputs.

    Hand-rolled because sklearn is not a dependency here. Full-batch gradient
    descent with a fixed step and fixed iteration count keeps it deterministic,
    which matters more than convergence speed for a few thousand rows.
    """
    x = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        grad = x.T @ (p - y) / len(x)
        grad[:-1] += l2 * w[:-1] / len(x)
        w -= lr * grad
    return w


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "best.pt"))
    p.add_argument("--config", default=None,
                   help="defaults to the config archived beside the checkpoint; see infer.py")
    p.add_argument("--dataset", default=str(SEG_DATASET_FILE), help="the segmenter dataset file")
    p.add_argument("--probe", default=None, help="defaults to the probe beside the checkpoint")
    p.add_argument("--n-patches", type=int, default=40)
    p.add_argument("--no-tta", action="store_true", help="faster, but no longer the production code path")
    p.add_argument("--cache-dir", default=".ood_eval_cache",
                   help="per-photo forward passes, keyed by checkpoint hash")
    p.add_argument("--out", default=None, help="write the report, every photo's score included, as JSON")
    args = p.parse_args()

    photos = load_seg_dataset(Path(args.dataset))
    checkpoint = Path(args.checkpoint)
    cfg, cfg_source = config_for_checkpoint(checkpoint, args.config)
    print(f"config: {cfg_source}")
    _, classes_file = cfg.resolve_paths()
    classes = load_classes(classes_file)
    model, head = load_model(checkpoint, cfg.model_name, len(classes), cfg.dropout)
    probe = load_ood_probe(checkpoint, Path(args.probe) if args.probe else None, head)
    if probe is None:
        sys.exit(f"No OOD probe at {args.probe or probe_path_for(checkpoint)}; fit one with coffeecv.fit_ood_probe")
    ckpt_sha = _sha(checkpoint)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    tta = False if args.no_tta else inference_tta_for(checkpoint, cfg.model_name)

    # The classifier's train split, by content: a segmenter positive copied from a pool photo was trained on.
    trained = {sha256_file(q) for q in id_photos(cfg, classes, "train")}

    def measure(path: str) -> Measured:
        try:
            _, embeds, _, diag = embed_photo(REPO_ROOT / path, cfg, model, head, args.n_patches, tta,
                                             cache_dir, ckpt_sha)
        except Unmeasurable:
            return Measured(None)
        return Measured(probe_score(embeds, probe), diag.get("seg_mask_empty"), diag.get("seg_fallback"))

    n_test = sum(e["split"] == "test" for e in photos)
    print(f"scoring {n_test} test photos of {args.dataset} with {'TTA' if tta else 'no TTA'}, "
          f"{args.n_patches} patches; the classifier's train split holds {len(trained)} photos\n")
    report = guard_report(photos, measure, trained,
                          min_area_frac=cfg.seg_min_area_frac if cfg.crop_method == "segment" else None)
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(json.dumps({
            "checkpoint": str(checkpoint), "checkpoint_sha": ckpt_sha, "config_source": cfg_source,
            "ood_probe": probe["_path"], "dataset": args.dataset, "n_patches": args.n_patches, "tta": tta,
            **report}, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
