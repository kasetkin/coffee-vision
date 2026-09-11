"""Score in-distribution photos and known negatives with one guard metric, so
candidate metrics can be compared against each other on real data instead of
against one remembered example.

The guard shipped in `infer.py` refuses on a single number -- penultimate-embedding
distance to the nearest class centroid, normalised by that class's spread, pooled
to a per-photo median and compared against `OOD_THRESHOLD`. That threshold came
from Phase 9's *one* out-of-rig photo (1.24 in-distribution max vs 1.92 on the
odd one out), which is thinner evidence than this repo accepts anywhere else, and
in three years of evaluation the guard has never once been scored against a photo
that contains no beans at all. Every negative it has ever seen was still a tray of
beans, just from an unseen camera. This file exists to close both gaps at once: it
puts a real negative set on one side, genuine in-distribution photos on the other,
and reports separation rather than anecdotes.

**Scoring runs through `patches_for_photo` and `forward_with_embeddings`, imported,
not reimplemented.** That is the same parity rule `infer.py` is built around: a
harness that sampled its own patches would be measuring a pipeline nobody deploys.
It also means the raw `dataset/<session>/...` photos are scored, not the offline
`data/cropped/...` crops -- live inference does its own crop, so scoring the
pre-cropped copies would skip a stage every real upload goes through.

The in-distribution side reuses the checkpoint's own `split_photos_by_class`, so
the photos scored here are the ones that checkpoint held out, not ones it trained
on. Train-rig and held-out-rig photos are reported as separate conditions and
never pooled: they answer different questions (a normal photo vs a photo from a
camera the model has never seen), and averaging them would hide the second inside
the first.

    python -m coffeecv.ood_eval --checkpoint models/allrigs_cam_s123.pt \\
        --negatives dataset/ood_negatives/2026-09__user_realworld --split dev
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from coffeecv.config import CHECKPOINTS_DIR, REPO_ROOT
from coffeecv.dataset import (find_class_dir, list_cropped_photos, load_class_labels,
                              resolve_rigs, split_photos_by_class)
from coffeecv.infer import (OOD_THRESHOLD, _sha, config_for_checkpoint, energy_score,
                            forward_with_embeddings, knn_score, load_model, mahalanobis_scores,
                            ood_scores, patches_for_photo, reference_path_for, shared_precision)
from coffeecv.transforms import build_eval_transform

# `centroid` first because it is the shipped one and the point of comparison.
METHODS = ("centroid", "knn", "mahalanobis_shared", "energy", "coverage",
           "patch_vote_agreement", "linear_probe", "ensemble")

# Fitted on the dev split rather than computed per photo, so they are scored
# separately after every photo's patches exist. `ensemble` additionally consumes
# the other methods' outputs, so it is scored last.
POSTHOC_METHODS = ("linear_probe", "ensemble")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct near 0 and 1, where normal-approx isn't.

    Copied rather than imported from photo_pooling_eval, which computes it inside
    an experiment-resolution flow this file has no business triggering.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged. scipy is not a dependency of this repo."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    srt = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(id_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """P(a negative scores higher than an in-distribution photo), ties at 0.5.

    Rank-sum rather than a swept threshold so the number does not depend on where
    the sweep happens to put its grid, which matters at these sample sizes.
    """
    n_neg, n_id = len(neg_scores), len(id_scores)
    if n_neg == 0 or n_id == 0:
        return float("nan")
    ranks = _average_ranks(np.concatenate([neg_scores, id_scores]))
    u = ranks[:n_neg].sum() - n_neg * (n_neg + 1) / 2.0
    return float(u / (n_neg * n_id))


def detection_at_fpr(id_scores: np.ndarray, neg_scores: np.ndarray,
                      fpr: float = 0.05) -> tuple[float, float]:
    """(threshold, fraction of negatives caught) at a fixed false-refusal budget.

    Phrased as a budget rather than the literature's FPR@95TPR because that is the
    decision actually being made here: a user whose genuine bean photo gets refused
    is the cost, so fix how often that is allowed to happen and ask what the guard
    catches in return. Higher score means more out-of-distribution, so the
    threshold is the ID distribution's (1 - fpr) quantile.
    """
    if len(id_scores) == 0 or len(neg_scores) == 0:
        return (float("nan"), float("nan"))
    thr = float(np.quantile(id_scores, 1.0 - fpr))
    return (thr, float((neg_scores > thr).mean()))


class Scorer:
    """Everything the metrics need that is fixed for a checkpoint, loaded once.

    The kNN embeddings and the pooled precision matrix are both derived from the
    same sidecar, so loading is shared -- and absent the sidecar the metrics that
    need it are refused up front rather than silently skipped, since a comparison
    table quietly missing a row is worse than one that fails.
    """

    def __init__(self, ref: dict, ref_path: Path, methods: list[str], knn_k: int = 5):
        self.ref = ref
        self.knn_k = knn_k
        self.train_embeds = self.train_labels = self.precision = None
        needs_sidecar = {"knn", "mahalanobis_shared"} & set(methods)
        if needs_sidecar:
            name = ref.get("knn_embeddings_path")
            if not name:
                raise SystemExit(
                    f"{sorted(needs_sidecar)} need training-patch embeddings, which "
                    f"{ref_path} does not reference. Rebuild it with "
                    f"`coffeecv.build_ood_reference --store-knn-embeddings`.")
            z = np.load(ref_path.parent / name)
            self.train_embeds = z["embeddings"].astype(np.float64)
            self.train_labels = z["labels"]
            if "mahalanobis_shared" in methods:
                self.precision = shared_precision(self.train_embeds, self.train_labels)

    def score(self, method: str, probs: np.ndarray, embeds: np.ndarray,
              logits: np.ndarray) -> float:
        """One photo's guard score. Higher is more out-of-distribution.

        Each method pools its per-patch scores to one number per photo. Median is
        the pooling `classify_one` already uses, and is kept for the distance
        metrics so the comparison is between *metrics*, not between pooling rules.
        """
        if method == "centroid":
            return float(np.median(ood_scores(embeds, self.ref)[0]))
        if method == "knn":
            return float(np.median(knn_score(embeds, self.train_embeds, self.knn_k)))
        if method == "mahalanobis_shared":
            return float(np.median(mahalanobis_scores(embeds, self.ref, self.precision)))
        if method == "energy":
            return float(np.median(energy_score(logits)))
        if method == "coverage":
            # The aggregator `frac_patches_over_threshold` has been computing in
            # classify_one all along without anything acting on it: what fraction
            # of patches individually look out-of-distribution, rather than where
            # the middle patch sits.
            return float((ood_scores(embeds, self.ref)[0] > OOD_THRESHOLD).mean())
        if method == "patch_vote_agreement":
            # Disagreement, so that (like every other metric here) larger means
            # more suspect. A real tray should have most patches voting for one
            # class; patches of something the model has no concept of have no
            # reason to agree on anything.
            votes = probs.argmax(axis=1)
            return float(1.0 - np.bincount(votes, minlength=probs.shape[1]).max() / len(votes))
        raise ValueError(f"unknown method: {method}")


def raw_photo_index() -> dict[str, Path]:
    """Filename stem -> raw photo, across every session in `dataset/`.

    An index rather than a path transform because the rigs a checkpoint trains on
    are *merged* ones (`data/cropped/cam_pixel` is three sessions stitched together
    by the merge_cam_* stages), so a cropped photo's directory no longer names the
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


def id_photos(cfg, class_ids: list[str], split: str) -> tuple[list[Path], list[Path]]:
    """(train-rig photos, held-out-rig photos) for `split`, as *raw* paths.

    Reuses split_photos_by_class with this checkpoint's own seed so the photos
    returned are the ones it actually held out. The held-out rig contributes every
    photo it has -- that is what heldout_rig means -- and is returned separately
    because pooling it with the train rigs would average a cross-camera question
    into a same-camera one.
    """
    train_dirs, heldout_dir, _ = cfg.resolve_paths()
    frac = {"train": cfg.train_photo_frac, "val": cfg.val_photo_frac, "test": cfg.test_photo_frac}
    index = raw_photo_index()

    train_photos: list[Path] = []
    for rig_idx, rig in enumerate(resolve_rigs(train_dirs)):
        for class_idx, cid in enumerate(class_ids):
            try:
                photos = list_cropped_photos(find_class_dir(rig.cropped_dir, cid))
            except FileNotFoundError:
                continue  # a rig need not carry every class -- cam_iphone does not
            chosen = split_photos_by_class(photos, cfg.seed, class_idx, frac, rig_idx)[split]
            train_photos.extend(raw_photo_for(p, index) for p in chosen)

    heldout_photos: list[Path] = []
    if heldout_dir is not None:
        rig = resolve_rigs([heldout_dir])[0]
        for cid in class_ids:
            try:
                photos = list_cropped_photos(find_class_dir(rig.cropped_dir, cid))
            except FileNotFoundError:
                continue
            heldout_photos.extend(raw_photo_for(p, index) for p in photos)
    return train_photos, heldout_photos


def negatives_from(batch_dirs: list[Path], split: str) -> list[dict]:
    """Manifest rows for `split`, with `path` resolved and `scenario_tag` kept.

    Read from manifest.csv rather than globbing the directory so that a file which
    was collected but deliberately excluded (or is still being vetted) cannot
    quietly enter an evaluation just by sitting on disk.
    """
    rows = []
    for batch in batch_dirs:
        # Sibling first: the photo directories are DVC-tracked, so a manifest kept
        # inside one would be invisible in git (splits and provenance are exactly
        # what wants reviewing) and would re-hash the whole directory on every
        # edit. Same split the crop.yaml files already use. The in-directory
        # location is still honoured so older batches keep working.
        manifest = batch.parent / f"{batch.name}.manifest.csv"
        if not manifest.exists():
            manifest = batch / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(
                f"No manifest for {batch}: looked for "
                f"{batch.parent / (batch.name + '.manifest.csv')} and {batch / 'manifest.csv'}")
        for row in csv.DictReader(manifest.open()):
            if row["split"] != split:
                continue
            path = batch / row["filename"]
            if not path.exists():
                raise FileNotFoundError(f"{manifest} lists {row['filename']}, which is missing")
            rows.append({"path": path, "scenario_tag": row["scenario_tag"],
                          "batch": batch.name, "notes": row.get("notes", "")})
    return rows


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
        # v2 in the key because entries written before logits were captured cannot
        # serve a request that needs them, and a stale hit would be silent.
        key = cache_dir / f"v2_{ckpt_sha[:12]}_{n_patches}_{int(tta)}_{path.stem}.npz"
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


def linear_probe_scores(per_photo: list[dict], patch_store: dict[str, np.ndarray],
                         n_folds: int = 5, seed: int = 0) -> dict[str, float]:
    """P(not-beans) per photo from a linear probe on frozen patch embeddings.

    Cross-validated **at the photo level**: each photo is scored by a probe fitted
    without it. Splitting by patch instead would put 39 of a photo's 40 patches in
    train and the 40th in test, which is not generalisation but memorisation, and
    it would report a near-perfect number.

    Only photos whose label is unambiguous at patch level are trained on -- every
    patch of an empty tray is a negative, but a photo of a *few* beans on a tray
    has patches of both kinds and no per-patch label to give them.
    """
    train_keys = [r["photo"] for r in per_photo if r.get("probe_label") is not None]
    if not train_keys:
        return {}
    labels = {r["photo"]: r["probe_label"] for r in per_photo if r.get("probe_label") is not None}

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_keys))
    folds = {train_keys[i]: int(j % n_folds) for j, i in enumerate(order)}

    out: dict[str, float] = {}
    for fold in range(n_folds):
        fit_keys = [k for k in train_keys if folds[k] != fold]
        held_keys = [k for k in train_keys if folds[k] == fold]
        if not held_keys or len({labels[k] for k in fit_keys}) < 2:
            continue
        x = np.concatenate([patch_store[k] for k in fit_keys])
        y = np.concatenate([np.full(len(patch_store[k]), labels[k], dtype=float)
                            for k in fit_keys])
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-6
        w = _fit_logistic((x - mu) / sd, y)
        for k in held_keys:
            z = ((patch_store[k] - mu) / sd) @ w[:-1] + w[-1]
            out[k] = float(np.mean(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))))

    # Photos excluded from training (ambiguous patch labels) still need a score:
    # fit once on everything and apply, which is honest for them because they were
    # never in any fit set.
    rest = [r["photo"] for r in per_photo
            if r["photo"] not in out and r["photo"] in patch_store]
    if rest and len({labels[k] for k in train_keys}) == 2:
        x = np.concatenate([patch_store[k] for k in train_keys])
        y = np.concatenate([np.full(len(patch_store[k]), labels[k], dtype=float)
                            for k in train_keys])
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-6
        w = _fit_logistic((x - mu) / sd, y)
        for k in rest:
            z = ((patch_store[k] - mu) / sd) @ w[:-1] + w[-1]
            out[k] = float(np.mean(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))))
    return out


def ensemble_scores(per_photo: list[dict], base_methods: list[str],
                     n_folds: int = 5, seed: int = 0) -> dict[str, float]:
    """Stack the per-photo scalar scores of the base metrics into one number.

    Fitted on the same evidence a deployed system could have -- photos from the
    checkpoint's own data plus collected negatives -- and cross-validated at the
    photo level, with the unseen positives held out of fitting entirely, exactly
    as the linear probe is. Only a handful of coefficients over ~6 inputs, so the
    overfitting risk is far below any method that touches raw embeddings, which
    is what makes this worth trying at these sample sizes at all.
    """
    usable = [r for r in per_photo
              if not r["unmeasurable"] and all(not np.isnan(r.get(m, np.nan)) for m in base_methods)]
    if not usable:
        return {}
    fit_rows = [r for r in usable if r.get("probe_label") is not None]
    if len({r["probe_label"] for r in fit_rows}) < 2:
        return {}

    feats = {r["photo"]: np.array([r[m] for m in base_methods], dtype=float) for r in usable}
    labels = {r["photo"]: r["probe_label"] for r in fit_rows}
    keys = [r["photo"] for r in fit_rows]

    rng = np.random.default_rng(seed)
    folds = {keys[i]: int(j % n_folds) for j, i in enumerate(rng.permutation(len(keys)))}

    def fit(train_keys):
        x = np.array([feats[k] for k in train_keys])
        y = np.array([labels[k] for k in train_keys], dtype=float)
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-9
        return mu, sd, _fit_logistic((x - mu) / sd, y, l2=1.0, iters=2000, lr=0.5)

    def apply(k, mu, sd, w):
        z = ((feats[k] - mu) / sd) @ w[:-1] + w[-1]
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))

    out: dict[str, float] = {}
    for fold in range(n_folds):
        tr = [k for k in keys if folds[k] != fold]
        te = [k for k in keys if folds[k] == fold]
        if not te or len({labels[k] for k in tr}) < 2:
            continue
        mu, sd, w = fit(tr)
        for k in te:
            out[k] = apply(k, mu, sd, w)

    rest = [r["photo"] for r in usable if r["photo"] not in out]
    if rest:
        mu, sd, w = fit(keys)
        for k in rest:
            out[k] = apply(k, mu, sd, w)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "best.pt"))
    p.add_argument("--config", default=None,
                   help="defaults to the config archived beside the checkpoint; see infer.py")
    p.add_argument("--negatives", nargs="+", required=True,
                   help="one or more ood_negatives batch directories (each with a manifest.csv)")
    p.add_argument("--split", choices=("dev", "holdout"), required=True,
                   help="no default on purpose: holdout is meant to be spent once, deliberately")
    p.add_argument("--methods", default="centroid",
                   help=f"comma-separated, from {','.join(METHODS)}. Defaults to the one "
                        f"metric that is actually shipped, so a bare run reproduces production.")
    p.add_argument("--id-split", default="test", choices=("val", "test"),
                   help="which of the checkpoint's own held-out splits to score as in-distribution")
    p.add_argument("--n-patches", type=int, default=40)
    p.add_argument("--no-tta", action="store_true",
                   help="faster, but no longer the production code path")
    p.add_argument("--limit-id", type=int, default=0,
                   help="score at most this many in-distribution photos per condition (0 = all)")
    p.add_argument("--cache-dir", default=".ood_eval_cache",
                   help="per-photo forward passes, keyed by checkpoint hash")
    p.add_argument("--positives", nargs="*", default=[],
                   help="batch directories of genuine bean photos the model has not seen. "
                        "Scored as their own in-distribution conditions, kept apart from the "
                        "checkpoint's own held-out split -- they are the population a guard "
                        "must not refuse, and the split photos are too easy to stand in for them.")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--out", default=None, help="write the full result table as JSON")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            sys.exit(f"unknown method {m!r}; available: {', '.join(METHODS)}")

    checkpoint = Path(args.checkpoint)
    cfg, cfg_source = config_for_checkpoint(checkpoint, args.config)
    print(f"config: {cfg_source}")
    _, _, classes_file = cfg.resolve_paths()
    class_ids = sorted(load_class_labels(classes_file))

    ref_path = reference_path_for(checkpoint)
    if not ref_path.exists():
        sys.exit(f"No OOD reference at {ref_path}; build one with coffeecv.build_ood_reference")
    ref = json.loads(ref_path.read_text())
    ckpt_sha = _sha(checkpoint)
    if ref.get("checkpoint_sha") and ref["checkpoint_sha"] != ckpt_sha:
        # Same refusal infer.py makes, for the same reason: centroids only mean
        # something in the embedding space of the weights they were built from.
        # Worth repeating here rather than trusting the caller -- an eval that
        # silently measures distances in the wrong space produces numbers that
        # look entirely reasonable and are entirely meaningless.
        sys.exit(f"OOD reference {ref_path} was built from a different checkpoint "
                 f"({ref['checkpoint_sha']} vs {ckpt_sha}); rebuild it with "
                 f"coffeecv.build_ood_reference --checkpoint {checkpoint}")

    model, head = load_model(checkpoint, cfg.model_name, len(class_ids), cfg.dropout)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    tta = not args.no_tta
    scorer = Scorer(ref, ref_path, methods, args.knn_k)

    train_ids, heldout_ids = id_photos(cfg, class_ids, args.id_split)
    if args.limit_id:
        train_ids = train_ids[:args.limit_id]
        heldout_ids = heldout_ids[:args.limit_id]
    negatives = negatives_from([Path(d) for d in args.negatives], args.split)
    positives = negatives_from([Path(d) for d in args.positives], args.split)

    conditions: dict[str, list[dict]] = {
        f"id_split[{args.id_split}]": [{"path": q, "scenario_tag": "-"} for q in train_ids],
    }
    if heldout_ids:
        conditions[f"id_heldout_rig[{cfg.heldout_rig}]"] = [
            {"path": q, "scenario_tag": "-"} for q in heldout_ids]
    # One condition per positive tag: photos the user shot on a training day and
    # photos from an unrelated day are different evidence and must not be pooled.
    for tag in sorted({r["scenario_tag"] for r in positives}):
        conditions[f"id_unseen[{tag}]"] = [r for r in positives if r["scenario_tag"] == tag]
    conditions[f"negatives[{args.split}]"] = negatives

    total = sum(len(v) for v in conditions.values())
    print(f"scoring {total} photos ({len(negatives)} negatives) "
          f"with {'TTA' if tta else 'no TTA'}, {args.n_patches} patches\n")

    # Tags whose every patch is unambiguously not-beans, so a patch-level probe can
    # be trained from the photo-level tag. An empty tray has no bean patches; a
    # sparse scattering of beans does, so those tags are deliberately absent.
    CLEAN_NEGATIVE_TAGS = {"empty_tray", "ground_coffee", "confusable_grain",
                            "other_nuts_seeds", "non_food_objects", "real_world_negatives"}

    per_photo: list[dict] = []
    patch_store: dict[str, np.ndarray] = {}
    live = [m for m in methods if m not in POSTHOC_METHODS]
    done = 0
    for cond, items in conditions.items():
        for item in items:
            path = item["path"].resolve()
            shown = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            row = {"condition": cond, "photo": str(shown),
                   "scenario_tag": item["scenario_tag"], "unmeasurable": False}
            try:
                probs, embeds, logits, diag = embed_photo(item["path"], cfg, model, head,
                                                           args.n_patches, tta, cache_dir, ckpt_sha)
            except Unmeasurable as exc:
                row.update({"unmeasurable": True, "error": str(exc), "beans_across": None})
                for m in methods:
                    row[m] = float("nan")
            else:
                # Free rider for docs/scale_guard_plan.md's open question about
                # whether framing and content are really two separate guards.
                # Recorded, never gated on -- see that document before using it.
                row["beans_across"] = diag.get("beans_across")
                for m in live:
                    row[m] = scorer.score(m, probs, embeds, logits)
                if set(POSTHOC_METHODS) & set(methods):
                    if "linear_probe" in methods:
                        patch_store[str(shown)] = embeds
                    # Trained only on what a deployed system could actually have:
                    # photos from the checkpoint's own data, plus collected
                    # negatives. The unseen positives are deliberately excluded --
                    # letting the probe learn from them would hand it a preview of
                    # the very distribution the other metrics are being judged on.
                    if cond.startswith("id_split["):
                        row["probe_label"] = 0.0
                    elif item["scenario_tag"] in CLEAN_NEGATIVE_TAGS:
                        row["probe_label"] = 1.0
            per_photo.append(row)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total}")

    if "linear_probe" in methods:
        probe = linear_probe_scores(per_photo, patch_store)
        for row in per_photo:
            row["linear_probe"] = probe.get(row["photo"], float("nan"))
    if "ensemble" in methods:
        base = [m for m in methods if m not in POSTHOC_METHODS]
        stacked = ensemble_scores(per_photo, base)
        for row in per_photo:
            row["ensemble"] = stacked.get(row["photo"], float("nan"))
        print(f"\nensemble stacks: {', '.join(base)}")
    for row in per_photo:
        row.pop("probe_label", None)

    id_conditions = [c for c in conditions if c.startswith("id_")]
    neg_condition = f"negatives[{args.split}]"
    results = []
    for m in methods:
        # Unmeasurable photos carry no score, so they are held out of every
        # score-based statistic and counted separately -- but they still count as
        # refusals in the rate, because that is what the user gets.
        scored = {c: np.array([r[m] for r in per_photo
                               if r["condition"] == c and not r["unmeasurable"]])
                  for c in conditions}
        n_unmeasurable = {c: sum(1 for r in per_photo
                                 if r["condition"] == c and r["unmeasurable"])
                          for c in conditions}
        neg = scored[neg_condition]
        for c in conditions:
            s = scored[c]
            n_total = len(s) + n_unmeasurable[c]
            k = int((s > OOD_THRESHOLD).sum()) + n_unmeasurable[c]
            lo, hi = wilson(k, n_total)
            entry = {"method": m, "condition": c, "n": n_total,
                     "n_unmeasurable": n_unmeasurable[c],
                     "median_score": round(float(np.median(s)), 4) if len(s) else None,
                     # Only meaningful for the shipped metric: 1.4 is a cut point in
                     # centroid-distance units and means nothing in an energy or
                     # probability scale, so it is not reported for the others.
                     "declined_at_production_threshold":
                         (round(k / n_total, 3) if n_total else None) if m == "centroid" else None,
                     "declined_ci95": [round(lo, 3), round(hi, 3)] if m == "centroid" else None}
            if c in id_conditions and len(s) and len(neg):
                thr, caught = detection_at_fpr(s, neg)
                entry["auroc_vs_negatives"] = round(auroc(s, neg), 3)
                entry["threshold_at_5pct_false_refusal"] = round(thr, 3)
                entry["negatives_caught_at_that_threshold"] = round(caught, 3)
            results.append(entry)

    print(f"\n{'method':<21} {'condition':<32} {'n':>4} {'unmeas':>6} {'median':>9} "
          f"{'AUROC':>7} {'catch@5%':>9}")
    for e in results:
        print(f"{e['method']:<21} {e['condition']:<32} {e['n']:>4} {e['n_unmeasurable']:>6} "
              f"{str(e['median_score']):>9} "
              f"{str(e.get('auroc_vs_negatives', '-')):>7} "
              f"{str(e.get('negatives_caught_at_that_threshold', '-')):>9}")

    # The headline. Separation against the checkpoint's own held-out split flatters
    # every metric, because those photos come from sessions it trained on; the
    # unseen-positive conditions are the ones that decide whether a guard is usable.
    unseen = [c for c in id_conditions if c.startswith("id_unseen[")]
    if unseen:
        pooled = {}
        for m in methods:
            s = np.array([r[m] for r in per_photo
                          if r["condition"] in unseen and not r["unmeasurable"]
                          and not np.isnan(r[m])])
            neg = np.array([r[m] for r in per_photo
                            if r["condition"] == neg_condition and not r["unmeasurable"]
                            and not np.isnan(r[m])])
            if len(s) and len(neg):
                thr, caught = detection_at_fpr(s, neg)
                pooled[m] = (auroc(s, neg), caught, len(s), len(neg))
        print(f"\nHEADLINE -- all unseen genuine bean photos (n={list(pooled.values())[0][2]}) "
              f"vs negatives (n={list(pooled.values())[0][3]}), {args.split} split:")
        print(f"  {'method':<21} {'AUROC':>7}   {'negatives caught at 5% false-refusal':>38}")
        for m, (a, caught, _, _) in sorted(pooled.items(), key=lambda kv: -kv[1][0]):
            print(f"  {m:<21} {a:>7.3f}   {caught*100:>37.1f}%")

    # Per-tag breakdown: the tags are not interchangeable difficulties, and a
    # pooled negative number hides which kind of photo a metric actually misses.
    tags = sorted({r["scenario_tag"] for r in per_photo if r["condition"] == neg_condition})
    if tags:
        print(f"\nnegatives by scenario tag ({args.split}):")
        # Per-method, per-tag catch rate at each method's *own* 5%-false-refusal
        # threshold, calibrated against the unseen positives -- a fixed 1.4 would
        # be a centroid-scale number applied to metrics measured in other units.
        for m in methods:
            ref_id = np.array([r[m] for r in per_photo
                               if (r["condition"] in unseen if unseen else
                                   r["condition"].startswith("id_"))
                               and not r["unmeasurable"] and not np.isnan(r[m])])
            if not len(ref_id):
                continue
            thr = float(np.quantile(ref_id, 0.95))
            for tag in tags:
                rows = [r for r in per_photo
                        if r["condition"] == neg_condition and r["scenario_tag"] == tag]
                s = np.array([r[m] for r in rows
                              if not r["unmeasurable"] and not np.isnan(r[m])])
                n_un = sum(1 for r in rows if r["unmeasurable"])
                k = int((s > thr).sum()) + n_un
                med = f"{np.median(s):.3f}" if len(s) else "-"
                print(f"  {m:<21} {tag:<24} n={len(rows):>3}  median={med:>9}  "
                      f"caught {k}/{len(rows)}"
                      + (f" ({n_un} unmeasurable)" if n_un else ""))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "checkpoint": str(checkpoint), "checkpoint_sha": ckpt_sha,
            "config_source": cfg_source, "ood_reference": str(ref_path),
            "split": args.split, "id_split": args.id_split, "methods": methods,
            "n_patches": args.n_patches, "tta": tta,
            "production_threshold": OOD_THRESHOLD,
            "results": results, "per_photo": per_photo,
        }, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
