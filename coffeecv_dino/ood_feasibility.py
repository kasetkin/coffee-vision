"""Plan §6.1 kill-switch: can the deployed OOD guard's method work in a frozen DINOv3 space?

The guard that ships is `linear_probe` (docs/ood_guard_eval.md): logistic regression on the per-patch
embeddings the classifier's forward pass already produces, bean patches vs patches of photos whose tag
makes every patch a negative, pooled to a per-photo mean P(not-beans). With a frozen backbone that
embedding space is fixed before any head exists, so the probe can be measured in it now, before the
backbone is adopted, rather than after.

This runs the probe's photo-level cross-validation -- `ood_eval.linear_probe_scores`, imported, the
protocol that chose probe v2 -- in two spaces, on the same day, on the identical photos and the identical
patches:

  r18    the deployed checkpoint's 512-d head input (what production's probe reads today)
  dino   a frozen backbone's readout (default: the selected dinov3_vitb16 x cls_mean, 1536-d)

The photos are the ones `ood_eval` scores: the deployed checkpoint's own test split as in-distribution
(probe label 0), the three negative batches (clean tags are probe label 1) and the three positive
batches, which are never fitted on. Patches come from `infer.patches_for_photo` with the deployed
config and seed key, so both spaces read the same pixels. R18 embeddings are independent of TTA
(`forward_with_embeddings` takes them from the untransformed pass), so no TTA is run.

**Dev split only.** The holdout is spent once, by `fit_ood_probe --verify`, on the checkpoint that
ships (plan §9.3); a feasibility check that read it would turn it into a second dev set. There is no
flag to change that.

Pass (plan §6.1): the dino space's probe AUROC on the same-rig batch (2026-09-11 positives vs the
same-rig negatives), and on its green legumes specifically, is no worse than the r18 space's.

    COFFEECV_TORCH_THREADS=all python -m coffeecv_dino.ood_feasibility
"""
from __future__ import annotations

import os

import torch

_threads_env = os.environ.get("COFFEECV_TORCH_THREADS")
if _threads_env == "all":
    torch.set_num_threads(os.cpu_count())
elif _threads_env:
    torch.set_num_threads(int(_threads_env))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from coffeecv.config import OUTPUTS_DIR, REPO_ROOT, build_env_block  # noqa: E402
from coffeecv.dataset import load_class_labels  # noqa: E402
from coffeecv.infer import _sha, config_for_checkpoint, forward_with_embeddings, load_model, patches_for_photo  # noqa: E402
from coffeecv.ood_eval import (CLEAN_NEGATIVE_TAGS, auroc, detection_at_fpr, id_photos,  # noqa: E402
                               linear_probe_scores, negatives_from)
from coffeecv.transforms import build_eval_transform  # noqa: E402
from coffeecv.backbones import SPECS, assert_input_size, build_backbone  # noqa: E402

DEPLOYED = REPO_ROOT / "models" / "allrigs_cam_s123.pt"
SPLIT = "dev"
NEGATIVES = ("dataset/ood_negatives/2026-09__internet_proxy", "dataset/ood_negatives/2026-09__user_realworld",
             "dataset/ood_negatives/2026-09-11__user_samerig")
POSITIVES = ("dataset/ood_positives", "dataset/ood_positives_2026-09-11", "dataset/ood_positives_internet")
SAMERIG_POS, SAMERIG_NEG = "ood_positives_2026-09-11", "2026-09-11__user_samerig"
USER_POS = {"ood_positives", SAMERIG_POS}
USER_NEG = {"2026-09__user_realworld", SAMERIG_NEG}
DEFAULT_OUT = OUTPUTS_DIR / "dino_ood_feasibility"


def log(msg: str) -> None:
    print(f"[ood-feasibility {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def photo_rows(cfg, class_ids: list[str]) -> list[dict]:
    """Every photo the comparison scores, with the fields ood_eval.main gives it."""
    train_ids = id_photos(cfg, class_ids, "test")
    rows = [{"condition": "id_split[test]", "path": p, "scenario_tag": "-", "batch": "id_split",
             "probe_label": 0.0} for p in train_ids]
    for r in negatives_from([REPO_ROOT / d for d in NEGATIVES], SPLIT):
        rows.append({"condition": f"negatives[{SPLIT}]", **r,
                     "probe_label": 1.0 if r["scenario_tag"] in CLEAN_NEGATIVE_TAGS else None})
    for r in negatives_from([REPO_ROOT / d for d in POSITIVES], SPLIT):
        rows.append({"condition": f"id_unseen[{r['scenario_tag']}]", **r, "probe_label": None})
    for r in rows:
        path = Path(r["path"]).resolve()
        r["photo"] = str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
    return rows


def embed_all(rows: list[dict], cfg, n_patches: int, r18, dino, readout: str, cache: Path) -> dict:
    """{space: {photo: [n_patches, D]}}; unmeasurable photos get no entry and are marked on the row,
    exactly as ood_eval does (production refuses them before any guard runs)."""
    model, head = r18
    transform = build_eval_transform(cfg.patch_resize)
    store: dict[str, dict[str, np.ndarray]] = {"r18": {}, "dino": {}}
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        key = cache / f"{Path(r['photo']).stem}__{hashlib.sha1(r['photo'].encode()).hexdigest()[:10]}.npz"
        if key.exists():
            z = np.load(key, allow_pickle=True)
            meta = json.loads(str(z["meta"]))
        else:
            try:
                patches, _ = patches_for_photo(Path(r["path"]), cfg, n_patches, [42, 0])
            except (ValueError, OSError) as exc:      # the catch classify_one uses
                meta = {"photo": r["photo"], "unmeasurable": True, "error": str(exc)}
                np.savez_compressed(key, meta=json.dumps(meta))
            else:
                x = torch.stack([transform(p) for p in patches])
                _, e18, _ = forward_with_embeddings(model, head, x, tta=False, return_logits=True)
                with torch.inference_mode():
                    ed = torch.cat([dino.features(x[j:j + 32])[readout] for j in range(0, len(x), 32)])
                meta = {"photo": r["photo"], "unmeasurable": False}
                np.savez_compressed(key, meta=json.dumps(meta), r18=e18.astype(np.float32),
                                    dino=ed.float().numpy())
                z = np.load(key, allow_pickle=True)
        if meta["photo"] != r["photo"]:
            raise RuntimeError(f"cache collision at {key}: {meta['photo']} vs {r['photo']}")
        r["unmeasurable"] = meta["unmeasurable"]
        if not meta["unmeasurable"]:
            store["r18"][r["photo"]] = z["r18"]
            store["dino"][r["photo"]] = z["dino"]
        if i % 25 == 0 or i == len(rows):
            log(f"embedded {i}/{len(rows)} photos ({time.perf_counter() - t0:.0f}s)")
    return store


def compare(rows: list[dict], scores: dict[str, dict[str, float]]) -> dict:
    """Separation per population, per space. Unmeasurable photos count as refusals in catch rates and
    are left out of AUROC, as in ood_eval."""
    def pick(pred):
        return [r for r in rows if pred(r)]

    unseen = lambda r: r["condition"].startswith("id_unseen[")   # noqa: E731
    neg = lambda r: r["condition"].startswith("negatives[")      # noqa: E731
    groups = {
        # the headline groups: plan §6.1's pass rule reads the first two
        "same_rig": (pick(lambda r: unseen(r) and r["batch"] == SAMERIG_POS),
                     pick(lambda r: neg(r) and r["batch"] == SAMERIG_NEG)),
        "same_rig_green_legume": (pick(lambda r: unseen(r) and r["batch"] == SAMERIG_POS),
                                  pick(lambda r: neg(r) and r["scenario_tag"] == "green_legume")),
        "user_matched": (pick(lambda r: unseen(r) and r["batch"] in USER_POS),
                         pick(lambda r: neg(r) and r["batch"] in USER_NEG)),
        "internet_matched": (pick(lambda r: unseen(r) and r["batch"] == "ood_positives_internet"),
                             pick(lambda r: neg(r) and r["batch"] == "2026-09__internet_proxy")),
        "pooled_unseen": (pick(unseen), pick(neg)),
        "own_test_split": (pick(lambda r: r["condition"] == "id_split[test]"), pick(neg)),
    }
    out: dict = {}
    for space, sc in scores.items():
        out[space] = {}
        for g, (pos, negs) in groups.items():
            p = np.array([sc[r["photo"]] for r in pos if not r["unmeasurable"]])
            n = np.array([sc[r["photo"]] for r in negs if not r["unmeasurable"]])
            entry = {"n_pos": len(pos), "n_neg": len(negs),
                     "n_unmeasurable": sum(r["unmeasurable"] for r in pos + negs)}
            if len(p) and len(n):
                thr, caught = detection_at_fpr(p, n)
                entry.update({"auroc": round(auroc(p, n), 4), "threshold_at_5pct_false_refusal": round(thr, 4),
                              "negatives_caught_at_that_threshold": round(caught, 4),
                              "pos_median": round(float(np.median(p)), 4), "neg_median": round(float(np.median(n)), 4)})
            out[space][g] = entry
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(DEPLOYED), help="the deployed ResNet18 (the r18 space)")
    p.add_argument("--backbone", default="dinov3_vitb16", choices=sorted(s for s in SPECS if s.startswith("dino")))
    p.add_argument("--readout", default="cls_mean")
    p.add_argument("--n-patches", type=int, default=40)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--smoke", action="store_true",
                   help="6 photos per condition and 4 patches, into <out>/smoke: tests the pipeline, measures nothing")
    args = p.parse_args()
    out = Path(args.out) / "smoke" if args.smoke else Path(args.out)
    if args.smoke:
        args.n_patches = 4

    checkpoint = Path(args.checkpoint)
    cfg, cfg_source = config_for_checkpoint(checkpoint, None)
    _, classes_file = cfg.resolve_paths()
    class_ids = sorted(load_class_labels(classes_file))
    r18 = load_model(checkpoint, cfg.model_name, len(class_ids), cfg.dropout)
    dino = build_backbone(args.backbone).eval()
    assert_input_size(dino, cfg.patch_resize)
    if args.readout not in dino.readouts:
        raise SystemExit(f"{args.backbone} has readouts {dino.readouts}, not {args.readout!r}")
    rows = photo_rows(cfg, class_ids)
    if args.smoke:
        seen: dict[str, int] = {}
        rows = [r for r in rows if (seen := {**seen, r["batch"]: seen.get(r["batch"], 0) + 1})[r["batch"]] <= 6]
    log(f"host {socket.gethostname()}, torch threads {torch.get_num_threads()}; r18 = {checkpoint.name} "
        f"({cfg_source}), dino = {args.backbone} x {args.readout}; {len(rows)} {SPLIT}-split photos, "
        f"{args.n_patches} patches each")

    cache = out / "cache" / f"{_sha(checkpoint)[:12]}_{args.backbone}_{args.readout}_{args.n_patches}"
    store = embed_all(rows, cfg, args.n_patches, r18, dino, args.readout, cache)

    scores = {}
    for space in ("r18", "dino"):
        per_photo = [{"photo": r["photo"], "probe_label": r["probe_label"]}
                     for r in rows if not r["unmeasurable"]]
        scores[space] = linear_probe_scores(per_photo, store[space])
    result = compare(rows, scores)

    print(f"\nlinear_probe, photo-level 5-fold CV, {SPLIT} split -- AUROC (catch at 5% false refusal)")
    print(f"{'population':24s}{'n pos/neg':>11s}{'r18 (deployed space)':>26s}{args.backbone + ' ' + args.readout:>30s}")
    for g in result["r18"]:
        a, b = result["r18"][g], result["dino"][g]
        fmt = lambda e: f"{e['auroc']:.3f} ({e['negatives_caught_at_that_threshold']:.2f})" if "auroc" in e else "-"  # noqa: E731
        print(f"{g:24s}{str(a['n_pos']) + '/' + str(a['n_neg']):>11s}{fmt(a):>26s}{fmt(b):>30s}")
    gate = {g: result["dino"][g].get("auroc", -1) >= result["r18"][g].get("auroc", 2)
            for g in ("same_rig", "same_rig_green_legume")}
    verdict = "PASS" if all(gate.values()) else "FAIL -- shipping is blocked until the guard has had its own work"
    print(f"\n§6.1 gate (dino AUROC >= r18 AUROC on same_rig and on its green legumes): {verdict}  {gate}")

    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.backbone}__{args.readout}.json").write_text(json.dumps({
        "plan": "docs/dinov3_integration_plan.md §6.1", "split": SPLIT, "checkpoint": str(checkpoint),
        "checkpoint_sha": _sha(checkpoint), "config_source": cfg_source, "backbone": args.backbone,
        "readout": args.readout, "weights_sha256": dino.weights_sha256, "n_patches": args.n_patches,
        "host": socket.gethostname(), "torch_threads": torch.get_num_threads(), "env": build_env_block(),
        "finished_at": datetime.now(timezone.utc).isoformat(), "gate": gate, "pass": all(gate.values()),
        "results": result,
        "per_photo": [{k: r[k] for k in ("photo", "condition", "batch", "scenario_tag", "unmeasurable")}
                      | {s: scores[s].get(r["photo"]) for s in scores} for r in rows],
    }, indent=1))
    log(f"wrote {out / f'{args.backbone}__{args.readout}.json'}")
    return 0                                      # a FAIL is a finding, not a crash: it is in the JSON


if __name__ == "__main__":
    sys.exit(main())
