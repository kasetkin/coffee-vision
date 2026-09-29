"""Experiment 1 of docs/dinov3_integration_plan.md: frozen backbones + a linear head under the REAL
leave-one-camera-out fold protocol.

For every (seed, held-out rig) this builds the fold with `coffeecv.fold_data.build_fold_datasets` --
the call train_baseline makes -- so at seed 42 the patches are the ones exp200-203 trained on and were
scored on. It embeds every split once per backbone (all readouts from one forward pass), fits an L2
logistic-regression head per (backbone, readout) with C chosen on val (coffeecv.linear_head), and scores
val / test / cross-rig with `coffeecv.metrics.compute_split_metrics`, the iPhone fold as its honest
8-class macro. The pre-registered primary cell is also scored photo by photo through
`coffeecv.xrig_eval.run_photowise` -- the /classify path, 40 patches per photo, no TTA.

Resumable: results accumulate in <out>/results.json, keyed by (seed, held-out rig, backbone), and a
restart skips whatever is already there. Nothing here writes params.yaml, dvc.lock, coffeecv/ or
experiments/. The primary cell's runs go into the experiment record afterwards with --archive
(plan §5.7), as exp220-231; the selected cell's (the owner's pick of ViT-B/16 after Experiment 1) with
--archive selected, as exp240-251.

    COFFEECV_TORCH_THREADS=all python -m coffeecv_dino.screen --seeds 42 123 7 \\
        --backbones resnet18 dinov2_vits14 dinov3_vits16 --photo-pool dinov3_vits16:cls_mean
    python -m coffeecv_dino.screen --summary            # print the tables and gates from results.json
    python -m coffeecv_dino.screen --archive            # primary cell -> experiments/exp220-231
    python -m coffeecv_dino.screen --archive selected   # selected cell -> experiments/exp240-251
"""
from __future__ import annotations

import os

import torch

# Same opt-in convention as coffeecv/train_baseline.py, read before any torch op: unset keeps torch's
# default (physical cores); "all" uses every logical CPU. For frozen ViT inference on the compute box
# "all" is ~23% faster (plan §6.2); thread count changes speed and float rounding, nothing else.
_threads_env = os.environ.get("COFFEECV_TORCH_THREADS")
if _threads_env == "all":
    torch.set_num_threads(os.cpu_count())
elif _threads_env:
    torch.set_num_threads(int(_threads_env))

import argparse  # noqa: E402
import csv  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from coffeecv.archive_experiment import archive  # noqa: E402
from coffeecv.config import OUTPUTS_DIR, REPO_ROOT, RunConfig, build_env_block, config_to_dict, set_seed  # noqa: E402
from coffeecv.fold_data import build_fold_datasets  # noqa: E402
from coffeecv.metrics import build_metrics_json, compute_split_metrics, write_predictions_csv  # noqa: E402
from coffeecv.photo_pooling_eval import pool_photos  # noqa: E402
from coffeecv.repro_utils import dirty_provenance_paths, stale_crop_stages  # noqa: E402
from coffeecv.run_folds import RIGS, train_rigs_for  # noqa: E402
from coffeecv.transforms import build_eval_transform  # noqa: E402
from coffeecv.xrig_eval import run_photowise  # noqa: E402
from coffeecv.backbones import READOUTS, SPECS, FrozenBackbone, assert_input_size, build_backbone  # noqa: E402
from coffeecv.linear_head import C_GRID, cross_entropy, fit_head, predict  # noqa: E402
from coffeecv.dino_classifier import DinoClassifier  # noqa: E402

DEFAULT_OUT = OUTPUTS_DIR / "dino_screen"
PRIMARY = ("dinov3_vits16", "cls_mean")          # pre-registered, plan §5.2 -- not chosen after the fact
# The owner's choice on 2026-09-26, made on Experiment 1's evidence (ViT-B/16 beat the primary cell on
# 11/12 pairs). The readout stays cls_mean by the pre-registered rule: CLS-only won 9/12, not 12/12.
SELECTED = ("dinov3_vitb16", "cls_mean")
# cell -> (first exp id, slug stem). Pre-assigned; exp232-239 are the ResNet18 seed 123/7 folds (plan §7.1).
ARCHIVE_SETS = {"primary": (PRIMARY, 220, "dino3s16"), "selected": (SELECTED, 240, "dino3b16")}
SEEDS = (42, 123, 7)
TEN_CLASS_FOLDS = ("cam_pixel", "cam_sony", "cam_oneplus")   # cam_iphone has 8 classes; never mixed in
PHOTO_PATCHES = 40                               # what /classify samples per photo
EMBED_BATCH = 32
# The fine-tuned ResNet18 + MixStyle camera-rig baseline, seed 42 only (plan §5.4).
BASELINE_EXPS = {"cam_pixel": 200, "cam_sony": 201, "cam_oneplus": 202, "cam_iphone": 203}
# Every RunConfig field that decides which photos and boxes a fold draws. If one differs from the
# baseline's archived config, "same patches as exp20X" is false and the summary says so.
SAMPLING_FIELDS = (
    "patch_crop_size", "patch_resize", "patch_store_size", "safety_margin", "train_patches_per_class",
    "val_patches_per_class", "test_patches_per_class", "xrig_patches_per_class", "train_photo_frac",
    "val_photo_frac", "test_photo_frac", "patch_scale_frac_min", "patch_scale_frac_max",
    "patch_beans_min", "patch_beans_max", "bean_k_lo", "bean_k_hi", "bean_analysis_frac",
    "bean_calibration_k", "rotation_jitter_degrees", "classes_file", "seed", "train_rigs", "heldout_rig",
)
SMOKE_BUDGET = dict(train_patches_per_class=6, val_patches_per_class=3, test_patches_per_class=3,
                    xrig_patches_per_class=4)
SMOKE_PHOTO_PATCHES = 4


def exp_id_for(seed: int, heldout: str, first: int = 220) -> int:
    """Pre-assigned per seed so two machines could never write the same expNNN (plan §5.5)."""
    return first + 4 * SEEDS.index(seed) + RIGS.index(heldout)


def rig_name(rig: str) -> str:
    return Path(rig).name


def log(msg: str) -> None:
    print(f"[screen {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------- results store

def load_results(out: Path) -> dict:
    f = out / "results.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"cells": {}, "folds": {}, "meta": {}}


def save_results(out: Path, results: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "results.json.tmp"
    tmp.write_text(json.dumps(results, indent=1))
    tmp.replace(out / "results.json")             # atomic: a kill mid-write cannot corrupt the store


def cell_key(seed: int, heldout: str, backbone: str, readout: str) -> str:
    return f"s{seed}/{rig_name(heldout)}/{backbone}/{readout}"


def backbone_done(results: dict, seed: int, heldout: str, backbone: str, photo_pool: set) -> bool:
    for readout in _readouts(backbone):
        cell = results["cells"].get(cell_key(seed, heldout, backbone, readout))
        if cell is None or ((backbone, readout) in photo_pool and "photo" not in cell):
            return False
    return True


def _readouts(backbone: str) -> tuple[str, ...]:
    return READOUTS[SPECS[backbone].family]


# ---------------------------------------------------------------------------------------- one fold

def pairing_check(cfg: RunConfig, heldout: str) -> dict:
    """Does this fold draw exactly the photos and boxes its fine-tuned baseline was scored on?"""
    exp = BASELINE_EXPS.get(rig_name(heldout))
    if cfg.seed != 42 or exp is None:
        return {"baseline_exp": None, "paired": False, "reason": "only seed 42 has a fine-tuned baseline"}
    [exp_dir] = list((REPO_ROOT / "experiments").glob(f"exp{exp}__*"))
    base = json.loads((exp_dir / "config.json").read_text())
    mine = config_to_dict(cfg)
    diffs = {k: [base.get(k), mine.get(k)] for k in SAMPLING_FIELDS
             if (list(base[k]) if isinstance(base.get(k), (list, tuple)) else base.get(k))
             != (list(mine[k]) if isinstance(mine.get(k), (list, tuple)) else mine.get(k))}
    return {"baseline_exp": exp, "paired": not diffs, "diffs": diffs}


@torch.no_grad()
def embed_split(ds, backbones: dict[str, FrozenBackbone]) -> tuple[dict, np.ndarray, dict]:
    """features[backbone][readout] -> [N, D] float32, labels [N], ms/img per backbone."""
    feats = {b: {r: [] for r in bb.readouts} for b, bb in backbones.items()}
    spent = {b: 0.0 for b in backbones}
    labels = []
    for start in range(0, len(ds), EMBED_BATCH):
        items = [ds[i] for i in range(start, min(start + EMBED_BATCH, len(ds)))]
        x = torch.stack([it[0] for it in items])
        labels.extend(int(it[1]) for it in items)
        for b, bb in backbones.items():
            t = time.perf_counter()
            out = bb.features(x)
            spent[b] += time.perf_counter() - t
            for r in bb.readouts:
                feats[b][r].append(out[r].float().numpy())
    n = max(len(labels), 1)
    return ({b: {r: np.concatenate(v) for r, v in d.items()} for b, d in feats.items()},
            np.array(labels), {b: 1000 * s / n for b, s in spent.items()})


def run_record_config(cfg: RunConfig, class_ids: list[str], bb: FrozenBackbone, readout: str,
                      fit, smoke: bool) -> dict:
    """config.json in the usual shape, stating what actually ran: the fields that describe a frozen
    backbone with a convex head are set to what happened, and a `dino` block records the rest."""
    import timm

    record_cfg = replace(cfg, model_name=bb.name, freeze_mode="full", mixstyle_p=0.0, dropout=0.0,
                         color_jitter_strength=0.0, random_erasing_p=0.0)
    return {
        **config_to_dict(record_cfg), "class_ids": class_ids, "env": build_env_block(),
        "dino": {
            "experiment": "docs/dinov3_integration_plan.md Experiment 1 (frozen, real fold protocol)",
            "backbone": bb.name, "readout": readout, "weights": bb.spec.weights,
            "weights_sha256": bb.weights_sha256,
            "timm_version": timm.__version__ if bb.spec.family == "dinov3" else None,
            "head": "sklearn LogisticRegression (L2, lbfgs) on standardised features, exported to nn.Linear; "
                    "the SGD-loop fields above (epochs, lr, optimizer, scheduler, ...) are unused",
            "C": fit.C, "C_grid": list(C_GRID), "val_macro_f1_by_C": {str(k): v for k, v in fit.val_macro_f1_by_C.items()},
            "not_converged_C": fit.not_converged,
            "train_transform": "eval transform (frozen arms get no photometric augmentation)",
            "smoke": smoke, "host": socket.gethostname(), "cpu": _cpu_model(),
            "torch_threads": torch.get_num_threads(),
        },
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run_fold(seed: int, heldout: str, backbones: dict[str, FrozenBackbone], photo_pool: set,
             base_cfg: RunConfig, out: Path, results: dict, smoke: bool) -> None:
    cfg = replace(base_cfg, seed=seed, train_rigs=tuple(train_rigs_for(heldout)), heldout_rig=heldout)
    if smoke:
        cfg = replace(cfg, **SMOKE_BUDGET)
    fold_key = f"s{seed}/{rig_name(heldout)}"
    pairing = pairing_check(cfg, heldout) if not smoke else {"paired": False, "reason": "smoke"}
    if pairing.get("diffs"):
        log(f"WARNING {fold_key}: sampling config differs from exp{pairing['baseline_exp']}: {pairing['diffs']}")

    set_seed(seed)
    eval_tf = build_eval_transform(cfg.patch_resize)
    t0 = time.perf_counter()
    fold = build_fold_datasets(cfg, eval_tf, eval_tf)
    build_s = time.perf_counter() - t0
    sizes = {s: len(d) for s, d in (("train", fold.train), ("val", fold.val), ("test", fold.test), ("xrig", fold.xrig))}
    log(f"{fold_key}: datasets built in {build_s:.0f}s  {sizes}  train rigs "
        f"{[r.name for r in fold.train_rigs]}  paired with baseline: {pairing.get('paired')}")

    feats, labels, ms = {}, {}, {}
    for split, ds in (("train", fold.train), ("val", fold.val), ("test", fold.test), ("xrig", fold.xrig)):
        t = time.perf_counter()
        f, y, m = embed_split(ds, backbones)
        labels[split] = y
        for b in backbones:
            feats.setdefault(b, {})[split] = f[b]
            ms.setdefault(b, {})[split] = round(m[b], 1)
        log(f"{fold_key}: embedded {split:5s} ({len(ds)} patches) in {time.perf_counter() - t:.0f}s  "
            + "  ".join(f"{b} {m[b]:.1f} ms/img" for b in backbones))
    present, missing = fold.xrig.present_class_idxs, fold.xrig.missing_classes
    class_ids, class_labels, heldout_rig = fold.class_ids, fold.class_labels, fold.heldout_rig
    rigs_record = {"train": [r.name for r in fold.train_rigs], "heldout": heldout_rig.name}
    del fold                                        # ~5 GB of stored patches; nothing below needs pixels
    gc.collect()

    fold_rec = results["folds"].setdefault(fold_key, {})
    fold_rec.update({"sizes": sizes, "build_seconds": round(build_s), "pairing": pairing,
                     "host": socket.gethostname(), "torch_threads": torch.get_num_threads()})
    fold_rec.setdefault("embed_ms_per_img", {}).update(ms)
    for b, bb in backbones.items():
        for readout in bb.readouts:
            X = {s: feats[b][s][readout] for s in labels}
            fit = fit_head(X["train"], labels["train"], X["val"], labels["val"], class_ids, class_labels)
            split_metrics, preds = {}, {}
            for s in ("val", "test", "xrig"):
                pred, probs, _ = predict(fit.linear, X[s])
                preds[s] = pred
                split_metrics[s] = compute_split_metrics(
                    labels[s], pred, cross_entropy(probs, labels[s]), class_ids, class_labels,
                    macro_labels=present if s == "xrig" else None)
            if missing:
                split_metrics["xrig"]["missing_classes"] = missing
            metrics_json = build_metrics_json(
                class_ids, class_labels, epochs_trained=None, best_epoch=None,
                val_metrics=split_metrics["val"], test_metrics=split_metrics["test"],
                xrig_metrics=split_metrics["xrig"], rigs=rigs_record)
            metrics_json["best_epoch_selection_metric"] = "val_macro_f1 over the C grid (head refit per C)"
            run_dir = out / "runs" / f"s{seed}" / rig_name(heldout) / f"{b}__{readout}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))
            (run_dir / "config.json").write_text(json.dumps(
                run_record_config(cfg, class_ids, bb, readout, fit, smoke), indent=2))
            for s, name in (("val", "predictions_val.csv"), ("test", "predictions_test.csv"),
                            ("xrig", "predictions_xrig.csv")):
                write_predictions_csv(run_dir / name, labels[s], preds[s], class_ids)
            torch.save(fit.linear.state_dict(), run_dir / "head.pt")

            cell = {
                "seed": seed, "heldout": rig_name(heldout), "backbone": b, "readout": readout,
                "dim": int(X["train"].shape[1]), "C": fit.C, "not_converged_C": fit.not_converged,
                # The optimum sitting on the grid's edge means the grid, not the data, chose C.
                "C_at_grid_edge": fit.C in (min(C_GRID), max(C_GRID)),
                "fit_seconds": round(fit.fit_seconds, 1),
                **{f"{s}_{k}": split_metrics[s][k] for s in ("val", "test", "xrig") for k in ("macro_f1", "mcc")},
                "xrig_macro_n": split_metrics["xrig"]["macro_n"],
                "run_dir": str(run_dir.relative_to(out)),
            }
            if (b, readout) in photo_pool:
                n_pp = SMOKE_PHOTO_PATCHES if smoke else PHOTO_PATCHES
                t = time.perf_counter()
                model = DinoClassifier(bb, readout, fit.linear).eval()
                probs, plabels = run_photowise(model, model.head, cfg, heldout_rig, class_ids, n_pp, seed,
                                               dihedral_tta=False)
                per_patch, pooled, photo_y = pool_photos(probs, plabels, n_pp)
                photo = compute_split_metrics(photo_y, pooled.argmax(axis=1), cross_entropy(pooled, photo_y),
                                              class_ids, class_labels, macro_labels=present)
                patch_pred = per_patch.argmax(axis=2).reshape(-1)
                patch = compute_split_metrics(np.repeat(photo_y, n_pp), patch_pred,
                                              np.zeros(len(patch_pred)), class_ids, class_labels,
                                              macro_labels=present)
                (run_dir / "photo_metrics.json").write_text(json.dumps(
                    {"n_photos": len(photo_y), "patches_per_photo": n_pp, "dihedral_tta": False,
                     "photo": photo, "patch_same_draw": patch}, indent=2))
                cell["photo"] = {"n_photos": len(photo_y), "macro_f1": photo["macro_f1"],
                                 "accuracy": photo["accuracy"], "patch_macro_f1": patch["macro_f1"],
                                 "seconds": round(time.perf_counter() - t)}
            results["cells"][cell_key(seed, heldout, b, readout)] = cell
            save_results(out, results)
            photo_txt = (f"  photo-pooled macro-F1 {cell['photo']['macro_f1']:.4f} "
                         f"(n={cell['photo']['n_photos']})" if "photo" in cell else "")
            log(f"{fold_key}: {b:17s} {readout:8s} dim {cell['dim']:4d}  C={fit.C:<8.4g} "
                f"val {cell['val_macro_f1']:.4f}  test {cell['test_macro_f1']:.4f}  "
                f"XRIG {cell['xrig_macro_f1']:.4f} (n={cell['xrig_macro_n']}){photo_txt}  "
                f"{'[C AT GRID EDGE] ' if cell['C_at_grid_edge'] else ''}"
                f"[fit {fit.fit_seconds:.0f}s{', NOT CONVERGED at C=' + str(fit.not_converged) if fit.not_converged else ''}]")


# ---------------------------------------------------------------------------------------- summary

def summarize(results: dict) -> dict:
    """Tables and gates from whatever is in results.json; works on a partial run too."""
    cells = results["cells"]
    lines: list[str] = []
    out: dict = {"cells": {}, "gates": {}}

    def xrig(seed, fold, b, r):
        c = cells.get(f"s{seed}/{fold}/{b}/{r}")
        return None if c is None else c["xrig_macro_f1"]

    folds = [rig_name(r) for r in RIGS]
    seeds = sorted({c["seed"] for c in cells.values()}, key=lambda s: SEEDS.index(s) if s in SEEDS else 99)
    combos = sorted({(c["backbone"], c["readout"]) for c in cells.values()},
                    key=lambda br: (list(SPECS).index(br[0]), br[1]))
    lines.append("cross-rig macro-F1 (patch level); '3-fold' = the three ten-class folds, 'all4' includes "
                 "cam_iphone (8-class)")
    lines.append(f"{'backbone':18s}{'readout':9s}{'seed':>5s}  " + "".join(f"{f[4:]:>9s}" for f in folds)
                 + f"{'3-fold':>9s}{'all4':>8s}")
    for b, r in combos:
        per_seed_3 = []
        for seed in seeds:
            vals = [xrig(seed, f, b, r) for f in folds]
            ten = [v for f, v in zip(folds, vals) if f in TEN_CLASS_FOLDS and v is not None]
            m3 = float(np.mean(ten)) if len(ten) == 3 else None
            m4 = float(np.mean(vals)) if all(v is not None for v in vals) else None
            if m3 is not None:
                per_seed_3.append(m3)
            lines.append(f"{b:18s}{r:9s}{seed:>5d}  " + "".join(f"{v:9.4f}" if v is not None else f"{'-':>9s}" for v in vals)
                         + (f"{m3:9.4f}" if m3 is not None else f"{'-':>9s}") + (f"{m4:8.4f}" if m4 is not None else f"{'-':>8s}"))
        if per_seed_3:
            mean3 = float(np.mean(per_seed_3))
            sd3 = float(np.std(per_seed_3, ddof=1)) if len(per_seed_3) > 1 else None
            out["cells"][f"{b}/{r}"] = {"mean_3fold": mean3, "seed_sd_3fold": sd3, "n_seeds": len(per_seed_3)}
            lines.append(f"{'':18s}{'':9s}{'mean':>5s}  {'':36s}{mean3:9.4f}   seed sd "
                         + (f"{sd3:.4f}" if sd3 is not None else "n/a") + f"  ({len(per_seed_3)} seeds)")

    def paired(a, b_):
        deltas = []
        for seed in seeds:
            for f in folds:
                va, vb = xrig(seed, f, *a), xrig(seed, f, *b_)
                if va is not None and vb is not None:
                    deltas.append(va - vb)
        return deltas

    def describe(name, d):
        if not d:
            return f"{name}: no pairs yet"
        pos = sum(x > 0 for x in d)
        return (f"{name}: {len(d)} pairs, {pos}/{len(d)} positive, mean {np.mean(d):+.4f}, "
                f"min {min(d):+.4f}, max {max(d):+.4f}")

    lines.append("")
    lines.append("paired (fold x seed) deltas, cross-rig macro-F1:")
    prim = PRIMARY
    d_r18 = paired(prim, ("resnet18", "avgpool"))
    d_v2 = paired(prim, ("dinov2_vits14", "cls_mean"))
    d_v2cls = paired(prim, ("dinov2_vits14", "cls"))
    d_readout = paired(("dinov3_vits16", "cls"), prim)
    for name, d in (("V3 cls_mean - R18 frozen", d_r18), ("V3 cls_mean - V2 cls_mean", d_v2),
                    ("V3 cls_mean - V2 cls (the screen's readout)", d_v2cls),
                    ("V3 cls - V3 cls_mean (readout lever)", d_readout),
                    ("V3-B cls_mean (selected) - V3 cls_mean", paired(SELECTED, prim)),
                    ("V3-B cls_mean (selected) - R18 frozen", paired(SELECTED, ("resnet18", "avgpool"))),
                    ("V3-B cls - V3-B cls_mean (readout lever, selected)", paired(("dinov3_vitb16", "cls"), SELECTED))):
        lines.append("  " + describe(name, d))

    # Against the fine-tuned ResNet18 + MixStyle baseline: seed 42 only, the only seed that exists.
    base = {}
    with open(REPO_ROOT / "experiments" / "index.csv") as f:
        for row in csv.DictReader(f):
            for fold, exp in BASELINE_EXPS.items():
                if row["exp"] == str(exp):
                    exp_dir = next((REPO_ROOT / "experiments").glob(f"exp{exp}__*"))
                    hist = json.loads((exp_dir / "history.json").read_text())
                    base[fold] = (float(row["xrig_macro_f1"]), float(np.mean([h["xrig_macro_f1"] for h in hist[-10:]])))
    d_ft = [(f, xrig(42, f, *prim) - base[f][0], xrig(42, f, *prim) - base[f][1])
            for f in folds if base.get(f) and xrig(42, f, *prim) is not None]
    if d_ft:
        lines.append(f"  V3 cls_mean - fine-tuned ResNet18 (exp200-203, seed 42): "
                     + "  ".join(f"{f[4:]} {dv:+.4f} (vs last-10 {dl:+.4f})" for f, dv, dl in d_ft))
        ten = [x for x in d_ft if x[0] in TEN_CLASS_FOLDS]
        if len(ten) == 3:
            m_prim = np.mean([xrig(42, f, *prim) for f, _, _ in ten])
            lines.append(f"    3 ten-class folds at seed 42: V3 {m_prim:.4f} vs fine-tuned val-peak "
                         f"{np.mean([base[f][0] for f, _, _ in ten]):.4f} / last-10 {np.mean([base[f][1] for f, _, _ in ten]):.4f}")

    for label, which in (("primary", PRIMARY), ("selected", SELECTED)):
        photo = [(c["seed"], c["heldout"], c["photo"]["macro_f1"], c["xrig_macro_f1"]) for c in cells.values()
                 if (c["backbone"], c["readout"]) == which and "photo" in c]
        if not photo:
            continue
        lines.append("")
        lines.append(f"{label} cell {which[0]} {which[1]}, photo-pooled cross-rig macro-F1 "
                     "(40 patches/photo, no TTA) vs patch level:")
        for seed, fold, pm, xm in sorted(photo, key=lambda t: (SEEDS.index(t[0]) if t[0] in SEEDS else 99, folds.index(t[1]))):
            lines.append(f"  s{seed:<4d}{fold:13s} photo {pm:.4f}  patch {xm:.4f}  delta {pm - xm:+.4f}")
        per_seed = {}
        for seed, fold, pm, _ in photo:
            if fold in TEN_CLASS_FOLDS:
                per_seed.setdefault(seed, []).append(pm)
        full_seeds = [float(np.mean(v)) for v in per_seed.values() if len(v) == 3]
        if full_seeds:
            lines.append(f"  3-fold photo-pooled mean {np.mean(full_seeds):.4f} over {len(full_seeds)} seed(s)")

    # Gates, plan §5.6. Each states whether it has its full evidence yet.
    full = len(seeds) >= 3 and all(xrig(s, f, *prim) is not None for s in SEEDS for f in folds)
    g1 = bool(d_r18) and all(x > 0 for x in d_r18)
    m3 = out["cells"].get(f"{prim[0]}/{prim[1]}", {}).get("mean_3fold")
    g2 = m3 is not None and m3 >= 0.75
    g3_stop = bool(d_v2) and all(x < 0 for x in d_v2)
    switch_readout = bool(d_readout) and all(x > 0 for x in d_readout)
    out["gates"] = {"complete": full, "G1_beats_frozen_R18_every_pair": g1, "G2_mean_3fold": m3,
                    "G2_pass": g2, "G3_stop_v2_wins_every_pair": g3_stop,
                    "readout_switch_to_cls": switch_readout, "n_pairs": len(d_r18)}
    lines.append("")
    lines.append(f"GATES ({'complete' if full else 'PARTIAL -- not all 12 fold x seed pairs are in yet'}):")
    lines.append(f"  G1 V3 cls_mean > frozen R18 on every pair: "
                 f"{('PASS' if g1 else 'FAIL') if d_r18 else 'no pairs yet'} ({len(d_r18)} pairs)")
    lines.append(f"  G2 V3 cls_mean 3-fold mean >= 0.75: {'PASS' if g2 else 'FAIL'} "
                 f"({m3:.4f})" if m3 is not None else "  G2: not enough folds yet")
    lines.append(f"  G3 DINOv2 beats DINOv3 (cls_mean) on every pair -> stop and ask: {'YES -- STOP' if g3_stop else 'no'}")
    lines.append(f"  readout lever: CLS-only beats cls_mean on every pair -> switch: {'YES' if switch_readout else 'no'}")
    out["text"] = "\n".join(lines)
    return out


# ---------------------------------------------------------------------------------------- archive

def archive_cell(out: Path, which: str) -> None:
    """One cell's 12 runs -> experiments/ through coffeecv's archive contract (plan §5.7): the primary
    cell as exp220-231, the selected one as exp240-251. photo_metrics.json, which the archive contract
    does not know about, is copied in beside the rest; the photo-level score is the /classify number."""
    (backbone, readout), first, stem = ARCHIVE_SETS[which]
    results = load_results(out)
    todo = []
    for seed in SEEDS:
        for heldout in RIGS:
            cell = results["cells"].get(cell_key(seed, heldout, backbone, readout))
            if cell is None:
                raise SystemExit(f"missing s{seed}/{rig_name(heldout)} for the {which} cell -- archive all 12 or none")
            if "photo" not in cell:
                raise SystemExit(f"s{seed}/{rig_name(heldout)} of the {which} cell has no photo-level score; "
                                 f"run it with --photo-pool {backbone}:{readout} first")
            todo.append((seed, heldout, out / cell["run_dir"]))
    for seed, heldout, run_dir in todo:
        exp = exp_id_for(seed, heldout, first)
        slug = f"{stem}_frozen_{readout.replace('_', '')}_s{seed}_heldout_{rig_name(heldout)}"
        pair = f"; paired with exp{BASELINE_EXPS[rig_name(heldout)]}" if seed == 42 else ""
        chosen = "" if which == "primary" else ", the owner's selected backbone (2026-09-26)"
        note = (f"Experiment 1 (docs/dinov3_integration_plan.md §5): frozen {backbone}{chosen}, readout {readout}, "
                f"L2 logistic-regression head (C on val), no MixStyle, no TTA{pair}")
        archive(str(exp), slug, note, src_dir=run_dir)
        exp_dir = next((REPO_ROOT / "experiments").glob(f"exp{exp}__*"))
        (exp_dir / "photo_metrics.json").write_bytes((run_dir / "photo_metrics.json").read_bytes())


# ---------------------------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--backbones", nargs="+", default=["resnet18", "dinov2_vits14", "dinov3_vits16"],
                   choices=sorted(SPECS))
    p.add_argument("--folds", nargs="+", default=None, help="held-out rigs to run (names, e.g. cam_iphone)")
    p.add_argument("--photo-pool", nargs="*", default=[f"{PRIMARY[0]}:{PRIMARY[1]}"],
                   help="backbone:readout cells to also score photo by photo")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--smoke", action="store_true", help="tiny patch budgets, for testing the pipeline")
    p.add_argument("--allow-dirty", action="store_true", help="skip the provenance checks (smoke only)")
    p.add_argument("--summary", action="store_true", help="print tables and gates from results.json and exit")
    p.add_argument("--archive", nargs="?", const="primary", choices=sorted(ARCHIVE_SETS),
                   help="archive a cell's 12 runs and exit: primary -> exp220-231, selected -> exp240-251")
    args = p.parse_args()
    out = Path(args.out)

    if args.summary:
        print(summarize(load_results(out))["text"])
        return 0
    if args.archive:
        archive_cell(out, args.archive)
        return 0

    if not args.allow_dirty:
        dirty = dirty_provenance_paths()
        if dirty:
            raise SystemExit("uncommitted source would make this run unreproducible:\n  " + "\n  ".join(dirty))
        stale = stale_crop_stages(RIGS)
        if stale:
            raise SystemExit(f"stale upstream data stages {stale}: the crops on disk are not the tracked ones")

    photo_pool = {tuple(s.split(":", 1)) for s in args.photo_pool}
    base_cfg = RunConfig.from_params_yaml()
    folds = [r for r in RIGS if args.folds is None or rig_name(r) in args.folds]
    results = load_results(out)
    results["meta"].setdefault("created_at", datetime.now(timezone.utc).isoformat())
    results["meta"]["last_invocation"] = {"argv": sys.argv, "host": socket.gethostname(), "cpu": _cpu_model(),
                                          "torch_threads": torch.get_num_threads(),
                                          "git_commit": build_env_block()["git_commit"],
                                          "started_at": datetime.now(timezone.utc).isoformat()}
    log(f"host {socket.gethostname()} ({_cpu_model()}), torch threads {torch.get_num_threads()}, "
        f"seeds {args.seeds}, folds {[rig_name(r) for r in folds]}, backbones {args.backbones}, "
        f"photo-pool {sorted(photo_pool)}, out {out}{'  [SMOKE]' if args.smoke else ''}")

    loaded: dict[str, FrozenBackbone] = {}
    for seed in args.seeds:
        for heldout in folds:
            todo = [b for b in args.backbones if not backbone_done(results, seed, heldout, b, photo_pool)]
            if not todo:
                log(f"s{seed}/{rig_name(heldout)}: already done, skipping")
                continue
            for b in todo:
                if b not in loaded:
                    loaded[b] = build_backbone(b)
                    assert_input_size(loaded[b], base_cfg.patch_resize)
                    log(f"loaded {b}: weights sha256 {loaded[b].weights_sha256[:16]}..., readouts {loaded[b].readouts}")
            t = time.perf_counter()
            run_fold(seed, heldout, {b: loaded[b] for b in todo}, photo_pool, base_cfg, out, results, args.smoke)
            log(f"s{seed}/{rig_name(heldout)}: fold done in {(time.perf_counter() - t) / 60:.1f} min")
    summary = summarize(results)
    results["summary"] = {k: v for k, v in summary.items() if k != "text"}
    save_results(out, results)
    print(summary["text"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
