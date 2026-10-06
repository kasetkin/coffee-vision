"""Frozen-backbone leave-one-rig-out probe.

Reads the tray heuristic's data/cropped pools, retired in ticket ML-3 P2b: a record of a finished experiment,
run at its own commit (or after `git checkout dbb4e62 && dvc checkout`).

Question: do features from a different backbone family transfer across CAMERAS
better than the ImageNet ResNet18 this project fine-tunes today?

This is a SCREEN, not an adoption test. It compares frozen features + a linear
classifier, which is not the pipeline (full fine-tune + MixStyle + TTA). What it
does give, cheaply and on CPU, is a paired comparison of backbone families on
the project's own metric (cross-rig macro-F1, leave-one-camera-out), with every
arm seeing byte-identical patches.

Deliberate simplifications, identical across arms so the comparison stays fair:
  - patches drawn with split="all" per rig (every photo), so training rigs
    contribute more photos than a real fold does. Held-out rig photos are never
    in training, which is the property that matters.
  - fixed logistic-regression C, no per-arm tuning.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from coffeecv.bean_scale import pitch_kwargs
from coffeecv.config import RunConfig
from coffeecv.dataset import MultiPhotoPatchDataset, discover_classes_multi, resolve_rigs
from coffeecv.transforms import build_eval_transform

torch.set_num_threads(6)

RIGS = ["cam_pixel", "cam_sony", "cam_oneplus", "cam_iphone"]
PATCHES_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEED = 42
OUT = Path(__file__).parent / "xrig_probe2_results.json"


def build_backbones():
    """Arms chosen to SEPARATE the three things DINOv2's lead could be:
      (a) pretraining recipe  -> resnet18 V1 vs resnet18.a1_in1k (same arch, same cost)
      (b) ViT architecture    -> vit_small supervised, ImageNet only
      (c) DINO self-supervision + LVD-142M -> dinov2_vits14 (carried over as the anchor)
    """
    import timm
    import torchvision.models as models

    def tv_r18():
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        return m

    def timm_r18_a1():
        return timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=0)

    def timm_vit_s():
        return timm.create_model("vit_small_patch16_224.augreg_in1k", pretrained=True, num_classes=0)

    def dinov2():
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)

    return {
        "resnet18 tv-V1 (control)": tv_r18,
        "resnet18 a1_in1k (modern recipe)": timm_r18_a1,
        "vit_small supervised in1k": timm_vit_s,
        "dinov2_vits14 (SSL, anchor)": dinov2,
    }


def patches_for_rig(rig_name: str, cfg: RunConfig, class_ids: list[str]) -> tuple[torch.Tensor, np.ndarray]:
    rigs = resolve_rigs([Path("data/cropped") / rig_name])
    ds = MultiPhotoPatchDataset(
        rigs=rigs,
        classes_file=Path(cfg.classes_file),
        split="all",
        class_ids=class_ids,
        seed=SEED,
        crop_size=cfg.patch_crop_size,
        resize=cfg.patch_resize,
        safety_margin=cfg.safety_margin,
        patches_per_class={"all": PATCHES_PER_CLASS},
        photo_frac={"train": cfg.train_photo_frac, "val": cfg.val_photo_frac, "test": cfg.test_photo_frac},
        transform=build_eval_transform(cfg.patch_resize),
        patch_store_size=cfg.patch_store_size,
        patch_beans=(cfg.patch_beans_min, cfg.patch_beans_max),
        pitch_geometry=pitch_kwargs(cfg),
    )
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), np.array(ys)


@torch.no_grad()
def embed(model, tensors: torch.Tensor, bs: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(tensors), bs):
        out.append(model(tensors[i:i + bs]).float().numpy())
    return np.concatenate(out)


def main():
    import os
    os.chdir(_REPO)  # rig paths below are repo-relative
    cfg = RunConfig.from_params_yaml()
    class_ids = sorted({c for r in RIGS for c in discover_classes_multi(Path("data/cropped") / r)})
    print(f"classes: {class_ids}", flush=True)
    print(f"patches/class/rig: {PATCHES_PER_CLASS}  (project xrig budget = {cfg.xrig_patches_per_class})", flush=True)

    backbones = {name: ctor() for name, ctor in build_backbones().items()}
    for m in backbones.values():
        m.eval()

    # Embed rig by rig so only one rig's patches are ever resident.
    emb: dict[str, dict[str, np.ndarray]] = {name: {} for name in backbones}
    labels: dict[str, np.ndarray] = {}
    for rig in RIGS:
        t = time.perf_counter()
        px, labels[rig] = patches_for_rig(rig, cfg, class_ids)
        print(f"[patches] {rig}: {tuple(px.shape)} in {time.perf_counter()-t:.0f}s", flush=True)
        for name, model in backbones.items():
            t = time.perf_counter()
            emb[name][rig] = embed(model, px)
            print(f"    [embed] {name:34s} dim={emb[name][rig].shape[1]:4d} "
                  f"{time.perf_counter()-t:.0f}s", flush=True)
        del px

    results = {}
    for name in backbones:
        e = emb[name]
        folds = {}
        for heldout in RIGS:
            tr = [r for r in RIGS if r != heldout]
            Xtr = np.concatenate([e[r] for r in tr])
            ytr = np.concatenate([labels[r] for r in tr])
            Xte, yte = e[heldout], labels[heldout]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
            clf.fit(sc.transform(Xtr), ytr)
            pred = clf.predict(sc.transform(Xte))
            f1 = f1_score(yte, pred, average="macro")
            mcc = matthews_corrcoef(yte, pred)
            id_f1 = f1_score(ytr, clf.predict(sc.transform(Xtr)), average="macro")
            folds[heldout] = {"xrig_macro_f1": round(float(f1), 4),
                              "xrig_mcc": round(float(mcc), 4),
                              "train_fit_macro_f1": round(float(id_f1), 4)}
            print(f"  {name:34s} heldout={heldout:12s} xrig_macro_f1={f1:.4f} mcc={mcc:.4f}", flush=True)
        mean = float(np.mean([v["xrig_macro_f1"] for v in folds.values()]))
        results[name] = {"dim": int(e[RIGS[0]].shape[1]), "folds": folds, "xrig_mean": round(mean, 4)}
        print(f"  ==> {name}: mean xrig macro-F1 = {mean:.4f}\n", flush=True)
        OUT.write_text(json.dumps({"patches_per_class": PATCHES_PER_CLASS, "seed": SEED,
                                   "results": results}, indent=2))

    print("\n=== SUMMARY (mean cross-rig macro-F1, frozen features + linear probe) ===")
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["xrig_mean"]):
        per = "  ".join(f"{k.replace('cam_','')}={v['xrig_macro_f1']:.3f}" for k, v in r["folds"].items())
        print(f"  {n:34s} dim={r['dim']:4d}  mean={r['xrig_mean']:.4f}   {per}")


if __name__ == "__main__":
    main()
