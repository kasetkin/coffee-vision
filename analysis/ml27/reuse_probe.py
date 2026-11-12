"""Ticket ML-27 D4 (b), the owner 2026-11-12: the candidate serves exp280's OOD probe, not a refit.

The probe reads the embedding a checkpoint feeds its head (`infer.forward_with_embeddings`): for a frozen DINOv3
head that is the backbone's readout, the same weights in both checkpoints (the cards' `dino.weights_sha256`), so
on the same patches the probe's scores, and serving's refusals, are the live model's. This checks that on real
photos first (the embeddings bit for bit, and the probe's score), then writes the live probe beside the candidate,
stamped with the candidate's sha (which `infer.load_ood_probe` requires) and a `reused_from` record, and loads it
back through `load_ood_probe`.

    python analysis/ml27/reuse_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coffeecv import model_io  # noqa: E402
from coffeecv.class_list import load_classes  # noqa: E402
from coffeecv.config import REPO_ROOT  # noqa: E402
from coffeecv.infer import (TorchClassifier, _sha, config_for_checkpoint, load_model, load_ood_probe,  # noqa: E402
                            patches_for_photo, probe_path_for, probe_score)

LIVE = REPO_ROOT / "models" / "dino3b16_segxl0v7r_country_aug8_s42.pt"
CANDIDATE = REPO_ROOT / "models" / "dino3b16_segxl0v7r_country_aug8_ml27_s42.pt"
PHOTOS = (  # a bean photo of each new session; the page's example photos (the cappuccino among them) are added
    "data/canonical/tasty_coffee/class_001__Ethiopia_Sidamo/20261110_113804_Pixel_9_Pro_Ethiopia_Sidamo.jpg.jxl",
    "data/canonical/massimo/class_017__Peru/20261107_101201_Pixel_9_Pro_Peru_Massimo.jpg.jxl",
)


def classifier(ckpt: Path, n_classes: int):
    cfg, _ = config_for_checkpoint(ckpt, None)
    model, head = load_model(ckpt, cfg.model_name, n_classes)
    return cfg, TorchClassifier(model, head)


def main() -> None:
    live_card = json.loads(LIVE.with_suffix(".json").read_text())["dino"]
    cand_card = json.loads(CANDIDATE.with_suffix(".json").read_text())["dino"]
    for k in ("backbone", "readout", "weights_sha256"):
        if live_card[k] != cand_card[k]:
            raise SystemExit(f"the backbones differ on {k}: the probe cannot be shared")
    n = len(load_classes(REPO_ROOT / "dataset" / "classes.txt"))
    cfg, live = classifier(LIVE, n)
    _, cand = classifier(CANDIDATE, n)
    probe = load_ood_probe(LIVE, embedding_dim=live.embedding_dim)
    photos = list(PHOTOS) + sorted(str(p.relative_to(REPO_ROOT)) for p in
                                   (REPO_ROOT / "webapp" / "static" / "examples").glob("*.jxl"))
    for rel in photos:
        patches, _ = patches_for_photo(REPO_ROOT / rel, cfg, 40, [42, 0])
        x = np.stack([model_io.eval_patch(p, cfg.patch_resize) for p in patches])
        _, e_live = live.probs_and_embeddings(x)
        _, e_cand = cand.probs_and_embeddings(x)
        if not np.array_equal(e_live, e_cand):
            raise SystemExit(f"{rel}: embeddings differ (max {np.abs(e_live - e_cand).max()}); refit the probe")
        print(f"{Path(rel).name}: {len(x)} patches, embeddings identical, probe {probe_score(e_live, probe):.4f}")

    out = json.loads(probe_path_for(LIVE).read_text())
    out["reused_from"] = {
        "checkpoint": out["checkpoint"], "checkpoint_sha": out["checkpoint_sha"],
        "why": ("ML-27 D4 (b), the owner 2026-11-12: the same frozen DINOv3 backbone, so the same embeddings and "
                "probe scores as the live model (analysis/ml27/reuse_probe.py); not refitted."),
    }
    out["checkpoint"] = str(CANDIDATE.relative_to(REPO_ROOT))
    out["checkpoint_sha"] = _sha(CANDIDATE)
    path = probe_path_for(CANDIDATE)
    path.write_text(json.dumps(out, indent=2))
    back = load_ood_probe(CANDIDATE, embedding_dim=cand.embedding_dim)
    print(f"wrote {path.relative_to(REPO_ROOT)}; load_ood_probe accepts it ({back['checkpoint_sha']})")


if __name__ == "__main__":
    main()
