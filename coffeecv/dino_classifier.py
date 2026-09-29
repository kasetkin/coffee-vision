"""Backbone -> readout -> linear head as ONE module whose forward returns logits, and its checkpoint.

That is the whole contract `coffeecv.infer.forward_with_embeddings` needs, so the OOD tools and the
web service score a DINO model with no changes on their side. The head is a real registered submodule that
`forward` calls, so the embedding pre-hook captures the readout vector -- the space the OOD guard lives
in (plan §9.2). Never hand out a slice or an inner layer of it as the head: a slice is a new wrapper
forward() never calls, and the hook would capture nothing (the old mobilenet bug).

**A checkpoint is the head plus the backbone's identity, never the backbone** (plan §8.2). At depth 0
the backbone never changes, so writing its 343 MB into every checkpoint would only scatter copies; it is
loaded from `models_pretrained/` (sha256 checked against the manifest by `build_backbone`) and then
checked against the digest the head was fitted on. A head in front of other backbone weights, another
readout or another timm build is not a smaller error, it is a different model -- so each is refused.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from coffeecv.backbones import SPECS, FrozenBackbone, build_backbone

CHECKPOINT_FORMAT = "coffeecv.frozen_head/1"


class DinoClassifier(nn.Module):
    def __init__(self, backbone: FrozenBackbone, readout: str, head: nn.Linear):
        super().__init__()
        if readout not in backbone.readouts:
            raise ValueError(f"{backbone.name} has readouts {backbone.readouts}, not {readout!r}")
        self.backbone = backbone
        self.readout = readout
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.features(x)[self.readout])


def is_frozen_model(model_name: str) -> bool:
    """True for a model_name that is a frozen backbone + fitted head rather than a fine-tuned network."""
    return model_name in SPECS and SPECS[model_name].family == "dinov3"


def _timm_version() -> str:
    import timm

    return timm.__version__


def save_frozen_checkpoint(path: Path, backbone: FrozenBackbone, readout: str, head: nn.Linear,
                           class_ids: list[str], C: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "backbone": backbone.name,
        "weights": backbone.spec.weights,
        "weights_sha256": backbone.weights_sha256,
        "timm_version": _timm_version(),
        "readout": readout,
        "class_ids": list(class_ids),
        "C": float(C),
        "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
    }, path)


def load_frozen_checkpoint(path: Path, model_name: str, num_classes: int) -> DinoClassifier:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or ck.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint (format={ck.get('format') if isinstance(ck, dict) else type(ck).__name__})")
    if ck["backbone"] != model_name:
        raise ValueError(f"{path} holds a head for {ck['backbone']!r}, but its config says model_name={model_name!r}")
    if ck["timm_version"] != _timm_version():
        raise ValueError(f"{path} was fitted under timm {ck['timm_version']} but timm {_timm_version()} is installed; "
                         f"a timm upgrade is an architecture change -- re-pass tests/test_dino_backbone.py and refit")
    backbone = build_backbone(model_name)           # refuses weights that do not match the manifest
    if backbone.weights_sha256 != ck["weights_sha256"]:
        raise ValueError(f"{path} was fitted on backbone weights {ck['weights_sha256'][:16]}..., but "
                         f"models_pretrained/{backbone.spec.weights} is {backbone.weights_sha256[:16]}...")
    w = ck["head"]["weight"]
    if w.shape[0] != num_classes:
        raise ValueError(f"{path}: head has {w.shape[0]} classes, the class list has {num_classes}")
    head = nn.Linear(w.shape[1], w.shape[0])
    head.load_state_dict(ck["head"], strict=True)
    return DinoClassifier(backbone, ck["readout"], head).eval()
