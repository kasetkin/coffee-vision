"""Build EfficientViT-SAM-L0 from the vendored upstream copy (ticket ML-2, plan §2.1).

The only module that imports `efficientvit`. Upstream is vendored unmodified apart from the import
patches listed in third_party/efficientvit/PATCHES.md (no triton, onnx or omegaconf on CPU), and it
is not installed as a package: this module puts third_party/efficientvit on sys.path itself.

Weights come from models_pretrained/ and are refused unless their sha256 matches manifest.json,
the same check the DINOv3 backbone gets. Never the model zoo's default path: that is relative to
the working directory (assets/checkpoints/...), so it would load whatever file sits there.
"""
from __future__ import annotations

import sys

import torch

from coffeecv.backbones import MODELS_PRETRAINED, verify_weights
from coffeecv.config import REPO_ROOT

EFFICIENTVIT_DIR = REPO_ROOT / "third_party" / "efficientvit"
L0_WEIGHTS = "efficientvit_sam/efficientvit_sam_l0.pt"   # under models_pretrained/, in its manifest
# Variants with pretrained weights in models_pretrained/ (L*: 512 px input, XL*: 1024 px);
# L0 is the ticket's choice.
VARIANTS = ("l0", "l1", "l2", "xl0", "xl1")


def weights_for(variant: str) -> str:
    return f"efficientvit_sam/efficientvit_sam_{variant}.pt"


def _import_efficientvit_sam():
    if str(EFFICIENTVIT_DIR) not in sys.path:
        sys.path.insert(0, str(EFFICIENTVIT_DIR))
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    return create_efficientvit_sam_model, EfficientViTSamPredictor


def build_sam_l0(weights: str = L0_WEIGHTS, variant: str = "l0"):
    """(EfficientViTSamPredictor around L0 (or `variant`) in eval mode with frozen weights, weights sha256).

    `weights` is a path under models_pretrained/ listed in its manifest.json. The load is strict,
    so a checkpoint for a different variant fails here rather than half-loading.
    """
    digest = verify_weights(weights)
    create, Predictor = _import_efficientvit_sam()
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {VARIANTS}")
    model = create(f"efficientvit-sam-{variant}", pretrained=False)
    sd = torch.load(MODELS_PRETRAINED / weights, map_location="cpu", weights_only=True)
    model.load_state_dict(sd.get("state_dict", sd), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Predictor(model), digest
