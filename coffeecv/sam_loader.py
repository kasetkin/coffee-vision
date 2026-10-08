"""Build EfficientViT-SAM from the vendored upstream copy (ticket ML-2, plan §2.1), at the variant its
weights file names (ticket ML-5 P4: L0 at 512 px, XL0 at 1024 px).

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
# Variants with pretrained weights in models_pretrained/ (L*: 512 px encoder input, XL*: 1024 px).
VARIANTS = ("l0", "l1", "l2", "xl0", "xl1")
_PREFIX, _SUFFIX = "efficientvit_sam/efficientvit_sam_", ".pt"


def weights_for(variant: str) -> str:
    return f"{_PREFIX}{variant}{_SUFFIX}"


def variant_of(weights: str) -> str:
    """The variant a weights path under models_pretrained/ holds, read from its name (weights_for's inverse).
    One source for both, so a config cannot name XL0 weights and build L0 around them."""
    v = weights.removeprefix(_PREFIX).removesuffix(_SUFFIX)
    if not (weights.startswith(_PREFIX) and weights.endswith(_SUFFIX)) or v not in VARIANTS:
        raise ValueError(f"{weights!r} names no EfficientViT-SAM variant; expected {_PREFIX}<variant>{_SUFFIX} "
                         f"with a variant in {VARIANTS}")
    return v


def _import_efficientvit_sam():
    if str(EFFICIENTVIT_DIR) not in sys.path:
        sys.path.insert(0, str(EFFICIENTVIT_DIR))
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    return create_efficientvit_sam_model, EfficientViTSamPredictor


def build_sam(weights: str):
    """(EfficientViTSamPredictor in eval mode with frozen weights, weights sha256), at variant_of(weights).

    `weights` is a path under models_pretrained/ listed in its manifest.json. The load is strict,
    so a file whose content is another variant's fails here rather than half-loading. The encoder's input
    side is model.image_size[1] (512 for L*, 1024 for XL*); prompts are in model.image_size[0], 1024 for all.
    """
    digest = verify_weights(weights)
    create, Predictor = _import_efficientvit_sam()
    model = create(f"efficientvit-sam-{variant_of(weights)}", pretrained=False)
    sd = torch.load(MODELS_PRETRAINED / weights, map_location="cpu", weights_only=True)
    model.load_state_dict(sd.get("state_dict", sd), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Predictor(model), digest
