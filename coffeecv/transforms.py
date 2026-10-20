"""The eval transform: a stored patch to the model's input tensor. Training's frozen head embeds every patch through
it (fit_frozen_head, fold_data), and serving runs the same model_io.eval_patch without torch.

The fine-tuned ResNet18's train transform (dihedral flips, ColorJitter, zoom, illumination ramp, random erasing)
lived here until 2026-10-20, when that path was removed; the frozen head's augmentation is params.yaml
head_augment (ADR 0025, ADR 0027).
"""
from __future__ import annotations

import torch

from coffeecv.model_io import eval_patch


class EvalTransform:
    """A patch to the model's input tensor: `model_io.eval_patch`, the port serving calls without torch (ticket
    ML-7 D5 (b)), so training's eval patches and serving's are one implementation. Bit-identical to the
    torchvision Resize, ToTensor and Normalize it replaced (tests/test_model_io.py)."""

    def __init__(self, resize: int):
        self.resize = resize

    def __call__(self, patch) -> torch.Tensor:
        return torch.from_numpy(eval_patch(patch, self.resize))


def build_eval_transform(resize: int) -> EvalTransform:
    return EvalTransform(resize)
