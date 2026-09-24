"""Backbone -> readout -> linear head as ONE module whose forward returns logits.

That is the whole contract `coffeecv.infer.forward_with_embeddings` and
`coffeecv.xrig_eval.run_photowise` need, so the photo-level path scores a DINO model with no changes
on either side. The head is a real registered submodule that `forward` calls, so the embedding
pre-hook captures the readout vector -- the space the OOD guard will live in (plan §9.2). Never hand
out a slice or an inner layer of it as the head: a slice is a new wrapper forward() never calls, and
the hook would capture nothing (the old mobilenet bug).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coffeecv_dino.backbone import FrozenBackbone


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
