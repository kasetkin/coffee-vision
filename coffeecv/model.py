"""Pretrained backbone + replaced classifier head for the coffee bean classes."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def build_model(name: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model_name: {name!r}")

    return model
