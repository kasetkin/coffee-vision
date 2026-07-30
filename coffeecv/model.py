"""Pretrained backbone + replaced classifier head for the coffee bean classes."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def _apply_freeze_mode(model: nn.Module, freeze_mode: str, last_block: nn.Module) -> None:
    if freeze_mode == "none":
        return  # everything trainable (default requires_grad=True from torchvision)
    for param in model.parameters():
        param.requires_grad = False
    if freeze_mode == "last_block":
        for param in last_block.parameters():
            param.requires_grad = True
    elif freeze_mode != "full":
        raise ValueError(f"Unknown freeze_mode: {freeze_mode!r}")


def build_model(name: str, num_classes: int, freeze_mode: str, dropout: float = 0.2) -> tuple[nn.Module, nn.Module]:
    """Returns (model, head_module). head_module is used by the caller to give
    the head its own (higher) learning rate, separate from any unfrozen backbone."""
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        _apply_freeze_mode(model, freeze_mode, last_block=model.features[-1])
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
        model.classifier[3] = nn.Linear(in_features, num_classes)
        head_module = model.classifier[2:4]
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        _apply_freeze_mode(model, freeze_mode, last_block=model.layer4)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        head_module = model.fc
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        _apply_freeze_mode(model, freeze_mode, last_block=model.features[-1])
        in_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head_module = model.classifier
    else:
        raise ValueError(f"Unknown model_name: {name!r}")

    return model, head_module
