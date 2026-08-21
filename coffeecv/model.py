"""Pretrained backbone + replaced classifier head for the coffee bean classes."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class MixStyle(nn.Module):
    """Domain-agnostic style mixing (Zhou et al., "Domain Generalization with
    MixStyle"). Mixes each sample's per-channel spatial mean/std with a randomly
    permuted sample from the same batch, at a random point between the two drawn
    from Beta(alpha, alpha). No learnable parameters -- purely a statistics swap,
    so it needs no optimizer changes.

    v1 is domain-agnostic: it mixes any two samples in the batch regardless of
    which rig they came from, which is simpler to wire (no rig-id needs to reach
    the forward pass) but less targeted than a domain-aware variant that only
    mixes *across* rigs would be. Screen this version first; only build the
    domain-aware one if this shows a promising-but-not-decisive signal, per the
    project's own precedent for full-freeze -> last_block.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0 or torch.rand(1).item() > self.p:
            return x
        batch = x.size(0)
        if batch < 2:
            return x  # nothing to mix with
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sigma = (var + self.eps).sqrt()
        x_normed = (x - mu) / sigma

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch, 1, 1, 1)).to(x.device)
        perm = torch.randperm(batch, device=x.device)
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sigma_mix = sigma * lam + sigma[perm] * (1 - lam)
        return x_normed * sigma_mix + mu_mix


def _install_mixstyle(model: nn.Module, mixstyle_p: float, mixstyle_alpha: float) -> None:
    """Registers MixStyle after resnet18's layer1 and layer2 (the paper's
    recommended low/mid-level placement, where "style" statistics live) via
    forward hooks rather than a reimplemented forward pass. Registered as real
    submodules (`model.mixstyle1`/`mixstyle2`), not just hook closures, so
    `model.train()`/`.eval()` correctly cascades into their `.training` flag --
    a hook closure over a bare `MixStyle()` would never see that toggle."""
    model.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
    model.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
    model.layer1.register_forward_hook(lambda _m, _i, out: model.mixstyle1(out))
    model.layer2.register_forward_hook(lambda _m, _i, out: model.mixstyle2(out))


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


def build_model(
    name: str, num_classes: int, freeze_mode: str, dropout: float = 0.2,
    mixstyle_p: float = 0.0, mixstyle_alpha: float = 0.1,
) -> tuple[nn.Module, nn.Module]:
    """Returns (model, head_module). head_module is used by the caller to give
    the head its own (higher) learning rate, separate from any unfrozen backbone.

    `mixstyle_p > 0` is scoped to resnet18 only for this first screen (see
    `_install_mixstyle`) -- matches how `freeze_mode`'s `last_block` already
    special-cases per architecture. Raises rather than silently ignoring the
    knob on an architecture it isn't wired for, consistent with this project's
    fail-loudly-on-config-mismatch convention (RunConfig.from_params_yaml,
    run_folds.py's CLI/config post-condition check)."""
    if mixstyle_p > 0 and name != "resnet18":
        raise ValueError(
            f"mixstyle_p={mixstyle_p} was requested but MixStyle is only wired for resnet18, "
            f"not {name!r}. Set mixstyle_p=0.0 or switch model_name to resnet18."
        )

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
        if mixstyle_p > 0:
            _install_mixstyle(model, mixstyle_p, mixstyle_alpha)
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
