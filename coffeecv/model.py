"""Pretrained backbone + replaced classifier head for the coffee bean classes."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class MixStyle(nn.Module):
    """Style mixing (Zhou et al., "Domain Generalization with MixStyle"). Mixes
    each sample's per-channel spatial mean/std with another sample from the same
    batch, at a random point between the two drawn from Beta(alpha, alpha). No
    learnable parameters -- purely a statistics swap, so it needs no optimizer
    changes.

    Domain-agnostic (v1): the partner is a uniformly random permutation of the
    batch, regardless of which capture dir each sample came from -- nothing about
    a sample's origin reaches the forward pass. Screened and adopted in Phase 16
    (mean +0.1400 cross-camera, 9/9 sign-consistent). A cross-domain variant (v2,
    partner restricted to a different camera) was screened as a confirmed null and
    removed with the rest of the camera-aware code (ticket ML-1, 2026-09-29).
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


SUPPORTED_MODELS = ("resnet18",)
# Frozen backbone + convex head (plan §8.2). Fitted by coffeecv.fit_frozen_head and loaded by
# coffeecv.infer.load_model -- never built or trained by train_baseline's SGD loop.
FROZEN_MODELS = ("dinov3_vitb16",)


def build_model(
    name: str, num_classes: int, freeze_mode: str, dropout: float = 0.2,
    mixstyle_p: float = 0.0, mixstyle_alpha: float = 0.1, mixstyle_mode: str = "agnostic",
    pretrained: bool = True,
) -> tuple[nn.Module, nn.Module]:
    """Returns (model, head_module). head_module is used by the caller to give
    the head its own (higher) learning rate, separate from any unfrozen backbone.

    `mixstyle_p > 0` is scoped to resnet18 only for this first screen (see
    `_install_mixstyle`) -- matches how `freeze_mode`'s `last_block` already
    special-cases per architecture. Raises rather than silently ignoring the
    knob on an architecture it isn't wired for, consistent with this project's
    fail-loudly-on-config-mismatch convention (RunConfig.from_params_yaml,
    run_all_rigs.py's CLI/config post-condition check).

    `pretrained=False` skips the ImageNet initialisation. Inference passes it: a checkpoint's strict
    load_state_dict overwrites every parameter and buffer anyway, and the served process runs with
    ProtectHome=true, where ~/.cache/torch is not visible (docs/ops1_release_isolation_plan.html §4.3).
    Training keeps the default."""
    # The camera-aware v2 mode was removed with ticket ML-1: it needed camera labels, and was a
    # confirmed null. An old config naming it must fail here rather than quietly train the agnostic one.
    if mixstyle_mode != "agnostic":
        raise ValueError(f"Unknown mixstyle_mode: {mixstyle_mode!r} (only 'agnostic' exists; the "
                         f"camera-aware v2 mode was removed on 2026-09-29, ticket ML-1)")
    if mixstyle_p > 0 and name != "resnet18":
        raise ValueError(
            f"mixstyle_p={mixstyle_p} was requested but MixStyle is only wired for resnet18, "
            f"not {name!r}. Set mixstyle_p=0.0 or switch model_name to resnet18."
        )

    if name in FROZEN_MODELS:
        # Depth 0 is a convex fit (plan §8.2): two ways of producing the same head is how arms stop being
        # comparable, so the SGD loop refuses it instead of quietly training a frozen network by SGD.
        raise ValueError(
            f"model_name={name!r} is a frozen backbone with a fitted linear head: fit it with "
            f"`python -m coffeecv.fit_frozen_head`, load it with coffeecv.infer.load_model. train_baseline "
            f"does not train it.")
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        _apply_freeze_mode(model, freeze_mode, last_block=model.layer4)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        head_module = model.fc
        if mixstyle_p > 0:
            _install_mixstyle(model, mixstyle_p, mixstyle_alpha)
    else:
        # An old checkpoint card restores its model_name faithfully (config_for_checkpoint), so a pruned
        # architecture must fail here, never fall back to a substitute.
        raise ValueError(
            f"Unknown model_name: {name!r}. Supported: {SUPPORTED_MODELS + FROZEN_MODELS}. mobilenet_v3_small and "
            f"efficientnet_b0 were removed on 2026-09-26 (EXPERIMENTS_LOG exp3/exp29/exp31: rejected "
            f"three times, used by none of the 166 archived experiments); to re-run an old experiment, check "
            f"out its own git_commit.")

    return model, head_module
