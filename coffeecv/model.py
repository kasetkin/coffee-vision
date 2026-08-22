"""Pretrained backbone + replaced classifier head for the coffee bean classes."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


def _cross_domain_perm(domain_ids: torch.Tensor) -> torch.Tensor:
    """For each sample, returns a partner index drawn uniformly from same-batch
    samples with a *different* domain id. A sample whose domain is the only one
    present in the batch falls back to pairing with itself -- mathematically a
    no-op regardless of the mixing coefficient (mixing a value with itself
    reproduces it exactly), not a crash. O(B^2) via broadcasting, fine at this
    project's batch sizes (32)."""
    batch = domain_ids.size(0)
    diff = domain_ids.unsqueeze(0) != domain_ids.unsqueeze(1)  # [B, B], diff[i, j] = domain_ids[j] != domain_ids[i]
    # Random score per (i, j), masked to -inf where same-domain, so argmax picks
    # a uniformly random cross-domain partner for each row (ties broken by argmax's
    # own first-max convention, immaterial since scores are continuous).
    scores = torch.rand(batch, batch, device=domain_ids.device)
    scores = scores.masked_fill(~diff, float("-inf"))
    has_partner = diff.any(dim=1)
    perm = torch.arange(batch, device=domain_ids.device)
    perm[has_partner] = scores[has_partner].argmax(dim=1)
    return perm


class MixStyle(nn.Module):
    """Style mixing (Zhou et al., "Domain Generalization with MixStyle"). Mixes
    each sample's per-channel spatial mean/std with another sample from the same
    batch, at a random point between the two drawn from Beta(alpha, alpha). No
    learnable parameters -- purely a statistics swap, so it needs no optimizer
    changes.

    Two modes, selected by `cross_domain`:
    - domain-agnostic (v1, default): partner is a uniformly random permutation of
      the batch, regardless of which rig each sample came from. Simpler to wire
      (no rig-id needs to reach the forward pass). Screened and adopted first
      (Phase 16, mean +0.1400 cross-rig, 9/9 sign-consistent).
    - cross-domain (v2): partner is drawn only from samples of a *different* rig
      (`_cross_domain_perm`), via `self.domain_ids` -- set externally on this
      module right before each forward pass, since a forward hook only receives
      `(module, input, output)` and has no other way to learn which rig each
      sample in the batch came from.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, cross_domain: bool = False):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.cross_domain = cross_domain
        # Set externally (train_baseline.py's train_one_epoch) before every
        # forward call when cross_domain=True; unused otherwise.
        self.domain_ids: torch.Tensor | None = None

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
        if self.cross_domain:
            if self.domain_ids is None:
                raise RuntimeError(
                    "MixStyle(cross_domain=True) requires domain_ids to be set before forward(); "
                    "train_one_epoch must set model.mixstyle1.domain_ids/model.mixstyle2.domain_ids "
                    "each batch when mixstyle_mode='cross_rig'."
                )
            if self.domain_ids.size(0) != batch:
                raise RuntimeError(
                    f"domain_ids has {self.domain_ids.size(0)} entries but the batch has {batch} -- "
                    "stale domain_ids from a previous batch were not updated before this forward call."
                )
            perm = _cross_domain_perm(self.domain_ids)
        else:
            perm = torch.randperm(batch, device=x.device)
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sigma_mix = sigma * lam + sigma[perm] * (1 - lam)
        return x_normed * sigma_mix + mu_mix


def _install_mixstyle(model: nn.Module, mixstyle_p: float, mixstyle_alpha: float, cross_domain: bool) -> None:
    """Registers MixStyle after resnet18's layer1 and layer2 (the paper's
    recommended low/mid-level placement, where "style" statistics live) via
    forward hooks rather than a reimplemented forward pass. Registered as real
    submodules (`model.mixstyle1`/`mixstyle2`), not just hook closures, so
    `model.train()`/`.eval()` correctly cascades into their `.training` flag --
    a hook closure over a bare `MixStyle()` would never see that toggle."""
    model.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, cross_domain=cross_domain)
    model.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, cross_domain=cross_domain)
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
    mixstyle_p: float = 0.0, mixstyle_alpha: float = 0.1, mixstyle_mode: str = "agnostic",
) -> tuple[nn.Module, nn.Module]:
    """Returns (model, head_module). head_module is used by the caller to give
    the head its own (higher) learning rate, separate from any unfrozen backbone.

    `mixstyle_p > 0` is scoped to resnet18 only for this first screen (see
    `_install_mixstyle`) -- matches how `freeze_mode`'s `last_block` already
    special-cases per architecture. Raises rather than silently ignoring the
    knob on an architecture it isn't wired for, consistent with this project's
    fail-loudly-on-config-mismatch convention (RunConfig.from_params_yaml,
    run_folds.py's CLI/config post-condition check)."""
    if mixstyle_mode not in ("agnostic", "cross_rig"):
        raise ValueError(f"Unknown mixstyle_mode: {mixstyle_mode!r} (want 'agnostic' or 'cross_rig')")
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
            _install_mixstyle(model, mixstyle_p, mixstyle_alpha, cross_domain=(mixstyle_mode == "cross_rig"))
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
