"""Frozen backbones: build them, verify their weights, and read their features out.

DINOv3 builds through **timm**, not facebookresearch/dinov3 (docs/dinov3_integration_plan.md §2).
timm's code is Apache-2.0 and pins with a hash, while Meta's is under the DINOv3 License and cannot
pass the webapp's `--require-hashes` install. With one correction -- copying the checkpoint's
bfloat16 RoPE periods over timm's fp32 recompute -- timm's output is bit-identical to Meta's
reference on one machine (verified at 6876159, 224 and 256 px); tests/test_dino_backbone.py pins it.

DINOv2 is a reference arm only, built exactly as the 2026-09-21 screen built it (torch.hub from the
cached hub checkout), so its numbers stay continuous with that screen. ResNet18 is the frozen control.

`features(x)` returns every readout a backbone offers from ONE forward pass, so a screen that scores
two readouts still embeds each patch once.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from coffeecv.config import REPO_ROOT

MODELS_PRETRAINED = REPO_ROOT / "models_pretrained"
DINOV2_HUB_DIR = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"


@dataclass(frozen=True)
class BackboneSpec:
    family: str               # "resnet18" | "dinov2" | "dinov3"
    weights: str              # path under models_pretrained/, as listed in its manifest.json
    timm_name: str | None = None


SPECS: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec("resnet18", "resnet18/resnet18-f37072fd.pth"),
    "dinov2_vits14": BackboneSpec("dinov2", "dinov2/dinov2_vits14_pretrain.pth"),
    "dinov3_vits16": BackboneSpec("dinov3", "dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                                  "vit_small_patch16_dinov3"),
    "dinov3_vits16plus": BackboneSpec("dinov3", "dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                                      "vit_small_plus_patch16_dinov3"),
    "dinov3_vitb16": BackboneSpec("dinov3", "dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                                  "vit_base_patch16_dinov3"),
}

# "cls" is what the model returns upstream; "cls_mean" (CLS concatenated with the mean of the patch
# tokens) is the input Meta's own linear-classification eval feeds its heads (dinov3/eval/linear.py).
READOUTS = {"resnet18": ("avgpool",), "dinov2": ("cls", "cls_mean"), "dinov3": ("cls", "cls_mean")}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_weights(rel: str) -> str:
    """sha256 of models_pretrained/<rel>, refusing any file that does not match manifest.json.

    A run that cannot say which weights it loaded cannot be paired against one that loaded another
    copy, so the digest goes into every run's config.json.
    """
    manifest = {e["path"]: e for e in json.loads((MODELS_PRETRAINED / "manifest.json").read_text())}
    path = MODELS_PRETRAINED / rel
    if rel not in manifest:
        raise ValueError(f"{rel} is not in models_pretrained/manifest.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing -- see models_pretrained/README.md for how to obtain it")
    digest = sha256_of(path)
    if digest != manifest[rel]["sha256"]:
        raise ValueError(f"{path}: sha256 {digest[:16]}... does not match the manifest's "
                         f"{manifest[rel]['sha256'][:16]}... -- run models_pretrained/verify.py")
    return digest


def _build_dinov3_timm(timm_name: str, sd: dict) -> nn.Module:
    import timm
    from timm.models.eva import checkpoint_filter_fn

    # timm's no-bias variants drop qkv.bias on load. That is lossless only because Meta's distilled
    # checkpoints carry all-zero biases (checked for S/16, S+/16, B/16) -- so check, don't assume.
    nonzero = [k for k, v in sd.items() if k.endswith("qkv.bias") and v.any()]
    if nonzero:
        raise ValueError(f"{timm_name}: non-zero qkv biases in the checkpoint ({nonzero[:2]}...); "
                         f"the no-bias timm variant would silently drop them")
    model = timm.create_model(timm_name, pretrained=False, num_classes=0,
                              global_pool="token")      # upstream's readout; timm's default is "avg"
    with contextlib.redirect_stdout(io.StringIO()):     # the filter prints one line per dropped bias
        converted = checkpoint_filter_fn(dict(sd), model)
    model.load_state_dict(converted, strict=True)       # a partial load is plausible garbage for a whole sweep
    # timm recomputes the RoPE periods in fp32; the checkpoint holds the bf16-rounded values the model
    # was trained with, and Meta's reference loads those. Copying them in makes the two bit-identical
    # (plan §2.1). It works only because dynamic_img_size=True leaves no cached sin/cos table -- a
    # timm release that starts caching must fail here rather than silently ignore the copy.
    if model.rope.feat_shape is not None:
        raise RuntimeError(f"{timm_name}: timm caches a RoPE table (feat_shape={model.rope.feat_shape}); "
                           f"copying the checkpoint's periods would not take effect")
    with torch.no_grad():
        model.rope.periods.copy_(sd["rope_embed.periods"].float())
    return model


class FrozenBackbone(nn.Module):
    """A pretrained backbone with every parameter frozen, in eval mode, exposing its readouts."""

    def __init__(self, name: str, model: nn.Module, weights_sha256: str):
        super().__init__()
        self.name = name
        self.spec = SPECS[name]
        self.model = model.eval()
        self.weights_sha256 = weights_sha256
        self.readouts = READOUTS[self.spec.family]
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # Frozen means frozen: no caller can flip it into train mode (dropout, RoPE augmentation).
        return super().train(False)

    @property
    def patch_size(self) -> int | None:
        if self.spec.family == "dinov2":
            return int(self.model.patch_size)
        if self.spec.family == "dinov3":
            return int(self.model.patch_embed.patch_size[0])
        return None

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.spec.family == "resnet18":
            return {"avgpool": self.model(x)}
        if self.spec.family == "dinov2":
            out = self.model.forward_features(x)
            cls, patches = out["x_norm_clstoken"], out["x_norm_patchtokens"]
        else:
            tokens = self.model.forward_features(x)      # [B, 1 + 4 registers + N, D], final norm applied
            cls, patches = tokens[:, 0], tokens[:, self.model.num_prefix_tokens:]
        return {"cls": cls, "cls_mean": torch.cat([cls, patches.mean(dim=1)], dim=1)}


def build_backbone(name: str) -> FrozenBackbone:
    spec = SPECS[name]
    digest = verify_weights(spec.weights)
    sd = torch.load(MODELS_PRETRAINED / spec.weights, map_location="cpu", weights_only=True)
    if spec.family == "resnet18":
        import torchvision

        model = torchvision.models.resnet18(weights=None)
        model.load_state_dict(sd, strict=True)
        model.fc = nn.Identity()
    elif spec.family == "dinov2":
        if not (DINOV2_HUB_DIR / "hubconf.py").exists():
            raise FileNotFoundError(
                f"{DINOV2_HUB_DIR} is missing; the DINOv2 reference arm builds from that hub checkout "
                f"(copy it from a machine that has it -- it is Apache-2.0 code)")
        model = torch.hub.load(str(DINOV2_HUB_DIR), "dinov2_vits14", source="local", pretrained=False)
        model.load_state_dict(sd, strict=True)
    else:
        model = _build_dinov3_timm(spec.timm_name, sd)
    return FrozenBackbone(name, model, digest)


def assert_input_size(backbone: FrozenBackbone, patch_resize: int) -> None:
    """Refuse a side that is not a whole number of patches, at config time rather than at the first
    forward pass. DINOv2 would raise there anyway, timm too, but Meta's DINOv3 code silently drops the
    remainder (plan §1.4) -- and either way the failure would come after every photo was decoded."""
    p = backbone.patch_size
    if p and patch_resize % p:
        lo, hi = patch_resize // p * p, (patch_resize // p + 1) * p
        raise ValueError(f"patch_resize={patch_resize} is not a multiple of {backbone.name}'s patch "
                         f"size {p}; nearest valid sizes: {lo} or {hi}")
