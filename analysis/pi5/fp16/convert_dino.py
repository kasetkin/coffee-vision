"""OPS-23 fp16 experiment: DINOv3 b8 graphs for XNNPACK FORCE_FP16 on the Pi.
prescaled : release forms (litert_gpu_forms + fp16_forms) + readout mean as sum(x * 1/N)
split<S>  : prescaled + every Linear over more than S inputs as a pairwise sum of S-wide slices, and attn @ v
            split over the tokens the same way (same function in real arithmetic)."""

import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch

sys.path.insert(0, "/workspace")
from coffeecv import litert_forms as LF
from coffeecv.backbones import build_backbone


class PreScaledReadout(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.backbone = readout.backbone

    def forward(self, x):
        m = self.backbone.model
        t = m.forward_features(x)
        p = t[:, m.num_prefix_tokens :]
        return torch.cat([t[:, 0], (p * (1.0 / p.shape[1])).sum(dim=1)], dim=1)


class SplitSumLinear(nn.Module):
    def __init__(self, lin: nn.Linear, s: int):
        super().__init__()
        self.bounds = [
            (a, min(a + s, lin.in_features)) for a in range(0, lin.in_features, s)
        ]
        w = lin.weight.detach()
        self.weights = nn.ParameterList(
            [
                nn.Parameter(w[:, a:b].clone(), requires_grad=False)
                for a, b in self.bounds
            ]
        )
        self.bias = (
            None
            if lin.bias is None
            else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        )

    def forward(self, x):
        y = LF._pairwise_sum(
            [F.linear(x[..., a:b], w) for (a, b), w in zip(self.bounds, self.weights)]
        )
        return y if self.bias is None else y + self.bias


def split_linears(model, s):
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and m.in_features > s:
            LF._set_submodule(model, name, SplitSumLinear(m, s))
            n += 1
    return n


def attention_4d_split(self, x, rope=None, attn_mask=None, is_causal=False):
    """LF._attention_4d (with score_shift) and attn @ v as a pairwise sum over token slices of SPLIT."""
    B, N, _ = x.shape
    q, k, v = (
        t.reshape(B, N, self.num_heads, -1).transpose(1, 2)
        for t in self.qkv(x).chunk(3, dim=-1)
    )
    q, k = self.q_norm(q), self.k_norm(k)
    from timm.layers import apply_rot_embed_cat

    npt = self.num_prefix_tokens
    half = getattr(self, "rotate_half", False)
    q = torch.cat(
        [q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)],
        dim=2,
    ).type_as(v)
    k = torch.cat(
        [k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)],
        dim=2,
    ).type_as(v)
    sh = self.score_shift
    s = torch.matmul(q * (self.scale * 2.0**-sh), k.transpose(-2, -1))
    attn = ((s - s.amax(dim=-1, keepdim=True)) * 2.0**sh).softmax(dim=-1)
    S = self.split
    x = LF._pairwise_sum(
        [
            torch.matmul(attn[..., a : a + S], v[:, :, a : a + S, :])
            for a in range(0, N, S)
        ]
    )
    x = x.transpose(1, 2).reshape(B, N, -1)
    return self.proj(self.norm(x))


# --- every mean pre-scaled: _layer_norm_scaled's two means as sum(x * 1/N) too (variant 'allpre')
def _layer_norm_scaled_pre(x, dim, weight, bias, eps):
    n = x.shape[dim]
    r = 1.0 / x.abs().amax(dim=dim, keepdim=True)
    y = x * r
    y = y - (y * (1.0 / n)).sum(dim=dim, keepdim=True)
    y = y * torch.rsqrt((y * y * (1.0 / n)).sum(dim=dim, keepdim=True) + eps * r * r)
    y = y if weight is None else y * weight
    return y if bias is None else y + bias


def _layer_norm_scaled_mean(x, dim, weight, bias, eps):
    """'lnmean': LayerNorm of x / max|x| through F.layer_norm (TFLite MEAN, XNNPACK's mean), its eps a constant
    1e-12 in place of eps * r^2 (not exact: eps is far below the variances; checked by D7 in fp32)."""
    if dim != -1:
        raise ValueError("lnmean covers the last dim only")
    y = x * (1.0 / x.abs().amax(dim=-1, keepdim=True))
    return F.layer_norm(y, (x.shape[-1],), weight, bias, 1e-12)


class PoolReadout(nn.Module):
    """[CLS] and the patch mean as an AVERAGE_POOL_2D over the 14 x 14 grid."""

    def __init__(self, readout):
        super().__init__()
        self.backbone = readout.backbone

    def forward(self, x):
        m = self.backbone.model
        t = m.forward_features(x)
        p = t[:, m.num_prefix_tokens :]
        g = p.transpose(1, 2).reshape(p.shape[0], p.shape[2], 14, 14)
        return torch.cat([t[:, 0], F.avg_pool2d(g, 14).flatten(1)], dim=1)


def build(variant):
    if variant.startswith("allpre"):
        LF._layer_norm_scaled = (
            _layer_norm_scaled_pre  # _layer_norm_fp16 looks it up at call time
        )
    if variant.startswith("lnmean"):
        LF._layer_norm_scaled = _layer_norm_scaled_mean
    bb = build_backbone("dinov3_vitb16")
    tm = bb.model.eval()
    LF.four_d_attention(tm)
    LF.constant_rope(tm, (14, 14))
    module = LF.DinoReadout(bb, "cls_mean").eval()
    print(LF.litert_gpu_forms(module, batch=8), LF.fp16_forms(module))
    if "split" in variant:
        s = int(variant.split("split")[1])
        from timm.models.eva import EvaAttention

        for m in tm.modules():
            if isinstance(m, EvaAttention):
                m.split = s
                m.forward = types.MethodType(attention_4d_split, m)
        print("split linears", split_linears(tm, s))
    return (
        PoolReadout(module)
        if variant.startswith("lnmean")
        else PreScaledReadout(module)
    ).eval()


if __name__ == "__main__":
    variant, out = sys.argv[1], sys.argv[2]
    import numpy as np

    model = build(variant)
    inp = np.load("/workspace/analysis/ml7_gpu/artifacts/inputs.npz")
    ref = np.load("/workspace/analysis/ml7_gpu/artifacts/refs.npz")
    x = torch.from_numpy(np.ascontiguousarray(inp["patches"][0, :8]))
    with torch.no_grad():
        y = model(x).double().numpy()
    r = ref["readouts"][0, :8].astype(np.float64)
    cos = (y * r).sum(1) / np.linalg.norm(y, axis=1) / np.linalg.norm(r, axis=1)
    print(f"{variant}: torch fp32 vs refs least cosine {cos.min():.9f}")
    litert_torch.convert(model, (torch.zeros(8, 3, 224, 224),)).export(out)
    print("exported", out)
