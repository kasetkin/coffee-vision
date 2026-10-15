"""ML-7 P1: the graphs a runtime serves, cut where serving's own code cuts them, with the exact rewrites.

Three graphs from the shipped checkpoint (plan P1; ticket §4: the same models, not approximations):

  dino      DinoReadout: the patches, ImageNet-normalised [B, 3, 224, 224], to DINOv3's cls_mean readout
            [B, 1536]. The head and the OOD guard stay in numpy, so parity is judged on what they consume.
  encoder   SegEncoder: the transformed photo [1, 3, 1024, 1024] to the XL0 embedding [1, 256, 64, 64].
  decoder   SegDecoder: the embedding and the transformed whole-image box [1, 4] to the three low-resolution
            masks [1, 3, 256, 256] and their predicted IoUs [1, 3]. SAM's prompt encoder turns the box into two
            corner points labelled 2 and 3, as serving's predict_torch does; multi3 is selected outside, as now.

Exact rewrites, applied by `prepare` to both formats (plan P1, ticket D14):

  matmul_neck        SamNeck resizes each stage to a fixed 64 x 64, bicubic. At a fixed input size that is a
                     constant linear map, A_h X A_w^T, with matrices made by passing an identity through torch's
                     own F.interpolate, so torch's coefficient, coordinate mapping and edge clamping are kept.
  four_d_attention   timm's qkv reshape to [B, N, 3, H, D] (5-D, which LiteRT's GPU cannot hold) becomes three
                     4-D slices; attention is written out (q k^T, softmax, @ v), the pattern ORT's fusions
                     match and the fp16 copies keep in fp32 (plan P1).
  constant_rope      DINOv3's RoPE table depends on the patch grid only; at the fixed 224 px input it is computed
                     once, by timm's own function, and kept as a buffer.

Each rewritten graph is checked against the original on the fixtures (make_refs.py).

fp16_forms (plan P7), for LiteRT's GPU in fp16, which returned NaN for both models: forms whose every tensor fits
fp16, the same functions in real arithmetic. LiteMLA's relu(k) and relu(q) scaled by 2^-6 and 2^6 with a
denominator that is never 0/0; LayerNorm on x / max|x| per token; DINOv3's attention scores made at 2^-4 and shifted
by their row's max before the softmax. And, for fp16's precision rather than its range: each convolution over more
than 128 input channels as the pairwise sum of convolutions over slices of 128 (SplitSumConv2d), since the GPU
accumulates a convolution's products in fp16 and the encoder's 4,096- and 12,288-channel point_convs drifted its
masks (least IoU 0.967).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# The repo, or a release laid out like it (ML7_REPO; P4's torch cell runs the release's own code on yappy).
REPO = Path(os.environ.get("ML7_REPO") or Path(__file__).resolve().parents[2])
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coffeecv.sam_loader import EFFICIENTVIT_DIR  # noqa: E402

if str(EFFICIENTVIT_DIR) not in sys.path:
    sys.path.insert(0, str(EFFICIENTVIT_DIR))

PATCH = 224                 # coffeecv's patch_resize for the shipped DINOv3 model
FRAME = 1024                # the XL0 encoder's input side
DINO_BATCHES = (8, 40)      # LiteRT has no dynamic batch (litert-torch #870): 5 calls of 8, or 1 of 40


# ------------------------------------------------------------------------------------------- the bicubic neck
def bicubic_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """[n_out, n_in]: torch's 1-D bicubic resize (align_corners=False) of a length-n_in signal, as a matrix.
    Each column is the resize of a unit impulse, computed by F.interpolate itself."""
    eye = torch.eye(n_in, dtype=torch.float64).reshape(n_in, 1, n_in, 1)
    out = F.interpolate(eye, size=(n_out, 1), mode="bicubic", align_corners=False)
    return out.reshape(n_in, n_out).T.contiguous().float()


class MatmulResize(nn.Module):
    """F.interpolate(x, out_hw, bicubic, align_corners=False) at a fixed input size, as A_h @ x @ A_w^T."""

    def __init__(self, in_hw: tuple[int, int], out_hw: tuple[int, int]):
        super().__init__()
        self.register_buffer("a_h", bicubic_matrix(in_hw[0], out_hw[0]))
        self.register_buffer("a_w_t", bicubic_matrix(in_hw[1], out_hw[1]).T.contiguous())

    constant_right = False   # litert_gpu_forms: A_h @ x as (x^T @ A_h^T)^T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.constant_right:
            x = torch.matmul(x.transpose(-1, -2), self.a_h.T).transpose(-1, -2)
        else:
            x = torch.matmul(self.a_h, x)
        return torch.matmul(x, self.a_w_t)


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, leaf = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, leaf, module)


def matmul_neck(root: nn.Module, example) -> list[dict]:
    """Replace every bicubic, fixed-size UpSampleLayer under `root` by its MatmulResize at the input size it sees
    on `example` (or by nothing, where that size already is the output's, as the layer itself returns x)."""
    from efficientvit.models.nn.ops import UpSampleLayer

    layers = {name: m for name, m in root.named_modules() if isinstance(m, UpSampleLayer)}
    for name, m in layers.items():
        if m.mode != "bicubic" or m.size is None or m.align_corners:
            raise ValueError(f"{name}: {m.mode} to {m.size}, align_corners={m.align_corners}; not the neck's resize")
    seen: dict[str, tuple[int, int]] = {}

    def record(name):
        def hook(mod, inp):              # returns None: a pre-hook's return value would replace the input
            seen.setdefault(name, tuple(inp[0].shape[-2:]))
        return hook
    hooks = [m.register_forward_pre_hook(record(name)) for name, m in layers.items()]
    try:
        with torch.no_grad():
            root(example)
    finally:
        for h in hooks:
            h.remove()
    replaced = []
    for name, m in layers.items():
        in_hw, out_hw = seen[name], tuple(m.size)
        _set_submodule(root, name, nn.Identity() if in_hw == out_hw else MatmulResize(in_hw, out_hw))
        replaced.append({"layer": name, "in_hw": in_hw, "out_hw": out_hw})
    return replaced


def _relu_linear_att_concat(self, qkv: torch.Tensor) -> torch.Tensor:
    """efficientvit's LiteMLA.relu_linear_att (fp32) with v's row of ones concatenated rather than padded in:
    LiteRT's WebGPU delegate refuses the pad (PADV2), and takes the concatenation."""
    B, _, H, W = list(qkv.size())
    qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
    q, k, v = qkv[:, :, 0:self.dim], qkv[:, :, self.dim:2 * self.dim], qkv[:, :, 2 * self.dim:]
    q, k = self.kernel_func(q), self.kernel_func(k)
    v = torch.cat([v, torch.ones_like(v[:, :, :1])], dim=2)
    out = torch.matmul(torch.matmul(v, k.transpose(-1, -2)), q)
    out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
    return torch.reshape(out, (B, -1, H, W))


# -------------------------------------------------------------------------------------------------- DINOv3
def _attention_4d(self, x, rope=None, attn_mask=None, is_causal=False):
    """timm's EvaAttention.forward for this model's configuration (no qkv bias, no gate, identity norms), with
    the qkv split in 4-D and the attention written out. With `score_shift` (fp16_forms) the scores are made at
    2^-shift and their row's max subtracted before they are scaled back: the softmax is the same for any shift of a
    row, and the shifted scores are <= 0 (those below -65504 become -inf in fp16, where exp of either is 0)."""
    if attn_mask is not None or is_causal:
        raise ValueError("the export graph has no attention mask")
    B, N, _ = x.shape
    q, k, v = (t.reshape(B, N, self.num_heads, -1).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
    q, k = self.q_norm(q), self.k_norm(k)
    if rope is not None:
        from timm.layers import apply_rot_embed_cat
        npt = self.num_prefix_tokens
        half = getattr(self, "rotate_half", False)
        q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
        k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
    shift = getattr(self, "score_shift", 0)    # fp16_forms
    if shift:
        s = torch.matmul(q * (self.scale * 2.0 ** -shift), k.transpose(-2, -1))
        attn = ((s - s.amax(dim=-1, keepdim=True)) * 2.0 ** shift).softmax(dim=-1)
    else:
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1)).softmax(dim=-1)
    x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)
    return self.proj(self.norm(x))


def four_d_attention(model: nn.Module) -> int:
    """Rebind every EvaAttention's forward to `_attention_4d`; the number rewritten. Refuses a configuration the
    rewrite does not reproduce."""
    from timm.models.eva import EvaAttention

    n = 0
    for name, m in model.named_modules():
        if isinstance(m, EvaAttention):
            if m.qkv is None or m.q_bias is not None or m.gate is not None:
                raise ValueError(f"{name}: qkv bias or gate or separate projections; _attention_4d does not cover it")
            m.forward = types.MethodType(_attention_4d, m)
            n += 1
    return n


def constant_rope(model: nn.Module, grid_hw: tuple[int, int]) -> None:
    """Freeze DINOv3's RoPE table for one patch grid: timm's own get_embed computes it once, and afterwards
    returns that buffer for this grid and refuses any other."""
    rope = model.rope
    with torch.no_grad():
        rope.register_buffer("ml7_table", rope.get_embed(shape=list(grid_hw)).clone(), persistent=False)

    def get_embed(self, shape=None):
        if shape is not None and tuple(shape) != tuple(grid_hw):
            raise ValueError(f"constant RoPE table is for grid {grid_hw}, not {tuple(shape)}")
        return self.ml7_table
    rope.get_embed = types.MethodType(get_embed, rope)


# ------------------------------------------------------------------------------------------- fp16's range
LITEMLA_SHIFT = 6       # relu(k) by 2^-6, relu(q) by 2^6
SCORE_SHIFT = 4         # DINOv3's attention scores made at 2^-4


def _relu_linear_att_fp16(self, qkv: torch.Tensor) -> torch.Tensor:
    """_relu_linear_att_concat inside fp16's range (plan P7). relu(k) by 2^-6 and relu(q) by 2^6: v relu(k)^T and
    relu(k)'s sum over the tokens (4.1e5 and 8.5e4 on the fixtures, stage 4's second block) shrink by 64, and the
    numerator and denominator are the same numbers. eps (1e-15) is 0 in fp16, so the denominator also gets
    [den == 0]: it is 0 only where the numerator is 0 too (relu(q) all zero in a head: 1,024 tokens a photo in
    that block), and the output there stays 0 rather than 0/0."""
    B, _, H, W = list(qkv.size())
    qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
    q, k, v = qkv[:, :, 0:self.dim], qkv[:, :, self.dim:2 * self.dim], qkv[:, :, 2 * self.dim:]
    q, k = self.kernel_func(q) * 2.0 ** LITEMLA_SHIFT, self.kernel_func(k) * 2.0 ** -LITEMLA_SHIFT
    v = torch.cat([v, torch.ones_like(v[:, :, :1])], dim=2)
    out = torch.matmul(torch.matmul(v, k.transpose(-1, -2)), q)
    den = out[:, :, -1:]
    out = out[:, :, :-1] / (den + self.eps + (den == 0).to(den.dtype))
    return torch.reshape(out, (B, -1, H, W))


def _layer_norm_scaled(x: torch.Tensor, dim: int, weight, bias, eps: float) -> torch.Tensor:
    """LayerNorm over `dim` of x / max|x| per token, with eps / max|x|^2: the same function, as LayerNorm does not
    change when its input and sqrt(eps) are scaled together (plan P7). Every entry is then in [-1, 1]. DINOv3's
    register tokens carry entries to 1.5e4, so (x - mean)^2 reaches 2.3e8 and the variance 5.1e5 (the encoder's last
    norm: 1.2e5 and 5.0e3). One power of two for every token does not fit: the variances span 2^20 and the sum over
    768 channels 2^10 more, fp16's whole normal range."""
    r = 1.0 / x.abs().amax(dim=dim, keepdim=True)
    y = x * r
    y = y - y.mean(dim=dim, keepdim=True)
    y = y * torch.rsqrt((y * y).mean(dim=dim, keepdim=True) + eps * r * r)
    y = y if weight is None else y * weight
    return y if bias is None else y + bias


def _layer_norm_fp16(self, x: torch.Tensor) -> torch.Tensor:
    return _layer_norm_scaled(x, -1, self.weight, self.bias, self.eps)


def _layer_norm_2d_fp16(self, x: torch.Tensor) -> torch.Tensor:
    """efficientvit's LayerNorm2d (over the channels) as _layer_norm_scaled."""
    w, b = (None if t is None else t.view(1, -1, 1, 1) for t in (self.weight, self.bias))
    return _layer_norm_scaled(x, 1, w, b, self.eps)


SPLIT_CHANNELS = 128    # a convolution's sum over its input channels, cut into slices of at most this many


def _pairwise_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    """parts[0] + parts[1] + ..., added in pairs, level by level (log2 of the count additions in a row)."""
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
    return parts[0]


class SplitSumConv2d(nn.Module):
    """An nn.Conv2d (groups 1) as the pairwise sum of convolutions over slices of its input channels, the bias added
    last: the same function in real arithmetic (plan P7). LiteRT's GPU at fp16 accumulates a convolution's products
    in fp16, one after another: over the encoder's 12,288 input channels (stage 5's downsampling point_conv; 4,096
    in stage 4's) the running sum wanders to 30-60 while the output, once its folded BatchNorm bias is added, is
    about 0.4, and the masks drift (least IoU 0.967 on the GPU). Cut into slices of `channels`, each running sum is
    over that many products only, and the slices' sums meet in a tree of additions."""

    def __init__(self, conv: nn.Conv2d, channels: int):
        super().__init__()
        if conv.groups != 1 or conv.padding_mode != "zeros" or isinstance(conv.padding, str):
            raise ValueError(f"{conv}: groups, a padding mode or a padding string SplitSumConv2d does not cover")
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation
        self.bounds = [(a, min(a + channels, conv.in_channels)) for a in range(0, conv.in_channels, channels)]
        w = conv.weight.detach()
        self.weights = nn.ParameterList([nn.Parameter(w[:, a:b].clone(), requires_grad=False) for a, b in self.bounds])
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _pairwise_sum([F.conv2d(x[:, a:b], w, None, self.stride, self.padding, self.dilation)
                           for (a, b), w in zip(self.bounds, self.weights)])
        return y if self.bias is None else y + self.bias.view(1, -1, 1, 1)


def split_sums(model: nn.Module, channels: int = SPLIT_CHANNELS) -> int:
    """Every nn.Conv2d of groups 1 over more than `channels` input channels as its SplitSumConv2d; the number
    replaced. An efficientvit ConvLayer's BatchNorm (eval) is folded into its conv first, as LiteRT's converter
    folds it, which it cannot once the conv is a sum of slices."""
    from efficientvit.models.nn.ops import ConvLayer
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    def eligible(m):
        return type(m) is nn.Conv2d and m.groups == 1 and m.in_channels > channels
    n = 0
    for layer in [m for m in model.modules() if isinstance(m, ConvLayer)]:
        if eligible(layer.conv) and isinstance(layer.norm, nn.BatchNorm2d):
            if layer.norm.training:
                raise ValueError("a BatchNorm in training mode cannot be folded")
            layer.conv, layer.norm = fuse_conv_bn_eval(layer.conv, layer.norm), None
    for name in [name for name, m in model.named_modules() if eligible(m)]:
        _set_submodule(model, name, SplitSumConv2d(model.get_submodule(name), channels))
        n += 1
    return n


def fp16_forms(model: nn.Module, split_channels: int | None = SPLIT_CHANNELS) -> dict[str, int]:
    """Both models in forms whose every tensor fits fp16 (plan P7; measured on the fixtures in fp32): each LiteMLA
    as _relu_linear_att_fp16, each LayerNorm as _layer_norm_scaled, and each 4-D attention's scores made at
    2^-SCORE_SHIFT and shifted by their row's max before the softmax (DINOv3's block 0 has a score of 1.3e5); and,
    for fp16's precision rather than its range, each convolution over more than `split_channels` input channels
    as split_sums makes it (None: none; 53 in the encoder at 128, none in DINOv3). The same functions in real
    arithmetic, at float rounding in fp32. Applied after the P1 rewrites (an EvaAttention still on timm's own
    forward is refused); counts what changed."""
    from efficientvit.models.nn.norm import LayerNorm2d
    from efficientvit.models.nn.ops import LiteMLA
    from timm.models.eva import EvaAttention

    n = {"linear_attention": 0, "attention": 0, "layer_norm": 0}
    n["split_conv"] = split_sums(model, split_channels) if split_channels else 0
    for name, m in model.named_modules():
        if isinstance(m, LiteMLA):
            m.relu_linear_att = types.MethodType(_relu_linear_att_fp16, m)
            n["linear_attention"] += 1
        elif isinstance(m, EvaAttention):
            if getattr(m.forward, "__func__", None) is not _attention_4d:
                raise ValueError(f"{name}: not rewritten by four_d_attention; the score shift lives in _attention_4d")
            m.score_shift = SCORE_SHIFT
            n["attention"] += 1
        elif isinstance(m, LayerNorm2d):
            m.forward = types.MethodType(_layer_norm_2d_fp16, m)
            n["layer_norm"] += 1
        elif isinstance(m, nn.LayerNorm):
            if len(m.normalized_shape) != 1:
                raise ValueError(f"{name}: LayerNorm over {m.normalized_shape}; _layer_norm_scaled covers one dim")
            m.forward = types.MethodType(_layer_norm_fp16, m)
            n["layer_norm"] += 1
    return n


def tiled_prefix_tokens(model: nn.Module, batch: int) -> int:
    """Hold DINOv3's class and register tokens already tiled to `batch`: timm expands each to the batch at every
    call (LiteRT's BROADCAST_TO), and an expand to the size a tensor has is no op. The same values; afterwards the
    model takes this batch only (another one fails the expand), until tiled again. The number of tensors tiled."""
    n = 0
    for name in ("cls_token", "reg_token"):
        t = getattr(model, name, None)
        if t is not None:
            setattr(model, name, nn.Parameter(t.detach()[:1].repeat(batch, 1, 1), requires_grad=False))
            n += 1
    return n


class DinoReadout(nn.Module):
    """Patches -> DINOv3's cls_mean readout, exactly coffeecv.backbones.FrozenBackbone.features."""

    def __init__(self, backbone: nn.Module, readout: str):
        super().__init__()
        self.backbone, self.readout = backbone, readout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(x)[self.readout]


# --------------------------------------------------------------------------------------------- the segmenter
class SegEncoder(nn.Module):
    def __init__(self, sam: nn.Module):
        super().__init__()
        self.image_encoder = sam.image_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(x)


class SegDecoder(nn.Module):
    """The embedding and the transformed box -> all three low-res masks and their IoUs, as predict_torch."""

    def __init__(self, sam: nn.Module):
        super().__init__()
        self.prompt_encoder, self.mask_decoder = sam.prompt_encoder, sam.mask_decoder
        self.register_buffer("image_pe", sam.prompt_encoder.get_dense_pe().clone())

    def forward(self, embedding: torch.Tensor, box: torch.Tensor):
        sparse, dense = self.prompt_encoder(points=None, boxes=box, masks=None)
        return self.mask_decoder(image_embeddings=embedding, image_pe=self.image_pe,
                                 sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                 multimask_output=True)


def _embed_boxes_whole(self, boxes: torch.Tensor) -> torch.Tensor:
    """segment_anything's PromptEncoder._embed_boxes without its writes into a tensor (LiteRT's SELECT_V2): the
    corners' x and y divided by the frame's width and height in one division, and both corners' embeddings added
    in one addition. The same operations on the same numbers."""
    h, w = self.input_image_size
    coords = (boxes + 0.5).reshape(-1, 2, 2) / torch.tensor([w, h], dtype=boxes.dtype, device=boxes.device)
    corners = torch.cat([self.point_embeddings[2].weight, self.point_embeddings[3].weight])
    return self.pe_layer._pe_encoding(coords) + corners


# -------------------------------------------------------------------------------------- LiteRT's GPU forms
def litert_gpu_forms(model: nn.Module, batch: int | None = None) -> dict[str, int]:
    """A graph in forms LiteRT's WebGPU delegate places whole on the GPU (probed on yappy, plans P2 and P7): each
    MatmulResize with its constants on the right, each LiteMLA's ones concatenated, DINOv3's prefix tokens tiled
    to `batch` (the batch the graph is converted at; required for DINOv3), SAM's box prompt without writes. The
    same model at float rounding or exactly; LiteRT's conversion only (ORT's WebGPU provider runs the plain graphs
    whole). Counts what changed, by form; a form with nothing to change is left out."""
    from efficientvit.models.nn.ops import LiteMLA
    from segment_anything.modeling.prompt_encoder import PromptEncoder
    from timm.models.eva import Eva

    n = {"linear_attention": 0, "resize": 0, "prefix_tokens": 0, "box_prompt": 0}
    for m in model.modules():
        if isinstance(m, MatmulResize):
            m.constant_right = True
            n["resize"] += 1
        elif isinstance(m, LiteMLA):
            m.relu_linear_att = types.MethodType(_relu_linear_att_concat, m)
            n["linear_attention"] += 1
        elif isinstance(m, Eva):
            if batch is None:
                raise ValueError("DINOv3's prefix tokens are tiled to the graph's batch: pass batch")
            n["prefix_tokens"] += tiled_prefix_tokens(m, batch)
        elif isinstance(m, PromptEncoder):
            m._embed_boxes = types.MethodType(_embed_boxes_whole, m)
            n["box_prompt"] += 1
    return {k: v for k, v in n.items() if v}


# ------------------------------------------------------------------------------------------------- loading
def repo_path(path: Path) -> Path:
    """A path as given on a command line: absolute, or relative to the repo."""
    return path if path.is_absolute() else REPO / path


def load_shipped(checkpoint: Path):
    """(DinoClassifier, BeanSegmenter, RunConfig) for a shipped checkpoint, built as webapp/app.py builds them."""
    from coffeecv.class_list import load_classes
    from coffeecv.infer import config_for_checkpoint, load_model
    from coffeecv.segment_beans import BeanSegmenter, seg_params

    cfg, _ = config_for_checkpoint(checkpoint, None)
    model, _ = load_model(checkpoint, cfg.model_name, len(load_classes(REPO / cfg.classes_file).keys), cfg.dropout)
    seg = BeanSegmenter(seg_params(cfg))
    return model.eval(), seg, cfg


def prepare(checkpoint: Path) -> tuple[dict, list[dict]]:
    """({graph name: (module, example args)}, the neck layers matmul_neck replaced) for the shipped checkpoint:
    dino_b8, dino_b40, encoder and decoder, with the three rewrites applied."""
    model, seg, cfg = load_shipped(repo_path(checkpoint))
    if cfg.patch_resize != PATCH:
        raise ValueError(f"patch_resize {cfg.patch_resize}, the graphs are built for {PATCH}")
    sam = seg.model
    if tuple(sam.image_size) != (FRAME, FRAME):
        raise ValueError(f"segmenter image_size {sam.image_size}, the graphs are built for {FRAME}")
    readout = DinoReadout(model.backbone, model.readout).eval()
    encoder, decoder = SegEncoder(sam).eval(), SegDecoder(sam).eval()
    image = torch.zeros(1, 3, FRAME, FRAME)
    out = {"encoder": (encoder, (image,)),
           "decoder": (decoder, (torch.zeros(1, 256, 64, 64), torch.tensor([[0.0, 0.0, 1023.0, 767.0]])))}
    timm_model = model.backbone.model
    four_d_attention(timm_model)
    constant_rope(timm_model, (PATCH // timm_model.patch_embed.patch_size[0],) * 2)
    neck = matmul_neck(encoder, image)
    for b in DINO_BATCHES:
        out[f"dino_b{b}"] = (readout, (torch.zeros(b, 3, PATCH, PATCH),))
    return out, neck
