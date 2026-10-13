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
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(self.a_h, x), self.a_w_t)


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


# -------------------------------------------------------------------------------------------------- DINOv3
def _attention_4d(self, x, rope=None, attn_mask=None, is_causal=False):
    """timm's EvaAttention.forward for this model's configuration (no qkv bias, no gate, identity norms), with
    the qkv split in 4-D and the attention written out."""
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


# ------------------------------------------------------------------------------------------------- loading
def load_shipped(checkpoint: Path):
    """(DinoClassifier, BeanSegmenter, RunConfig) for a shipped checkpoint, built as webapp/app.py builds them."""
    from coffeecv.class_list import load_classes
    from coffeecv.infer import config_for_checkpoint, load_model
    from coffeecv.segment_beans import BeanSegmenter, seg_params

    cfg, _ = config_for_checkpoint(checkpoint, None)
    model, _ = load_model(checkpoint, cfg.model_name, len(load_classes(REPO / cfg.classes_file).keys), cfg.dropout)
    seg = BeanSegmenter(seg_params(cfg))
    return model.eval(), seg, cfg


def prepare(checkpoint: Path, rewrite: bool = True) -> dict:
    """{graph name: (module, example args)} for the shipped checkpoint: dino_b8, dino_b40, encoder and decoder,
    with the rewrites applied unless rewrite=False (the originals, to check the rewrites against)."""
    model, seg, cfg = load_shipped(checkpoint)
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
    if rewrite:
        timm_model = model.backbone.model
        four_d_attention(timm_model)
        constant_rope(timm_model, (PATCH // timm_model.patch_embed.patch_size[0],) * 2)
        out["neck"] = matmul_neck(encoder, image)
    for b in DINO_BATCHES:
        out[f"dino_b{b}"] = (readout, (torch.zeros(b, 3, PATCH, PATCH),))
    return out
