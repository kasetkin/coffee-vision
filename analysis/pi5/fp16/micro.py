"""OPS-23 fp16 experiment: one DINOv3 block and its parts, plain and in ADR 0022's fp16 forms, as small graphs, to
find which form loses XNNPACK's fp16 speed on the Pi. Input: block 0's real input on 8 patches."""

import os
import sys

sys.path.insert(0, "/workspace")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import litert_torch  # noqa: E402
from coffeecv import litert_forms as LF  # noqa: E402
from coffeecv.backbones import build_backbone  # noqa: E402

OUT = os.environ.get("PI16_OUT", "out/pi16")
os.makedirs(OUT + "/micro", exist_ok=True)  # where the graphs and arrays go

inp = np.load("/workspace/analysis/ml7_gpu/artifacts/inputs.npz")
x8 = torch.from_numpy(np.ascontiguousarray(inp["patches"][0, :8]))


def model(forms):
    bb = build_backbone("dinov3_vitb16")
    tm = bb.model.eval()
    LF.four_d_attention(tm)
    LF.constant_rope(tm, (14, 14))
    mod = LF.DinoReadout(bb, "cls_mean").eval()
    LF.litert_gpu_forms(mod, batch=8)
    if forms:
        LF.fp16_forms(mod)
    return tm


plain, forms = model(False), model(True)
with torch.no_grad():
    t = plain.patch_embed(x8)
    t, rope = plain._pos_embed(t) if hasattr(plain, "_pos_embed") else (None, None)
print(
    "tokens",
    None if t is None else tuple(t.shape),
    "rope",
    None if rope is None else tuple(rope.shape),
    flush=True,
)
np.save(f"{OUT}/micro/micro_x.npy", t.numpy())


class Block(nn.Module):
    def __init__(s, m):
        super().__init__()
        s.b = m.blocks[0]
        s.rope = rope

    def forward(s, x):
        return s.b(x, rope=s.rope)


class Norm(nn.Module):
    def __init__(s, m):
        super().__init__()
        s.n = m.blocks[0].norm1

    def forward(s, x):
        return s.n(x)


class Attn(nn.Module):
    def __init__(s, m):
        super().__init__()
        s.a = m.blocks[0].attn
        s.n = m.blocks[0].norm1
        s.rope = rope

    def forward(s, x):
        return s.a(x, rope=s.rope)


class FC(nn.Module):
    def __init__(s, m):
        super().__init__()
        s.f = m.blocks[0].mlp.fc1

    def forward(s, x):
        return s.f(x)


gens = [("block", Block), ("norm", Norm), ("attn", Attn), ("fc1", FC)]
for tag, m in (("plain", plain), ("forms", forms)):
    for name, cls in gens:
        mod = cls(m).eval()
        with torch.no_grad():
            mod(t)
        litert_torch.convert(mod, (t,)).export(f"{OUT}/micro/{name}_{tag}.tflite")
        print("exported", name, tag, flush=True)
