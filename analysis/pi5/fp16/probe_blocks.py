import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import litert_torch  # noqa: E402
import convert_dino as C  # noqa: E402

OUT = os.environ.get("PI16_OUT", "out/pi16")
os.makedirs(OUT, exist_ok=True)  # where the graphs and arrays go

model = C.build("allpre")  # PreScaledReadout around DinoReadout, with all forms
m = model.backbone.model


class Blocks(nn.Module):
    def __init__(s):
        super().__init__()
        s.m = m

    def forward(s, x):
        t = s.m.patch_embed(x)
        t, rope = s.m._pos_embed(t)
        t = s.m.norm_pre(t)
        outs = []
        for b in s.m.blocks:
            t = b(t, rope=rope)
            outs.append(t)
        return tuple(outs)


p = Blocks().eval()
with torch.no_grad():
    o = p(torch.zeros(8, 3, 224, 224))
    print(len(o), tuple(o[0].shape))
litert_torch.convert(p, (torch.zeros(8, 3, 224, 224),)).export(
    f"{OUT}/dino_b8_allpre_blocks.tflite"
)
print("exported")
