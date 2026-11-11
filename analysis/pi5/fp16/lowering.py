import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch
from collections import Counter
from ai_edge_litert.tools import flatbuffer_utils as fu
from ai_edge_litert import schema_py_generated as sch

OUT = os.environ.get("PI16_OUT", "out/pi16")
os.makedirs(OUT, exist_ok=True)  # where the graphs and arrays go

names = {v: k for k, v in vars(sch.BuiltinOperator).items() if not k.startswith("_")}
x = torch.randn(8, 201, 768)
w, b = torch.randn(768), torch.randn(768)
cands = {
    "mean_keepdim": lambda x: x.mean(dim=-1, keepdim=True),
    "mean_nokeep": lambda x: x.mean(dim=1),
    "layer_norm": lambda x: F.layer_norm(x, (768,), w, b, 1e-6),
    "layer_norm_scaled": lambda x: F.layer_norm(
        x * (1.0 / x.abs().amax(-1, keepdim=True)), (768,), w, b, 1e-12
    ),
    "avgpool1d": lambda x: F.adaptive_avg_pool1d(x[:, 5:].transpose(1, 2), 1).squeeze(
        -1
    ),
    "avg_pool2d": lambda x: F.avg_pool2d(
        x[:, 5:].transpose(1, 2).reshape(8, 768, 14, 14), 14
    ).flatten(1),
}


class M(nn.Module):
    def __init__(s, f):
        super().__init__()
        s.f = f

    def forward(s, x):
        return s.f(x)


for k, f in cands.items():
    try:
        litert_torch.convert(M(f).eval(), (x,)).export(f"{OUT}/low_{k}.tflite")
        m = fu.read_model(f"{OUT}/low_{k}.tflite")
        g = m.subgraphs[0]
        c = Counter(
            names.get(
                max(
                    m.operatorCodes[o.opcodeIndex].builtinCode,
                    m.operatorCodes[o.opcodeIndex].deprecatedBuiltinCode,
                )
            )
            for o in g.operators
        )
        print(f"{k:18s}", dict(c), flush=True)
    except Exception as e:
        print(k, "FAILED", type(e).__name__, str(e)[:120])
