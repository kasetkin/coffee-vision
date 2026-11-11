"""OPS-23 fp16 experiment: the segmenter's encoder and decoder with litert_gpu_forms + fp16_forms (as
export_litert does for a graph served in fp16), from the checkpoint ML-7's references were made with."""

import os
import sys
from pathlib import Path
import litert_torch

sys.path.insert(0, "/workspace")
from coffeecv import litert_forms as LF  # noqa: E402

OUT = os.environ.get("PI16_OUT", "out/pi16")
os.makedirs(OUT, exist_ok=True)  # where the graphs and arrays go

graphs, _ = LF.prepare(Path("models/allrigs_dino3b16_segxl0v3_country_s42.pt"))
for name in ("encoder", "decoder"):
    module, example = graphs[name]
    print(
        name,
        LF.litert_gpu_forms(module, batch=example[0].shape[0]),
        LF.fp16_forms(module),
        flush=True,
    )
    litert_torch.convert(module.eval(), example).export(
        f"{OUT}/{name}_fp16forms.tflite"
    )
    print("exported", name, flush=True)
