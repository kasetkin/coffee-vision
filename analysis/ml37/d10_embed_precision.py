"""ML-37 D10: how far does each storage precision move DINOv3's embedding of a patch?

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d10_embed_precision.py

Reads tmp/ml37/d10/patches.npz (float32 448 px patches of the float read-back, made by d10_crop_forms.py). Each patch
is stored four ways: float32 (the reference), uint16 (round(x * 65535) / 65535), float16, uint8 (round(x * 255) /
255, today's); then resized to 224 in float (PIL "F" bilinear per channel, ML-37 D11 (a)), ImageNet-normalised and
embedded by the frozen dinov3_vitb16 (cls_mean, the shipped readout), in float32 on 2 CPU threads. Reported: 1 - cosine
of each form's embedding to the float32 one, and, as yardsticks, the float32 input rounded to float16 after the
normalisation (what an fp16 graph's first cast does) and experiments.md's cross-CPU limit (cosine >= 0.9999998).
Writes analysis/ml37/d10_embed_precision.txt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from coffeecv.backbones import build_backbone
from coffeecv.config import REPO_ROOT
from coffeecv.model_io import IMAGENET_MEAN, IMAGENET_STD

PATCHES = REPO_ROOT / "tmp" / "ml37" / "d10" / "patches.npz"
OUT = Path(__file__).with_suffix(".txt")
MEAN = np.asarray(IMAGENET_MEAN, np.float32)[:, None, None]
STD = np.asarray(IMAGENET_STD, np.float32)[:, None, None]


def to_input(x: np.ndarray) -> np.ndarray:
    small = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(x[..., c])).resize((224, 224), Image.BILINEAR))
                      for c in range(3)])
    return (small.astype(np.float32) - MEAN) / STD


FORMS = {
    "float32": lambda x: x,
    "uint16": lambda x: (np.rint(x * 65535) / 65535).astype(np.float32),
    "float16": lambda x: x.astype(np.float16).astype(np.float32),
    "uint8": lambda x: (np.rint(x * 255) / 255).astype(np.float32),
}


def main() -> None:
    torch.set_num_threads(2)
    patches = np.load(PATCHES)["patches"]
    bb = build_backbone("dinov3_vitb16")
    emb = {k: [] for k in (*FORMS, "fp16 input")}
    with torch.no_grad():
        for p in patches:
            batch = {k: to_input(f(p)) for k, f in FORMS.items()}
            batch["fp16 input"] = batch["float32"].astype(np.float16).astype(np.float32)
            x = torch.from_numpy(np.stack(list(batch.values())))
            f = bb.features(x)["cls_mean"].float().numpy()
            for k, v in zip(batch, f):
                emb[k].append(v)
    E = {k: np.stack(v) for k, v in emb.items()}
    ref = E["float32"]

    def cos(a, b):
        return np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)

    lines = [f"{len(patches)} patches (448 px, float read-back), dinov3_vitb16 cls_mean, float32 on CPU, 2 threads",
             "form          1-cos median   1-cos max     rel. L2 median   (vs the float32 patch's embedding)"]
    for k in E:
        if k == "float32":
            continue
        c = 1 - cos(E[k], ref)
        rel = np.linalg.norm(E[k] - ref, axis=1) / np.linalg.norm(ref, axis=1)
        lines.append(f"{k:12s}  {np.median(c):12.3e}  {c.max():11.3e}   {np.median(rel):12.3e}")
    # Scale: two different patches of the same photo (consecutive patches come from one crop)
    pair = 1 - cos(ref[0::2], ref[1::2])
    lines.append(f"yardstick: two patches of the same crop, 1-cos median {np.median(pair):.3e}; experiments.md's "
                 f"cross-CPU limit 1-cos <= 2e-7")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
