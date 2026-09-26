"""Forward-pass latency of the frozen backbones on real bean patches (plan §5.5 / §6.2's table).

Batch 32 at 224 px, the real weights, the real eval-transformed cam_iphone patches from
`coffeecv_dino.reference`. One process per thread count, because torch's intra-op pool is sized once.
Reports ms per image; model time per /classify photo is that times 40 patches (no TTA) for a frozen
DINO arm, and times 320 (40 patches x 8 dihedral views) for today's ResNet18.

    python -m coffeecv_dino.bench --threads 4 --backbones resnet18 dinov3_vits16 dinov3_vitb16
"""
from __future__ import annotations

import argparse
import json
import socket
import time

import torch

from coffeecv_dino.backbone import SPECS, build_backbone
from coffeecv_dino.reference import reference_patches


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, required=True, help="torch intra-op threads (production: 4)")
    p.add_argument("--backbones", nargs="+", default=["resnet18", "dinov3_vits16", "dinov3_vitb16"],
                   choices=sorted(SPECS))
    p.add_argument("--batches", type=int, default=8, help="timed batches per backbone, after 2 warm-up")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    x, _, _, _ = reference_patches(per_class=4)            # 32 real patches
    x = x[:32]
    rows = {}
    for name in args.backbones:
        bb = build_backbone(name)
        with torch.inference_mode():
            for _ in range(2):
                bb.features(x)
            t = time.perf_counter()
            for _ in range(args.batches):
                bb.features(x)
        rows[name] = round(1000 * (time.perf_counter() - t) / (args.batches * len(x)), 1)
        print(f"{name:18s} {rows[name]:6.1f} ms/img   40 patches: {rows[name] * 40 / 1000:.2f} s/photo   "
              f"320 (8-view TTA): {rows[name] * 320 / 1000:.2f} s/photo", flush=True)
    print(json.dumps({"host": socket.gethostname(), "threads": torch.get_num_threads(), "batch": len(x),
                      "ms_per_img": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
