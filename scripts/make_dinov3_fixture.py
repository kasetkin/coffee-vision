"""Regenerate tests/fixtures/dinov3_vits16_reference.npz from Meta's reference implementation.

The fixture is what lets tests/test_dino_backbone.py check the timm-built DINOv3 against Meta's own
code without Meta's code being in this repo (docs/dinov3_integration_plan.md §2.3-2.4): the DINOv3
License covers the code as well as the weights, and a public repo must not redistribute either. The
fixture holds only model *outputs*, which the licence does not restrict. The input is real: one bean
patch per class from cam_iphone, drawn by coffeecv_dino.reference exactly as the test redraws it.

    python scripts/make_dinov3_fixture.py                 # clones facebookresearch/dinov3 into a temp dir
    python scripts/make_dinov3_fixture.py --repo /path/to/existing/clone

The clone is checked out at the pinned commit, and the script refuses any other.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coffeecv_dino.reference import REFERENCE_RIG, REFERENCE_SEED, reference_patches  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
META_URL = "https://github.com/facebookresearch/dinov3.git"
META_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"   # verified against timm 1.0.29 on 2026-09-24
WEIGHTS = REPO_ROOT / "models_pretrained" / "dinov3" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
OUT = REPO_ROOT / "tests" / "fixtures" / "dinov3_vits16_reference.npz"
PER_CLASS = 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None, help="an existing facebookresearch/dinov3 clone")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(args.repo) if args.repo else Path(tmp) / "dinov3"
        if not args.repo:
            subprocess.run(["git", "clone", "-q", META_URL, str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", META_COMMIT], check=True)
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if head != META_COMMIT:
            raise SystemExit(f"{repo} is at {head}, not the pinned {META_COMMIT}")
        sys.path.insert(0, str(repo))
        from dinov3.hub.backbones import dinov3_vits16   # not via hubconf.py, which needs torchmetrics

        model = dinov3_vits16(pretrained=False)            # weights=<path> would copy into the torch cache
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=True)
        model.eval()
        x, _, class_ids, _ = reference_patches(PER_CLASS)
        with torch.no_grad():
            out = model.forward_features(x)
        cls = out["x_norm_clstoken"].numpy()
        cls_mean = np.concatenate([cls, out["x_norm_patchtokens"].mean(dim=1).numpy()], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, cls=cls, cls_mean=cls_mean, rig=REFERENCE_RIG, seed=REFERENCE_SEED, per_class=PER_CLASS,
             class_ids=np.array(class_ids), meta_commit=META_COMMIT, weights=WEIGHTS.name)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: cls {cls.shape}, cls_mean {cls_mean.shape} "
          f"(Meta dinov3 @ {META_COMMIT[:7]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
