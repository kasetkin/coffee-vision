"""Run configuration: loaded from params.yaml, overridable via CLI flags in train_baseline.py."""
from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = REPO_ROOT / "params.yaml"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TENSORBOARD_DIR = REPO_ROOT / "tensorboard"


@dataclass
class RunConfig:
    seed: int = 42

    dataset_dir: str = "dataset/2026-08-07__box_pictures_all_classes"
    classes_file: str = "dataset/classes.txt"

    patch_crop_size: int = 512
    patch_resize: int = 224
    safety_margin: float = 0.97
    train_patches_per_class: int = 150
    val_patches_per_class: int = 40
    test_patches_per_class: int = 40

    # Photo-level split sizes for the multi-photo box-rig dataset (must sum to
    # the number of cropped photos per class -- 20 for 2026-08-07).
    train_photos_per_class: int = 14
    val_photos_per_class: int = 3
    test_photos_per_class: int = 3

    model_name: str = "mobilenet_v3_small"
    freeze_mode: str = "full"  # full | last_block | none
    dropout: float = 0.2
    label_smoothing: float = 0.0
    color_jitter_strength: float = 0.2

    batch_size: int = 32
    epochs: int = 20
    early_stop_patience: int = 8  # stop if val_macro_f1 hasn't improved in this many epochs
    optimizer: str = "adamw"  # adamw | sgd
    lr: float = 1e-3
    backbone_lr: float = 1e-5  # used only when freeze_mode != "full"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"

    @classmethod
    def from_params_yaml(cls, path: Path = PARAMS_FILE) -> "RunConfig":
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def resolve_paths(self) -> tuple[Path, Path]:
        return REPO_ROOT / self.dataset_dir, REPO_ROOT / self.classes_file


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env_block() -> dict:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = None
    import torchvision

    return {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "git_commit": git_commit,
        "argv": sys.argv,
    }


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)
