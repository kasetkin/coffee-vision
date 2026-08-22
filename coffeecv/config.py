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

    # Training reads the *cropped* tree, which the `crop` stage produces from the
    # raw session. The raw photos are `dvc add`-tracked data and are not read here;
    # which session the crops came from is recorded by the crop stage in dvc.yaml
    # and in data/cropped/<session>/crop_manifest.json.
    #
    # Leave-one-rig-out: `train_rigs` are split into train/val/test at the photo
    # level; `heldout_rig` contributes every one of its photos as a second,
    # cross-rig test set and is never seen in training. Setting `heldout_rig` to
    # "" disables the cross-rig split and reproduces a plain single-rig run.
    train_rigs: tuple[str, ...] = (
        "data/cropped/2026-08-07__box_pictures_all_classes",
        "data/cropped/2026-08-09__pixel_cam",
    )
    heldout_rig: str = "data/cropped/2026-08-09__sony_cam"
    classes_file: str = "dataset/classes.txt"

    patch_crop_size: int = 512
    patch_resize: int = 224
    safety_margin: float = 0.97
    # Edge length patches are stored at after extraction. Bounds memory: the
    # 2026-08-09 rigs decode to 37-57 MB per photo, so holding whole photos would
    # need ~18 GB across the four datasets. 0 means "keep full patch_crop_size"
    # (single-rig, pre-Phase-11 behaviour). Keep this well above patch_resize so
    # zoom augmentation crops into detail rather than upsampling.
    patch_store_size: int = 448
    train_patches_per_class: int = 150
    val_patches_per_class: int = 40
    test_patches_per_class: int = 40
    # The held-out rig gets a bigger patch budget than the in-distribution
    # splits: it is the headline number of the whole phase, and at 40/class the
    # measured noise band on macro-F1 is +/-0.048, wide enough to hide the effect
    # sizes being chased. It also draws from 20 photos/class rather than 3, so
    # the extra patches are spread over more independent photos rather than
    # resampling the same few.
    xrig_patches_per_class: int = 120

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
    # Phase 13: overrides only the brightness bound of the ColorJitter call above
    # (contrast/saturation/hue keep color_jitter_strength). 0.0 is the no-op
    # default -- brightness falls back to color_jitter_strength, reproducing the
    # exact pre-Phase-13 pipeline. Rigs measure 1.48-1.59x apart in grey mean
    # (see EXPERIMENTS_LOG.md Phase 11), which color_jitter_strength=0.2's
    # 0.8-1.2x range cannot reach; this widens brightness alone to test whether
    # augmentation can close that gap without perturbing the other channels.
    brightness_jitter_strength: float = 0.0

    # Phase 8 augmentation knobs. Every default here is a no-op, so the whole
    # block off reproduces the Phase 7 pipeline exactly (see transforms.py).
    # Scale augmentation (Phase 11). Patch side is drawn log-uniformly between
    # these fractions of each photo's short side, instead of the fixed
    # `patch_crop_size` in pixels. 0/0 disables it -- the no-op default, so the
    # baseline arm and the treatment arm run identical code.
    #
    # Applies to eval splits too, unlike other augmentation: patch side sets how
    # many beans a patch covers, so a fixed pixel size would score each rig at a
    # different bean coverage. See MultiPhotoPatchDataset.
    patch_scale_frac_min: float = 0.0
    patch_scale_frac_max: float = 0.0

    # Bean-unit patch sizing (Phase 12). Patch side = B * measured bean pitch,
    # with B drawn log-uniformly in [min, max]. B is side/pitch = sqrt(bean count)
    # over the square, so 6-9 across means roughly 36-81 beans per patch.
    #
    # This is the only sizing mode that works at inference, because it needs no
    # knowledge of how the rig was framed -- the pitch is measured from the photo
    # by the same estimator used in training (coffeecv/bean_scale.py). Takes
    # precedence over patch_scale_frac. 0/0 disables it.
    #
    # 9 is the ceiling these rigs can deliver: they frame 10.7-12.5 beans across,
    # and a patch must leave room to be placed. Reaching 12 would need capture
    # about 1.3x wider.
    patch_beans_min: float = 0.0
    patch_beans_max: float = 0.0

    rotation_jitter_degrees: float = 0.0  # +/- jitter around each right angle, applied at patch sampling
    zoom_scale_min: float = 1.0  # RandomResizedCrop min *area* fraction; 1.0 = plain resize
    random_erasing_p: float = 0.0  # probability of erasing a rectangle per patch
    illum_gradient_strength: float = 0.0  # +/- fraction of a corner-to-corner luminance ramp
    mixup_alpha: float = 0.0  # Beta(a, a) mixing strength; applied in the train loop, not a transform

    # MixStyle (Zhou et al., domain-agnostic v1): probability of mixing each
    # training batch's per-sample channel statistics with a randomly permuted
    # copy of the batch, inside the model's forward pass (see model.py). 0.0 is
    # the no-op default, same convention as every other Phase 8+ knob here.
    mixstyle_p: float = 0.0
    mixstyle_alpha: float = 0.1  # Beta(a, a) shape for the mixing coefficient; the paper's default
    # "agnostic" (v1, adopted): partner is any random batch sample, regardless of rig.
    # "cross_rig" (v2, screening): partner is restricted to a different rig (model.py's
    # _cross_domain_perm). Irrelevant when mixstyle_p == 0.0.
    mixstyle_mode: str = "agnostic"

    batch_size: int = 32
    epochs: int = 20
    early_stop_patience: int = 8  # stop if val_macro_f1 hasn't improved in this many epochs
    optimizer: str = "adamw"  # adamw | sgd
    lr: float = 1e-3
    backbone_lr: float = 1e-5  # used only when freeze_mode != "full"
    weight_decay: float = 1e-4

    @classmethod
    def from_params_yaml(cls, path: Path = PARAMS_FILE) -> "RunConfig":
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        # Unknown keys are a hard error, not something to drop. Silently ignoring
        # them is what made the Phase 13 provenance failure invisible: the commits
        # for exp 72-95 pair a params.yaml naming `brightness_jitter_strength`
        # with a RunConfig predating the field, so checking one out and re-running
        # quietly trained at the default brightness and looked like it had worked.
        # A parameter this file names but the code cannot honour means the two are
        # out of sync, and every number produced from that pair is mislabelled --
        # so it has to fail loudly rather than pick a value nobody asked for.
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"{path} sets parameters this RunConfig does not define: {', '.join(unknown)}. "
                "The config file and the code are out of sync -- most likely the source change "
                "implementing them is uncommitted or was never written. Refusing to run at "
                "defaults, which would silently produce mislabelled results."
            )
        values = {k: v for k, v in raw.items() if k in known}
        # YAML gives a list; the field is a tuple so the config stays hashable
        # and cannot be mutated in place by a caller.
        if isinstance(values.get("train_rigs"), list):
            values["train_rigs"] = tuple(values["train_rigs"])
        return cls(**values)

    def resolve_paths(self) -> tuple[list[Path], Path | None, Path]:
        """(train rig dirs, held-out rig dir or None, classes file)."""
        train = [REPO_ROOT / d for d in self.train_rigs]
        heldout = (REPO_ROOT / self.heldout_rig) if self.heldout_rig else None
        return train, heldout, REPO_ROOT / self.classes_file


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
