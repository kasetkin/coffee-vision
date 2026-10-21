"""Run configuration: loaded from params.yaml (RunConfig.from_params_yaml, strict) or from a model's card
(infer.config_for_checkpoint)."""
from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = REPO_ROOT / "params.yaml"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"


@dataclass
class RunConfig:
    seed: int = 42

    # Training reads the *cropped* tree, which the `segcrop` stage produces from the
    # raw session (and the merge_segpool stage pools, ticket ML-8). The raw photos are `dvc
    # add`-tracked data and are not read here; which session the crops came from is
    # recorded in dvc.yaml and in data/segcropped/<session>/segcrop_manifest.json.
    #
    # The capture dirs whose photos are pooled per class and split 70/15/15 into
    # train/val/test (ticket ML-1, D1(b)): since ticket ML-8 (D2, ADR 0024) one, the
    # pool of every session. The same list as dataset.CAPTURES. Kept as
    # a literal so this module does not import the dataset stack; a test checks
    # the two agree. Overridden by params.yaml on every real run, so this is a sane
    # resting shape rather than a live configuration. Kept current anyway: a
    # default naming dirs that no longer exist is the kind of drift that makes a
    # reader trust the wrong thing. Since ticket ML-3 P2b this is the segmenter's
    # crops (the tray heuristic's data/cropped pools are retired), while
    # crop_method below still defaults to "tray_heuristic" for old cards; so a
    # config that sets neither is refused by resolve_paths, never read.
    train_capture_dirs: tuple[str, ...] = (
        "data/segcropped/pool",
    )
    classes_file: str = "dataset/classes.txt"

    patch_crop_size: int = 512
    patch_resize: int = 224
    safety_margin: float = 0.97
    # Edge length patches are stored at after extraction. Bounds memory: the
    # 2026-08-09 sessions decode to 37-57 MB per photo, so holding whole photos
    # would need ~18 GB across the datasets. 0 means "keep full patch_crop_size"
    # (single-dir, pre-Phase-11 behaviour). Kept above patch_resize, so head_augment's patch pipeline acts on
    # more pixels than the model's input (ADR 0025).
    patch_store_size: int = 448
    # Patches per class, spread over the class's photos in the split (see
    # MultiPhotoPatchDataset; ticket ML-8 D8). Until ML-8, 150/40/40 per class *per
    # capture dir*, over four camera pools.
    train_patches_per_class: int = 600
    val_patches_per_class: int = 160
    test_patches_per_class: int = 160

    # Photo-level split, as fractions of however many cropped photos a class has
    # pooled across all capture dirs (not a fixed count, and not per dir: dirs
    # differ in photos per class, and since ticket ML-1 the split ignores which
    # dir a photo came from). val/test each floor at 1 photo; train gets the
    # remainder. See split_photos_by_class in dataset.py.
    train_photo_frac: float = 0.70
    val_photo_frac: float = 0.15
    test_photo_frac: float = 0.15

    # The frozen backbone the head is fitted on (coffeecv/backbones.py SPECS; fit_frozen_head --backbone sets it
    # per run). The fine-tuned ResNet18 path and its fields -- freeze_mode, dropout, the ColorJitter, zoom,
    # erasing, illumination, mixup and MixStyle knobs, and the SGD loop's epochs, LR schedule and optimizer --
    # were removed on 2026-10-20; old cards still carry them, and config_for_checkpoint drops unknown keys.
    model_name: str = "dinov3_vitb16"

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

    # Bean-pitch estimator geometry (coffeecv/bean_scale.py). These were module
    # constants until 2026-09-03, which made them impossible to vary per run and
    # impossible to restore per checkpoint: changing one silently re-sized patches
    # for every model ever trained, including the deployed one, with no record in
    # any config.json that anything had moved. Since the whole point of
    # config_for_checkpoint is that patch geometry is the one thing inference must
    # get right, geometry that inference depends on belongs here.
    #
    # Defaults are exactly the shipped module constants, so an unset config
    # reproduces every existing run bit-for-bit. Sweeping bean_k_lo needs
    # bean_calibration_k refitted alongside -- they are not separable; see
    # analysis/bean_scale/README.md.
    bean_k_lo: int = 4
    bean_k_hi: int = 80
    bean_analysis_frac: float = 0.40
    bean_calibration_k: float = 1.18

    rotation_jitter_degrees: float = 0.0  # +/- jitter around each right angle, applied at patch sampling

    # Ticket ML-2 (D21): the seed of the segmenter's own photo split, used only by
    # seg_lists.py to build labels/ml2/photo_lists.yaml. It is none of the head seeds
    # (42/123/7), so no head seed's split is the segmenter's. No training run reads it.
    seg_split_seed: int = 239
    # Ticket ML-2 (D5, owner 2026-10-01): the segmenter's fixed prompt is one whole-image box, and
    # the widest multimask output is kept. P0 found it gives the whole frame on frame-filling photos
    # and no output gives the pile on tray photos, so P2 measures exactly that baseline.
    seg_prompt: str = "box"
    seg_mask_select: str = "multi3"
    # Ticket ML-2 (D18, owner 2026-10-01): a mask covering less of the photo than this is treated as empty
    # and the whole photo is used. Half the smallest owner-accepted base mask (0.1657), not the plan's half
    # of the smallest judge-accepted pretrained mask, which is 0.45 because only frame-filling masks pass.
    seg_min_area_frac: float = 0.083
    # Ticket ML-2 P3 (plan §2.2, §6): how the bean region is found, the same way at both ends.
    # "tray_heuristic": crop_tray -- training read the data/cropped pools the crop stage made (both retired
    # in ticket ML-3 P2b), serving runs infer.crop_to_bean_region. "segment": the segmenter below
    # (segment_beans.py) -- training reads the data/segcropped pools the segcrop stage made, serving segments
    # the photo live. A checkpoint written before this field restores as "tray_heuristic", so every shipped
    # model keeps its own path; that is why the default stays here while params.yaml rests at "segment".
    crop_method: str = "tray_heuristic"
    # The segmenter: pretrained L0 weights (under models_pretrained/) and, from P5, a fine-tuned mask decoder
    # (repo-relative; "" = the pretrained decoder). Both sha256s are recorded so a checkpoint names the exact
    # segmenter its patches came from, and loading refuses any other.
    seg_weights: str = "efficientvit_sam/efficientvit_sam_l0.pt"
    seg_weights_sha256: str = ""
    seg_decoder: str = ""
    seg_decoder_sha256: str = ""
    # D4: the crop keeps >= this share of the bean region (a quantile trim of (1 - keep) / 4 per side).
    seg_keep_frac: float = 0.95
    # D17: a patch box with less bean region than this share is rejected and redrawn; after
    # patch_max_attempts_factor x n candidates the best rejected boxes top the photo up to n. Only on the
    # "segment" path, and never on a D18 fallback photo (no mask: every box is accepted).
    patch_min_bean_share: float = 0.80
    patch_max_attempts_factor: int = 20
    # Ticket ML-2 P5 (D7 (a), plan §8): the decoder-only fine-tuning recipe, a block read and checked key by
    # key only by coffeecv/seg_finetune.py (FtParams). Empty = FtParams' defaults.
    seg_ft: dict = field(default_factory=dict)
    # Ticket ML-5 P5 (D11-D13): the segmenter dataset's pool-draw and split seed and its split shares, a
    # block read and checked only by coffeecv/seg_dataset.py (Params). No training run reads it.
    seg_dataset: dict = field(default_factory=dict)
    # Ticket ML-8 D4 (ADR 0025): the frozen head fit's augmentation, a block read and checked only by
    # coffeecv/fit_frozen_head.py (HeadAugment): `views` augmented views of every training patch, drawn per run
    # and held in memory as embeddings, through the albumentations pipelines `patch` (on the stored patch) and
    # `input` (on the patch resized to patch_resize). Empty = no augmentation, as every head before ML-8.
    head_augment: dict = field(default_factory=dict)
    # ADR 0027: augmentation groups both fits share, {name: [transforms]}. seg_ft.augment and head_augment's
    # pipelines name them as {use: <name>}; from_params_yaml expands them in place (augment.expand), so the two
    # blocks above always hold the transforms that run, and a recorded config never holds a reference.
    augment: dict = field(default_factory=dict)
    # Ticket ML-5 P6/P10 (D22): the seed that draws, per photo, which side of the blind paired review shows which
    # model's mask (coffeecv/seg_review.py paired). No training run reads it.
    seg_paired_seed: int = 0

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
        if isinstance(values.get("train_capture_dirs"), list):
            values["train_capture_dirs"] = tuple(values["train_capture_dirs"])
        # Imported here: serving reads a model's card, never params.yaml, and a release ships no coffeecv/augment.py
        # (webapp/deploy/release_manifest.py LIBRARY, checked by tests/test_release_manifest.py).
        from coffeecv.augment import expand
        groups = values.get("augment") or {}
        if "augment" in (values.get("seg_ft") or {}):
            values["seg_ft"] = {**values["seg_ft"],
                                "augment": expand(values["seg_ft"]["augment"], groups, where="seg_ft.augment")}
        if values.get("head_augment"):
            values["head_augment"] = {**values["head_augment"], **{
                k: expand(values["head_augment"][k], groups, where=f"head_augment.{k}")
                for k in ("patch", "input") if k in values["head_augment"]}}
        return cls(**values)

    def resolve_paths(self) -> tuple[list[Path], Path]:
        """(capture dirs, classes file). Refuses the segmenter's data/segcropped pools under any crop method
        but "segment" (ticket ML-3 P2b): sampled as tray-heuristic crops they would skip their masks' patch
        rule (D17). Nothing would fail; the patches would just be wrong."""
        for d in self.train_capture_dirs:
            if Path(d).parent.name == "segcropped" and self.crop_method != "segment":
                raise ValueError(f"capture dir {d} needs crop_method 'segment', not {self.crop_method!r}")
        return [REPO_ROOT / d for d in self.train_capture_dirs], REPO_ROOT / self.classes_file


def set_seed(seed: int) -> None:
    import torch     # here, not at the top: the web service imports this module and no torch (ML-7 P8)
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
    import torch
    import torchvision

    return {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "git_commit": git_commit,
        "argv": sys.argv,
    }


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)
