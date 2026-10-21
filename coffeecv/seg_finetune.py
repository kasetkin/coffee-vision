"""Ticket ML-2 P5: decoder-only fine-tuning of the segmenter (D7 (a), plan §8); ticket ML-5 P8: on the owner's
labels of a labelling session, with point prompts (D15); ticket ML-6 P5: augmented as it trains (D12). One DVC
stage:

    python -m coffeecv.seg_finetune train --labels pass3 --seed 123 --name xl0_v3 --threads 8
                                                        # XL0-v3's stage (ML-6 P6)

Labels come from data/ml5_labels/<session>/labels.csv (seg_review.py, the owner's verdicts): every accepted
positive and every negative (an empty label, D14) of the training and validation splits. Dropped photos are left
out, a pending one stops the run, and a test photo never appears there (D4). Roles, by split (D8):

    train       training positives
    val         validation positives: checkpoint selection only (mean IoU)
    neg_train   training negatives
    neg_select  validation negatives: checkpoint selection only (share empty or tiny, ML-2 D18 threshold)

ML-2's fine-tunes (ft_s{42,123,7}, code at a1e73e3) and ML-5's XL0-v1 and v2 (an embedding cache of 4 dihedral
views, code at 46ec14c) left dvc.yaml on 2026-10-21; they are in git history.

Data (ML-6 D12, docs/adr/0018): nothing is cached, and nothing augmented is written. At the start of a run each
training photo (train, neg_train) is checked against labels.csv's sha256, resized to the encoder's frame by
serving's own resize (SamResize, the first step of model.transform: the long side to 512 for L0, 1024 for XL0,
which the encoder's transform then leaves as it is), and held in memory with
its label at the same size (INTER_AREA, >= 0.5). Each sample is augmented when it is drawn, by seg_ft.augment
(albumentations transforms built by name, strictly by coffeecv.augment since ML-8 P2, seeded from the run's seed; a spatial transform moves the label with
the photo, and a warp fills the label's exposed border with IGNORE), its label padded with IGNORE as SamPad pads
the image, and encoded by the frozen encoder. The validation photos are not augmented: they are encoded once per
run at full resolution, as at serving, and held in memory, so checkpoint selection is deterministic.

Training (seg_ft in params.yaml): the image encoder and prompt encoder are frozen, the mask decoder trains. The
prompt is the serving prompt, one whole-image box, and the loss is on the output serving keeps (seg_mask_select,
multi3), upsampled to the encoder's frame. A point_share of the batches add 1 to max_points points to the box
(ML-5 D15, SAM's recipe): drawn uniformly from the error region of that sample's box-only prediction, an include
point where the label is bean and an exclude point where it is not, and decoded through the same output (D23),
as a redraw on the review page decodes them. Checkpoint selection stays box-only. The loss is focal_weight x focal
+ dice_weight x soft dice (smooth 1) + iou_weight x MSE of the IoU head against the IoU it achieves. Each epoch
shows every train positive once, augmented; each batch holds neg_share negatives, drawn with replacement. AdamW
with cosine decay to lr_min; the epoch with the best selection score is kept: (mean val IoU + neg_select
empty-or-tiny share) / 2, tiny meaning under seg_ft.select_min_area_frac (the fine-tune's own; params.yaml
seg_min_area_frac is serving's, ML-5 P11). Stops after `patience` epochs without a better score.

Out: models/seg/<name>.pt ({"mask_decoder", "base_weights_sha256", ...}) + .json (params with the pipeline,
albumentations' version, label session and label-set hash, history, the chosen epoch).
segment_beans.SegParams(weights=..., decoder=...) loads it over the pretrained weights it was fine-tuned from,
which the card's base_weights names (segment_beans.decoder_base).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffeecv import augment, model_io
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.repo_files import read_csv, sha256_file
from coffeecv.sam_loader import build_sam, variant_of

LABELS_ROOT = REPO_ROOT / "data" / "ml5_labels"     # seg_review.MASK_ROOT: <session>/labels.csv
MODEL_DIR = REPO_ROOT / "models" / "seg"
LOW = 256                         # the decoder's low-res logits
PROMPT_FRAME = 1024               # SAM's prompt coordinates (model.image_size[0])
IGNORE = 255
ROLES = {("accepted", "train"): "train", ("accepted", "validation"): "val",
         ("negative", "train"): "neg_train", ("negative", "validation"): "neg_select"}
TRAIN_ROLES, SELECT_ROLES = ("train", "neg_train"), ("val", "neg_select")


@dataclass(frozen=True)
class FtParams:
    weights: str                  # the pretrained base under models_pretrained/ (params.yaml); its variant sets the frame
    augment: tuple[dict, ...] = ()        # ML-6 D12, D16: albumentations transforms, each {transform, p, args}
    epochs: int = 40
    patience: int = 8
    batch_size: int = 8
    lr: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    focal_weight: float = 20.0
    dice_weight: float = 1.0
    iou_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    neg_share: float = 0.5
    point_share: float = 0.0      # share of batches that add points to the box (D15)
    max_points: int = 3
    select_min_area_frac: float = 0.083   # selection's tiny-mask rule on the validation negatives (ML-2 D18)

    @classmethod
    def from_config(cls, cfg: RunConfig) -> "FtParams":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg.seg_ft) - known)
        if unknown:
            raise ValueError(f"params.yaml seg_ft sets keys seg_finetune does not define: {', '.join(unknown)}")
        if "weights" not in cfg.seg_ft:
            raise ValueError("params.yaml seg_ft.weights must name the pretrained base (no default: it is the one "
                             "place the fine-tune's base is written)")
        p = cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.seg_ft.items()})
        if not 0 <= p.point_share <= 1 or p.max_points < 1:
            raise ValueError("seg_ft.point_share must be in [0, 1] and max_points at least 1")
        variant_of(p.weights)
        build_augment(p.augment, seed=0)
        return p


# ---------------------------------------------------------------- augmentation (ML-6 D12, D16)

def _fill_mask_is_ignore(cls, args: dict) -> str | None:
    """The fine-tune's own rule on top of coffeecv.augment's: a transform that fills a mask (a warp's exposed
    border) fills it with IGNORE."""
    if "fill_mask" in inspect.signature(cls).parameters and args.get("fill_mask") != IGNORE:
        return f"must set fill_mask: {IGNORE} (the label's ignore value)"
    return None


def compose_augment(spec, seed: int):
    """seg_ft.augment as an albumentations Compose, built strictly by coffeecv.augment (ML-8 P2)."""
    return augment.build(spec, where="seg_ft.augment", seed=seed, check=_fill_mask_is_ignore)


def build_augment(spec, seed: int) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """(photo HxWx3 uint8, label HxW uint8) -> the same, augmented by `spec` (seg_ft.augment) in a stream seeded
    by `seed`: a spatial transform moves the label with the photo, a pixel transform leaves it as it is."""
    if not spec:
        return lambda rgb, label: (rgb, label)
    pipe = compose_augment(spec, seed)

    def augment_sample(rgb: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = pipe(image=rgb, mask=label)
        return out["image"], out["mask"]
    return augment_sample


# ---------------------------------------------------------------- geometry

def frame_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """SAM's ResizeLongestSide.get_preprocess_shape."""
    s = long_side / max(h, w)
    return int(h * s + 0.5), int(w * s + 0.5)


def encoder_frame(model) -> int:
    """The side of the padded square the encoder sees, and the frame labels and the loss live in: 512 for L*,
    1024 for XL* (SamResize/SamPad at model.image_size[1])."""
    return int(model.image_size[1])


def label_at(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A full-resolution bool mask -> uint8 at `size` (h, w): 1 bean, 0 not."""
    return (cv2.resize(mask.astype(np.float32), size[::-1], interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)


def pad_label(label: np.ndarray, frame: int) -> np.ndarray:
    """A label at the photo's resized size -> frame x frame, IGNORE on the padding (bottom/right, as SamPad pads
    the image)."""
    out = np.full((frame, frame), IGNORE, np.uint8)
    out[:label.shape[0], :label.shape[1]] = label
    return out


def label_in_frame(mask: np.ndarray, frame: int) -> np.ndarray:
    """A full-resolution bool mask -> uint8 frame x frame: 1 bean, 0 not, IGNORE on the padding."""
    return pad_label(label_at(mask, frame_size(*mask.shape, frame)), frame)


def whole_box(h: int, w: int) -> np.ndarray:
    """The serving prompt, one whole-image box [x0, y0, x1, y1], in the prompt frame (BeanSegmenter._decode)."""
    fh, fw = frame_size(h, w, PROMPT_FRAME)
    return np.array([0.0, 0.0, (w - 1) * fw / w, (h - 1) * fh / h], np.float32)


# ---------------------------------------------------------------- the label set, held in memory

def label_rows(session: str) -> list[dict]:
    """A labelling session's labels.csv rows that train or select, each with its role (ROLES). Dropped photos are
    left out; a pending photo or a test photo is an error."""
    rows = read_csv(LABELS_ROOT / session / "labels.csv")
    if bad := [r["path"] for r in rows if r["split"] not in ("train", "validation")]:
        raise ValueError(f"session {session}: {len(bad)} photos outside training and validation, e.g. {bad[0]}")
    if pending := [r["path"] for r in rows if r["status"] == "pending"]:
        raise ValueError(f"session {session}: {len(pending)} photos still pending, e.g. {pending[0]}")
    return [{**r, "role": ROLES[r["status"], r["split"]]} for r in rows if r["status"] != "dropped"]


def label_set_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["id"]):
        h.update(f"{r['id']} {r['photo_sha256']} {r['mask_sha256']} {r['role']}\n".encode())
    return h.hexdigest()


def label_set(session: str, rows: list[dict]) -> dict:
    """What the card records of the labels: the session, the label-set hash and the photos per role."""
    return {"session": session, "label_set_sha256": label_set_sha256(rows), "n": len(rows),
            "roles": {k: sum(r["role"] == k for r in rows) for k in (*TRAIN_ROLES, *SELECT_ROLES)}}


def load_pair(r: dict) -> tuple[np.ndarray, np.ndarray]:
    """(photo, bool label) of a labels.csv row at full resolution, the photo checked against its sha256."""
    photo = REPO_ROOT / r["path"]
    if sha256_file(photo) != r["photo_sha256"]:
        raise ValueError(f"{r['path']}: sha256 differs from labels.csv")
    rgb = load_rgb_image(photo)
    mask = np.array(Image.open(REPO_ROOT / r["mask"])) > 0
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"{r['id']}: label {mask.shape} does not match photo {rgb.shape[:2]}")
    return rgb, mask


class TrainSet:
    """The training photos (train, neg_train) and their labels, loaded once and held in memory at the encoder's
    frame; `pos` and `neg` index them. `sample` augments one as it is drawn. The resize is serving's
    (model_io.sam_resize to `frame`), so the encoder's input port leaves the photo as it is."""

    def __init__(self, rows: list[dict], frame: int):
        self.frame = frame
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.pos, self.neg = [], []
        for r in rows:
            if r["role"] not in TRAIN_ROLES:
                continue
            rgb, mask = load_pair(r)
            small = np.ascontiguousarray(model_io.sam_resize(rgb, frame))
            (self.pos if r["role"] == "train" else self.neg).append(len(self.items))
            self.items.append((small, label_at(mask, small.shape[:2]), whole_box(*rgb.shape[:2])))

    def sample(self, i: int, aug) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """(photo at the frame, augmented by `aug` unless None; its label padded to frame x frame; the whole-image
        box 1 x 4 in the prompt frame, the photo's own, which a spatial transform keeps)."""
        rgb, label, box = self.items[i]
        if aug is not None:
            shape = rgb.shape
            rgb, label = aug(rgb, label)
            if rgb.shape != shape or label.shape != shape[:2]:
                raise ValueError(f"seg_ft.augment changed a photo's shape {shape} -> {rgb.shape}: the box and the "
                                 "padding would no longer fit it")
        return rgb, pad_label(label, self.frame), torch.from_numpy(box)[None]


@torch.no_grad()
def encode(model, rgb: np.ndarray) -> torch.Tensor:
    """The frozen encoder's embedding (1 x 256 x 64 x 64) of a photo, as at serving (model_io.sam_input)."""
    return model.image_encoder(torch.from_numpy(model_io.sam_input(rgb, encoder_frame(model))[0]))


class Selection:
    """The validation photos (val, neg_select), unaugmented, encoded once at full resolution as at serving and
    held in memory: `val` and `neg`, lists of (embedding, label frame x frame, box)."""

    def __init__(self, rows: list[dict], model):
        frame = encoder_frame(model)
        self.val, self.neg = [], []
        for r in rows:
            if r["role"] not in SELECT_ROLES:
                continue
            rgb, mask = load_pair(r)
            item = (encode(model, rgb), torch.from_numpy(label_in_frame(mask, frame)),
                    torch.from_numpy(whole_box(*rgb.shape[:2]))[None])
            (self.val if r["role"] == "val" else self.neg).append(item)


def batch_split(data: TrainSet, p: FtParams) -> tuple[int, int]:
    """(positives, negatives) per batch."""
    n_neg = round(p.batch_size * p.neg_share) if data.neg else 0
    return p.batch_size - n_neg, n_neg


def epoch_batches(data: TrainSet, p: FtParams, rng: np.random.Generator, aug) -> Iterator[tuple[list, bool]]:
    """One epoch, drawn and augmented as it goes: (samples, with_points) a step, each sample `TrainSet.sample`'s.
    Every training positive once in a random order; negatives drawn with replacement; a point_share of the
    batches add points."""
    n_pos, n_neg = batch_split(data, p)
    order = rng.permutation(data.pos)
    for s in range(math.ceil(len(data.pos) / n_pos)):
        batch = list(order[s * n_pos:(s + 1) * n_pos]) + (list(rng.choice(data.neg, n_neg)) if n_neg else [])
        with_points = bool(rng.random() < p.point_share)
        yield [data.sample(int(i), aug) for i in batch], with_points


# ---------------------------------------------------------------- the decoder pass and the loss

def decode(model, emb: torch.Tensor, box: torch.Tensor, k: int,
           points: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Output k's logits in the encoder's padded frame (encoder_frame) and its predicted IoU (scalar), for one image.
    `points` (coordinates 1 x n x 2 in the prompt frame, labels 1 x n: 1 include, 0 exclude) join the box, as
    SamPredictor.predict_torch passes them (the review page's redraw, segment_beans.decode_points)."""
    sparse, dense = model.prompt_encoder(points=points, boxes=box, masks=None)
    low, iou = model.mask_decoder(image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
                                  sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                  multimask_output=k > 0)
    j = k - 1 if k > 0 else 0
    frame = encoder_frame(model)
    up = F.interpolate(low[:, j:j + 1], (frame, frame), mode="bilinear", align_corners=False)
    return up[0, 0], iou[0, j]


def frame_to_prompt(xy: np.ndarray, frame: int) -> np.ndarray:
    """Pixel (x, y) in the encoder's frame -> SAM's prompt frame (pixel centres; both frames scale the photo's
    long side, to `frame` and to PROMPT_FRAME)."""
    return (np.asarray(xy, np.float32) + 0.5) * (PROMPT_FRAME / frame)


def error_points(logits: torch.Tensor, label: torch.Tensor, n: int, rng: np.random.Generator):
    """`n` points drawn uniformly from where the prediction (logits > 0) disagrees with the label on the valid
    pixels: include (1) where the label is bean, exclude (0) where it is not. None when there is no error.
    Returns (coordinates 1 x n x 2 in the prompt frame, labels 1 x n)."""
    valid = label != IGNORE
    target = label == 1
    ys, xs = torch.nonzero(valid & ((logits > 0) != target), as_tuple=True)
    if len(ys) == 0:
        return None
    pick = rng.integers(len(ys), size=n)
    xy = np.stack([xs[pick].numpy(), ys[pick].numpy()], axis=1)
    lab = target[ys[pick], xs[pick]].to(torch.int)
    return torch.from_numpy(frame_to_prompt(xy, label.shape[-1]))[None], lab[None]


def output_index(mask_select: str) -> int:
    """0 = the single-mask token, 1-3 = multi1..multi3. best_iou has no fixed output to train."""
    if mask_select == "single":
        return 0
    if mask_select.startswith("multi"):
        return int(mask_select[-1])
    raise ValueError(f"seg_mask_select {mask_select!r}: fine-tuning needs a fixed output")


def losses(logits: torch.Tensor, iou_pred: torch.Tensor, label: torch.Tensor, p: FtParams) -> dict:
    """Focal + soft dice on the valid (non-padding) pixels, MSE of the IoU head against the achieved IoU."""
    valid = label != IGNORE
    x = logits[valid]
    t = (label[valid] == 1).float()
    prob = torch.sigmoid(x)
    ce = F.binary_cross_entropy_with_logits(x, t, reduction="none")
    pt = prob * t + (1 - prob) * (1 - t)
    alpha = p.focal_alpha * t + (1 - p.focal_alpha) * (1 - t)
    focal = (alpha * (1 - pt) ** p.focal_gamma * ce).mean()
    dice = 1 - (2 * (prob * t).sum() + 1) / (prob.sum() + t.sum() + 1)
    with torch.no_grad():
        iou_true = hard_iou(x > 0, t > 0)
    mse = (iou_pred - iou_true) ** 2
    total = p.focal_weight * focal + p.dice_weight * dice + p.iou_weight * mse
    return {"total": total, "focal": focal.detach(), "dice": dice.detach(), "iou_mse": mse.detach()}


def hard_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU of two bool masks; 1 when both are empty (a correct empty mask on a negative)."""
    union = (pred | target).sum()
    return (pred & target).sum() / union if union > 0 else torch.tensor(1.0)


@torch.no_grad()
def evaluate(model, sel: Selection, k: int, min_area_frac: float) -> dict:
    """Selection metrics, unaugmented: mean IoU on val, share of neg_select empty or tiny."""
    ious = []
    for emb, lab, box in sel.val:
        logits, _ = decode(model, emb, box, k)
        valid = lab != IGNORE
        ious.append(float(hard_iou(logits[valid] > 0, lab[valid] == 1)))
    empty = []
    for emb, lab, box in sel.neg:
        logits, _ = decode(model, emb, box, k)
        empty.append(float((logits[lab != IGNORE] > 0).float().mean()) < min_area_frac)
    miou = float(np.mean(ious)) if ious else math.nan
    neg = float(np.mean(empty)) if empty else math.nan
    return {"val_miou": miou, "neg_select_empty": neg, "score": float(np.nanmean([miou, neg]))}


def train(seed: int, cfg: RunConfig, p: FtParams, rows: list[dict]) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    predictor, wsha = build_sam(p.weights)
    model = predictor.model
    frame = encoder_frame(model)
    t0 = time.perf_counter()
    data, sel = TrainSet(rows, encoder_frame(model)), Selection(rows, model)
    print(f"loaded {len(data.pos)} positives and {len(data.neg)} negatives to train on, encoded {len(sel.val)} + "
          f"{len(sel.neg)} validation photos: {time.perf_counter() - t0:.0f} s", flush=True)
    aug = build_augment(p.augment, seed)
    model.mask_decoder.train()
    for q in model.mask_decoder.parameters():
        q.requires_grad_(True)
    k = output_index(cfg.seg_mask_select)
    n_pos, n_neg = batch_split(data, p)
    steps = math.ceil(len(data.pos) / n_pos)
    opt = torch.optim.AdamW(model.mask_decoder.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p.epochs * steps, eta_min=p.lr_min)
    model.mask_decoder.eval()
    history = [{"epoch": 0, **evaluate(model, sel, k, p.select_min_area_frac)}]
    print(f"seed {seed}: batch {n_pos}+{n_neg}, {steps} steps/epoch, point batches {p.point_share}; augment "
          f"{[t['transform'] for t in p.augment]}; epoch 0 {history[0]}", flush=True)
    best = dict(history[0], state={n: t.clone() for n, t in model.mask_decoder.state_dict().items()})
    for epoch in range(1, p.epochs + 1):
        model.mask_decoder.train()
        t0 = time.perf_counter()
        sums = {"total": 0.0, "focal": 0.0, "dice": 0.0, "iou_mse": 0.0}
        point_steps = 0
        for s, (samples, with_points) in enumerate(epoch_batches(data, p, rng, aug), 1):
            point_steps += with_points
            opt.zero_grad()
            for rgb, lab, box in samples:
                lab = torch.from_numpy(lab)
                emb = encode(model, rgb)
                points = None
                if with_points:
                    with torch.no_grad():
                        box_only, _ = decode(model, emb, box, k)
                    points = error_points(box_only, lab, int(rng.integers(1, p.max_points + 1)), rng)
                logits, iou = decode(model, emb, box, k, points)
                ls = losses(logits, iou, lab, p)
                (ls["total"] / len(samples)).backward()
                for key in sums:
                    sums[key] += float(ls[key].detach()) / len(samples) / steps
            opt.step()
            sched.step()
            if s % 10 == 0:
                print(f"  epoch {epoch} step {s}/{steps}  {time.perf_counter() - t0:.0f} s", flush=True)
        model.mask_decoder.eval()
        ev = evaluate(model, sel, k, p.select_min_area_frac)
        row = {"epoch": epoch, **{f"train_{a}": round(b, 5) for a, b in sums.items()}, "point_steps": point_steps, **ev,
               "lr": sched.get_last_lr()[0], "seconds": round(time.perf_counter() - t0, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if ev["score"] > best["score"]:
            best = dict(ev, epoch=epoch, state={n: t.clone() for n, t in model.mask_decoder.state_dict().items()})
        elif epoch - best["epoch"] >= p.patience:
            print(f"no better score for {p.patience} epochs; stopping", flush=True)
            break
    return {"seed": seed, "best": {a: b for a, b in best.items() if a != "state"}, "state": best["state"],
            "history": history, "weights_sha256": wsha, "k": k, "frame": frame}


def write_model(res: dict, cfg: RunConfig, p: FtParams, labels: dict, name: str) -> Path:
    """The decoder and its card; `labels` is `label_set`'s."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"{name}.pt"
    torch.save({"mask_decoder": res["state"], "base_weights_sha256": res["weights_sha256"],
                "output": cfg.seg_mask_select, "prompt": cfg.seg_prompt}, out)
    card = {"name": name, "seed": res["seed"], "labels": labels["session"], "label_set": labels,
            "base_weights": p.weights, "base_weights_sha256": res["weights_sha256"],
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "seg_ft": asdict(p),
            "seg_prompt": cfg.seg_prompt, "seg_mask_select": cfg.seg_mask_select, "frame": res["frame"],
            "albumentations": augment.version(),
            "chosen": res["best"], "history": res["history"], "threads": torch.get_num_threads()}
    out.with_suffix(".json").write_text(json.dumps(card, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train"])
    ap.add_argument("--labels", required=True, help="the labelling session (data/ml5_labels/<session>)")
    ap.add_argument("--seed", type=int, required=True, help="the fine-tune seed")
    ap.add_argument("--name", required=True, help="the model's name, models/seg/<name>.pt")
    ap.add_argument("--threads", type=int, required=True, help="torch.set_num_threads")
    ap.add_argument("--lr", type=float, help="overrides params.yaml seg_ft.lr for this run, one candidate's own stage "
                                             "(ML-9 D8); the card records the rate used")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    cfg = RunConfig.from_params_yaml()
    p = FtParams.from_config(cfg)
    if args.lr is not None:
        if not args.lr > 0:
            ap.error("--lr must be positive")
        p = replace(p, lr=args.lr)
    rows = label_rows(args.labels)
    res = train(args.seed, cfg, p, rows)
    out = write_model(res, cfg, p, label_set(args.labels, rows), args.name)
    print(f"wrote {out.relative_to(REPO_ROOT)}: epoch {res['best']['epoch']}, score {res['best']['score']:.4f}")


if __name__ == "__main__":
    main()
