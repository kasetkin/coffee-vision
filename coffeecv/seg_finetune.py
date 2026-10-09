"""Ticket ML-2 P5: decoder-only fine-tuning of the segmenter (D7 (a), plan §8); ticket ML-5 P8: on the owner's
labels of a labelling session, with point prompts (D15). Two DVC stages:

    python -m coffeecv.seg_finetune cache --labels pass1 --threads 4     # seg_ml5_cache@pass1
    python -m coffeecv.seg_finetune train --labels pass1 --seed 123 --name xl0_v1 --threads 4
                                                                          # seg_ml5_finetune@xl0_v1

Labels come from data/ml5_labels/<session>/labels.csv (seg_review.py, the owner's verdicts): every accepted
positive and every negative (an empty label, D14) of the training and validation splits. Dropped photos are left
out, a pending one stops the cache, and a test photo never appears there (D4). Roles, by split (D8):

    train       training positives
    val         validation positives: checkpoint selection only (mean IoU)
    neg_train   training negatives
    neg_select  validation negatives: checkpoint selection only (share empty or tiny, ML-2 D18 threshold)

ML-2's fine-tune (labels/ml2 lists, data/seg_labels, its random 20% of negative groups held for selection) made
models/seg/ft_s{42,123,7}.pt; its stages are frozen in dvc.yaml and the code that ran them is at a1e73e3.

Cache: for each view in seg_ft.views (dihedral: id, hflip, vflip, rot180, rot90, rot270, transpose,
antitranspose), the photo and its label are transformed at full resolution, the photo goes through the frozen
encoder of seg_ft.weights exactly as at serving (resize to the encoder's frame on the long side, normalise, pad),
and the embedding is stored in fp16 with the label resized into the same frame (255 = padding, ignored). The frame
follows the weights' variant (ticket ML-5 P4): 512 for L0, 1024 for XL0. data/seg_cache/<view>/emb.npy
(N, 256, 64, 64) and label.npy (N, frame, frame) are memory-mapped; index.csv holds the rows and roles.

Training (seg_ft in params.yaml): the image encoder and prompt encoder are frozen, the mask decoder trains. The
prompt is the serving prompt, one whole-image box, and the loss is on the output serving keeps (seg_mask_select,
multi3), upsampled to the encoder's frame. A point_share of the batches add 1 to max_points points to the box
(ML-5 D15, SAM's recipe): drawn uniformly from the error region of that photo's box-only prediction, an include
point where the label is bean and an exclude point where it is not, and decoded through the same output (D23),
as a redraw on the review page decodes them. Checkpoint selection stays box-only. The loss is focal_weight x focal + dice_weight x soft dice (smooth 1) + iou_weight x MSE
of the IoU head against the IoU it achieves. Each epoch shows every train positive once, in a random view; each
batch holds neg_share negatives, drawn with replacement. AdamW with cosine decay to lr_min; the epoch with the
best selection score is kept: (mean val IoU + neg_select empty-or-tiny share) / 2, tiny meaning under
seg_ft.select_min_area_frac (the fine-tune's own; params.yaml seg_min_area_frac is serving's, ML-5 P11). Stops
after `patience` epochs without a better score.

Out: models/seg/<name>.pt ({"mask_decoder", "base_weights_sha256", ...}) + .json (params, label session and
label-set hash, history, the chosen epoch). segment_beans.SegParams(weights=..., decoder=...) loads it over the pretrained
weights it was fine-tuned from, which the card's base_weights names (segment_beans.decoder_base).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.repo_files import read_csv, sha256_file
from coffeecv.sam_loader import build_sam, variant_of

LABELS_ROOT = REPO_ROOT / "data" / "ml5_labels"     # seg_review.MASK_ROOT: <session>/labels.csv
CACHE_ROOT = REPO_ROOT / "data" / "seg_cache_ml5"   # <session>/
MODEL_DIR = REPO_ROOT / "models" / "seg"
LOW = 256                         # the decoder's low-res logits
PROMPT_FRAME = 1024               # SAM's prompt coordinates (model.image_size[0])
IGNORE = 255
EMB_SHAPE = (256, 64, 64)
ROLES = {("accepted", "train"): "train", ("accepted", "validation"): "val",
         ("negative", "train"): "neg_train", ("negative", "validation"): "neg_select"}
VIEWS = {
    "id": lambda a: a,
    "hflip": lambda a: a[:, ::-1],
    "vflip": lambda a: a[::-1],
    "rot180": lambda a: a[::-1, ::-1],
    "rot90": lambda a: np.rot90(a, 1),
    "rot270": lambda a: np.rot90(a, 3),
    "transpose": lambda a: np.swapaxes(a, 0, 1),
    "antitranspose": lambda a: np.rot90(np.swapaxes(a, 0, 1), 2),
}


@dataclass(frozen=True)
class FtParams:
    weights: str                  # the pretrained base under models_pretrained/ (params.yaml); its variant sets the frame
    views: tuple[str, ...] = ("id", "hflip", "vflip", "rot180")
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
        if bad := [v for v in p.views if v not in VIEWS]:
            raise ValueError(f"seg_ft.views: unknown {bad}; known {list(VIEWS)}")
        if p.views[0] != "id":
            raise ValueError("seg_ft.views must start with id (selection reads the identity view)")
        if not 0 <= p.point_share <= 1 or p.max_points < 1:
            raise ValueError("seg_ft.point_share must be in [0, 1] and max_points at least 1")
        variant_of(p.weights)
        return p


# ---------------------------------------------------------------- geometry shared by cache and training

def frame_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """SAM's ResizeLongestSide.get_preprocess_shape."""
    s = long_side / max(h, w)
    return int(h * s + 0.5), int(w * s + 0.5)


def encoder_frame(model) -> int:
    """The side of the padded square the encoder sees, and the frame labels and the loss live in: 512 for L*,
    1024 for XL* (SamResize/SamPad at model.image_size[1])."""
    return int(model.image_size[1])


def label_in_frame(mask: np.ndarray, frame: int) -> np.ndarray:
    """A full-resolution bool mask -> uint8 frame x frame: 1 bean, 0 not, IGNORE on the padding (bottom/right,
    as SamPad pads the image)."""
    fh, fw = frame_size(*mask.shape, frame)
    small = cv2.resize(mask.astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA) >= 0.5
    out = np.full((frame, frame), IGNORE, np.uint8)
    out[:fh, :fw] = small
    return out


def whole_box(h: int, w: int) -> np.ndarray:
    """The serving prompt, one whole-image box [x0, y0, x1, y1], in the prompt frame (BeanSegmenter._decode)."""
    fh, fw = frame_size(h, w, PROMPT_FRAME)
    return np.array([0.0, 0.0, (w - 1) * fw / w, (h - 1) * fh / h], np.float32)


# ---------------------------------------------------------------- the label set and the cache

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


@torch.inference_mode()
def build_cache(session: str, p: FtParams) -> None:
    rows = label_rows(session)
    root = CACHE_ROOT / session
    predictor, wsha = build_sam(p.weights)
    model = predictor.model
    frame = encoder_frame(model)
    n = len(rows)
    t0 = time.perf_counter()
    arrays = {}
    for v in p.views:
        d = root / v
        d.mkdir(parents=True, exist_ok=True)
        arrays[v] = (np.lib.format.open_memmap(d / "emb.npy", "w+", np.float16, (n, *EMB_SHAPE)),
                     np.lib.format.open_memmap(d / "label.npy", "w+", np.uint8, (n, frame, frame)), [])
    for i, r in enumerate(rows):
        photo = REPO_ROOT / r["path"]
        if sha256_file(photo) != r["photo_sha256"]:
            raise ValueError(f"{r['path']}: sha256 differs from labels.csv")
        rgb = load_rgb_image(photo)
        mask = np.array(Image.open(REPO_ROOT / r["mask"])) > 0
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"{r['id']}: label {mask.shape} does not match photo {rgb.shape[:2]}")
        for v in p.views:
            im = np.ascontiguousarray(VIEWS[v](rgb))
            m = np.ascontiguousarray(VIEWS[v](mask))
            emb, lab, meta = arrays[v]
            emb[i] = model.image_encoder(model.transform(im).unsqueeze(0))[0].numpy().astype(np.float16)
            lab[i] = label_in_frame(m, frame)
            meta.append({"h": m.shape[0], "w": m.shape[1]})
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}  {time.perf_counter() - t0:.0f}s", flush=True)
    for v, (emb, lab, meta) in arrays.items():
        emb.flush()
        lab.flush()
        (root / v / "meta.json").write_text(json.dumps(meta) + "\n")
    fields_ = ["id", "source", "split", "role", "path", "photo_sha256", "mask_sha256"]
    with open(root / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["i", *fields_])
        wr.writeheader()
        for i, r in enumerate(rows):
            wr.writerow({"i": i, **{k: r[k] for k in fields_}})
    (root / "cache.json").write_text(json.dumps({
        "labels": session, "views": list(p.views), "n": n, "weights": p.weights, "weights_sha256": wsha, "frame": frame,
        "label_set_sha256": label_set_sha256(rows),
        "roles": {k: sum(r["role"] == k for r in rows) for k in ("train", "val", "neg_train", "neg_select")},
        "threads": torch.get_num_threads()}, indent=2) + "\n")


class Cache:
    def __init__(self, session: str):
        root = CACHE_ROOT / session
        info = json.loads((root / "cache.json").read_text())
        self.info = info
        self.rows = read_csv(root / "index.csv")
        self.views = info["views"]
        self.emb = {v: np.load(root / v / "emb.npy", mmap_mode="r") for v in self.views}
        self.label = {v: np.load(root / v / "label.npy", mmap_mode="r") for v in self.views}
        self.meta = {v: json.loads((root / v / "meta.json").read_text()) for v in self.views}
        self.frame = int(self.label["id"].shape[-1])

    def role(self, name: str) -> list[int]:
        return [int(r["i"]) for r in self.rows if r["role"] == name]

    def sample(self, i: int, view: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(embedding 1x256x64x64 float32, label frame x frame uint8, box 1x4 in the prompt frame)."""
        m = self.meta[view][i]
        return (torch.from_numpy(np.asarray(self.emb[view][i], np.float32))[None],
                torch.from_numpy(np.array(self.label[view][i])),
                torch.from_numpy(whole_box(m["h"], m["w"]))[None])


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
def evaluate(model, cache: Cache, k: int, min_area_frac: float) -> dict:
    """Selection metrics on the identity view: mean IoU on val, share of neg_select empty or tiny."""
    ious = []
    for i in cache.role("val"):
        emb, lab, box = cache.sample(i, "id")
        logits, _ = decode(model, emb, box, k)
        valid = lab != IGNORE
        ious.append(float(hard_iou(logits[valid] > 0, lab[valid] == 1)))
    empty = []
    for i in cache.role("neg_select"):
        emb, lab, box = cache.sample(i, "id")
        logits, _ = decode(model, emb, box, k)
        empty.append(float((logits[lab != IGNORE] > 0).float().mean()) < min_area_frac)
    miou = float(np.mean(ious)) if ious else math.nan
    neg = float(np.mean(empty)) if empty else math.nan
    return {"val_miou": miou, "neg_select_empty": neg, "score": float(np.nanmean([miou, neg]))}


def train(seed: int, cfg: RunConfig, p: FtParams, session: str) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cache = Cache(session)
    predictor, wsha = build_sam(p.weights)
    if wsha != cache.info["weights_sha256"]:
        raise ValueError("the cache was built from other encoder weights")
    model = predictor.model
    model.mask_decoder.train()
    for q in model.mask_decoder.parameters():
        q.requires_grad_(True)
    k = output_index(cfg.seg_mask_select)
    pos, neg = cache.role("train"), cache.role("neg_train")
    n_neg = round(p.batch_size * p.neg_share) if neg else 0
    n_pos = p.batch_size - n_neg
    steps = math.ceil(len(pos) / n_pos)
    opt = torch.optim.AdamW(model.mask_decoder.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p.epochs * steps, eta_min=p.lr_min)
    model.mask_decoder.eval()
    history = [{"epoch": 0, **evaluate(model, cache, k, p.select_min_area_frac)}]
    print(f"seed {seed}: {len(pos)} positives, {len(neg)} negatives, batch {n_pos}+{n_neg}, {steps} steps/epoch, "
          f"point batches {p.point_share}; epoch 0 {history[0]}", flush=True)
    best = dict(history[0], state={n: t.clone() for n, t in model.mask_decoder.state_dict().items()})
    for epoch in range(1, p.epochs + 1):
        model.mask_decoder.train()
        t0 = time.perf_counter()
        order = rng.permutation(pos)
        sums = {"total": 0.0, "focal": 0.0, "dice": 0.0, "iou_mse": 0.0}
        point_steps = 0
        for s in range(steps):
            batch = list(order[s * n_pos:(s + 1) * n_pos]) + list(rng.choice(neg, n_neg)) if n_neg else \
                list(order[s * n_pos:(s + 1) * n_pos])
            with_points = bool(rng.random() < p.point_share)
            point_steps += with_points
            opt.zero_grad()
            for i in batch:
                emb, lab, box = cache.sample(int(i), cache.views[rng.integers(len(cache.views))])
                points = None
                if with_points:
                    with torch.no_grad():
                        box_only, _ = decode(model, emb, box, k)
                    points = error_points(box_only, lab, int(rng.integers(1, p.max_points + 1)), rng)
                logits, iou = decode(model, emb, box, k, points)
                ls = losses(logits, iou, lab, p)
                (ls["total"] / len(batch)).backward()
                for key in sums:
                    sums[key] += float(ls[key].detach()) / len(batch) / steps
            opt.step()
            sched.step()
        model.mask_decoder.eval()
        ev = evaluate(model, cache, k, p.select_min_area_frac)
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
            "history": history, "weights_sha256": wsha, "k": k}


def write_model(res: dict, cfg: RunConfig, p: FtParams, cache: Cache, name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"{name}.pt"
    torch.save({"mask_decoder": res["state"], "base_weights_sha256": res["weights_sha256"],
                "output": cfg.seg_mask_select, "prompt": cfg.seg_prompt}, out)
    card = {"name": name, "seed": res["seed"], "labels": cache.info["labels"], "base_weights": p.weights, "base_weights_sha256": res["weights_sha256"],
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "seg_ft": asdict(p),
            "seg_prompt": cfg.seg_prompt, "seg_mask_select": cfg.seg_mask_select,
            "cache": cache.info,
            "chosen": res["best"], "history": res["history"], "threads": torch.get_num_threads()}
    out.with_suffix(".json").write_text(json.dumps(card, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["cache", "train"])
    ap.add_argument("--labels", required=True, help="the labelling session (data/ml5_labels/<session>)")
    ap.add_argument("--seed", type=int, help="train: the fine-tune seed")
    ap.add_argument("--name", help="train: the model's name, models/seg/<name>.pt")
    ap.add_argument("--threads", type=int, required=True, help="torch.set_num_threads")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    cfg = RunConfig.from_params_yaml()
    p = FtParams.from_config(cfg)
    if args.command == "cache":
        build_cache(args.labels, p)
        return
    if args.seed is None or not args.name:
        ap.error("train needs --seed and --name")
    res = train(args.seed, cfg, p, args.labels)
    out = write_model(res, cfg, p, Cache(args.labels), args.name)
    print(f"wrote {out.relative_to(REPO_ROOT)}: epoch {res['best']['epoch']}, score {res['best']['score']:.4f}")


if __name__ == "__main__":
    main()
