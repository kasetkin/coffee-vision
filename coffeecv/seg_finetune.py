"""Ticket ML-2 P5: decoder-only fine-tuning of the segmenter (D7 (a), plan §8). Two DVC stages:

    python -m coffeecv.seg_finetune cache --threads 4        # seg_embed_cache: L0 encoder embeddings + labels
    python -m coffeecv.seg_finetune train --seed 42 --threads 4   # seg_finetune@<seed>: models/seg/ft_s<seed>.pt

Labels come from data/seg_labels/labels.csv (seg_labels.py, the judge loop): every accepted positive and every
seg-train negative (an empty label, D11). Pending and dropped photos are left out. Roles:

    train       seg_train + pos_seg_train accepted masks
    val         seg_val accepted masks: checkpoint selection only (mean IoU)
    neg_train   seg-train negatives, ~80% by near-duplicate group
    neg_select  the other ~20%: checkpoint selection only (share empty or tiny, D18 threshold)

The eval lists and the judge are never used here.

Cache: for each view in seg_ft.views (dihedral: id, hflip, vflip, rot180, rot90, rot270, transpose,
antitranspose), the photo and its label are transformed at full resolution, the photo goes through the frozen
encoder exactly as at serving (resize to 512 on the long side, normalise, pad), and the embedding is stored in
fp16 with the label resized into the same 512 frame (255 = padding, ignored). data/seg_cache/<view>/emb.npy
(N, 256, 64, 64) and label.npy (N, 512, 512) are memory-mapped; index.csv holds the rows and roles.

Training (seg_ft in params.yaml): the image encoder and prompt encoder are frozen, the mask decoder trains. The
prompt is the serving prompt, one whole-image box, and the loss is on the output serving keeps (seg_mask_select,
multi3), upsampled to the 512 frame: focal_weight x focal + dice_weight x soft dice (smooth 1) + iou_weight x MSE
of the IoU head against the IoU it achieves. Each epoch shows every train positive once, in a random view; each
batch holds neg_share negatives, drawn with replacement. AdamW with cosine decay to lr_min; the epoch with the
best selection score is kept: (mean val IoU + neg_select empty-or-tiny share) / 2. Stops after `patience`
epochs without a better score.

Out: models/seg/ft_s<seed>.pt ({"mask_decoder", "base_weights_sha256", ...}) + .json (params, label-set hash,
history, the chosen epoch). segment_beans.SegParams(decoder=...) loads it over the pretrained L0.
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

from coffeecv import seg_lists
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.sam_loader import L0_WEIGHTS, build_sam_l0
from coffeecv.seg_labels import LABELS_CSV

CACHE_ROOT = REPO_ROOT / "data" / "seg_cache"
MODEL_DIR = REPO_ROOT / "models" / "seg"
FRAME = 512                       # L0's encoder input; the loss is computed in this padded frame
LOW = 256                         # the decoder's low-res logits
PROMPT_FRAME = 1024               # SAM's prompt coordinates (model.image_size[0])
IGNORE = 255
EMB_SHAPE = (256, 64, 64)
_NEG_STREAM = 8                   # under seg_split_seed; seg_lists uses 1-4, 6-7 and seg_base_masks 5
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
    neg_select_frac: float = 0.2

    @classmethod
    def from_config(cls, cfg: RunConfig) -> "FtParams":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg.seg_ft) - known)
        if unknown:
            raise ValueError(f"params.yaml seg_ft sets keys seg_finetune does not define: {', '.join(unknown)}")
        p = cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.seg_ft.items()})
        if bad := [v for v in p.views if v not in VIEWS]:
            raise ValueError(f"seg_ft.views: unknown {bad}; known {list(VIEWS)}")
        if p.views[0] != "id":
            raise ValueError("seg_ft.views must start with id (selection reads the identity view)")
        return p


# ---------------------------------------------------------------- geometry shared by cache and training

def frame_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """SAM's ResizeLongestSide.get_preprocess_shape."""
    s = long_side / max(h, w)
    return int(h * s + 0.5), int(w * s + 0.5)


def label_in_frame(mask: np.ndarray) -> np.ndarray:
    """A full-resolution bool mask -> uint8 FRAME x FRAME: 1 bean, 0 not, IGNORE on the padding (bottom/right,
    as SamPad pads the image)."""
    fh, fw = frame_size(*mask.shape, FRAME)
    small = cv2.resize(mask.astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA) >= 0.5
    out = np.full((FRAME, FRAME), IGNORE, np.uint8)
    out[:fh, :fw] = small
    return out


def whole_box(h: int, w: int) -> np.ndarray:
    """The serving prompt, one whole-image box [x0, y0, x1, y1], in the prompt frame (BeanSegmenter._decode)."""
    fh, fw = frame_size(h, w, PROMPT_FRAME)
    return np.array([0.0, 0.0, (w - 1) * fw / w, (h - 1) * fh / h], np.float32)


# ---------------------------------------------------------------- the label set and the cache

def label_rows(cfg: RunConfig, p: FtParams) -> list[dict]:
    """labels.csv rows that train or select, each with its role."""
    rows = [r for r in csv.DictReader(LABELS_CSV.read_text().splitlines()) if r["status"] in ("accepted", "negative")]
    _, lists = seg_lists.load_lists()
    group = {e["path"]: e["group"] for e in lists["neg_seg_train"]}
    groups = sorted({group[r["path"]] for r in rows if r["status"] == "negative"})
    rng = np.random.default_rng([cfg.seg_split_seed, _NEG_STREAM])
    held = set(rng.choice(groups, round(p.neg_select_frac * len(groups)), replace=False)) if groups else set()
    for r in rows:
        if r["status"] == "negative":
            r["role"] = "neg_select" if group[r["path"]] in held else "neg_train"
        else:
            r["role"] = "val" if r["list"] == "seg_val" else "train"
    return rows


def label_set_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["id"]):
        h.update(f"{r['id']} {r['photo_sha256']} {r['mask_sha256']} {r['role']}\n".encode())
    return h.hexdigest()


@torch.inference_mode()
def build_cache(cfg: RunConfig, p: FtParams) -> None:
    rows = label_rows(cfg, p)
    predictor, wsha = build_sam_l0(L0_WEIGHTS)
    model = predictor.model
    n = len(rows)
    t0 = time.perf_counter()
    arrays = {}
    for v in p.views:
        d = CACHE_ROOT / v
        d.mkdir(parents=True, exist_ok=True)
        arrays[v] = (np.lib.format.open_memmap(d / "emb.npy", "w+", np.float16, (n, *EMB_SHAPE)),
                     np.lib.format.open_memmap(d / "label.npy", "w+", np.uint8, (n, FRAME, FRAME)), [])
    for i, r in enumerate(rows):
        photo = REPO_ROOT / r["path"]
        if seg_lists.sha256_file(photo) != r["photo_sha256"]:
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
            lab[i] = label_in_frame(m)
            meta.append({"h": m.shape[0], "w": m.shape[1]})
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}  {time.perf_counter() - t0:.0f}s", flush=True)
    for v, (emb, lab, meta) in arrays.items():
        emb.flush()
        lab.flush()
        (CACHE_ROOT / v / "meta.json").write_text(json.dumps(meta) + "\n")
    with open(CACHE_ROOT / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["i", "id", "list", "role", "path", "photo_sha256", "mask_sha256"])
        wr.writeheader()
        for i, r in enumerate(rows):
            wr.writerow({"i": i, **{k: r[k] for k in ("id", "list", "role", "path", "photo_sha256", "mask_sha256")}})
    (CACHE_ROOT / "cache.json").write_text(json.dumps({
        "views": list(p.views), "n": n, "weights_sha256": wsha, "label_set_sha256": label_set_sha256(rows),
        "roles": {k: sum(r["role"] == k for r in rows) for k in ("train", "val", "neg_train", "neg_select")},
        "threads": torch.get_num_threads()}, indent=2) + "\n")


class Cache:
    def __init__(self):
        info = json.loads((CACHE_ROOT / "cache.json").read_text())
        self.info = info
        self.rows = list(csv.DictReader((CACHE_ROOT / "index.csv").read_text().splitlines()))
        self.views = info["views"]
        self.emb = {v: np.load(CACHE_ROOT / v / "emb.npy", mmap_mode="r") for v in self.views}
        self.label = {v: np.load(CACHE_ROOT / v / "label.npy", mmap_mode="r") for v in self.views}
        self.meta = {v: json.loads((CACHE_ROOT / v / "meta.json").read_text()) for v in self.views}

    def role(self, name: str) -> list[int]:
        return [int(r["i"]) for r in self.rows if r["role"] == name]

    def sample(self, i: int, view: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(embedding 1x256x64x64 float32, label FRAME x FRAME uint8, box 1x4 in the prompt frame)."""
        m = self.meta[view][i]
        return (torch.from_numpy(np.asarray(self.emb[view][i], np.float32))[None],
                torch.from_numpy(np.array(self.label[view][i])),
                torch.from_numpy(whole_box(m["h"], m["w"]))[None])


# ---------------------------------------------------------------- the decoder pass and the loss

def decode(model, emb: torch.Tensor, box: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Output k's logits in the padded FRAME (FRAME x FRAME) and its predicted IoU (scalar), for one image."""
    sparse, dense = model.prompt_encoder(points=None, boxes=box, masks=None)
    low, iou = model.mask_decoder(image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
                                  sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                  multimask_output=k > 0)
    j = k - 1 if k > 0 else 0
    up = F.interpolate(low[:, j:j + 1], (FRAME, FRAME), mode="bilinear", align_corners=False)
    return up[0, 0], iou[0, j]


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


def train(seed: int, cfg: RunConfig, p: FtParams) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cache = Cache()
    predictor, wsha = build_sam_l0(L0_WEIGHTS)
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
    history = [{"epoch": 0, **evaluate(model, cache, k, cfg.seg_min_area_frac)}]
    print(f"seed {seed}: {len(pos)} positives, {len(neg)} negatives, batch {n_pos}+{n_neg}, {steps} steps/epoch; "
          f"epoch 0 {history[0]}", flush=True)
    best = dict(history[0], state={n: t.clone() for n, t in model.mask_decoder.state_dict().items()})
    for epoch in range(1, p.epochs + 1):
        model.mask_decoder.train()
        t0 = time.perf_counter()
        order = rng.permutation(pos)
        sums = {"total": 0.0, "focal": 0.0, "dice": 0.0, "iou_mse": 0.0}
        for s in range(steps):
            batch = list(order[s * n_pos:(s + 1) * n_pos]) + list(rng.choice(neg, n_neg)) if n_neg else \
                list(order[s * n_pos:(s + 1) * n_pos])
            opt.zero_grad()
            for i in batch:
                emb, lab, box = cache.sample(int(i), cache.views[rng.integers(len(cache.views))])
                logits, iou = decode(model, emb, box, k)
                ls = losses(logits, iou, lab, p)
                (ls["total"] / len(batch)).backward()
                for key in sums:
                    sums[key] += float(ls[key].detach()) / len(batch) / steps
            opt.step()
            sched.step()
        model.mask_decoder.eval()
        ev = evaluate(model, cache, k, cfg.seg_min_area_frac)
        row = {"epoch": epoch, **{f"train_{a}": round(b, 5) for a, b in sums.items()}, **ev,
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


def write_model(res: dict, cfg: RunConfig, p: FtParams, cache: Cache) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"ft_s{res['seed']}.pt"
    torch.save({"mask_decoder": res["state"], "base_weights_sha256": res["weights_sha256"],
                "output": cfg.seg_mask_select, "prompt": cfg.seg_prompt}, out)
    card = {"seed": res["seed"], "base_weights": L0_WEIGHTS, "base_weights_sha256": res["weights_sha256"],
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "seg_ft": asdict(p),
            "seg_prompt": cfg.seg_prompt, "seg_mask_select": cfg.seg_mask_select,
            "seg_min_area_frac": cfg.seg_min_area_frac, "cache": cache.info,
            "chosen": res["best"], "history": res["history"], "threads": torch.get_num_threads()}
    out.with_suffix(".json").write_text(json.dumps(card, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["cache", "train"])
    ap.add_argument("--seed", type=int, help="train: the fine-tune seed")
    ap.add_argument("--threads", type=int, required=True, help="torch.set_num_threads")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    cfg = RunConfig.from_params_yaml()
    p = FtParams.from_config(cfg)
    if args.command == "cache":
        build_cache(cfg, p)
        return
    if args.seed is None:
        ap.error("train needs --seed")
    res = train(args.seed, cfg, p)
    out = write_model(res, cfg, p, Cache())
    print(f"wrote {out.relative_to(REPO_ROOT)}: epoch {res['best']['epoch']}, score {res['best']['score']:.4f}")


if __name__ == "__main__":
    main()
