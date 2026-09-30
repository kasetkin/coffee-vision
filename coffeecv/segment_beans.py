"""The bean segmentation stage (ticket ML-2 §3 steps 2-3): predict the bean region, fill the rest,
crop to the rectangle that keeps >= 95% of it, or fall back to the whole photo (D18).

One implementation behind both ends, per the train/inference parity rule: the training crop stage
and infer.patches_for_photo call the same `segment_and_crop` (plan §2.2).

    BeanSegmenter.predict_mask   full-resolution bool mask from L0 with a whole-image box prompt (D5)
    mask_and_crop                pure numpy: D3 fill, D4 quantile crop, D18 fallback, diagnostics

`python -m coffeecv.segment_beans --list FILE --time` is the P0/P6 latency report (D8: measured and
reported, never gated). `--candidates OUT_DIR` writes every decoder output of a box prompt at full
resolution, for fixing the mask-selection rule (plan §3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from coffeecv.config import REPO_ROOT
from coffeecv.sam_loader import L0_WEIGHTS, VARIANTS, build_sam_l0, weights_for

# SAM's box prompt yields either the single-mask token (multimask_output=False) or three multimask
# tokens. "best_iou" picks the multimask output with the highest predicted IoU.
MASK_SELECT = ("single", "multi1", "multi2", "multi3", "best_iou")
# D5: (a) "box", one whole-image box; (b) "grid3", a 3x3 grid of foreground points at 20/50/80% of
# each side (the crop_tray_sam.py prototype's layout, here over the whole photo); "box+grid3" both.
# P0 probes: "center_box", a centred square of side min(H, W) / 10; "center_point", one foreground
# point at the photo centre; "center_grid4", 16 foreground points on a centred 4x4 grid spanning a
# square of side min(H, W) / 8 (outer points on the square's edges).
PROMPTS = ("box", "grid3", "box+grid3", "center_box", "center_point", "center_grid4")


@dataclass(frozen=True)
class SegParams:
    mask_select: str
    weights: str = L0_WEIGHTS          # under models_pretrained/, sha256-checked at load
    variant: str = "l0"                # EfficientViT-SAM size; must match `weights`
    prompt: str = "box"                # D5 (a): one fixed whole-image box
    keep_frac: float = 0.95            # D4: cut (1 - keep_frac) / 4 of the bean pixels per side
    fill_rgb: tuple[int, int, int] = (124, 116, 104)   # D3: ImageNet mean, rounded to uint8
    min_area_frac: float = 0.0         # D18: set in P2

    def __post_init__(self):
        if self.mask_select not in MASK_SELECT:
            raise ValueError(f"mask_select {self.mask_select!r} not in {MASK_SELECT}")
        if self.prompt not in PROMPTS:
            raise ValueError(f"prompt {self.prompt!r} not in {PROMPTS}")


@dataclass
class BeanCrop:
    rgb: np.ndarray                    # filled crop, or the whole original photo on fallback
    mask: np.ndarray | None            # bool, aligned with rgb; None = fallback (accept every patch)
    info: dict = field(default_factory=dict)


def mask_sha256(mask: np.ndarray) -> str:
    """Key of a mask array (plan §2.3): the packed bits plus the shape, never the PNG bytes, so
    re-encoding a mask file cannot orphan the verdicts stored against it."""
    h = hashlib.sha256()
    h.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    h.update(np.packbits(mask.astype(bool, copy=False)).tobytes())
    return h.hexdigest()


def _quantile_span(counts: np.ndarray, cut: float) -> tuple[int, int]:
    """Inclusive [lo, hi] such that at most `cut` bean pixels lie before lo and at most `cut`
    after hi. `counts` is the bean-pixel count per column (or row)."""
    lo = int(np.searchsorted(np.cumsum(counts), cut, side="right"))
    hi = counts.size - 1 - int(np.searchsorted(np.cumsum(counts[::-1]), cut, side="right"))
    return lo, hi


def mask_and_crop(rgb: np.ndarray, mask: np.ndarray, p: SegParams) -> BeanCrop:
    """D3 fill + D4 quantile crop, or the D18 fallback. `mask` is bool HxW aligned with `rgb`.

    The crop cuts at most (1 - keep_frac) / 4 of the bean pixels from each side, so by the union
    bound it keeps >= keep_frac of them. Counting pixels per column and row gives the same
    quantiles as sorting the coordinates, without materialising them on a 50 MP photo.
    """
    h, w = mask.shape
    if rgb.shape[:2] != (h, w):
        raise ValueError(f"mask {mask.shape} does not match photo {rgb.shape[:2]}")
    n = int(np.count_nonzero(mask))
    area = n / (h * w)
    info = {"mask_area_frac": area, "mask_sha256": mask_sha256(mask)}
    if n == 0 or area < p.min_area_frac:
        return BeanCrop(rgb, None, {**info, "fallback": True, "box": None,
                                    "bean_frac_in_crop": None, "retained_frac": None})
    cut = (1.0 - p.keep_frac) / 4 * n
    x0, x1 = _quantile_span(mask.sum(axis=0, dtype=np.int64), cut)
    y0, y1 = _quantile_span(mask.sum(axis=1, dtype=np.int64), cut)
    m = mask[y0:y1 + 1, x0:x1 + 1]
    out = rgb[y0:y1 + 1, x0:x1 + 1].copy()
    out[~m] = p.fill_rgb
    kept = int(np.count_nonzero(m))
    return BeanCrop(out, m, {**info, "fallback": False, "box": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                             "bean_frac_in_crop": kept / m.size, "retained_frac": kept / n})


class BeanSegmenter:
    """L0 loaded once; one photo in, one full-resolution bool mask out."""

    def __init__(self, p: SegParams):
        self.p = p
        self.predictor, self.weights_sha256 = build_sam_l0(p.weights, p.variant)
        self.model = self.predictor.model
        self.timing_ms: dict[str, float] = {}

    @torch.inference_mode()
    def _encode(self, rgb: np.ndarray) -> None:
        """EfficientViTSamPredictor.set_image, split so the resize and the encoder time apart."""
        from segment_anything.utils.transforms import ResizeLongestSide
        t0 = time.perf_counter()
        x = self.model.transform(rgb).unsqueeze(0)
        t1 = time.perf_counter()
        pr = self.predictor
        pr.reset_image()
        pr.original_size = rgb.shape[:2]
        pr.input_size = ResizeLongestSide.get_preprocess_shape(*rgb.shape[:2], self.model.image_size[0])
        pr.features = self.model.image_encoder(x)
        pr.is_image_set = True
        t2 = time.perf_counter()
        self.timing_ms.update(resize=(t1 - t0) * 1e3, encoder=(t2 - t1) * 1e3)

    def prompt_geometry(self, h: int, w: int) -> dict:
        """The fixed prompt in photo pixels: {"box": [x0, y0, x1, y1] | None, "points": [[x, y], ...],
        "labels": [1 = include, 0 = exclude, ...]}. Also written to --candidates output for the sheets."""
        prompt, box, pts = self.p.prompt, None, []
        cx, cy = (w - 1) / 2, (h - 1) / 2
        if prompt == "center_box":
            half = min(h, w) / 10 / 2
            box = [cx - half, cy - half, cx + half, cy + half]
        elif prompt == "center_point":
            pts = [[cx, cy]]
        elif prompt == "center_grid4":
            half = min(h, w) / 8 / 2
            off = np.linspace(-half, half, 4)
            pts = [[cx + dx, cy + dy] for dy in off for dx in off]
        else:
            if "box" in prompt:
                box = [0, 0, w - 1, h - 1]
            if "grid3" in prompt:
                f = np.linspace(0.2, 0.8, 3)
                pts = [[fx * (w - 1), fy * (h - 1)] for fy in f for fx in f]
        return {"box": box, "points": [[float(x), float(y)] for x, y in pts], "labels": [1] * len(pts)}

    @torch.inference_mode()
    def _decode(self, h: int, w: int, multimask: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Low-res logits (C x 256 x 256) and predicted IoUs (C) for the fixed D5 prompt."""
        pr = self.predictor
        g = self.prompt_geometry(h, w)
        box = pr.apply_boxes_torch(torch.tensor([g["box"]], dtype=torch.float)) if g["box"] else None
        pts = labels = None
        if g["points"]:
            pts = pr.apply_coords_torch(torch.tensor([g["points"]], dtype=torch.float))
            labels = torch.tensor([g["labels"]], dtype=torch.int)
        _, iou, low = pr.predict_torch(point_coords=pts, point_labels=labels, boxes=box,
                                       multimask_output=multimask, return_logits=True)
        return low[0], iou[0]

    @torch.inference_mode()
    def _upsample(self, low: torch.Tensor) -> np.ndarray:
        """SAM's own postprocess on one channel: to the padded 1024 frame, drop the bottom/right
        padding, resize to the photo, threshold at 0."""
        pr = self.predictor
        logits = self.model.postprocess_masks(low[None, None], pr.input_size, pr.original_size)
        return (logits[0, 0] > self.model.mask_threshold).numpy()

    def predict_mask(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        self._encode(rgb)
        t0 = time.perf_counter()
        sel = self.p.mask_select
        low, iou = self._decode(h, w, multimask=sel != "single")
        k = 0 if sel == "single" else int(iou.argmax()) if sel == "best_iou" else int(sel[-1]) - 1
        t1 = time.perf_counter()
        mask = self._upsample(low[k])
        t2 = time.perf_counter()
        self.timing_ms.update(decoder=(t1 - t0) * 1e3, upsample=(t2 - t1) * 1e3)
        return mask

    def candidates(self, rgb: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
        """Every box-prompt output at full resolution: {name: (mask, predicted IoU)}."""
        h, w = rgb.shape[:2]
        self._encode(rgb)
        out = {}
        for multimask, names in ((False, ["single"]), (True, ["multi1", "multi2", "multi3"])):
            low, iou = self._decode(h, w, multimask)
            for k, name in enumerate(names):
                out[name] = (self._upsample(low[k]), float(iou[k]))
        return out


def segment_and_crop(rgb: np.ndarray, seg: BeanSegmenter) -> BeanCrop:
    mask = seg.predict_mask(rgb)
    t0 = time.perf_counter()
    crop = mask_and_crop(rgb, mask, seg.p)
    seg.timing_ms["fill_crop"] = (time.perf_counter() - t0) * 1e3
    crop.info["timing_ms"] = dict(seg.timing_ms)
    return crop


def _time(photos: list[Path], p: SegParams, runs: int) -> dict:
    from coffeecv.dataset import load_rgb_image
    seg = BeanSegmenter(p)
    report = {"variant": p.variant, "threads": torch.get_num_threads(), "runs": runs, "prompt": p.prompt, "mask_select": p.mask_select,
              "weights_sha256": seg.weights_sha256, "photos": {}}
    for path in photos:
        rows, shas = [], set()
        for _ in range(runs):
            t0 = time.perf_counter()
            rgb = load_rgb_image(path)
            decode_ms = (time.perf_counter() - t0) * 1e3
            crop = segment_and_crop(rgb, seg)
            rows.append({"decode": decode_ms, **crop.info["timing_ms"]})
            shas.add(crop.info["mask_sha256"])
        med = {k: round(statistics.median(r[k] for r in rows), 1) for k in rows[0]}
        med["total"] = round(sum(med.values()), 1)
        key = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        report["photos"][key] = {"shape": list(rgb.shape[:2]), "median_ms": med,
                                       "mask_area_frac": round(crop.info["mask_area_frac"], 4),
                                       "mask_sha256": sorted(shas),
                                       "identical_masks_across_runs": len(shas) == 1}
        print(f"{path.name}  {rgb.shape[1]}x{rgb.shape[0]}  " +
              "  ".join(f"{k} {v:.0f}" for k, v in med.items()) + f"  (ms, median of {runs})"
              f"  masks identical: {len(shas) == 1}")
    return report


def _write_candidates(photos: list[Path], p: SegParams, out_dir: Path) -> None:
    from PIL import Image

    from coffeecv.dataset import load_rgb_image
    seg = BeanSegmenter(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for path in photos:
        rgb = load_rgb_image(path)
        cands = seg.candidates(rgb)
        index[str(path)] = {"prompt": seg.prompt_geometry(*rgb.shape[:2])}
        for name, (mask, iou) in cands.items():
            Image.fromarray(mask).save(out_dir / f"{path.stem}__{name}.png", optimize=True)
            index[str(path)][name] = {"pred_iou": round(iou, 4),
                                      "area_frac": round(float(mask.mean()), 4),
                                      "mask_sha256": mask_sha256(mask)}
        print(path.name, {k: (v["pred_iou"], v["area_frac"]) for k, v in index[str(path)].items() if k != "prompt"})
    (out_dir / "candidates.json").write_text(json.dumps(index, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, required=True,
                    help="text file, one repo-relative photo path per line (# comments allowed)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--time", action="store_true", help="per-stage latency, median of --runs")
    mode.add_argument("--candidates", type=Path, metavar="OUT_DIR",
                      help="write every box-prompt output as a full-resolution PNG mask")
    # Required, not defaulted: both are open owner decisions (P0 finding, D5) until params.yaml
    # carries them, and a silent default would pick one.
    ap.add_argument("--mask-select", choices=MASK_SELECT, required=True)
    ap.add_argument("--prompt", choices=PROMPTS, required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="l0", help="EfficientViT-SAM size (L0 is the ticket's)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch's)")
    ap.add_argument("--out", type=Path, help="write the --time report as JSON")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    photos = [REPO_ROOT / s for line in args.list.read_text().splitlines()
              if (s := line.split("#", 1)[0].strip())]
    p = SegParams(mask_select=args.mask_select, prompt=args.prompt, variant=args.variant,
                  weights=weights_for(args.variant))
    if args.candidates:
        _write_candidates(photos, p, args.candidates)
        return
    report = _time(photos, p, args.runs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
