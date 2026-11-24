"""ML-37, bits for the segmenter: the shared pieces of the bits_seg_* scripts (not a script itself).

The segmenter is the live one: params.yaml's seg_* block, the EfficientViT-SAM XL0 encoder
(efficientvit_sam/efficientvit_sam_xl0.pt) with the fine-tuned decoder models/seg/xl0_v7r_s123.pt, the whole-image
box prompt and the multi3 output (segment_beans.seg_params), torch on CPU in float32.

The float picture: the photo's canonical photo decoded at 16 bits (canonical.decode_canonical, uint16 Rec.2020,
BT.709 curve) and read back to sRGB by the transform load_canonical uses with float32 out, clipped to [0, 1]
(bits_cls_common.readback): the same stand-in for ML-37's read-back as the classifier study, here on the whole photo.

Where the arms quantize: where today's 8-bit step sits on the segmenter's path, load_canonical's uint8 output, which
model_io.sam_input then resizes (sam_resize, PIL's uint8 bilinear) and normalises. Each arm rounds the whole float
photo there, and everything after it is float32: the resize to the encoder's 1024 px long side by PIL's bilinear in
"F" mode channel by channel (ML-37 D11 (a); d14_mask_sensitivity.float_resize), the EfficientViT-SAM normalisation
and the zero pad. So only the bit depth of the photo differs between f32, f16, u12, u10 and u8.

  f32    the float photo itself (the reference)
  f16    rounded to float16 (the owner's "16-bit float")
  u12    round(x * 4095) / 4095
  u10    round(x * 1023) / 1023
  u8     round(x * 255) / 255, with the resize and the rest in float
  today  today's path exactly: load_canonical's uint8 (canonical.rec2020_to_srgb, LittleCMS with 8-bit output) through
         model_io.sam_input (sam_resize on uint8, rounded again, / 255, normalised, padded)

Yardsticks (changes that do not matter), each from the f32 photo and the float path unless said:
  fp16in  the f32 arm's normalised encoder input cast to float16 and back
  shift1  the float photo moved left by one photo pixel (last column repeated), same size
  gen2    the canonical photo re-encoded at the store's quality and effort (q95 e7, jxl_encode.encode_rec2020, as
          bits_cls_noise.jxl) and decoded, then read back as f32: a second JPEG XL generation
"""
from __future__ import annotations

import csv
import resource
import sys
from pathlib import Path

import imagecodecs as ic
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import bits_cls_common as bc  # noqa: E402

from coffeecv import canonical as cn  # noqa: E402
from coffeecv import model_io  # noqa: E402
from coffeecv.canonical_store import canonical_for  # noqa: E402
from coffeecv.config import REPO_ROOT, RunConfig  # noqa: E402
from coffeecv.jxl_encode import encode_rec2020  # noqa: E402
from coffeecv.seg_finetune import IGNORE, label_in_frame  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, d4_box, seg_params  # noqa: E402

SCRATCH = REPO_ROOT / "tmp" / "ml37" / "bits_seg"
THREADS = 8
ARMS = ("f32", "f16", "u12", "u10", "u8", "today")
YARDS = ("fp16in", "shift1", "gen2")
STORED = REPO_ROOT / "data" / "seg_masks_ml19" / "xl0_v7r_s123"
LABELS = REPO_ROOT / "data" / "ml5_labels" / "pass17" / "labels.csv"
SELECT_MIN_AREA = 0.083       # params.yaml seg_ft.select_min_area_frac: seg_finetune.evaluate's tiny rule


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def camera(path: str, source: str) -> str:
    """A coarse camera group from the dataset path (bits_cls_predict.camera's rule), negatives and internet apart."""
    if source == "negative":
        return "negative"
    if source == "internet_positive":
        return "internet"
    return bc_camera(path)


def bc_camera(path: str) -> str:
    parts = path.lower().split("/")
    s, p = parts[1] if len(parts) > 1 else "", parts[-1]
    for k in ("pixel", "sony", "iphone", "oneplus"):
        if k in s:
            return k
    for k, tag in (("pixel", "pxl_"), ("pixel", "pixel"), ("oneplus", "oneplus"), ("canon", "canon"),
                   ("iphone", "iphone"), ("iphone", "img_"), ("sony", "dsc"), ("sony", "pic_")):
        if tag in p:
            return k
    return "other"


def segmenter() -> BeanSegmenter:
    torch.set_num_threads(THREADS)
    return BeanSegmenter(seg_params(RunConfig.from_params_yaml()))


def decode_rec(path: str, sha256: str | None) -> tuple[np.ndarray, bytes]:
    data = canonical_for(path, sha256).read_bytes()
    return cn.decode_canonical(data, THREADS), data


def gen2_rec(rec: np.ndarray) -> np.ndarray:
    """A second JPEG XL generation of the decoded canonical photo at the store's quality and effort."""
    b = encode_rec2020(rec.shape[0], rec.shape[1], lambda y, x, ys, xs: rec[y:y + ys, x:x + xs],
                       level=cn.QUALITY, effort=cn.EFFORT, numthreads=THREADS)
    return ic.jpegxl_decode(b, numthreads=THREADS)


def float_resize(rgb: np.ndarray, side: int) -> np.ndarray:
    """sam_resize in float: PIL's bilinear in mode F, one channel at a time (ML-37 D11 (a))."""
    h, w = rgb.shape[:2]
    if max(h, w) == side:
        return rgb
    nh, nw = model_io.sam_preprocess_shape(h, w, side)
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c]), mode="F")
                                .resize((nw, nh), Image.BILINEAR)) for c in range(3)], axis=-1)


def float_sam_input(rgb01: np.ndarray, side: int) -> np.ndarray:
    """model_io.sam_input on a float [0, 1] photo: float resize, normalise, zero pad. float32 [1, 3, side, side]."""
    small = float_resize(rgb01, side)
    ih, iw = small.shape[:2]
    m = np.asarray(model_io.SAM_MEAN, np.float32)[:, None, None]
    s = np.asarray(model_io.SAM_STD, np.float32)[:, None, None]
    x = np.zeros((1, 3, side, side), np.float32)
    x[0, :, :ih, :iw] = (np.ascontiguousarray(small.transpose(2, 0, 1)) - m) / s
    return x


def arm_input(x01: np.ndarray, arm: str, side: int, today_u8: np.ndarray | None = None) -> np.ndarray:
    """The encoder's input for one arm. `x01`: the float photo; `today_u8`: load_canonical's uint8 (the today arm)."""
    if arm == "today":
        return model_io.sam_input(today_u8, side)[0]
    return float_sam_input(bc.quantize(x01, arm), side)


class Run:
    """The live segmenter, one photo at a time: encode an input, decode the box prompt's multi3 output."""

    def __init__(self):
        self.seg = segmenter()
        self.side = self.seg.engine.image_side
        self.frame = self.seg.engine.frame
        self.k = int(self.seg.p.mask_select[-1]) - 1     # multi3 -> low[2]
        self.min_area = self.seg.p.min_area_frac

    def embed(self, x: np.ndarray) -> torch.Tensor:
        self.seg.engine.set_image(x)
        return self.seg.engine.features.clone()

    def decode(self, feats: torch.Tensor, h: int, w: int) -> tuple[np.ndarray, float]:
        """(low-res logits [256, 256] of the selected output, its predicted IoU), as BeanSegmenter.predict_mask."""
        s = self.seg
        s.original_size = (h, w)
        s.input_size = model_io.sam_preprocess_shape(h, w, self.frame)
        s.engine.features = feats
        low, iou = s._decode(h, w, multimask=True)
        return low[self.k], float(iou[self.k])

    def full_mask(self, low: np.ndarray) -> np.ndarray:
        return self.seg._upsample(low)

    def frame_logits(self, low: np.ndarray) -> torch.Tensor:
        """seg_finetune.decode's view: the low-res logits bilinearly to the encoder's frame (frame x frame)."""
        t = torch.from_numpy(np.ascontiguousarray(low))[None, None]
        return F.interpolate(t, (self.side, self.side), mode="bilinear", align_corners=False)[0, 0]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return 1.0 if u == 0 else float(np.count_nonzero(a & b) / u)


def label_scores(logits: torch.Tensor, lab: torch.Tensor) -> dict:
    """seg_finetune.evaluate's two measures for one photo: hard IoU on the valid pixels (positives), and the
    predicted area share of the valid frame and whether it is empty or tiny (< select_min_area_frac, negatives)."""
    valid = lab != IGNORE
    p, t = logits[valid] > 0, lab[valid] == 1
    u = int((p | t).sum())
    return {"iou": 1.0 if u == 0 else int((p & t).sum()) / u, "area": float(p.float().mean()),
            "tiny": float(p.float().mean()) < SELECT_MIN_AREA}


def frame_label(mask_png: Path, side: int) -> torch.Tensor:
    m = np.array(Image.open(mask_png)) > 0
    return torch.from_numpy(label_in_frame(m, side))


def feat_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """1 - cosine and relative L2 of the whole embedding (flattened 256 x 64 x 64), float64; and the per-location
    (64 x 64, over the 256 channels) 1 - cosine: median and max."""
    a64, b64 = a.double().flatten(), b.double().flatten()
    cos = float(a64 @ b64 / a64.norm() / b64.norm())
    rel = float((a64 - b64).norm() / b64.norm())
    pa, pb = a.double()[0].flatten(1), b.double()[0].flatten(1)       # 256 x 4096
    pc = 1 - (pa * pb).sum(0) / pa.norm(dim=0) / pb.norm(dim=0)
    return {"cos": 1 - cos, "rel": rel, "loc_med": float(pc.median()), "loc_max": float(pc.max())}


def mask_diff(m: np.ndarray, ref: np.ndarray, min_area: float, keep_frac: float = 0.95) -> dict:
    """A mask against the reference mask: IoU, pixels changed, empty and fallback flips, and the D4 box's move."""
    a, r = float(m.mean()), float(ref.mean())
    bm = d4_box(m, keep_frac) if m.any() else None
    br = d4_box(ref, keep_frac) if ref.any() else None
    move = None if bm is None or br is None else int(np.abs(np.subtract(bm, br)).max())
    fb_m, fb_r = (a == 0 or a < min_area), (r == 0 or r < min_area)
    return {"iou": iou(m, ref), "changed": int(np.count_nonzero(m != ref)), "area": a,
            "empty_flip": (a == 0) != (r == 0), "fallback_flip": fb_m != fb_r, "box_move": move, "box": bm}


def read_labels() -> dict[str, dict]:
    return {r["id"]: r for r in csv.DictReader(LABELS.open())}


def shift1(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
