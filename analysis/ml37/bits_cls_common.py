"""ML-37, bits for the classifier: the shared pieces of the bits_cls_* scripts (not a script itself).

The float picture: today's canonical photo decoded at 16 bits (canonical.decode_canonical, uint16 Rec.2020, BT.709
curve), the crop's box (segcrop_report.json) cut out and read back to sRGB by the transform load_canonical uses
(canonical.REC2020 -> canonical.SRGB, relative, NOOPTIMIZE) with float32 out, clipped to [0, 1], and filled outside
its __beanmask.png with fill_rgb / 255: research_d10_crop_and_patch_storage.md's stand-in for ML-37's read-back,
threaded over strips here (imagecodecs releases the GIL; the strips do not depend on the thread count).

The arms quantize that float crop where today's 8-bit step sits (the stored crop, ML-19 D32), and everything after
it stays float32: the box cut, the resize to patch_store_size 448 and then to patch_resize 224 by PIL's bilinear in
"F" mode channel by channel (ML-37 D11 (a)), and the ImageNet normalisation. Quantization is pointwise, so it is
applied to the box's pixels, which equals quantizing the whole crop and then cutting the box.

  f32    the float crop itself (the reference)
  f16    the crop rounded to float16 (D10 (b)'s half-float crop): the "16-bit float" arm
  u12    round(x * 4095) / 4095
  u10    round(x * 1023) / 1023
  u8     round(x * 255) / 255, with the rest of the path in float
  today  today's path: the crop rounded to uint8, PIL's uint8 bilinear to 448 (rounded again), and
         model_io.eval_patch (uint8 bilinear to 224, rounded again, / 255, normalised)

Patch boxes are drawn exactly as dataset.MultiPhotoPatchDataset._extract_photo draws them, from
round(x * 255) as the uint8 picture the pitch is measured on (equal to the stored crop in >= 0.999993 of samples,
D10's note); bits_cls_check_boxes.py checks the boxes against _extract_photo on the stored crops.
"""
from __future__ import annotations

import json
import resource
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import imagecodecs as ic
import numpy as np
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import crop_io
from coffeecv.bean_scale import estimate_bean_pitch, pitch_kwargs
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import SPLIT_SEED_COMPONENT, load_bean_mask
from coffeecv.geometry import Region, compute_valid_region_rect, sample_masked_bean_unit_boxes
from coffeecv.infer import grayscale_like_training
from coffeecv.model_io import IMAGENET_MEAN, IMAGENET_STD, eval_patch

SEG = REPO_ROOT / "data" / "segcropped"
SCRATCH = REPO_ROOT / "tmp" / "ml37" / "bits_cls"
FILL = np.array([124, 116, 104], np.float32) / 255     # segment_beans fill_rgb, as D10's stand-in
STRIP = 128
THREADS = 8
MEAN = np.asarray(IMAGENET_MEAN, np.float32)[:, None, None]
STD = np.asarray(IMAGENET_STD, np.float32)[:, None, None]
STORE, RESIZE = 448, 224
ARMS = ("f32", "f16", "u12", "u10", "u8", "today")


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def pool_entries() -> dict[str, dict]:
    """Each pool crop's name -> its session's segcrop_report entry (canonical, box, out_path, fallback, size)."""
    pool = {p.name: p for p in (SEG / "pool").rglob("*" + crop_io.CROP_SUFFIX)}
    out = {}
    for rep in sorted(SEG.glob("*/class_*/segcrop_report.json")):
        if rep.parent.parent.name == "pool":
            continue
        for e in json.loads(rep.read_text()):
            name = Path(e["out_path"]).name
            if name in pool:
                out[name] = {**e, "pool_path": pool[name], "session": rep.parent.parent.name}
    return out


def readback(rec: np.ndarray, threads: int = THREADS) -> np.ndarray:
    """uint16 Rec.2020/BT.709 -> float32 sRGB in [0, 1]: load_canonical's transform with float output, clipped."""
    out = np.empty(rec.shape, np.float32)

    def conv(y: int) -> None:
        ic.cms_transform(rec[y:y + STRIP], cn.REC2020, cn.SRGB, intent=cn.RELATIVE, flags=cn.NOOPTIMIZE,
                         outdtype=np.float32, out=out[y:y + STRIP])

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(conv, range(0, rec.shape[0], STRIP)))
    np.clip(out, 0.0, 1.0, out=out)
    return out


def box_of(rec: np.ndarray, e: dict) -> np.ndarray:
    if e["box"] is None:
        return rec
    x0, y0, w, h = e["box"]
    return np.ascontiguousarray(rec[y0:y0 + h, x0:x0 + w])


def float_crop(e: dict, threads: int = THREADS, rec: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """The float crop of one pool photo and its bean mask (None for a D18 fallback). `rec`: the decoded canonical
    photo, if the caller has it (bits_cls_noise passes a re-encoded one)."""
    if rec is None:
        rec = cn.decode_canonical((REPO_ROOT / e["canonical"]).read_bytes(), threads)
    x = readback(box_of(rec, e), threads)
    rec = None
    mask = load_bean_mask(e["pool_path"], x.shape[:2])
    if mask is not None:
        x[~mask] = FILL
    return x, mask


def quantize(x: np.ndarray, arm: str) -> np.ndarray:
    if arm == "f32":
        return x
    if arm == "f16":
        return x.astype(np.float16).astype(np.float32)
    bits = {"u12": 12, "u10": 10, "u8": 8}[arm]
    top = np.float32(2 ** bits - 1)
    return np.rint(x * top) / top


def resize_f(x: np.ndarray, side: int) -> np.ndarray:
    """PIL bilinear in "F" mode, channel by channel (ML-37 D11 (a)): HxWx3 float32 -> side x side x 3 float32."""
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(x[..., c])).resize((side, side),
                                                                                        Image.BILINEAR))
                     for c in range(3)], axis=-1)


def normalise(hwc: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(hwc.transpose(2, 0, 1)) - MEAN) / STD


def model_input(box_px: np.ndarray, arm: str) -> np.ndarray:
    """One box's pixels (float32 crop, HxWx3) -> the model's input for `arm`, float32 [3, 224, 224]."""
    if arm == "today":
        u8 = np.rint(box_px * 255).astype(np.uint8)
        patch = Image.fromarray(u8)
        if patch.size[0] != STORE:
            patch = patch.resize((STORE, STORE), Image.BILINEAR)
        return eval_patch(Image.fromarray(np.asarray(patch, dtype=np.uint8)), RESIZE)
    q = quantize(box_px, arm)
    p448 = q if q.shape[0] == STORE else resize_f(q, STORE)
    return normalise(resize_f(p448, RESIZE))


@dataclass(frozen=True)
class Geometry:
    cfg: RunConfig

    @classmethod
    def from_params(cls) -> "Geometry":
        return cls(RunConfig.from_params_yaml())

    def boxes(self, x_u8: np.ndarray, mask: np.ndarray | None, seed: int, class_idx: int, photo_idx: int,
              split: str, n: int, pitch: float | None = None) -> tuple[list[tuple[Region, float, int]], float]:
        """_extract_photo's draw for one photo (crop_method segment, bean-unit sizing, no rotation)."""
        c = self.cfg
        h, w = x_u8.shape[:2]
        region = compute_valid_region_rect(h, w, c.safety_margin)
        if pitch is None:
            pitch = estimate_bean_pitch(grayscale_like_training(x_u8), **pitch_kwargs(c))
        rng = np.random.default_rng([seed, class_idx, photo_idx, SPLIT_SEED_COMPONENT[split]])
        boxes, _, _ = sample_masked_bean_unit_boxes(rng, region, n, pitch, c.patch_beans_min, c.patch_beans_max,
                                                    mask, c.patch_min_bean_share, c.patch_max_attempts_factor, 0.0)
        return boxes, pitch


def cut(x: np.ndarray, box: Region) -> np.ndarray:
    return x[box.y0:box.y1, box.x0:box.x1]


def cos_l2(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: 1 - cosine and relative L2 (|a - b| / |b|), in float64."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return 1 - np.sum(a * b, 1) / na / nb, np.linalg.norm(a - b, axis=1) / nb
