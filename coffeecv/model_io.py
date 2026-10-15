"""What goes into the models and what comes out of them, in numpy, PIL and OpenCV: no torch (ticket ML-7 D5 (b)).

Training and serving both call these, so the train/inference parity rule keeps one implementation of each step
while the release drops torch (plan P8). Each function replaces torch code it is checked against in
tests/test_model_io.py:

    eval_patch       transforms.build_eval_transform: torchvision's Resize on a PIL image is PIL's own resize, so
                     PIL is called directly; ToTensor and Normalize in float32
    sam_input        EfficientViT-SAM's transform (SamResize, ToTensor, Normalize, SamPad): SamResize is PIL already,
                     and sam_resize is it alone
    sam_box, sam_coords
                     SamPredictor.apply_boxes_torch and apply_coords_torch: the prompt scaled from the photo to
                     the resized frame
    upsample_mask    EfficientViTSam.postprocess_masks and the threshold: bilinear to the padded frame, cropped,
                     bilinear to the photo (cv2.resize, INTER_LINEAR), then > 0
    head_logits      the frozen model's Linear head on DINOv3's readout, and `softmax` for its probabilities

The resizes, normalisations, pad and box are bit-identical to the torch code. The mask's upsampling and the head's
matmul round differently, so they are held to ML-7 D7's bars, not to bits.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# EfficientViT-SAM's Normalize (its sam.py), on [0, 1] values.
SAM_MEAN = [123.675 / 255, 116.28 / 255, 103.53 / 255]
SAM_STD = [58.395 / 255, 57.12 / 255, 57.375 / 255]
SAM_MASK_THRESHOLD = 0.0


def _to_chw_normalised(hwc_uint8: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """ToTensor then Normalize, as torchvision does them: uint8 to float32 / 255, then (x - mean) / std with mean
    and std in float32."""
    x = np.ascontiguousarray(hwc_uint8.transpose(2, 0, 1)).astype(np.float32) / np.float32(255)
    m = np.asarray(mean, dtype=np.float32)[:, None, None]
    s = np.asarray(std, dtype=np.float32)[:, None, None]
    return (x - m) / s


def eval_patch(patch: Image.Image, resize: int) -> np.ndarray:
    """A stored patch to the model's input, float32 [3, resize, resize], ImageNet-normalised."""
    if patch.size != (resize, resize):
        patch = patch.resize((resize, resize), Image.BILINEAR)
    return _to_chw_normalised(np.asarray(patch), IMAGENET_MEAN, IMAGENET_STD)


def sam_preprocess_shape(h: int, w: int, long_side: int) -> tuple[int, int]:
    """(h, w) with the long side scaled to `long_side`, rounded as SamResize and ResizeLongestSide round it."""
    scale = long_side * 1.0 / max(h, w)
    return int(h * scale + 0.5), int(w * scale + 0.5)


def sam_resize(rgb: np.ndarray, side: int) -> np.ndarray:
    """SamResize: the photo with its long side at `side`, by PIL's bilinear; unchanged if it is already."""
    h, w = rgb.shape[:2]
    if max(h, w) == side:
        return rgb
    nh, nw = sam_preprocess_shape(h, w, side)
    return np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))


def sam_input(rgb: np.ndarray, side: int) -> tuple[np.ndarray, tuple[int, int]]:
    """(the encoder's input, float32 [1, 3, side, side]; the resized photo's (h, w) inside it). The photo is resized
    (sam_resize), normalised, and padded with zeros at the bottom and right."""
    rgb = sam_resize(rgb, side)
    ih, iw = rgb.shape[:2]
    x = np.zeros((1, 3, side, side), dtype=np.float32)
    x[0, :, :ih, :iw] = _to_chw_normalised(rgb, SAM_MEAN, SAM_STD)
    return x, (ih, iw)


def sam_coords(points, original_size: tuple[int, int], input_size: tuple[int, int]) -> np.ndarray:
    """Points [[x, y], ...] in photo pixels to the resized frame's, float32 [N, 2] (apply_coords_torch)."""
    (oh, ow), (nh, nw) = original_size, input_size
    c = np.array(points, dtype=np.float32).reshape(-1, 2)
    c[:, 0] = c[:, 0] * (nw / ow)
    c[:, 1] = c[:, 1] * (nh / oh)
    return c


def sam_box(box: list[float], original_size: tuple[int, int], input_size: tuple[int, int]) -> np.ndarray:
    """A box [x0, y0, x1, y1] in photo pixels to the resized frame's, float32 [1, 4] (apply_boxes_torch)."""
    return sam_coords(box, original_size, input_size).reshape(-1, 4)


def upsample_mask(low: np.ndarray, input_size: tuple[int, int], original_size: tuple[int, int],
                  frame: int = 1024) -> np.ndarray:
    """One low-resolution mask logit map [256, 256] to the photo's bool mask [H, W]."""
    up = cv2.resize(np.ascontiguousarray(low, dtype=np.float32), (frame, frame), interpolation=cv2.INTER_LINEAR)
    up = np.ascontiguousarray(up[:input_size[0], :input_size[1]])
    h, w = original_size
    return cv2.resize(up, (w, h), interpolation=cv2.INTER_LINEAR) > SAM_MASK_THRESHOLD


def head_logits(readout: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """The Linear head, float32 [N, classes]."""
    return (np.asarray(readout, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T
            + np.asarray(bias, dtype=np.float32))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
