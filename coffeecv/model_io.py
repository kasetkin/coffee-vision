"""What goes into the models and what comes out of them, in numpy, PIL and OpenCV: no torch (ticket ML-7 D5 (b)).

Training and serving both call these, so the train/inference parity rule keeps one implementation of each step
while the release drops torch (plan P8). Each function replaces torch code it is checked against in
tests/test_model_io.py:

    eval_patch       transforms.build_eval_transform: Resize, ToTensor and Normalize; the resize is eval_resize,
                     which the frozen head's augmented views share, and the normalisation eval_input
    sam_input        EfficientViT-SAM's transform (SamResize, ToTensor, Normalize, SamPad); sam_resize is SamResize's
                     geometry
    sam_box, sam_coords
                     SamPredictor.apply_boxes_torch and apply_coords_torch: the prompt scaled from the photo to
                     the resized frame
    upsample_mask    EfficientViTSam.postprocess_masks and the threshold: bilinear to the padded frame, cropped,
                     bilinear to the photo (cv2.resize, INTER_LINEAR), then > 0
    head_logits      the frozen model's Linear head on DINOv3's readout, and `softmax` for its probabilities

The two resizes take the 8-bit picture and give float32 in [0, 1] (ML-37 D11 (b)): PIL's bilinear in mode "F", the
filter PIL's "RGB" resize runs, with no rounding to 8 bits after it. They are held to torch's antialiased bilinear,
the same filter, within 0.01 of a level, and lie within one level of the 8-bit resize they replaced. The
normalisations, pad and box are bit-identical to the torch code. The mask's upsampling and the head's matmul round
differently, so they are held to ML-7 D7's bars, not to bits.
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


def _to_chw_normalised(hwc01: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """A float32 HxWx3 picture in [0, 1] (a resize's output) to CHW, then Normalize as torchvision does it:
    (x - mean) / std with mean and std in float32."""
    if hwc01.dtype != np.float32:
        raise TypeError(f"normalising takes a float32 picture in [0, 1] (a resize's output), not {hwc01.dtype}")
    x = np.ascontiguousarray(hwc01.transpose(2, 0, 1))
    m = np.asarray(mean, dtype=np.float32)[:, None, None]
    s = np.asarray(std, dtype=np.float32)[:, None, None]
    return (x - m) / s


def eval_resize(patch: Image.Image | np.ndarray, resize: int) -> np.ndarray:
    """A stored 8-bit patch (a PIL image or HxWx3 uint8) at the model's input size, by PIL's bilinear, float32 in
    [0, 1] with no rounding after the resize (ML-37 D11 (b)); only scaled to [0, 1] if it is at size already.
    eval_patch's resize, and the one the frozen head's augmented views take between their two pipelines (ticket ML-8
    D4)."""
    return _resize_float(np.asarray(patch), resize, resize)


def eval_input(picture: np.ndarray) -> np.ndarray:
    """A float32 picture in [0, 1] at the model's input size (eval_resize's output, or an augmented view of it) to
    the model's input, float32 [3, H, W], ImageNet-normalised."""
    return _to_chw_normalised(picture, IMAGENET_MEAN, IMAGENET_STD)


def eval_patch(patch: Image.Image | np.ndarray, resize: int) -> np.ndarray:
    """A stored 8-bit patch to the model's input, float32 [3, resize, resize], ImageNet-normalised."""
    return eval_input(eval_resize(patch, resize))


def sam_preprocess_shape(h: int, w: int, long_side: int) -> tuple[int, int]:
    """(h, w) with the long side scaled to `long_side`, rounded as SamResize and ResizeLongestSide round it."""
    scale = long_side * 1.0 / max(h, w)
    return int(h * scale + 0.5), int(w * scale + 0.5)


def _check_8_bit(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TypeError(f"a model's resize takes the 8-bit picture, HxWx3 uint8, not {rgb.dtype} {rgb.shape}")
    return rgb


def _resize_float(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """HxWx3 uint8 -> h x w x 3 float32 in [0, 1]: PIL's bilinear in mode "F", channel by channel (ML-37 D11 (b)).
    The same filter as on "RGB" (one precompute_coeffs, widened on a downscale), with no rounding to 8 bits after
    each pass or at the end; its weights are non-negative and sum to 1, so it stays in [0, 1]. At (h, w) already,
    the picture is only scaled to [0, 1]."""
    rgb = _check_8_bit(rgb)
    if rgb.shape[:2] == (h, w):
        return rgb.astype(np.float32) / np.float32(255)
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        plane = Image.fromarray(np.ascontiguousarray(rgb[:, :, c], dtype=np.float32), mode="F")
        out[:, :, c] = np.asarray(plane.resize((w, h), Image.BILINEAR))
    out *= np.float32(1 / 255)
    return np.clip(out, 0, 1, out=out)      # float32 rounding of a sum of weights near 1, no more


def sam_resize(rgb: np.ndarray, side: int) -> np.ndarray:
    """SamResize's geometry on the 8-bit photo: its long side at `side`, by PIL's bilinear, float32 in [0, 1] with
    no rounding after the resize (ML-37 D11 (b))."""
    h, w = rgb.shape[:2]
    return _resize_float(rgb, *sam_preprocess_shape(h, w, side))


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
