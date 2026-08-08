"""Train/eval transforms. Full dihedral-group augmentation is valid here since
these are top-down photos of a bean pile with no canonical "up".

Every augmentation below defaults to a no-op, so `build_train_transform` with
default arguments composes exactly the pre-Phase-8 pipeline and consumes exactly
the same RNG draws — that keeps Phase 7's recorded baselines valid without
re-running them.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _mean_fill(img) -> list[int]:
    """Per-patch mean RGB, used as the fill colour for geometric transforms that
    leave undefined pixels. A bean pile's own mean colour blends into the border
    far better than a constant black, which would otherwise put a hard
    high-contrast frame around every rotated patch."""
    return [int(c) for c in np.asarray(img).reshape(-1, 3).mean(axis=0)]


class RandomRightAngleRotation:
    """Rotate by one of {0, 90, 180, 270} degrees. Exact and lossless on a
    square image — no interpolation/border artifacts, unlike arbitrary-angle
    rotation."""

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        if angle:
            img = TF.rotate(img, angle)
        return img


class RandomArbitraryRotation:
    """Rotate by a random angle in [-degrees, +degrees], keeping the patch size
    fixed and filling the resulting corner wedges with the patch's mean colour.

    The alternative — oversampling a larger source crop so the rotation can be
    cropped back down artifact-free — is not available on this dataset: the
    cropped photos are only ~1050x1520px, so after `safety_margin` the valid
    region is ~1018px wide and a 900px patch already has just ~118px of
    horizontal placement room. There is no headroom to oversample into.
    """

    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return TF.rotate(
            img, angle, interpolation=T.InterpolationMode.BILINEAR, fill=_mean_fill(img)
        )


class RandomPerspectiveMeanFill:
    """`T.RandomPerspective` with the same mean-colour fill as
    `RandomArbitraryRotation`, instead of torchvision's constant fill."""

    def __init__(self, distortion_scale: float, p: float = 0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        width, height = img.size
        startpoints, endpoints = T.RandomPerspective.get_params(
            width, height, self.distortion_scale
        )
        return TF.perspective(
            img, startpoints, endpoints,
            interpolation=T.InterpolationMode.BILINEAR, fill=_mean_fill(img),
        )


class RandomIlluminationGradient:
    """Multiply the image by a linear luminance ramp of random direction,
    spanning 1-strength .. 1+strength corner to corner.

    Motivated by something observed in this project rather than generic advice:
    real directional lighting drift over the ~2h 2026-08-07 capture session is
    what broke the original adaptive crop heuristic. Deliberately a *spatial*
    gradient — uniform brightness is already covered by `T.ColorJitter`, so a
    flat offset here would add nothing new. No radial vignette either: a patch
    is a 900px crop from a 1050x1520 photo, so any real vignette is centred
    somewhere outside the patch, and a linear ramp is the honest local
    approximation of that.

    Operates on a [0, 1] tensor (before normalization) so the gain is applied in
    image space, where "20% brighter" actually means that.
    """

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        _, h, w = tensor.shape
        theta = random.uniform(0, 2 * np.pi)
        ys = torch.linspace(-0.5, 0.5, h).view(h, 1)
        xs = torch.linspace(-0.5, 0.5, w).view(1, w)
        ramp = np.cos(theta) * xs + np.sin(theta) * ys  # in [-0.5, 0.5] along the drawn direction
        gain = 1.0 + 2.0 * self.strength * ramp
        return (tensor * gain).clamp(0.0, 1.0)


def build_train_transform(
    resize: int,
    jitter_strength: float = 0.2,
    rotation_degrees: float = 0.0,
    zoom_scale_min: float = 1.0,
    random_erasing_p: float = 0.0,
    perspective_distortion: float = 0.0,
    illum_gradient_strength: float = 0.0,
) -> T.Compose:
    steps = [
        RandomRightAngleRotation(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ]
    if rotation_degrees > 0:
        steps.append(RandomArbitraryRotation(rotation_degrees))
    if perspective_distortion > 0:
        steps.append(RandomPerspectiveMeanFill(perspective_distortion))
    if jitter_strength > 0:
        steps.append(T.ColorJitter(
            brightness=jitter_strength, contrast=jitter_strength,
            saturation=jitter_strength, hue=min(jitter_strength * 0.1, 0.5),
        ))
    # `scale` is an *area* fraction: 0.7 means zooming into a ~0.84-linear
    # sub-region. `ratio` is pinned to 1.0 so this stays a pure zoom and doesn't
    # smuggle in aspect-ratio distortion as a second variable.
    if zoom_scale_min < 1.0:
        steps.append(T.RandomResizedCrop(
            resize, scale=(zoom_scale_min, 1.0), ratio=(1.0, 1.0),
            interpolation=T.InterpolationMode.BILINEAR,
        ))
    else:
        steps.append(T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR))
    steps.append(T.ToTensor())
    if illum_gradient_strength > 0:
        steps.append(RandomIlluminationGradient(illum_gradient_strength))
    steps.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    # After Normalize, so value=0 erases to the ImageNet mean colour rather than
    # to black — the same reasoning as the mean fill used for rotation above.
    if random_erasing_p > 0:
        steps.append(T.RandomErasing(p=random_erasing_p, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0))
    return T.Compose(steps)


def build_eval_transform(resize: int) -> T.Compose:
    return T.Compose([
        T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
