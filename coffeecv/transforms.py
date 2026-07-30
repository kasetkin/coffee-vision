"""Train/eval transforms. Full dihedral-group augmentation is valid here since
these are top-down photos of a bean pile with no canonical "up"."""
from __future__ import annotations

import random

import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomRightAngleRotation:
    """Rotate by one of {0, 90, 180, 270} degrees. Exact and lossless on a
    square image — no interpolation/border artifacts, unlike arbitrary-angle
    rotation."""

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        if angle:
            img = TF.rotate(img, angle)
        return img


def build_train_transform(resize: int) -> T.Compose:
    return T.Compose([
        RandomRightAngleRotation(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(resize: int) -> T.Compose:
    return T.Compose([
        T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
