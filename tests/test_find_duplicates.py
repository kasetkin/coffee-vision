"""coffeecv.find_duplicates (2026-10-24): a messenger-style copy (downscaled, recompressed, rotation baked in,
mirrored) scores above the threshold; another shot of the same scene (shifted framing) and an unrelated photo
score below it, and the CLI exits 1 only on a copy. Plain unittest, synthetic photos, no data needed.

    python -m unittest discover -s tests -p 'test_find_duplicates.py'
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv.find_duplicates import THRESHOLD, main, signature, similarity


def scene(seed: int, w: int = 1600, h: int = 1200) -> Image.Image:
    """Blobs on a gradient: structure at a thumbnail's scale, like beans in a hopper."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 60 + 80 * xx / w
    for _ in range(40):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(40, 160)
        img += rng.uniform(-90, 90) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    g = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([g, g // 2 + 40, 255 - g], -1))


class TestFindDuplicates(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.original = self.dir / "20261023_125135.jpg"
        scene(1).save(self.original, quality=95)

    def save(self, im: Image.Image, name: str, quality: int = 70) -> Path:
        p = self.dir / name
        im.save(p, quality=quality)
        return p

    def score(self, other: Path) -> float:
        return similarity(signature(other), signature(self.original))[0]

    def test_messenger_copy_is_a_copy(self):
        copy = scene(1).rotate(90, expand=True).resize((480, 640), Image.Resampling.LANCZOS)
        self.assertGreaterEqual(self.score(self.save(copy, "photo_1.jpg")), THRESHOLD)

    def test_mirrored_copy_is_a_copy(self):
        copy = scene(1).transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((640, 480))
        self.assertGreaterEqual(self.score(self.save(copy, "photo_2.jpg")), THRESHOLD)

    def test_another_shot_of_the_scene_is_not(self):
        big = scene(1, 1800, 1350)
        shifted = big.crop((160, 120, 1760, 1320))       # the camera moved ~10% between shots
        self.assertLess(self.score(self.save(shifted, "shot2.jpg", quality=95)), THRESHOLD)

    def test_unrelated_photo_is_not(self):
        self.assertLess(self.score(self.save(scene(2), "other.jpg")), 0.5)

    def test_aspect_mismatch_is_never_compared(self):
        square = scene(1).resize((1200, 1200))
        self.assertEqual(self.score(self.save(square, "square.jpg")), -1.0)

    def test_cli_exits_1_only_on_a_copy(self):
        new = self.dir / "new"
        new.mkdir()
        scene(3).save(new / "unique.jpg")
        quiet = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
        with quiet[0], quiet[1]:
            self.assertEqual(main([str(new), "--against", str(self.original), "--jobs", "1"]), 0)
            scene(1).rotate(270, expand=True).resize((480, 640)).save(new / "photo_3.jpg", quality=60)
            self.assertEqual(main([str(new), "--against", str(self.original), "--jobs", "1"]), 1)


if __name__ == "__main__":
    unittest.main()
