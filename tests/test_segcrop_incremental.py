"""coffeecv.segcrop_session's incremental run (ticket ML-26), with a stand-in segmenter: after a recorded run only new
and changed photos are cropped, a kept photo's report entry is the recorded one, a gone photo's crop, mask and class
folder are deleted, and the session's crops are the same bytes as a full rebuild's.

Plain unittest; the segmenter is replaced by a crop made from the canonical photo's bytes, so no weights are needed.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from coffeecv import segcrop_session as sc
from coffeecv.config import REPO_ROOT
from coffeecv.segment_beans import BeanCrop

from tests import _dvc_lock as dvc_lock

SEGCROP_YAML = """\
stages:
  segcrop:
    foreach:
      - sess
    do:
      cmd: python -m coffeecv.segcrop_session --session ${item} --threads 4
      deps:
        - data/canonical/${item}
        - code.py
      params:
        - params.yaml:
            - seg_prompt
      outs:
        - data/segcropped/${item}:
            persist: true
"""


class FakeSegmenter:
    weights_sha256, decoder_sha256 = "w" * 64, "d" * 64

    def __init__(self, params):
        pass


def fake_crop(path: Path, seg) -> BeanCrop:
    """A crop that depends on the photo's bytes only; a photo named fb_* is a D18 fallback (no mask)."""
    seed = int(hashlib.md5(Path(path).read_bytes()).hexdigest()[:8], 16)
    rgb = np.random.default_rng(seed).integers(0, 255, (12, 16, 3), dtype=np.uint8)
    fallback = Path(path).name.startswith("fb_")
    mask = None if fallback else rgb[..., 0] > 127
    info = {"mask_area_frac": 0.01 if fallback else 0.5, "bean_frac_in_crop": 0.9, "retained_frac": 0.95,
            "fallback": fallback, "box": [0, 0, 16, 12], "mask_sha256": f"{seed:064x}"}
    return BeanCrop(rgb=rgb, mask=mask, info=info)


class Incremental(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / "dvc.yaml").write_text(SEGCROP_YAML)
        shutil.copy(REPO_ROOT / "params.yaml", self.root / "params.yaml")
        (self.root / "code.py").write_text("x = 1\n")
        self.canon = self.root / "data" / "canonical" / "sess"
        for rel in ["class_001/a.jpg.jxl", "class_001/b.jpg.jxl", "class_001/fb_c.jpg.jxl", "class_002/gone.jpg.jxl"]:
            self.photo(rel, rel.encode())
        self.calls: list[str] = []
        for p in (mock.patch.object(sc, "BeanSegmenter", FakeSegmenter),
                  mock.patch.object(sc, "crop_photo", self.counted_crop)):
            p.start()
            self.addCleanup(p.stop)

    def photo(self, rel: str, data: bytes) -> None:
        (self.canon / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.canon / rel).write_bytes(data)

    def counted_crop(self, path, seg):
        self.calls.append(Path(path).name)
        return fake_crop(path, seg)

    def run_stage(self, full: bool = False) -> None:
        self.calls.clear()
        sc.segcrop_session("sess", self.root / "data" / "segcropped", full=full, root=self.root,
                           canonical_root=self.root / "data" / "canonical")

    def crops(self) -> dict[str, bytes]:
        out = self.root / "data" / "segcropped" / "sess"
        return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}

    def commit(self) -> None:
        prompt = yaml.safe_load((self.root / "params.yaml").read_text())["seg_prompt"]
        dvc_lock.commit(self.root, "segcrop@sess", "python -m coffeecv.segcrop_session --session sess --threads 4",
                        ["data/canonical/sess", "code.py"], ["data/segcropped/sess"], {"seg_prompt": prompt})

    def test_only_new_and_changed_photos_are_cropped_and_the_crops_are_a_full_rebuilds(self):
        self.run_stage()
        self.assertEqual(len(self.calls), 4)
        self.commit()
        self.photo("class_001/d.jpg.jxl", b"new photo")
        self.photo("class_001/b.jpg.jxl", b"b, edited")
        shutil.rmtree(self.canon / "class_002")
        self.run_stage()
        self.assertEqual(sorted(self.calls), ["b.jpg.jxl", "d.jpg.jxl"])
        incremental = self.crops()
        self.assertFalse(any(k.startswith("class_002/") for k in incremental))
        self.run_stage(full=True)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(incremental, self.crops())

    def test_a_kept_fallback_stays_without_a_mask(self):
        self.run_stage()
        self.commit()
        self.run_stage()
        self.assertEqual(self.calls, [])
        self.assertNotIn("class_001/fb_c__beanmask.png", self.crops())

    def test_a_crop_that_differs_from_the_record_is_cropped_again(self):
        self.run_stage()
        self.commit()
        crop = self.root / "data/segcropped/sess/class_001/a__cropped.jxl"
        good = crop.read_bytes()
        crop.write_bytes(b"not as recorded")
        self.run_stage()
        self.assertEqual(self.calls, ["a.jpg.jxl"])
        self.assertEqual(crop.read_bytes(), good)

    def test_a_changed_param_crops_everything(self):
        self.run_stage()
        self.commit()
        params = self.root / "params.yaml"
        params.write_text(params.read_text().replace("seg_prompt: box\n", "seg_prompt: grid3\n"))
        self.run_stage()
        self.assertEqual(len(self.calls), 4)


if __name__ == "__main__":
    unittest.main()
