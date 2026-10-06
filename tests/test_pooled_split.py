"""The pooled photo split (ticket ML-1, D1(b)): one 70/15/15 split per class over every capture dir's
photos, deterministic, independent of the order the dirs are listed in, and -- because a dir is no
longer guaranteed a share of each split -- tolerant of a dir that lands no photos in one. Since ticket
ML-3 a class pools several class folders (a country's coffees). Plain unittest.

    python -m unittest discover -s tests -p 'test_pooled_split.py'

Runs on synthetic crops written to a temp dir, so it needs no DVC data.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv.class_list import ClassList, folder_classes
from coffeecv.config import RunConfig
from coffeecv.dataset import (CAPTURES, MultiPhotoPatchDataset, pooled_class_photos, resolve_captures,
                              split_census, split_photos_by_class)

FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}


def write_capture(root: Path, name: str, counts: dict[str, int], per_folder_names: bool = False) -> Path:
    """<root>/<name>/class_<id>__X/<name>_<i>__cropped.jpg, `counts[id]` photos per class. The names repeat
    across a capture's folders unless `per_folder_names`, which puts the folder id in them too."""
    rng = np.random.default_rng(len(name))
    for cid, n in counts.items():
        d = root / name / f"class_{cid}__X"
        d.mkdir(parents=True)
        stem = f"{name}_{cid}" if per_folder_names else name
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)).save(
                d / f"{stem}_{i:03d}__cropped.jpg")
    return root / name


class TestPooledSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        # 20 + 10 photos of class 001; only capA has class 002.
        cls.a = write_capture(cls.tmp, "capA", {"001": 20, "002": 12})
        cls.b = write_capture(cls.tmp, "capB", {"001": 10})
        (cls.tmp / "classes.txt").write_text("001;one\n002;two\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def split(self, dirs, seed=7, class_idx=0, cid="001"):
        pool, _ = pooled_class_photos(resolve_captures(dirs), [cid])
        return split_photos_by_class(pool, seed, class_idx, FRAC)

    def test_fractions_apply_to_the_pooled_count(self):
        s = self.split([self.a, self.b])
        # Pooled 30 -> val round(4.5)=4, test 4, train 22. Two per-dir splits would give
        # 14/3/3 + 6/2/2 = 20/5/5 instead, so this tells the two apart.
        self.assertEqual([len(s[k]) for k in ("train", "val", "test")], [22, 4, 4])
        everything = s["train"] + s["val"] + s["test"]
        self.assertEqual(len(set(everything)), 30)                 # disjoint, and nothing lost

    def test_deterministic_across_calls(self):
        self.assertEqual(self.split([self.a, self.b]), self.split([self.a, self.b]))

    def test_independent_of_capture_dir_order(self):
        self.assertEqual(self.split([self.a, self.b]), self.split([self.b, self.a]))

    def test_seed_and_class_move_the_split(self):
        base = self.split([self.a, self.b], seed=7)
        self.assertNotEqual(base["val"], self.split([self.a, self.b], seed=8)["val"])
        self.assertNotEqual(base["val"], self.split([self.a, self.b], seed=7, class_idx=1)["val"])

    def test_each_split_comes_back_sorted(self):
        for photos in self.split([self.a, self.b]).values():
            self.assertEqual(photos, sorted(photos))

    def test_too_few_pooled_photos_raise(self):
        pool, _ = pooled_class_photos(resolve_captures([self.b]), ["001"])
        with self.assertRaisesRegex(ValueError, "too few"):
            split_photos_by_class(pool[:2], 7, 0, FRAC)

    def test_duplicate_capture_names_are_refused(self):
        other = self.tmp / "elsewhere"
        write_capture(other, "capA", {"001": 3})
        try:
            with self.assertRaisesRegex(ValueError, "unique"):
                resolve_captures([self.a, other / "capA"])
        finally:
            shutil.rmtree(other)

    def test_census_reports_every_dir_per_split(self):
        census = split_census(resolve_captures([self.a, self.b]), folder_classes(self.tmp / "classes.txt"), 7, FRAC)
        c1 = census["001"]
        self.assertEqual(c1["pooled"], 30)
        self.assertEqual(c1["absent"], [])
        for split, n in (("train", 22), ("val", 4), ("test", 4)):
            self.assertEqual(set(c1[split]), {"capA", "capB"})     # zero counts are listed, not dropped
            self.assertEqual(sum(c1[split].values()), n)
        s = self.split([self.a, self.b])
        self.assertEqual(c1["val"]["capB"], sum(1 for p in s["val"] if p.capture == "capB"))
        self.assertEqual(census["002"]["absent"], ["capB"])
        self.assertEqual(set(census["002"]["val"]), {"capA"})      # an absent dir is not "starved"


class TestStarvedCaptureDoesNotCrash(unittest.TestCase):
    """A dir can land zero photos in a split for a class. That used to be impossible (every dir was
    floored at one val and one test photo) and would now divide by zero when the patch budget is spread
    over that dir's photos; it must instead be skipped, warned about and recorded."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.a = write_capture(cls.tmp, "capA", {"001": 20})
        cls.b = write_capture(cls.tmp, "capB", {"001": 1})     # one photo: it lands in exactly one split
        (cls.tmp / "classes.txt").write_text("001;one\n")
        cls.captures = resolve_captures([cls.a, cls.b])
        pool, _ = pooled_class_photos(cls.captures, ["001"])
        cls.where = {s: [p for p in ps if p.capture == "capB"]
                     for s, ps in split_photos_by_class(pool, 7, 0, FRAC).items()}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def build(self, split):
        return MultiPhotoPatchDataset(
            captures=self.captures, classes=folder_classes(self.tmp / "classes.txt"), split=split,
            seed=7, crop_size=32, resize=32, safety_margin=0.97,
            patches_per_class={"train": 6, "val": 4, "test": 4}, photo_frac=FRAC)

    def test_starved_split_is_skipped_and_recorded(self):
        [home] = [s for s, ps in self.where.items() if ps]
        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                ds = self.build(split)
                dirs = {m.capture for m in ds._meta}
                if split == home:
                    self.assertEqual(ds.starved, [])
                    self.assertEqual(dirs, {"capA", "capB"})
                    self.assertEqual(sum(m.capture == "capB" for m in ds._meta),
                                     {"train": 6, "val": 4, "test": 4}[split])  # the whole dir budget
                else:
                    self.assertEqual(ds.starved, [("capB", "001")])
                    self.assertEqual(dirs, {"capA"})
                self.assertEqual(ds.present_class_idxs, [0])

    def test_all_split_keeps_every_photo_in_name_order(self):
        """split="all" (the DINOv3 fixture's path): each dir's photos in file-name order, photo_idx from 0."""
        ds = MultiPhotoPatchDataset(
            captures=[self.captures[0]], classes=folder_classes(self.tmp / "classes.txt"), split="all",
            seed=7, crop_size=32, resize=32, safety_margin=0.97,
            patches_per_class={"train": 1, "val": 1, "test": 1, "all": 20}, photo_frac=FRAC)
        names = [m.photo_name for m in ds._meta]
        self.assertEqual(names, sorted(p.name for p in (self.a / "class_001__X").glob("*__cropped.jpg")))


class TestCountryPool(unittest.TestCase):
    """Ticket ML-3: a country's class folders pool in each capture (D6), and the split key stays
    (capture, photo name), so a name two of its folders share in one capture is refused."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        # Country "B" is folders 002 and 003: 12 + 6 photos in capA, 3 of 002 in capB. Country "A" is 001.
        cls.a = write_capture(cls.tmp, "capA", {"001": 12, "002": 12, "003": 6}, per_folder_names=True)
        cls.b = write_capture(cls.tmp, "capB", {"001": 4, "002": 3}, per_folder_names=True)
        (cls.tmp / "classes.txt").write_text("001;A,One;\n002;B,Two;\n003;B,Three;x\n")
        cls.captures = resolve_captures([cls.a, cls.b])
        cls.classes = ClassList(("A", "B"), {"A": "A", "B": "B"}, {"A": ("001",), "B": ("002", "003")})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_two_folders_form_one_pool(self):
        pool, absent = pooled_class_photos(self.captures, ("002", "003"))
        self.assertEqual(len(pool), 21)
        self.assertEqual(absent, [])
        self.assertEqual(pool, sorted(pool))
        self.assertEqual({(p.capture, p.folder_id) for p in pool},
                         {("capA", "002"), ("capA", "003"), ("capB", "002")})
        # absent only when a capture has none of the folders
        self.assertEqual(pooled_class_photos(self.captures, ("003",))[1], ["capB"])

    def test_one_id_as_a_string_is_refused(self):
        with self.assertRaises(TypeError):
            pooled_class_photos(self.captures, "002")

    def test_budget_spreads_over_both_folders(self):
        ds = MultiPhotoPatchDataset(
            captures=self.captures, classes=self.classes, split="train", seed=7, crop_size=32, resize=32,
            safety_margin=0.97, patches_per_class={"train": 30, "val": 4, "test": 4}, photo_frac=FRAC)
        b_a = [m for m in ds._meta if m.class_id == "B" and m.capture == "capA"]
        self.assertEqual(len(b_a), 30)                              # one per-(capture, class) budget
        self.assertEqual({m.folder_id for m in b_a}, {"002", "003"})
        self.assertTrue(all(m.photo_name.startswith(f"capA_{m.folder_id}_") for m in b_a))   # provenance
        labels = {ds[i][1] for i in range(len(ds)) if ds._meta[i].class_id == "B"}
        self.assertEqual(labels, {1})

    def test_split_is_per_country(self):
        census = split_census(self.captures, self.classes, 7, FRAC)
        self.assertEqual(list(census), ["A", "B"])
        self.assertEqual(census["B"]["pooled"], 21)
        self.assertEqual(sum(census["B"]["val"].values()), round(0.15 * 21))

    def test_a_name_repeated_across_folders_raises(self):
        clash = write_capture(self.tmp / "clash", "capC", {"002": 3, "003": 3})   # capC_000 in both
        try:
            pool, _ = pooled_class_photos(resolve_captures([clash]), ("002", "003"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                split_photos_by_class(pool, 7, 0, FRAC)
        finally:
            shutil.rmtree(self.tmp / "clash")


class TestRestingCaptureDirs(unittest.TestCase):
    def test_config_default_and_params_rest_at_captures(self):
        """RunConfig keeps a literal copy of dataset.CAPTURES (to avoid importing the dataset stack), and
        params.yaml rests at it; all three must agree, order included -- the order seeds patch boxes."""
        self.assertEqual(RunConfig().train_capture_dirs, tuple(CAPTURES))
        self.assertEqual(RunConfig.from_params_yaml().train_capture_dirs, tuple(CAPTURES))


if __name__ == "__main__":
    unittest.main()
