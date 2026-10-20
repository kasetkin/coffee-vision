"""Ticket ML-8 P2: coffeecv.augment, the one builder of every albumentations pipeline, the segmenter's fine-tune
and the classifier's head fit alike. Built strictly: albumentations 2.0.8 only warns on an argument it does not
know and runs with its default (ColorJitter(brightness_range=...) runs at brightness (0.8, 1.2)), so a warning
while a pipeline is built is an error here."""
from __future__ import annotations

import json
import unittest

import numpy as np
import yaml

from coffeecv import augment
from coffeecv.config import PARAMS_FILE, REPO_ROOT, RunConfig


def one(transform: str, **args) -> list[dict]:
    return [{"transform": transform, "p": 1.0, "args": args}]


def photo(h: int = 64, w: int = 80, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


class Strict(unittest.TestCase):
    def test_an_argument_the_pinned_version_does_not_know_fails_naming_it(self):
        with self.assertRaisesRegex(ValueError, r"head_augment: ColorJitter: .*brightness_range"):
            augment.build(one("ColorJitter", brightness_range=[0.5, 0.6]), where="head_augment", seed=0)

    def test_an_unknown_argument_inside_a_oneof_fails(self):
        spec = [{"transform": "OneOf", "p": 1.0,
                 "args": {"transforms": [{"transform": "GaussianBlur", "p": 1.0, "args": {"sigma_range": [1, 2],
                                                                                          "blur_range": [3, 5]}}]}}]
        with self.assertRaisesRegex(ValueError, "GaussianBlur: .*blur_range"):
            augment.build(spec, where="x", seed=0)

    def test_an_unknown_transform_or_a_malformed_entry_fails(self):
        for spec in (one("NoSuchTransform"), one("to_dict"), [{"transform": "HorizontalFlip"}],
                     [{"transform": "HorizontalFlip", "p": 1.0, "argz": {}}], ["HorizontalFlip"]):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                augment.build(spec, where="x", seed=0)

    def test_a_value_the_transform_refuses_fails_with_where_and_name(self):
        with self.assertRaisesRegex(ValueError, r"^x: GaussNoise: "):
            augment.build(one("GaussNoise", std_range=[0.5, 0.1]), where="x", seed=0)

    def test_the_callers_check_is_applied_to_nested_transforms_too(self):
        def no_affine(cls, args):
            return "is not allowed here" if cls.__name__ == "Affine" else None
        nested = [{"transform": "OneOf", "p": 1.0, "args": {"transforms": one("Affine", scale=[0.9, 1.1])}}]
        with self.assertRaisesRegex(ValueError, "^x: Affine is not allowed here"):
            augment.build(nested, where="x", seed=0, check=no_affine)
        augment.build(one("HorizontalFlip"), where="x", seed=0, check=no_affine)

    def test_lists_become_tuples_and_mappings_stay_mappings(self):
        pipe = augment.build(one("Downscale", scale_range=[0.5, 0.9],
                                 interpolation_pair={"downscale": 3, "upscale": 1}), where="x", seed=0)
        t = pipe.transforms[0]
        self.assertEqual((tuple(t.scale_range), t.interpolation_pair), ((0.5, 0.9), {"downscale": 3, "upscale": 1}))

    def test_an_empty_spec_builds_an_empty_pipeline(self):
        rgb = photo()
        np.testing.assert_array_equal(augment.build([], where="x", seed=0)(image=rgb)["image"], rgb)


class Seeded(unittest.TestCase):
    SPEC = [*one("ColorJitter", brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            *one("GaussNoise", std_range=[0.01, 0.05])]

    def draws(self, seed: int, n: int = 4) -> list[bytes]:
        pipe = augment.build(self.SPEC, where="x", seed=seed)
        return [pipe(image=photo(seed=i))["image"].tobytes() for i in range(n)]

    def test_one_seed_repeats_its_stream_and_another_seed_differs(self):
        self.assertEqual(self.draws(3), self.draws(3))
        self.assertNotEqual(self.draws(3), self.draws(4))

    def test_the_update_check_is_off(self):
        import os
        augment.albumentations()
        self.assertEqual(os.environ.get("NO_ALBUMENTATIONS_UPDATE"), "1")
        self.assertEqual(augment.version(), "2.0.8")


class Shared(unittest.TestCase):
    """params.yaml `augment` (ADR 0027): groups both fits take by reference, {use: <group>}, expanded in place."""
    GROUPS = {"colour": one("ColorJitter", brightness=0.2), "noise": one("GaussNoise")}

    def test_a_use_entry_is_replaced_by_its_group_in_place(self):
        spec = one("HorizontalFlip") + [{"use": "colour"}] + one("Downscale") + [{"use": "noise"}]
        self.assertEqual(augment.expand(spec, self.GROUPS, where="x"),
                         one("HorizontalFlip") + self.GROUPS["colour"] + one("Downscale") + self.GROUPS["noise"])

    def test_a_pipeline_without_use_is_unchanged(self):
        self.assertEqual(augment.expand(one("HorizontalFlip"), self.GROUPS, where="x"), one("HorizontalFlip"))
        self.assertEqual(augment.expand([], {}, where="x"), [])

    def test_an_unknown_group_is_refused_naming_the_pipeline(self):
        with self.assertRaisesRegex(ValueError, r"seg_ft.augment: .*'color'.*colour, noise"):
            augment.expand([{"use": "color"}], self.GROUPS, where="seg_ft.augment")

    def test_a_malformed_use_or_a_group_that_uses_another_is_refused(self):
        for spec, groups in (([{"use": "colour", "p": 1.0}], self.GROUPS),
                             ([{"use": "a"}], {"a": [{"use": "b"}], "b": one("GaussNoise")}),
                             ([{"use": "a"}], {"a": {"transform": "GaussNoise", "p": 1.0}})):
            with self.subTest(spec=spec, groups=groups), self.assertRaises(ValueError):
                augment.expand(spec, groups, where="x")


class SharedInParams(unittest.TestCase):
    """params.yaml as written: both fits name every shared group, so neither holds a copy that can drift."""

    @classmethod
    def setUpClass(cls):
        cls.raw = yaml.safe_load(PARAMS_FILE.read_text())
        cls.cfg = RunConfig.from_params_yaml()

    def test_both_fits_take_every_group_by_reference(self):
        groups = set(self.raw["augment"])
        self.assertEqual(groups, {"colour", "compression", "noise"})
        uses = lambda spec: {t["use"] for t in spec if "use" in t}  # noqa: E731
        self.assertEqual(uses(self.raw["seg_ft"]["augment"]), groups)
        self.assertEqual(uses(self.raw["head_augment"]["patch"]) | uses(self.raw["head_augment"]["input"]), groups)

    def test_the_loaded_config_holds_the_expanded_pipelines(self):
        for spec in (self.cfg.seg_ft["augment"], self.cfg.head_augment["patch"], self.cfg.head_augment["input"]):
            self.assertFalse([t for t in spec if "use" in t])
        seg = self.cfg.seg_ft["augment"]
        for name, group in self.cfg.augment.items():
            with self.subTest(group=name):
                self.assertTrue(any(seg[i:i + len(group)] == group for i in range(len(seg))))

    def test_the_classifiers_pipeline_is_the_one_exp270_272_were_fitted_with(self):
        [run] = (REPO_ROOT / "experiments").glob("exp271__*")
        self.assertEqual(self.cfg.head_augment, json.loads((run / "config.json").read_text())["head_augment"])

    def test_the_segmenter_is_the_classifier_and_more(self):
        # ADR 0027: the segmenter adds its own on top of the shared groups (a hue shift, gamma), never weaker ones.
        names = [t["transform"] for t in self.cfg.seg_ft["augment"]]
        self.assertNotIn("RandomBrightnessContrast", names)
        self.assertLess(names.index("ColorJitter"), names.index("HueSaturationValue"))
        self.assertIn("RandomGamma", names)


if __name__ == "__main__":
    unittest.main()
