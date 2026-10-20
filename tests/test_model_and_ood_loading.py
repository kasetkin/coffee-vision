"""The checked OOD-artifact loaders, and what config_for_checkpoint reads from a card. Plain unittest. (The
ResNet18 embedding contract went with its code, 2026-10-20; the frozen head's is tests/test_frozen_checkpoint.py.)

    python -m unittest discover -s tests -p 'test_model_and_ood_loading.py'

On real artifacts: the live checkpoint `models/dino3b16_segxl0v3_country_aug8_s123.pt` (ticket ML-8) with its own OOD
reference and probe, ML-3's card (`models/allrigs_dino3b16_seg_country_s123.json`) for the pools before ML-8, and real bean patches from `coffeecv_dino.reference`. The checkpoint is DVC-tracked and
its ViT-B/16 backbone is not in git, so the tests that need them skip with a reason when either is absent.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path


from coffeecv.backbones import MODELS_PRETRAINED, SPECS
from coffeecv.config import REPO_ROOT
from coffeecv.dataset import CAPTURES
from coffeecv.infer import (LEGACY_CONFIG_NOTE, config_for_checkpoint, embedding_dim_of, load_model,
                            load_ood_probe, load_ood_reference, probe_path_for, reference_path_for)


DEPLOYED = REPO_ROOT / "models" / "dino3b16_segxl0v3_country_aug8_s123.pt"
# Fitted on the four camera pools before ticket ML-8 (D2) merged them into one: its card still names them.
FOUR_POOLS = REPO_ROOT / "models" / "allrigs_dino3b16_seg_country_s123.pt"
HAVE_DEPLOYED = (DEPLOYED.exists() and reference_path_for(DEPLOYED).exists() and probe_path_for(DEPLOYED).exists()
                 and (MODELS_PRETRAINED / SPECS["dinov3_vitb16"].weights).exists())


@unittest.skipUnless(HAVE_DEPLOYED, f"{DEPLOYED}, its OOD sidecars or the ViT-B/16 weights are not present")
class TestOodLoaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.head = load_model(DEPLOYED, "dinov3_vitb16", 10)
        cls.dim = embedding_dim_of(cls.head)

    def _copy_with(self, tmp: Path, suffix: str, **overrides) -> Path:
        ckpt = tmp / DEPLOYED.name
        if not ckpt.exists():
            shutil.copy2(DEPLOYED, ckpt)
        src = DEPLOYED.with_suffix(suffix)
        data = json.loads(src.read_text())
        data.update(overrides)
        (tmp / src.name).write_text(json.dumps(data))
        return ckpt

    def test_deployed_pairing_loads(self):
        self.assertEqual(embedding_dim_of(self.head), 1536)
        self.assertEqual(load_ood_reference(DEPLOYED, self.dim)["embedding_dim"], 1536)
        self.assertEqual(load_ood_probe(DEPLOYED, embedding_dim=self.dim)["embedding_dim"], 1536)

    def test_width_mismatch_refused(self):
        # A ResNet18 (512-d) reference beside this DINOv3 ViT-B/16 cls_mean model: the artifact's width is
        # not what this head is fed.
        with tempfile.TemporaryDirectory() as d:
            ckpt = self._copy_with(Path(d), ".ood_reference.json", embedding_dim=512)
            with self.assertRaisesRegex(SystemExit, "512-d but this model emits 1536-d"):
                load_ood_reference(ckpt, self.dim)
            ckpt = self._copy_with(Path(d), ".ood_probe.json", embedding_dim=512)
            with self.assertRaisesRegex(SystemExit, "512-d but this model emits 1536-d"):
                load_ood_probe(ckpt, embedding_dim=self.dim)

    def test_other_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = self._copy_with(Path(d), ".ood_reference.json", checkpoint_sha="0000000000000000")
            with self.assertRaisesRegex(SystemExit, "different checkpoint"):
                load_ood_reference(ckpt, self.dim)

    def test_absent_reference_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "x.pt"
            ckpt.write_bytes(b"")
            self.assertIsNone(load_ood_reference(ckpt, self.dim))


class TestLegacyTrainRigsAlias(unittest.TestCase):
    """Ticket ML-1 (D4) renamed `train_rigs` to `train_capture_dirs`; every card and run config written
    before then still says `train_rigs` (the retired models' cards, the archived experiments' configs).
    config_for_checkpoint must map it, or the key is dropped as unknown and an old model silently resolves
    to the default dirs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def card(self, training_config: dict) -> Path:
        ckpt = self.tmp / "m.pt"
        ckpt.with_suffix(".json").write_text(json.dumps({"training_config": training_config}))
        return ckpt

    def test_old_key_is_read_and_flagged(self):
        dirs = ["data/cropped/cam_sony", "data/cropped/cam_iphone"]
        cfg, source = config_for_checkpoint(self.card({"train_rigs": dirs, "heldout_rig": ""}), None)
        self.assertEqual(list(cfg.train_capture_dirs), dirs)
        self.assertIn(LEGACY_CONFIG_NOTE, source)

    def test_new_key_wins_and_is_not_flagged(self):
        cfg, source = config_for_checkpoint(
            self.card({"train_capture_dirs": ["data/cropped/cam_sony"], "train_rigs": ["x"]}), None)
        self.assertEqual(cfg.train_capture_dirs, ("data/cropped/cam_sony",))
        self.assertNotIn(LEGACY_CONFIG_NOTE, source)

    def test_a_shipped_card_resolves_to_the_pools_it_trained_on(self):
        cfg, _ = config_for_checkpoint(FOUR_POOLS, None)
        self.assertEqual([str(d.relative_to(REPO_ROOT)) for d in cfg.resolve_paths()[0]],
                         [f"data/segcropped/cam_{c}" for c in ("pixel", "sony", "oneplus", "iphone")])
        cfg, _ = config_for_checkpoint(DEPLOYED, None)
        self.assertEqual([str(d.relative_to(REPO_ROOT)) for d in cfg.resolve_paths()[0]], CAPTURES)
        self.assertEqual(CAPTURES, ["data/segcropped/pool"])


class TestRecordedClassIds(unittest.TestCase):
    """Ticket ML-3: a run records `classes_file` as a path, so after classes.txt changed format a pre-ML-3
    run rebuilt at a later commit would read countries where it was fitted on folder ids. A config whose
    recorded class_ids differ from what its class list now reads is refused, naming the commit."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.classes = self.tmp / "classes.txt"

    def config(self, class_ids) -> str:
        path = self.tmp / "config.json"
        path.write_text(json.dumps({"classes_file": str(self.classes), "class_ids": class_ids,
                                    "env": {"git_commit": "abc123"}}))
        return str(path)

    def test_same_classes_load(self):
        self.classes.write_text("001;Ethiopia,Sidamo\n006;Brazil,Cerrado\n")
        cfg, _ = config_for_checkpoint(self.tmp / "unused.pt", self.config(["001", "006"]))
        self.assertEqual(cfg.classes_file, str(self.classes))

    def test_folder_run_on_a_country_list_is_refused(self):
        self.classes.write_text("001;Ethiopia,Sidamo;\n006;Brazil,Cerrado;\n007;Brazil,Cerrado;MonteCristo\n")
        with self.assertRaisesRegex(ValueError, "rebuild it at abc123"):
            config_for_checkpoint(self.tmp / "unused.pt", self.config(["001", "006", "007"]))

    def test_shipped_cards_read_their_frozen_lists(self):
        for card in sorted((REPO_ROOT / "models").glob("*.classes.txt")):
            ckpt = card.with_name(card.name.replace(".classes.txt", ".pt"))
            if ckpt.with_suffix(".json").exists():
                with self.subTest(model=ckpt.name):
                    cfg, _ = config_for_checkpoint(ckpt, None)
                    self.assertEqual(Path(cfg.classes_file), card.resolve())


class TestPredictionsProvenance(unittest.TestCase):
    """metrics.write_predictions_csv (ticket ML-3): DVC's confusion template reads true_label, pred_label;
    with meta and probabilities each row also says where the patch came from and what the head gave."""

    def test_columns(self):
        import csv

        import numpy as np

        from coffeecv.dataset import PatchMeta
        from coffeecv.geometry import Region
        from coffeecv.metrics import write_predictions_csv

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        keys = ["Ethiopia", "Brazil"]
        meta = [PatchMeta("Brazil", "cam_pixel", "a__cropped.jpg", Region(0, 1, 0, 1), 0.0, 1, "013"),
                PatchMeta("Ethiopia", "cam_sony", "b__cropped.jpg", Region(0, 1, 0, 1), 0.0, 1, "008")]
        probs = np.array([[0.25, 0.75], [0.9, 0.1]], dtype=np.float32)
        write_predictions_csv(tmp / "p.csv", np.array([1, 0]), np.array([1, 0]), keys, meta, probs)
        rows = list(csv.DictReader((tmp / "p.csv").open()))
        self.assertEqual(list(rows[0]), ["true_label", "pred_label", "capture", "photo", "folder_id",
                                         "p_Ethiopia", "p_Brazil"])
        self.assertEqual((rows[0]["true_label"], rows[0]["folder_id"], rows[0]["p_Brazil"]), ("Brazil", "013", "0.75"))
        write_predictions_csv(tmp / "q.csv", np.array([1]), np.array([0]), keys)
        self.assertEqual((tmp / "q.csv").read_text().splitlines(), ["true_label,pred_label", "Brazil,Ethiopia"])


if __name__ == "__main__":
    unittest.main()
