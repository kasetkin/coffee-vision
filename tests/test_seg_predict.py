"""The seg_predict stage's photo set (ticket ML-2 P1, plan §4.2): the three eval lists, one unique id per
photo, and every base candidate among them, so each has a stored mask. Plain unittest; needs only the
committed photo_lists.yaml.

    python -m unittest discover -s tests -p 'test_seg_predict.py'
"""
from __future__ import annotations

import unittest

from coffeecv import seg_lists
from coffeecv.config import RunConfig
from coffeecv.sam_loader import weights_for
from coffeecv.seg_predict import LISTS, model_params, photo_entries


class PhotoEntries(unittest.TestCase):
    def setUp(self):
        _, self.lists = seg_lists.load_lists()

    def test_covers_the_eval_lists_once(self):
        entries = photo_entries(self.lists)
        self.assertEqual(len(entries), sum(len(self.lists[k]) for k in LISTS))
        self.assertEqual(len({e["id"] for e in entries}), len(entries))

    def test_every_base_candidate_has_a_mask(self):
        paths = {e["path"] for e in photo_entries(self.lists)}
        self.assertEqual({e["path"] for e in self.lists["base_candidates"]} - paths, set())

    def test_repeated_id_is_an_error(self):
        row = self.lists["seg_eval"][0]
        lists = {"seg_eval": [row], "neg_seg_eval": [], "pos_seg_eval": [dict(row)]}
        with self.assertRaises(ValueError):
            photo_entries(lists)



class Models(unittest.TestCase):
    """ML-5 P4: "pretrained" is the fine-tune's base, params.yaml seg_ft.weights; a fine-tuned decoder runs over
    the base its card records, so moving the fine-tune to XL0 is one edit and ML-2's decoders stay on L0."""

    def cfg(self, variant: str) -> RunConfig:
        return RunConfig(seg_ft={"weights": weights_for(variant)}, seg_mask_select="multi3", seg_prompt="box")

    def test_pretrained_is_the_fine_tunes_base(self):
        for v in ("l0", "xl0"):
            p = model_params("pretrained", self.cfg(v))
            self.assertEqual((p.weights, p.decoder, p.mask_select, p.prompt), (weights_for(v), None, "multi3", "box"))

    def test_a_decoder_runs_over_the_base_its_card_records(self):
        p = model_params("ft_s123", self.cfg("xl0"))
        self.assertEqual((p.weights, p.decoder), ("efficientvit_sam/efficientvit_sam_l0.pt", "models/seg/ft_s123.pt"))


if __name__ == "__main__":
    unittest.main()
