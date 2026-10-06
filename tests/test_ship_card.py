"""The ship card's all-seeds summary (ticket ML-3 P8), against the archived country fits exp261-263."""
import json
import unittest

from coffeecv.archive_experiment import EXPERIMENTS_DIR
from coffeecv.fit_frozen_head import seed_summary


def config(exp: int) -> dict:
    [d] = list(EXPERIMENTS_DIR.glob(f"exp{exp}__*"))
    return json.loads((d / "config.json").read_text())


class TestSeedSummary(unittest.TestCase):
    def test_every_seed_with_mean_and_range(self):
        out = seed_summary(262, [261, 262, 263], config(262))
        self.assertEqual(sorted(out["runs"]), ["s123", "s42", "s7"])
        val = out["val_macro_f1"]
        self.assertAlmostEqual(val["s123"], 0.9790, places=4)
        self.assertAlmostEqual(val["mean"], (val["s42"] + val["s123"] + val["s7"]) / 3)
        self.assertEqual((val["min"], val["max"]), (min(val["s42"], val["s123"], val["s7"]),
                                                     max(val["s42"], val["s123"], val["s7"])))

    def test_refuses_a_list_without_the_shipped_run(self):
        with self.assertRaises(SystemExit):
            seed_summary(262, [261, 263], config(262))

    def test_refuses_a_run_on_another_class_list(self):
        with self.assertRaisesRegex(SystemExit, "class list"):
            seed_summary(262, [258, 262, 263], config(262))


if __name__ == "__main__":
    unittest.main()
