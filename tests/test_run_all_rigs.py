"""run_all_rigs.py's lever defaults are the adopted recipe. Plain unittest -- the repo has no pytest.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import unittest

from coffeecv.config import RunConfig
from coffeecv.run_all_rigs import ADOPTED, build_parser

# Each lever flag's argparse dest -> the RunConfig field it writes.
FLAGS = {
    "brightness_jitter": "brightness_jitter_strength",
    "mixstyle_p": "mixstyle_p",
    "mixstyle_mode": "mixstyle_mode",
    "freeze_mode": "freeze_mode",
    "eta_min": "eta_min",
    "scheduler": "scheduler",
}


class TestAdoptedDefaults(unittest.TestCase):
    def test_omitted_flags_train_the_adopted_recipe(self):
        # An omitted flag is written into params.yaml, so its default must be the adopted value
        # (feedback-cli-defaults-overwrite-config: 0.0 defaults once turned MixStyle off unasked).
        args = build_parser().parse_args(["--seeds", "7", "--start-exp", "1"])
        for dest, field in FLAGS.items():
            self.assertEqual(getattr(args, dest), ADOPTED[field], dest)

    def test_adopted_matches_params_yaml_resting_values(self):
        # params.yaml's committed values are what a bare `dvc repro train` reproduces. If this fails,
        # either a screen left params.yaml unrestored or a lever was adopted in only one of the two.
        cfg = RunConfig.from_params_yaml()
        for field, value in ADOPTED.items():
            self.assertEqual(getattr(cfg, field), value, field)

    def test_every_lever_has_a_flag(self):
        self.assertEqual(set(FLAGS.values()), set(ADOPTED))


if __name__ == "__main__":
    unittest.main()
