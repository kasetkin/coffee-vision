"""Test tiers (scripts/check.sh). `real_data` marks a test that runs real photos or real model weights and
costs seconds. The pre-push hook's fast tier (COFFEECV_FAST_TESTS=1) skips it; `scripts/check.sh --full`,
which every deploy runs (scripts/deploy_webapp.sh), includes it. Mark any test or class slower than ~1 s
(`python -m unittest discover -s tests --durations 20` lists them; class setUp time is not in that list).
Imported as `tests._tiers`, so run unittest from the repo root, as every test's docstring says."""
import os
import unittest

FAST = os.environ.get("COFFEECV_FAST_TESTS") == "1"

real_data = unittest.skipIf(FAST, "real-data tier: runs under scripts/check.sh --full")
