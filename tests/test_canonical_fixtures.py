"""webapp/deploy/canonical_fixtures.py: the deploy's fixtures as the inference host reads them (ML-19 D34), and since
OPS-22 the photos as the front sends them: the smoke photo (D6, --verify's public check) and each example
(D7, checked here before a deploy). Plain unittest.

    python -m unittest tests.test_canonical_fixtures
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp" / "deploy"))
import canonical_fixtures  # noqa: E402

from coffeecv import canonical  # noqa: E402
from tests._tiers import real_data  # noqa: E402


def front(data: bytes, name: str) -> bytes:
    return canonical.encode(canonical.decode(data, name, settings=canonical.FRONT_SETTINGS),
                            settings=canonical.FRONT_SETTINGS)


def jpeg(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


class CanonicalFixtures(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.photos = {}
        lines = []
        for role, name, seed in (("smoke", "smoke.jpg", 1), ("compare", "one.jpg", 2), ("compare", "two.jpg", 3)):
            data = jpeg(seed)
            (self.dir / name).write_bytes(data)
            self.photos[name] = data
            lines.append(f"{role} {hashlib.sha256(data).hexdigest()} {self.dir / name}")
        self.fixtures = self.dir / "fixtures.txt"
        self.fixtures.write_text("# a comment\n" + "\n".join(lines) + "\n")
        self.out = self.dir / "out"

    def test_every_fixture_as_the_store_makes_it_and_the_smoke_photo_as_the_front_does(self):
        canonical_fixtures.write(self.fixtures, self.out)
        for name, data in self.photos.items():
            self.assertEqual((self.out / f"{name}.jxl").read_bytes(), canonical.canonicalize(data, name))
        listed = (self.out / canonical_fixtures.CANONICAL_LIST).read_text().split("\n")
        self.assertEqual([line.split()[2] for line in listed if line], ["smoke.jpg.jxl", "one.jpg.jxl", "two.jpg.jxl"])
        made = self.out / canonical_fixtures.FRONT_DIR / "smoke.jpg.jxl"
        self.assertEqual(made.read_bytes(), front(self.photos["smoke.jpg"], "smoke.jpg"))
        self.assertNotEqual(made.read_bytes(), (self.out / "smoke.jpg.jxl").read_bytes())
        self.assertEqual([p.name for p in (self.out / canonical_fixtures.FRONT_DIR).iterdir()], ["smoke.jpg.jxl"])

    def test_a_fixture_that_does_not_match_its_sha256_stops_it(self):
        (self.dir / "one.jpg").write_bytes(jpeg(9))
        with self.assertRaises(SystemExit):
            canonical_fixtures.write(self.fixtures, self.out)

    @real_data   # about 6 s: the 50 MP grinder example re-encoded among them
    def test_each_example_as_the_front_sends_it(self):
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                check=True).stdout.strip()
        canonical_fixtures.write(self.fixtures, self.out, examples_from=commit)
        listing = json.loads(subprocess.run(["git", "show", f"{commit}:webapp/static/examples/examples.json"],
                                            cwd=REPO, capture_output=True, check=True).stdout)
        names = [e["name"] for e in listing["examples"]]
        made = self.out / canonical_fixtures.FRONT_DIR / canonical_fixtures.EXAMPLES_DIR
        self.assertEqual(sorted(p.name for p in made.iterdir()), sorted(f"{n}.jxl" for n in names))
        for name in names:
            with self.subTest(name):
                original = subprocess.run(["git", "show", f"{commit}:webapp/static/examples/{name}.jxl"], cwd=REPO,
                                          capture_output=True, check=True).stdout
                got = (made / f"{name}.jxl").read_bytes()
                self.assertEqual(got, front(original, f"{name}.jxl"))
                self.assertTrue(canonical.is_canonical(got))


if __name__ == "__main__":
    unittest.main()
