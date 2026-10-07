"""coffeecv.ruff_staged: Ruff lints the Python files staged for commit as they stand in the index, so a
finding only on disk passes and one only in the index is refused; excluded paths, deletions and other file
types are not linted. Plain unittest on a throwaway git repo holding the repo's [tool.ruff] tables.

    python -m unittest discover -s tests -p 'test_ruff_staged.py'
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

from coffeecv import ruff_staged as rs

# The P1 tables of the root pyproject.toml (OPS-4): the core rule set, third_party/ excluded even when named.
PYPROJECT = b"""[tool.ruff]
extend-exclude = ["third_party"]
force-exclude = true

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]
"""
CLEAN = b"import os\n\nprint(os.sep)\n"
UNUSED = b"import os\nimport sys\n\nprint(os.sep)\n"


class StagedRepo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.write("pyproject.toml", PYPROJECT)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel: str, data: bytes) -> None:
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_bytes(data)

    def git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)

    def stage(self, rel: str, data: bytes) -> None:
        self.write(rel, data)
        self.git("add", rel)

    def run_main(self) -> tuple[int, list[str]]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rs.main([], root=self.root)
        return code, out.getvalue().splitlines()

    def test_staged_unused_import_is_reported(self):
        self.stage("pkg/mod.py", UNUSED)
        code, lines = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(lines, ["pkg/mod.py:2:8: F401 [*] `sys` imported but unused",
                                 "ruff_staged: 1 finding(s) in 1 staged Python files"])

    def test_clean_in_the_index_passes_whatever_is_on_disk(self):
        self.stage("pkg/mod.py", CLEAN)
        self.write("pkg/mod.py", UNUSED)
        self.assertEqual(self.run_main(), (0, ["ruff_staged: 1 staged Python files clean"]))

    def test_finding_in_the_index_is_reported_though_fixed_on_disk(self):
        self.stage("pkg/mod.py", UNUSED)
        self.write("pkg/mod.py", CLEAN)
        code, lines = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(lines[-1], "ruff_staged: 1 finding(s) in 1 staged Python files")

    def test_staged_file_under_third_party_passes(self):
        self.stage("third_party/vendored/mod.py", UNUSED)
        self.assertEqual(self.run_main(), (0, ["ruff_staged: 1 staged Python files clean"]))

    def test_deletion_and_other_file_types_are_not_linted(self):
        self.stage("gone.py", UNUSED)
        self.git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "x")
        self.git("rm", "-q", "gone.py")
        self.stage("notes.md", b"import sys\n")
        self.assertEqual(self.run_main(), (0, ["ruff_staged: 0 staged Python files clean"]))

    def test_staged_symlink_is_skipped_by_its_index_mode(self):
        self.write("real.txt", UNUSED)
        (self.root / "link.py").symlink_to("real.txt")
        self.git("add", "link.py")
        self.assertEqual(self.run_main(), (0, ["ruff_staged: 0 staged Python files clean"]))

    def test_ruff_error_fails_with_its_message(self):
        self.write("pyproject.toml", b'[tool.ruff.lint]\nselect = ["NOPE"]\n')
        self.stage("pkg/mod.py", CLEAN)
        code, lines = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(lines[0], "pkg/mod.py: ruff exited 2")
        self.assertIn("Unknown rule selector `NOPE`", "\n".join(lines))
        self.assertEqual(lines[-1], "ruff_staged: 1 finding(s) in 1 staged Python files")

    def test_nothing_staged_exits_0(self):
        self.write("pkg/mod.py", UNUSED)
        self.assertEqual(self.run_main(), (0, ["ruff_staged: 0 staged Python files clean"]))


if __name__ == "__main__":
    unittest.main()
