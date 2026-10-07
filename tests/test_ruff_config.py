"""The Ruff configuration (OPS-4): the root pyproject.toml's [tool.ruff] is the only one, so nothing shadows it
for a file deeper in the tree, and it lints the project's code while leaving the vendored third_party/ out.
Plain unittest on the tracked files and on `ruff check --show-files`.

    python -m unittest discover -s tests -p 'test_ruff_config.py'
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tracked(*patterns: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *patterns], cwd=ROOT, capture_output=True, check=True)
    return [p for p in out.stdout.decode().split("\0") if p]


class RuffConfig(unittest.TestCase):
    def test_root_pyproject_is_the_only_ruff_config(self):
        self.assertEqual(tracked("ruff.toml", "**/ruff.toml", ".ruff.toml", "**/.ruff.toml"), [])
        with_ruff = [p for p in tracked("pyproject.toml", "**/pyproject.toml")
                     if "ruff" in tomllib.loads((ROOT / p).read_text()).get("tool", {})]
        self.assertEqual(with_ruff, ["pyproject.toml"])

    def test_lints_project_code_and_not_third_party(self):
        out = subprocess.run([sys.executable, "-m", "ruff", "check", "--show-files"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        files = {Path(line).relative_to(ROOT).as_posix() for line in out.stdout.splitlines() if line.strip()}
        self.assertIn("webapp/app.py", files)
        self.assertIn("tests/test_ruff_config.py", files)
        self.assertEqual([f for f in files if f.startswith("third_party/")], [])


if __name__ == "__main__":
    unittest.main()
