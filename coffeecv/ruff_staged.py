"""Ruff over the Python files staged for commit, as they will be committed. .githooks/pre-commit runs it
beside coffeecv.leak_check (OPS-4).

    python -m coffeecv.ruff_staged

The files: those added, copied, modified or renamed in the index (leak_check.published_files) with an
extension `ruff check` includes (.py, .pyi, .ipynb). Symlinks and submodules are skipped by their index
mode. A deletion is not linted.

Each staged blob (`git show :<path>`) is piped into `python -m ruff check --stdin-filename <path> -` from
the repo root, so a file staged in part (`git add -p`) is linted as it is staged: a finding only on disk
passes, one in the index is refused, even when fixed on disk. leak_check --staged reads the disk instead,
until OPS-5. The rules are pyproject.toml's [tool.ruff], resolved from <path>; with force-exclude, a staged
file under third_party/ passes. scripts/check.sh lints the whole working tree before a push or a deploy.

Cost: about 85 ms per staged file, most of it the start of `python -m ruff`, plus about 1.7 s for this
module's own start (leak_check imports torch through coffeecv.config). Measured 2026-10-07.

Exit status is non-zero on any finding, and on any other nonzero Ruff exit (a bad config). A Ruff missing
from the interpreter fails, naming the fix, and never skips the check.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from coffeecv.config import REPO_ROOT
from coffeecv.leak_check import published_files

EXTENSIONS = {".py", ".pyi", ".ipynb"}   # what `ruff check` includes by default
SKIPPED_MODES = {"120000", "160000"}     # symlinks and submodules, which leak_check skips on disk too


def staged_python_files(root: Path = REPO_ROOT) -> list[str]:
    """The Python files added or changed in the index, symlinks and submodules left out."""
    files = [p for p in published_files(root, staged=True) if Path(p).suffix in EXTENSIONS]
    if not files:
        return []
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-s", "-z", "--", *files],
                         capture_output=True, check=True).stdout.decode()
    skipped = {e.split("\t", 1)[1] for e in out.split("\0") if e and e.split(" ", 1)[0] in SKIPPED_MODES}
    return [p for p in files if p not in skipped]


def lint(root: Path, path: str) -> list[str]:
    """Ruff's findings in the staged blob of `path`, one concise line each. A Ruff error (exit 2, a bad
    config) is one finding: its exit status and message."""
    blob = subprocess.run(["git", "-C", str(root), "show", f":{path}"], capture_output=True, check=True).stdout
    proc = subprocess.run([sys.executable, "-m", "ruff", "check", "--output-format", "concise",
                           "--stdin-filename", path, "-"], input=blob, capture_output=True, cwd=root)
    err = proc.stderr.decode().strip()
    if "No module named ruff" in err:
        raise SystemExit(f"ruff_staged: no Ruff in {sys.executable}: run `uv sync --locked`")
    found = [ln for ln in proc.stdout.decode().splitlines() if ln.startswith(f"{path}:")]
    if proc.returncode and not found:
        found = [f"{path}: ruff exited {proc.returncode}" + (f"\n{err}" if err else "")]
    return found


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    files = staged_python_files(root)
    found = [ln for path in files for ln in lint(root, path)]
    for ln in found:
        print(ln)
    what = f"{len(files)} staged Python files"
    if found:
        print(f"ruff_staged: {len(found)} finding(s) in {what}")
        return 1
    print(f"ruff_staged: {what} clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
