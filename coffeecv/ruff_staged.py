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

Known limit: Ruff reads its config from the pyproject.toml on disk, not the one in the index, so the staged
files are linted with the [tool.ruff] settings on disk: a change to them staged in part, or not staged, applies
to this commit's check, and the settings being committed may differ. Not fixed (OPS-4 code review, R6).

Cost: about 85 ms per staged file, most of it the start of `python -m ruff`, plus about 1.7 s for this
module's own start (leak_check imports torch through coffeecv.config). Measured 2026-10-07.

Exit status is non-zero on any finding, and on any other nonzero Ruff exit (a bad config), which is printed
with Ruff's message and counted apart as a Ruff error. A Ruff missing from the interpreter fails before any
file is read, naming the fix, and never skips the check.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

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


class Lint(NamedTuple):
    findings: list[str]   # Ruff's findings, one concise line each
    error: str | None     # a nonzero Ruff exit with no finding (a bad config): its status and Ruff's message


def lint(root: Path, path: str) -> Lint:
    """Ruff on the staged blob of `path`."""
    blob = subprocess.run(["git", "-C", str(root), "show", f":{path}"], capture_output=True, check=True).stdout
    proc = subprocess.run([sys.executable, "-m", "ruff", "check", "--output-format", "concise",
                           "--stdin-filename", path, "-"], input=blob, capture_output=True, cwd=root)
    findings = [ln for ln in proc.stdout.decode().splitlines() if ln.startswith(f"{path}:")]
    error = None
    if proc.returncode and not findings:
        err = proc.stderr.decode().strip()
        error = f"{path}: ruff exited {proc.returncode}" + (f"\n{err}" if err else "")
    return Lint(findings, error)


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    if importlib.util.find_spec("ruff") is None:
        print(f"ruff_staged: no Ruff in {sys.executable}: run `uv sync --locked`", file=sys.stderr)
        return 1
    files = staged_python_files(root)
    findings: list[str] = []
    errors: list[str] = []
    for path in files:
        result = lint(root, path)
        for ln in result.findings:
            print(ln)
        findings += result.findings
        if result.error:
            print(result.error)
            errors.append(result.error)
    counts = ([f"{len(findings)} finding(s)"] if findings else []) + \
        ([f"{len(errors)} Ruff error(s)"] if errors else [])
    if counts:
        summary = f"ruff_staged: {' and '.join(counts)} in {len(files)} staged Python files"
    else:
        summary = f"ruff_staged: {len(files)} staged Python files clean"
    print(summary)
    return 1 if counts else 0


if __name__ == "__main__":
    sys.exit(main())
