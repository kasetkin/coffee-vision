"""Refuse a package held at different versions by two of the repo's uv locks (ticket ML-19 D40, D41).

    python -m coffeecv.lock_check [LOCK ...]      # default: the three locks below

Training, the web service and the front each have their own uv project and lock (ADR 0007). A photo is decoded,
colour-converted and encoded by the same libraries in all three (ADR 0033), so a library that two locks hold must be
at one version in both: otherwise the store's canonical photos and the front's differ. uv has no shared pins file,
so this check compares the resolved versions instead. It covers every package two locks share, not only the photo
stack. `convert/` stays out: it holds an older torch on purpose (litert-torch). Standard library only, so the
pre-commit hook runs it without importing torch.
"""
from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

LOCKS = ("uv.lock", "webapp/uv.lock", "webapp/front/uv.lock")
REPO = Path(__file__).resolve().parents[1]
_LOCAL_SOURCES = {"virtual", "editable", "directory", "path"}   # a project itself, not a pinned package


@dataclass(frozen=True)
class Mismatch:
    name: str
    versions: dict[str, tuple[str, ...]]   # lock -> the versions it holds, sorted


def normalize(name: str) -> str:
    """PEP 503's normalized name: `Typing_Extensions` and `typing.extensions` are one package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_versions(path: Path) -> dict[str, frozenset[str]]:
    """Each package a lock resolves, by normalized name, to the versions it holds (usually one)."""
    with open(path, "rb") as f:
        lock = tomllib.load(f)
    held: dict[str, set[str]] = {}
    for pkg in lock.get("package", []):
        if _LOCAL_SOURCES & set(pkg.get("source", {})):
            continue
        held.setdefault(normalize(pkg["name"]), set()).add(pkg["version"])
    return {name: frozenset(versions) for name, versions in held.items()}


def mismatches(locks: dict[str, dict[str, frozenset[str]]]) -> list[Mismatch]:
    """Every package that two or more locks hold at different versions, by name."""
    names = sorted({name for held in locks.values() for name in held})
    found = []
    for name in names:
        holders = {label: held[name] for label, held in locks.items() if name in held}
        if len(holders) > 1 and len(set(holders.values())) > 1:
            found.append(Mismatch(name, {label: tuple(sorted(v)) for label, v in holders.items()}))
    return found


def report(found: list[Mismatch]) -> str:
    return "\n".join(f"  {m.name}: " + ", ".join(f"{label} {' '.join(v)}" for label, v in m.versions.items())
                     for m in found)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    paths = [Path(a) for a in args] if args else [REPO / p for p in LOCKS]
    labels = [str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p) for p in paths]
    locks = {label: read_versions(path) for label, path in zip(labels, paths)}
    found = mismatches(locks)
    shared = sum(1 for name in {n for held in locks.values() for n in held}
                 if sum(name in held for held in locks.values()) > 1)
    if found:
        print(f"lock_check: {len(found)} of {shared} shared packages held at different versions "
              "(ADR 0033, ML-19 D41; align the pyproject.toml pins and re-lock):\n" + report(found))
        return 1
    print(f"lock_check: {shared} packages shared by {len(paths)} locks, each at one version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
