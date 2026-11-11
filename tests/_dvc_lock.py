"""A stage's run recorded in a made-up repository's dvc.lock and cache, as `dvc commit` writes them (ticket ML-26)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def cache(root: Path, data: bytes, suffix: str = "") -> str:
    h = md5(data) + suffix
    obj = root / ".dvc/cache/files/md5" / h[:2] / h[2:]
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(data)
    return h


def entry(root: Path, rel: str) -> dict:
    """A dep or out entry: a file's md5, or a folder's .dir md5 with its listing (and its files) in the cache."""
    p = root / rel
    if p.is_file():
        return {"path": rel, "hash": "md5", "md5": md5(p.read_bytes())}
    files = sorted((f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts),
                   key=lambda f: f.relative_to(p).as_posix())
    for f in files:
        cache(root, f.read_bytes())
    body = json.dumps([{"md5": md5(f.read_bytes()), "relpath": f.relative_to(p).as_posix()} for f in files])
    return {"path": rel, "hash": "md5", "md5": cache(root, body.encode(), ".dir"), "nfiles": len(files)}


def commit(root: Path, stage: str, cmd: str, deps: list[str], outs: list[str], params: dict | None = None) -> None:
    """Record `stage` in root/dvc.lock (other stages kept)."""
    path = root / "dvc.lock"
    lock = yaml.safe_load(path.read_text()) if path.exists() else {"schema": "2.0", "stages": {}}
    e = {"cmd": cmd, "deps": [entry(root, d) for d in sorted(deps)], "outs": [entry(root, o) for o in outs]}
    if params:
        e["params"] = {"params.yaml": params}
    lock["stages"][stage] = e
    path.write_text(yaml.safe_dump(lock))
