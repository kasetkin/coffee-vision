"""A folder stage that processes only its new photos (ticket ML-26): which inputs' recorded outputs still stand.

The canonicalize@<folder> and segcrop@<session> stages each turn one folder of photos into one folder of outputs, one
photo at a time. DVC re-runs such a stage whenever any input changes, a photo or the code, and cannot tell a script
which. dvc.lock can: it holds the stage's command, every input's md5 (a folder's as a per-file listing in the cache),
its params and every output's md5, as of the last run (a folder's md5 is its listing's, `dir_md5`). `recorded` compares that entry with today's files:

  - the same command, inputs, params and outputs as dvc.yaml names now, every input but the photo folder unchanged,
    and every listing in the cache: a `Record` of the photos (path under the photo folder -> md5) that are unchanged
    since that run, and of the outputs it recorded (path under the output folder -> md5). The stage keeps a photo's
    outputs when they are on disk as recorded, and processes every other photo;
  - anything else (the code, a params value, the model, dvc.yaml's stage, no entry yet): None and the reason, and the
    stage rebuilds its whole output folder, as before ML-26.

seg_ml19_predict@<model> (coffeecv.seg_predict) reads two inputs that change, the segmenter dataset's file and the
whole canonical store, whose listing DVC keeps nowhere: `recorded_outputs` gives it the recorded outputs, and it keeps a
photo's mask by the photo's sha256 in its recorded index.csv.

The stages' outputs are `persist: true` in dvc.yaml, so DVC leaves them in place for this. Nothing here runs DVC: the
stage runs inside `dvc repro`, which holds the repository's lock. A file DVC ignores (.dvcignore's `name/` lines) is no
part of a folder's listing.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from coffeecv.config import REPO_ROOT


@dataclass(frozen=True)
class Record:
    inputs: dict[str, str]      # photo folder path -> md5, for the photos unchanged since the recorded run
    outputs: dict[str, str]     # output folder path -> md5, as the recorded run wrote them


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ignored_names(root: Path) -> set[str]:
    """The folder names .dvcignore leaves out. Any other kind of pattern is refused: this reads only `name/`."""
    path = root / ".dvcignore"
    lines = [ln.strip() for ln in path.read_text().splitlines()] if path.exists() else []
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    bad = [ln for ln in lines if not re.fullmatch(r"[^/*?\[\]!]+/", ln)]
    if bad:
        raise ValueError(f".dvcignore patterns this module cannot read: {bad}")
    return {ln[:-1] for ln in lines}


def folder_md5s(folder: Path, ignored: set[str]) -> dict[str, str]:
    """{path under `folder`: md5} of every file DVC would list."""
    return {p.relative_to(folder).as_posix(): md5_file(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and not ignored & set(p.relative_to(folder).parts)}


def dir_md5(files: dict[str, str]) -> str:
    """DVC's md5 of a folder: of its listing, serialized as DVC writes the .dir object ("<md5>.dir")."""
    body = json.dumps([{"md5": m, "relpath": p} for p, m in sorted(files.items())])
    return hashlib.md5(body.encode()).hexdigest() + ".dir"


def listing(root: Path, md5: str) -> dict[str, str] | None:
    """A recorded folder's {path: md5}, from its .dir object in the cache; None if the cache does not have it."""
    obj = root / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]
    if not obj.exists():
        return None
    return {e["relpath"]: e["md5"] for e in json.loads(obj.read_text())}


def _lookup(params: dict, key: str):
    node = params
    for part in key.split("."):
        node = node[part]
    return node


def _interpolate(text: str, key: str, item, params: dict) -> str:
    """dvc.yaml's ${...} as DVC fills it in a foreach stage: ${key}, ${item}, ${item.<field>}, else params.yaml."""
    def value(name: str) -> str:
        if name == "key":
            return key
        if name == "item":
            return str(item)
        if name.startswith("item."):
            return str(_lookup(item, name[len("item."):]))
        return str(_lookup(params, name))
    for _ in range(5):   # an item's value may hold a ${...} of its own (seg_ml19_predict's ${seg_ft.weights})
        filled = re.sub(r"\$\{([^}]+)\}", lambda m: value(m.group(1)), text)
        if filled == text:
            break
        text = filled
    return text


def stage_spec(root: Path, stage: str, params: dict) -> dict | None:
    """dvc.yaml's `name@key` stage resolved as DVC resolves it here: cmd, dep paths, out paths, params keys. A foreach
    over a list takes the key as its item; over a mapping, the key's value."""
    name, _, key = stage.partition("@")
    spec = yaml.safe_load((root / "dvc.yaml").read_text())["stages"].get(name)
    if spec is None or "foreach" not in spec or key not in spec["foreach"]:
        return None
    item = spec["foreach"][key] if isinstance(spec["foreach"], dict) else key
    do = spec["do"]
    keys = [k for block in do.get("params", []) for k in (block["params.yaml"] if isinstance(block, dict) else [block])]
    fill = lambda text: _interpolate(text, key, item, params)   # noqa: E731
    return {"cmd": fill(do["cmd"]),
            "deps": sorted(fill(d) for d in do.get("deps", [])),
            "outs": sorted(fill(next(iter(o)) if isinstance(o, dict) else o) for o in do.get("outs", [])),
            "params": sorted(keys)}


def recorded_outputs(stage: str, inputs: tuple[str, ...], out: str,
                     root: Path = REPO_ROOT) -> tuple[dict[str, str] | None, dict[str, str], str]:
    """({output folder path: md5} of `stage`'s last run, {input: recorded md5} of `inputs`, "") when only `inputs` may
    have changed since, else (None, {}, why the stage rebuilds in full). Paths are repo-relative, as dvc.yaml names
    them."""
    entry = yaml.safe_load((root / "dvc.lock").read_text()).get("stages", {}).get(stage)
    if entry is None:
        return None, {}, f"{stage} is not in dvc.lock"
    params = yaml.safe_load((root / "params.yaml").read_text())
    spec = stage_spec(root, stage, params)
    if spec is None:
        return None, {}, f"{stage} is not a foreach stage of dvc.yaml"
    locked_params = (entry.get("params") or {}).get("params.yaml", {})
    if (" ".join(entry["cmd"].split()) != " ".join(spec["cmd"].split())
            or sorted(d["path"] for d in entry["deps"]) != spec["deps"]
            or sorted(o["path"] for o in entry["outs"]) != spec["outs"] or sorted(locked_params) != spec["params"]):
        return None, {}, "its dvc.yaml stage changed since dvc.lock"
    for key, value in locked_params.items():
        if _lookup(params, key) != value:
            return None, {}, f"params.yaml {key} changed"
    deps = {d["path"]: d["md5"] for d in entry["deps"]}
    outs = {o["path"]: o["md5"] for o in entry["outs"]}
    if not set(inputs) <= set(deps) or list(outs) != [out]:
        return None, {}, f"dvc.lock's {stage} does not read {', '.join(inputs)} into {out}"
    ignored = ignored_names(root)
    for path, md5 in deps.items():
        if path in inputs:
            continue
        here = root / path
        if md5.endswith(".dir"):           # by its listing's md5: DVC keeps no listing of a code folder in the cache
            if not here.is_dir() or dir_md5(folder_md5s(here, ignored)) != md5:
                return None, {}, f"{path} changed"
        elif not here.is_file() or md5_file(here) != md5:
            return None, {}, f"{path} changed"
    was_out = listing(root, outs[out])
    if was_out is None:
        return None, {}, "the recorded output listing is not in the cache"
    return was_out, {p: deps[p] for p in inputs}, ""


def recorded(stage: str, photos: str, out: str, root: Path = REPO_ROOT) -> tuple[Record | None, str]:
    """(the Record of `stage`'s last run, "") when only its photo folder `photos` may have changed since, else
    (None, why the stage rebuilds in full). `photos` and `out` are repo-relative, as dvc.yaml names them."""
    was_out, was, why = recorded_outputs(stage, (photos,), out, root)
    if was_out is None:
        return None, why
    was_in = listing(root, was[photos])
    if was_in is None:
        return None, "a recorded listing is not in the cache"
    now = folder_md5s(root / photos, ignored_names(root)) if (root / photos).is_dir() else {}
    return Record({p: m for p, m in now.items() if was_in.get(p) == m}, was_out), ""


def stage_of_output(out: str, root: Path = REPO_ROOT) -> str | None:
    """The dvc.lock stage whose one output is `out` (repo-relative), if any."""
    stages = yaml.safe_load((root / "dvc.lock").read_text()).get("stages", {})
    return next((name for name, e in stages.items() if [o["path"] for o in e.get("outs", [])] == [out]), None)


def recorded_output(root: Path, md5: str) -> bytes | None:
    """A recorded output file's bytes, from the cache; None if the cache does not have it."""
    obj = root / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]
    return obj.read_bytes() if obj.exists() else None


def keeps(record: Record, out_dir: Path, rel: str) -> bool:
    """Whether output `rel` (under the output folder `out_dir`) is on disk as the recorded run wrote it."""
    want = record.outputs.get(rel)
    return want is not None and (out_dir / rel).is_file() and md5_file(out_dir / rel) == want
