"""Preflight checks and the `dvc repro` runner shared by the drivers that train and commit.

`fit_frozen_head.py` refuses to start on uncommitted source or stale crops, and shells out to dvc/git
(so did the ResNet driver, run_all_rigs.py, until 2026-10-20). These helpers lived in `run_folds.py` until 2026-09-29 by accident of
file history -- they were never fold-specific -- and moved here when the fold driver was retired
(ticket ML-1), so nothing imports from a module that no longer exists.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from coffeecv.config import REPO_ROOT
from coffeecv.merge_rig import merge_cmd_args


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


# Paths a driver rewrites itself, so they are expected to be dirty at launch
# and are committed per run. Everything else -- source, dvc.yaml, the
# per-session crop configs -- describes what the run *is*.
SWEEP_WRITES = ("params.yaml", "dvc.lock", "outputs/", "experiments/")


def dirty_provenance_paths() -> list[str]:
    """`git status --porcelain` entries that would leave a sweep unreproducible.

    The per-run commit stages only params.yaml, dvc.lock, the metrics files and
    experiments/. Anything else edited but uncommitted therefore runs for hours
    and lands in no commit at all.

    Phase 13 lost an entire feature this way: `brightness_jitter_strength` was
    implemented in config.py/transforms.py and never committed, so exp 72-95 each
    recorded a git_commit whose tree has no such field in RunConfig -- and
    `from_params_yaml` filtered params.yaml to known fields *silently* back then,
    so checking one out re-runs at the default brightness and looks like it
    worked. 18 paired runs, ~30h of compute, reproducible only from the archived
    config.json.
    """
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode()
    dirty = []
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:  # rename: the destination is what would be running
            path = path.split(" -> ", 1)[1]
        if path and not path.startswith(SWEEP_WRITES):
            dirty.append(line.rstrip())
    return dirty


# A capture dir's upstream stages: its merge stage and the segcrop stages feeding it (dataset.CAPTURES is
# the merged pool). Both matter, and the merge stage alone is not enough: if segment_beans.py changes,
# segcrop@<session> goes stale while the merge's dependency (the crop output directory, not yet
# regenerated) still looks clean -- so a `dvc repro train` would re-crop and then re-merge, changing the
# pixels under a sweep that had been told everything was up to date. Keyed "segcropped/<name>" (see
# `_stage_key`). Since ticket ML-8 (D2) the one pool, merge_segpool, whose segcrop stages are read from its
# own command in dvc.yaml, so the list cannot drift from what the merge reads. Until then the four camera
# pools (merge_segcam_*); until ticket ML-3 P2b also the tray heuristic's data/cropped pools.
POOL_MERGE_STAGE = "merge_segpool"


def capture_stage_overrides() -> dict[str, list[str]]:
    """{"segcropped/pool": [merge_segpool, segcrop@<each session it merges>]}, read from dvc.yaml when asked."""
    stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())["stages"]
    _, sessions = merge_cmd_args(stages[POOL_MERGE_STAGE]["cmd"])
    return {"segcropped/pool": [POOL_MERGE_STAGE, *(f"segcrop@{s}" for s in sessions)]}


def _stage_key(capture: str) -> str:
    """A capture dir's key in `capture_stage_overrides()`: "<parent>/<basename>"
    (data/segcropped/pool -> "segcropped/pool")."""
    p = Path(capture)
    return f"{p.parent.name}/{p.name}"


def stale_crop_stages(capture_dirs: list[str]) -> list[str]:
    """The stages that make the crops a fit reads (the pool's merge stage and its
    segcrop stages -- see `capture_stage_overrides`) which `dvc status` reports as
    stale, for the capture dirs `capture_dirs` names. A session's own segcrop dir
    maps to its segcrop stage; any other dir is refused, since no stage here could
    tell whether it is stale. fit_frozen_head refuses to start on any (but under
    --allow-dirty, for a smoke run).

    Only these stages, not the whole `dvc status`: frozen stages and the outputs
    dvc.md names as expectedly missing are not the fit's inputs.

    A stale one means the crops on disk are not what today's code, segmenter and
    params would cut: the fit reads data/segcropped/pool as it stands, so it would
    train on different pixels than the runs it is compared against, or than a
    rebuild of the same commit -- a silent comparison-invalidating event. It is
    also the one failure git cannot see: `data/segcropped/` is gitignored, so
    on-disk loss or corruption of the crops shows up in `dvc status` and nowhere
    else. An interrupted `dvc repro` deletes the stage's outs, which is exactly how
    this happens in practice; so does a byte change to a file every segcrop stage
    depends on, such as segment_beans.py (dc021bf).
    """
    stale = []
    overrides = capture_stage_overrides()          # dvc.yaml, read once
    for capture in capture_dirs:
        name, key = Path(capture).name, _stage_key(capture)
        if key in overrides:
            stages = overrides[key]
        elif key.startswith("segcropped/"):
            stages = [f"segcrop@{name}"]
        else:
            raise ValueError(f"{capture} is not one of the segmenter's crop dirs; no stage tracks it")
        for stage in stages:
            if stage in stale:
                continue  # sessions feed more than one dir; report each stage once
            out = subprocess.check_output(["dvc", "status", "--json", stage], cwd=REPO_ROOT).decode()
            if json.loads(out or "{}"):
                stale.append(stage)
    return stale
