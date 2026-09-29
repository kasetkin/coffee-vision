"""Preflight checks and the `dvc repro` runner shared by the drivers that train and commit.

`run_all_rigs.py` and `fit_frozen_head.py` both refuse to start on uncommitted source or stale crops,
and both shell out to dvc/git. These helpers lived in `run_folds.py` until 2026-09-29 by accident of
file history -- they were never fold-specific -- and moved here when the fold driver was retired
(ticket ML-1), so nothing imports from a module that no longer exists.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from coffeecv.config import REPO_ROOT


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


# Most capture dirs' upstream stage is `crop@<name>`, derived straight from the
# dir's own name. A merged capture dir (see merge_rig.py) is the exception: its
# data comes from its own differently-named stage instead of a 1:1 crop.
# A merged dir maps to its merge stage *and* to the crop stages feeding it. Both
# matter, and the merge stage alone is not enough: if crop_tray.py changes,
# crop@<session> goes stale while the merge's dependency (the crop output
# directory, not yet regenerated) still looks clean -- so a `dvc repro train`
# would re-crop and then re-merge, changing the pixels under a sweep that had
# been told everything was up to date.
RIG_STAGE_OVERRIDES = {
    "oneplus_combined": ["merge_oneplus",
                         "crop@2026-08-25__oneplus", "crop@2026-08-27__oneplus_flash"],
    "cam_pixel":   ["merge_cam_pixel",
                    "crop@2026-08-07__box_pictures_all_classes",
                    "crop@2026-08-09__pixel_cam", "crop@2026-08-30__pixel"],
    "cam_sony":    ["merge_cam_sony",
                    "crop@2026-08-09__sony_cam", "crop@2026-08-30__sony"],
    "cam_oneplus": ["merge_cam_oneplus",
                    "crop@2026-08-25__oneplus", "crop@2026-08-27__oneplus_flash",
                    "crop@2026-08-30__oneplus"],
    "cam_iphone":  ["merge_cam_iphone", "crop@2026-08-25__iphone"],
}


def stale_crop_stages(capture_dirs: list[str]) -> list[str]:
    """Upstream-of-train stages that `dvc repro train` would regenerate before
    training (crop stages for most dirs, merge stages for a merged one -- see
    RIG_STAGE_OVERRIDES), for the capture dirs `capture_dirs` names.

    Deliberately *not* a check on overall `dvc status`, which is dirty by design
    here: a driver rewrites params.yaml precisely so the train stage re-runs, so
    "train is out of date" is the required state at launch, not a fault.

    These upstream stages are different. If one is stale, `dvc repro train`
    regenerates the dataset first, and every run then trains on different pixels
    than the reference runs it is about to be compared against -- a silent
    comparison-invalidating event. It is also the one failure git cannot see:
    `data/cropped/` is gitignored, so on-disk loss or corruption of the crops
    shows up in `dvc status` and nowhere else. An interrupted `dvc repro` deletes
    the stage's outs, which is exactly how this happens in practice.
    """
    stale = []
    for capture in capture_dirs:
        name = Path(capture).name
        for stage in RIG_STAGE_OVERRIDES.get(name, [f"crop@{name}"]):
            if stage in stale:
                continue  # sessions feed more than one dir; report each stage once
            out = subprocess.check_output(["dvc", "status", "--json", stage], cwd=REPO_ROOT).decode()
            if json.loads(out or "{}"):
                stale.append(stage)
    return stale
