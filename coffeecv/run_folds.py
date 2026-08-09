"""Drive the leave-one-rig-out sweep: hold out each rig in turn, train, archive.

The protocol lives here rather than in a shell history because it *is* the
experiment: which rigs train, which is held out, and that every fold runs
identical code with only the fold rotating. A single held-out rig is n=1 and
cannot distinguish a real transfer result from the luck of which rig was held
out, so the unit of evidence is the set of three folds, not any one of them.

params.yaml is rewritten line-by-line rather than round-tripped through a YAML
parser, because the file is documentation as much as configuration and a
round-trip would strip every comment in it.

    python -m coffeecv.run_folds --arm baseline --start-exp 48
    python -m coffeecv.run_folds --arm scale --start-exp 51
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

from coffeecv.config import PARAMS_FILE, REPO_ROOT

RIGS = [
    "data/cropped/2026-08-07__box_pictures_all_classes",
    "data/cropped/2026-08-09__pixel_cam",
    "data/cropped/2026-08-09__sony_cam",
]

# frac_min, frac_max for each arm. The baseline keeps the fixed pixel patch size
# so its cross-rig number answers "what does the current model actually do".
ARMS = {
    "baseline": (0.0, 0.0),
    "scale": (0.15, 0.60),
}


def set_fold(heldout: str, frac_min: float, frac_max: float) -> None:
    """Point params.yaml at one fold, preserving comments and ordering."""
    text = PARAMS_FILE.read_text()
    train = [r for r in RIGS if r != heldout]

    block = "train_rigs:\n" + "".join(f"  - {r}\n" for r in train)
    text = re.sub(r"train_rigs:\n(?:  - .*\n)+", block, text, count=1)
    text = re.sub(r"^heldout_rig: .*$", f"heldout_rig: {heldout}", text, count=1, flags=re.M)
    text = re.sub(r"^patch_scale_frac_min: .*$", f"patch_scale_frac_min: {frac_min}",
                  text, count=1, flags=re.M)
    text = re.sub(r"^patch_scale_frac_max: .*$", f"patch_scale_frac_max: {frac_max}",
                  text, count=1, flags=re.M)
    PARAMS_FILE.write_text(text)

    # Read it back through the real loader: a silently-failed regex would
    # otherwise run the wrong fold and look like a result.
    from coffeecv.config import RunConfig
    cfg = RunConfig.from_params_yaml()
    assert cfg.heldout_rig == heldout, f"heldout_rig is {cfg.heldout_rig!r}, wanted {heldout!r}"
    assert list(cfg.train_rigs) == train, f"train_rigs is {cfg.train_rigs!r}, wanted {train!r}"
    assert cfg.patch_scale_frac_min == frac_min and cfg.patch_scale_frac_max == frac_max
    assert heldout not in cfg.train_rigs, "held-out rig leaked into training"


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--start-exp", type=int, required=True, help="experiment number of the first fold")
    p.add_argument("--only", help="run just this held-out rig (substring match)")
    args = p.parse_args()

    frac_min, frac_max = ARMS[args.arm]
    heldouts = [r for r in RIGS if not args.only or args.only in r]

    for i, heldout in enumerate(heldouts):
        exp_id = args.start_exp + i
        short = Path(heldout).name
        slug = f"lorio_{args.arm}_heldout_{short.split('__')[-1]}"
        print(f"\n{'=' * 72}\nfold {i + 1}/{len(heldouts)}  exp{exp_id}  arm={args.arm}  "
              f"held out: {short}\n{'=' * 72}", flush=True)
        set_fold(heldout, frac_min, frac_max)

        t0 = time.time()
        if run(["dvc", "repro", "train"]) != 0:
            print(f"fold {short} FAILED; stopping so the failure is not buried")
            return
        mins = (time.time() - t0) / 60
        print(f"fold {short} finished in {mins:.0f} min", flush=True)

        note = (f"leave-one-rig-out, arm={args.arm}, held out {short}; "
                f"scale_frac={frac_min}-{frac_max}")
        run(["python", "-m", "coffeecv.archive_experiment",
             "--id", str(exp_id), "--slug", slug, "--note", note])


if __name__ == "__main__":
    main()
