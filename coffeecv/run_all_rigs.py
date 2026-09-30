"""Train ResNet18 on every capture dir, one archived and committed run per seed.

The one training driver (ticket ML-1, 2026-09-29): it points params.yaml at
`dataset.CAPTURES`, states every lever explicitly, runs `dvc repro train`, archives
the run under experiments/ and commits it. Photos are pooled across the capture
dirs and split 70/15/15 per class; no camera is held out, trained on selectively
or reported separately. The leave-one-camera-out fold driver that used to sit
beside this one, and decided configurations on a cross-camera metric, is retired.

How a change is judged now (ticket ML-1 §4): screen it here at 3 seeds, paired per
seed against the unchanged baseline at the same seed, and adopt on val macro-F1
deltas that are sign-consistent over the seeds; test (patch) is a reported
control, read once, never used to choose. The best-val run of an adopted screen is
already a shipping candidate -- there is no separate refit.

**Every number this produces is in-distribution.** Val and test photos come from
the same bean bags and the same cameras as train, so they say nothing about a new
camera or a new scoop of beans (ML-1 R2), and they sit near ceiling (R1: before the
first screen is adopted on this metric, measure its seed-to-seed noise). Never
compare them with the fold-era cross-camera numbers in index.csv (R3).

    python -m coffeecv.run_all_rigs --seeds 42 123 7 --start-exp 255
    python -m coffeecv.run_all_rigs --seeds 42 123 7 --mixstyle-p 0.0 --start-exp 258 --tag no_mixstyle

Every lever flag defaults to the adopted recipe (`ADOPTED`), so a screen names only the lever it
changes. The script writes every lever into params.yaml on every run, flag given or not: it never
inherits params.yaml's resting value, which a screen can leave at a variant that was not adopted.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

from coffeecv.config import PARAMS_FILE, REPO_ROOT, RunConfig
from coffeecv.lr_schedules import SCHEDULERS
from coffeecv.dataset import CAPTURES
from coffeecv.repro_utils import dirty_provenance_paths, run, stale_crop_stages

# The adopted training recipe: each lever flag's default. Until 2026-09-30 --mixstyle-p and --eta-min
# defaulted to 0.0 instead, so a run that omitted either silently trained MixStyle off and a zero LR
# floor (see feedback-cli-defaults-overwrite-config). tests/test_run_all_rigs.py keeps this equal to
# params.yaml's committed resting values; change both together when a lever is adopted.
ADOPTED = {
    "brightness_jitter_strength": 0.0,
    "mixstyle_p": 0.5,
    "mixstyle_mode": "agnostic",
    "freeze_mode": "none",
    "eta_min": 1e-5,
    "scheduler": "cosine",
}


def set_all_rigs(
    seed: int, epochs: int | None, brightness_jitter: float, mixstyle_p: float, freeze_mode: str,
    mixstyle_mode: str, eta_min: float, scheduler: str,
) -> RunConfig:
    """Point params.yaml at every capture dir and state every lever, preserving comments."""
    text = PARAMS_FILE.read_text()
    block = "train_capture_dirs:\n" + "".join(f"  - {c}\n" for c in CAPTURES)
    text = re.sub(r"train_capture_dirs:\n(?:  - .*\n)+", block, text, count=1)
    text = re.sub(r"^seed: .*$", f"seed: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^brightness_jitter_strength: .*$",
                  f"brightness_jitter_strength: {brightness_jitter}", text, count=1, flags=re.M)
    # Stated on every invocation, like brightness_jitter above -- not "leave
    # whatever params.yaml had". The Phase 13 provenance failure was exactly
    # this class of bug for brightness; a variant of it (relying on params.yaml's
    # resting freeze_mode instead of stating it explicitly) is exactly what
    # silently trained exp124-129 with the wrong freeze_mode -- see
    # feedback-experiment-provenance and project-phase16-screens.
    text = re.sub(r"^mixstyle_p: .*$", f"mixstyle_p: {mixstyle_p}", text, count=1, flags=re.M)
    # Same rationale as mixstyle_p/freeze_mode above -- discovered missing while
    # relaunching the all-rigs sweep after a MixStyle screen left params.yaml
    # resting at a variant that was never adopted (since removed, ticket ML-1).
    text = re.sub(r"^mixstyle_mode: \S+", f"mixstyle_mode: {mixstyle_mode}", text, count=1, flags=re.M)
    text = re.sub(r"^freeze_mode: \S+", f"freeze_mode: {freeze_mode}", text, count=1, flags=re.M)
    # PyYAML's SafeLoader float regex requires a literal decimal point -- "1e-05"
    # round-trips as the *string* "1e-05", not the float. str(1e-05) omits the
    # dot, so it must be inserted here.
    eta_min_str = f"{eta_min:.10g}"
    if "e" in eta_min_str:
        mantissa, exp = eta_min_str.split("e")
        if "." not in mantissa:
            eta_min_str = f"{mantissa}.0e{exp}"
    elif "." not in eta_min_str:
        eta_min_str += ".0"
    text = re.sub(r"^eta_min: .*$", f"eta_min: {eta_min_str}", text, count=1, flags=re.M)
    # Stated on every invocation, for the same reason as mixstyle_mode above: a plateau screen
    # leaves params.yaml resting at `plateau`, and inheriting it here would train a shipping
    # model under a scheduler that has not been adopted.
    text = re.sub(r"^scheduler: \S+", f"scheduler: {scheduler}", text, count=1, flags=re.M)
    if epochs is not None:
        text = re.sub(r"^epochs: .*$", f"epochs: {epochs}", text, count=1, flags=re.M)
    PARAMS_FILE.write_text(text)

    # Read back through the real loader: a regex that silently failed would
    # otherwise train the wrong thing and look like a result.
    cfg = RunConfig.from_params_yaml()
    assert cfg.seed == seed, f"seed is {cfg.seed}, wanted {seed}"
    assert list(cfg.train_capture_dirs) == CAPTURES, f"train_capture_dirs is {cfg.train_capture_dirs!r}"
    assert cfg.brightness_jitter_strength == brightness_jitter
    assert cfg.mixstyle_p == mixstyle_p, f"mixstyle_p is {cfg.mixstyle_p}, wanted {mixstyle_p}"
    assert cfg.mixstyle_mode == mixstyle_mode, f"mixstyle_mode is {cfg.mixstyle_mode!r}, wanted {mixstyle_mode!r}"
    assert cfg.freeze_mode == freeze_mode, f"freeze_mode is {cfg.freeze_mode!r}, wanted {freeze_mode!r}"
    assert cfg.eta_min == eta_min, f"eta_min is {cfg.eta_min}, wanted {eta_min}"
    assert cfg.scheduler == scheduler, f"scheduler is {cfg.scheduler!r}, wanted {scheduler!r}"
    if epochs is not None:
        assert cfg.epochs == epochs
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="one run per seed; several make an ensemble, if the ensemble study says that helps")
    p.add_argument("--start-exp", type=int, required=True)
    p.add_argument("--epochs", type=int, default=None,
                   help="epoch budget; also the cosine T_max, but only a cap under --scheduler plateau. "
                        "Default None leaves params.yaml's resting value "
                        "untouched -- this script used to default "
                        "to a hardcoded 80, which silently overrode the adopted epochs=100 (see "
                        "project-epochs100-patience20-relaunch) on any invocation that didn't pass "
                        "--epochs explicitly.")
    p.add_argument("--brightness-jitter", type=float, default=ADOPTED["brightness_jitter_strength"])
    p.add_argument("--mixstyle-p", type=float, default=ADOPTED["mixstyle_p"],
                   help="per-batch probability of MixStyle (resnet18 only; default: the adopted %(default)s). "
                        "States its value on EVERY run, like --brightness-jitter, not 'leave whatever "
                        "params.yaml had'.")
    p.add_argument("--mixstyle-mode", default=ADOPTED["mixstyle_mode"], choices=["agnostic"],
                   help="MixStyle partner selection. 'agnostic' is the only mode: the camera-aware v2 needed "
                        "camera labels and was removed with ticket ML-1 (it was a confirmed null), so a stale "
                        "invocation naming it fails here rather than deep inside model.py. States its value "
                        "on EVERY run like --mixstyle-p, not 'leave whatever params.yaml had'.")
    p.add_argument("--freeze-mode", default=ADOPTED["freeze_mode"], choices=["none", "last_block", "full"],
                   help="how much of the backbone to fine-tune. Stated on EVERY run -- exp124-129 "
                        "silently trained with the wrong "
                        "freeze_mode because this script used to inherit whatever params.yaml rested at "
                        "instead of stating it. Default 'none' matches what the MixStyle screen validated.")
    p.add_argument("--eta-min", type=float, default=ADOPTED["eta_min"],
                   help="CosineAnnealingLR floor (default: the adopted %(default)g). States its value on "
                        "EVERY run, like --mixstyle-p, not 'leave whatever params.yaml had'.")
    p.add_argument("--scheduler", default=ADOPTED["scheduler"], choices=list(SCHEDULERS),
                   help="LR schedule. Stated on EVERY run (default: the adopted 'cosine'), never inherited "
                        "from params.yaml -- see set_all_rigs. 'plateau' is a screening arm "
                        "(docs/lr_scheduler_plan.md) and is not adopted.")
    p.add_argument("--tag", default="allrigs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Provenance gate: this writes commits, and a sweep on uncommitted source
    # produces experiment commits that cannot re-run it.
    dirty = dirty_provenance_paths()
    if dirty and not args.allow_dirty:
        print("Uncommitted changes outside params.yaml/dvc.lock/outputs/experiments:\n")
        for line in dirty:
            print(f"    {line}")
        print("\nCommit these before starting. Override with --allow-dirty.")
        raise SystemExit(1)
    stale = stale_crop_stages(CAPTURES)
    if stale and not args.allow_dirty:
        print(f"Crop stages out of date: {', '.join(stale)}. The dataset would be regenerated "
              f"mid-run. Run `dvc repro crop` deliberately first, or --allow-dirty.")
        raise SystemExit(1)
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                     cwd=REPO_ROOT).decode().strip()
    if branch == "HEAD":
        print("HEAD is detached; commits would land off-branch. Check out a branch first.")
        raise SystemExit(1)

    for i, seed in enumerate(args.seeds):
        exp_id = args.start_exp + i
        slug = f"{args.tag}_s{seed}"
        done = REPO_ROOT / "experiments" / f"exp{exp_id}__{slug}" / "metrics.json"
        if done.exists() and not args.force:
            print(f"exp{exp_id} ({slug}) already archived, skipping", flush=True)
            continue

        print(f"\n{'=' * 72}\nexp{exp_id}  all capture dirs  seed={seed}\n{'=' * 72}", flush=True)
        cfg = set_all_rigs(seed, args.epochs, args.brightness_jitter, args.mixstyle_p, args.freeze_mode,
                            args.mixstyle_mode, args.eta_min, args.scheduler)

        # Post-condition on the CLI contract: reads what actually loaded rather than
        # trusting the write. A forwarding mistake once ran six folds (~15h) of the
        # plain config under slugs that claimed otherwise (exp100-105).
        for name, wanted, got in (
            ("mixstyle_p", args.mixstyle_p, cfg.mixstyle_p),
            ("mixstyle_mode", args.mixstyle_mode, cfg.mixstyle_mode),
            ("freeze_mode", args.freeze_mode, cfg.freeze_mode),
            ("eta_min", args.eta_min, cfg.eta_min),
            ("scheduler", args.scheduler, cfg.scheduler),
        ):
            if got != wanted:
                raise SystemExit(
                    f"{name} is {got!r} in the config that will train, but {wanted!r} was requested. "
                    f"Refusing to start."
                )

        t0 = time.time()
        if run(["dvc", "repro", "train"]) != 0:
            print(f"exp{exp_id} FAILED; stopping so the failure is not buried")
            raise SystemExit(1)
        print(f"exp{exp_id} finished in {(time.time() - t0) / 60:.0f} min", flush=True)

        note = (f"all capture dirs, pooled per-class split: shipping candidate. "
                f"beans={cfg.patch_beans_min}-{cfg.patch_beans_max}, epochs={cfg.epochs}, "
                f"seed={cfg.seed}, brightness_jitter={cfg.brightness_jitter_strength}, "
                f"mixstyle_p={cfg.mixstyle_p}, mixstyle_mode={cfg.mixstyle_mode}, freeze_mode={cfg.freeze_mode}, "
                f"eta_min={cfg.eta_min}, scheduler={cfg.scheduler}. "
                f"In-distribution val/test only (not a new-camera estimate; ticket ML-1).")
        if dirty:
            note += " WARNING: uncommitted source at launch."
        if run(["python", "-m", "coffeecv.archive_experiment",
                "--id", str(exp_id), "--slug", slug, "--note", note]) != 0:
            print(f"archiving exp{exp_id} FAILED; stopping before the next run overwrites outputs/")
            raise SystemExit(1)

        run(["git", "add", "-A", "params.yaml", "dvc.lock", "outputs/metrics.json",
             "outputs/summary.json", "experiments"])
        if run(["git", "commit", "-q", "-m",
                f"exp{exp_id}: {slug}\n\n{note}\n\n"
                f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"]) != 0:
            print(f"committing exp{exp_id} FAILED; stopping")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
