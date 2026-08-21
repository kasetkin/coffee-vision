"""Train the shipping model: every rig, no held-out rig.

Every run in this project so far is a leave-one-rig-out fold, so no model has
ever seen all three rigs -- each was deliberately blind to a third of the data so
that a cross-rig number could be measured. That is the right trade for deciding a
*configuration* and the wrong one for shipping *weights*.

**This run cannot be scored on cross-rig transfer, by construction.** With every
rig in training there is nothing left to transfer to, so the only metrics are
in-distribution. That is not a gap to apologise for, it is the standard split of
labour: the folds choose the configuration, the final fit uses all the data. What
this run must not do is quietly become the thing a configuration claim rests on
-- any *change* still has to be validated through run_folds, where the cross-rig
metric exists.

So the in-distribution test score here is a sanity check ("did adding a third rig
break anything?"), not evidence of an improvement. Compare it to the folds'
in-distribution numbers, never to their cross-rig ones.

    python -m coffeecv.run_all_rigs --seeds 42 123 7 --start-exp 96
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

from coffeecv.config import PARAMS_FILE, REPO_ROOT, RunConfig
from coffeecv.run_folds import RIGS, dirty_provenance_paths, run, stale_crop_stages


def set_all_rigs(
    seed: int, epochs: int | None, brightness_jitter: float, mixstyle_p: float, freeze_mode: str,
) -> RunConfig:
    """Point params.yaml at every rig with no held-out rig, preserving comments."""
    text = PARAMS_FILE.read_text()
    block = "train_rigs:\n" + "".join(f"  - {r}\n" for r in RIGS)
    text = re.sub(r"train_rigs:\n(?:  - .*\n)+", block, text, count=1)
    # "" is the documented disable switch in RunConfig: no heldout rig, no
    # test_xrig split, no cross-rig metric.
    text = re.sub(r"^heldout_rig: .*$", 'heldout_rig: ""', text, count=1, flags=re.M)
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
    text = re.sub(r"^freeze_mode: \S+", f"freeze_mode: {freeze_mode}", text, count=1, flags=re.M)
    if epochs is not None:
        text = re.sub(r"^epochs: .*$", f"epochs: {epochs}", text, count=1, flags=re.M)
    PARAMS_FILE.write_text(text)

    # Read back through the real loader, same as run_folds: a regex that silently
    # failed would otherwise train the wrong thing and look like a result.
    cfg = RunConfig.from_params_yaml()
    assert cfg.seed == seed, f"seed is {cfg.seed}, wanted {seed}"
    assert cfg.heldout_rig == "", f"heldout_rig is {cfg.heldout_rig!r}, wanted empty"
    assert list(cfg.train_rigs) == RIGS, f"train_rigs is {cfg.train_rigs!r}"
    assert cfg.brightness_jitter_strength == brightness_jitter
    assert cfg.mixstyle_p == mixstyle_p, f"mixstyle_p is {cfg.mixstyle_p}, wanted {mixstyle_p}"
    assert cfg.freeze_mode == freeze_mode, f"freeze_mode is {cfg.freeze_mode!r}, wanted {freeze_mode!r}"
    if epochs is not None:
        assert cfg.epochs == epochs
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="one run per seed; several make an ensemble, if the ensemble study says that helps")
    p.add_argument("--start-exp", type=int, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--brightness-jitter", type=float, default=0.0)
    p.add_argument("--mixstyle-p", type=float, default=0.0,
                   help="per-batch probability of MixStyle (resnet18 only). States its value on EVERY "
                        "run, like --brightness-jitter, not 'leave whatever params.yaml had'.")
    p.add_argument("--freeze-mode", default="none", choices=["none", "last_block", "full"],
                   help="how much of the backbone to fine-tune. Stated on EVERY run, matching "
                        "run_folds.py's own convention -- exp124-129 silently trained with the wrong "
                        "freeze_mode because this script used to inherit whatever params.yaml rested at "
                        "instead of stating it. Default 'none' matches what the MixStyle screen validated.")
    p.add_argument("--tag", default="allrigs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()

    # Same launch gate as run_folds: this writes commits, and a sweep on
    # uncommitted source produces experiment commits that cannot re-run it.
    dirty = dirty_provenance_paths()
    if dirty and not args.allow_dirty:
        print("Uncommitted changes outside params.yaml/dvc.lock/outputs/experiments:\n")
        for line in dirty:
            print(f"    {line}")
        print("\nCommit these before starting. Override with --allow-dirty.")
        raise SystemExit(1)
    stale = stale_crop_stages()
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

        print(f"\n{'=' * 72}\nexp{exp_id}  ALL RIGS  seed={seed}  (no held-out rig)\n{'=' * 72}", flush=True)
        cfg = set_all_rigs(seed, args.epochs, args.brightness_jitter, args.mixstyle_p, args.freeze_mode)

        # Post-condition on the CLI contract, same rationale as run_folds.py's own
        # check: reads what actually loaded rather than trusting the write.
        for name, wanted, got in (
            ("mixstyle_p", args.mixstyle_p, cfg.mixstyle_p),
            ("freeze_mode", args.freeze_mode, cfg.freeze_mode),
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

        note = (f"ALL RIGS (no held-out rig): shipping candidate. "
                f"beans={cfg.patch_beans_min}-{cfg.patch_beans_max}, epochs={cfg.epochs}, "
                f"seed={cfg.seed}, brightness_jitter={cfg.brightness_jitter_strength}, "
                f"mixstyle_p={cfg.mixstyle_p}, freeze_mode={cfg.freeze_mode}. "
                f"NO cross-rig metric exists for this run by construction; in-distribution only.")
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
