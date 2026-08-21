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
    python -m coffeecv.run_folds --arm beans --epochs 80 --brightness-jitter 0.6 \
        --tag bright06 --start-exp 72

`--brightness-jitter` is independent of `--arm`: patch sizing (baseline/scale/
beans/beans69) and brightness augmentation are orthogonal knobs, not a combined
arm matrix, so it composes with whichever sizing arm is selected instead of
needing its own ARMS entries.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from coffeecv.config import PARAMS_FILE, REPO_ROOT, RunConfig

RIGS = [
    "data/cropped/2026-08-07__box_pictures_all_classes",
    "data/cropped/2026-08-09__pixel_cam",
    "data/cropped/2026-08-09__sony_cam",
]

# frac_min, frac_max for each arm. The baseline keeps the fixed pixel patch size
# so its cross-rig number answers "what does the current model actually do".
# (frac_min, frac_max, beans_min, beans_max). Only one sizing mode is active per
# arm; bean-unit takes precedence in the dataset when its max is > 0.
ARMS = {
    "baseline": (0.0, 0.0, 0.0, 0.0),
    "scale": (0.15, 0.60, 0.0, 0.0),
    # 16-49 beans per patch. 4-7 rather than the 6-9 first proposed: per-photo
    # pitch estimation carries ~24% noise (kept deliberately, so training matches
    # inference), and at 6-9 that noise pushes 32% of patches past the frame edge
    # -- 52% on old_box. Clamped patches all collapse to the frame size and lose
    # placement freedom, so the range that fits is the range that keeps the scale
    # variety it is there to provide.
    "beans": (0.0, 0.0, 4.0, 7.0),
    # 36-81 beans per patch, the size originally wanted. Viable at 4% clamping
    # once bean pitch is measured on a 0.40 analysis window rather than the whole
    # frame; the 32% clamp rate that ruled it out earlier was an estimator bug.
    "beans69": (0.0, 0.0, 6.0, 9.0),
}


def set_fold(heldout: str, frac_min: float, frac_max: float,
             beans_min: float, beans_max: float, epochs: int | None = None,
             seed: int | None = None, brightness_jitter: float | None = None,
             freeze_mode: str | None = None, mixup_alpha: float | None = None,
             mixstyle_p: float | None = None) -> "RunConfig":
    """Point params.yaml at one fold, preserving comments and ordering.

    `epochs` is not just a cap: it is also `T_max` for the cosine LR schedule, so
    changing it changes the LR trajectory as well as how long training may run.
    Both arms of a comparison therefore have to share it -- scale@80 vs
    baseline@50 would confound "trained longer" with "annealed differently".
    """
    text = PARAMS_FILE.read_text()
    train = [r for r in RIGS if r != heldout]

    block = "train_rigs:\n" + "".join(f"  - {r}\n" for r in train)
    text = re.sub(r"train_rigs:\n(?:  - .*\n)+", block, text, count=1)
    text = re.sub(r"^heldout_rig: .*$", f"heldout_rig: {heldout}", text, count=1, flags=re.M)
    text = re.sub(r"^patch_scale_frac_min: .*$", f"patch_scale_frac_min: {frac_min}",
                  text, count=1, flags=re.M)
    text = re.sub(r"^patch_scale_frac_max: .*$", f"patch_scale_frac_max: {frac_max}",
                  text, count=1, flags=re.M)
    text = re.sub(r"^patch_beans_min: .*$", f"patch_beans_min: {beans_min}", text, count=1, flags=re.M)
    text = re.sub(r"^patch_beans_max: .*$", f"patch_beans_max: {beans_max}", text, count=1, flags=re.M)
    if epochs is not None:
        text = re.sub(r"^epochs: .*$", f"epochs: {epochs}", text, count=1, flags=re.M)
    if seed is not None:
        text = re.sub(r"^seed: .*$", f"seed: {seed}", text, count=1, flags=re.M)
    if brightness_jitter is not None:
        text = re.sub(r"^brightness_jitter_strength: .*$", f"brightness_jitter_strength: {brightness_jitter}",
                      text, count=1, flags=re.M)
    # \S+ rather than .*$ so the explanatory trailing comments on these two lines
    # survive the rewrite -- params.yaml is documentation as much as configuration.
    if freeze_mode is not None:
        text = re.sub(r"^freeze_mode: \S+", f"freeze_mode: {freeze_mode}", text, count=1, flags=re.M)
    if mixup_alpha is not None:
        text = re.sub(r"^mixup_alpha: \S+", f"mixup_alpha: {mixup_alpha}", text, count=1, flags=re.M)
    if mixstyle_p is not None:
        text = re.sub(r"^mixstyle_p: \S+", f"mixstyle_p: {mixstyle_p}", text, count=1, flags=re.M)
    PARAMS_FILE.write_text(text)

    # Read it back through the real loader: a silently-failed regex would
    # otherwise run the wrong fold and look like a result.
    cfg = RunConfig.from_params_yaml()
    assert cfg.heldout_rig == heldout, f"heldout_rig is {cfg.heldout_rig!r}, wanted {heldout!r}"
    assert list(cfg.train_rigs) == train, f"train_rigs is {cfg.train_rigs!r}, wanted {train!r}"
    assert cfg.patch_scale_frac_min == frac_min and cfg.patch_scale_frac_max == frac_max
    assert cfg.patch_beans_min == beans_min and cfg.patch_beans_max == beans_max
    assert heldout not in cfg.train_rigs, "held-out rig leaked into training"
    if epochs is not None:
        assert cfg.epochs == epochs, f"epochs is {cfg.epochs}, wanted {epochs}"
    if seed is not None:
        assert cfg.seed == seed, f"seed is {cfg.seed}, wanted {seed}"
    if brightness_jitter is not None:
        assert cfg.brightness_jitter_strength == brightness_jitter, \
            f"brightness_jitter_strength is {cfg.brightness_jitter_strength}, wanted {brightness_jitter}"
    if freeze_mode is not None:
        assert cfg.freeze_mode == freeze_mode, f"freeze_mode is {cfg.freeze_mode!r}, wanted {freeze_mode!r}"
    if mixup_alpha is not None:
        assert cfg.mixup_alpha == mixup_alpha, f"mixup_alpha is {cfg.mixup_alpha}, wanted {mixup_alpha}"
    if mixstyle_p is not None:
        assert cfg.mixstyle_p == mixstyle_p, f"mixstyle_p is {cfg.mixstyle_p}, wanted {mixstyle_p}"
    # Returned so the caller records what was actually loaded rather than what was
    # asked for -- the omitted-flag case has no value in args to report.
    return cfg


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


# Paths this script rewrites itself, so they are expected to be dirty at launch
# and are committed per fold. Everything else -- source, dvc.yaml, the
# per-session crop configs -- describes what the run *is*.
SWEEP_WRITES = ("params.yaml", "dvc.lock", "outputs/", "experiments/")


def dirty_provenance_paths() -> list[str]:
    """`git status --porcelain` entries that would leave this sweep unreproducible.

    The per-fold commit stages only params.yaml, dvc.lock, the metrics files and
    experiments/. Anything else edited but uncommitted therefore runs for hours
    and lands in no commit at all.

    Phase 13 lost an entire feature this way: `brightness_jitter_strength` was
    implemented in config.py/transforms.py and never committed, so exp 72-95 each
    recorded a git_commit whose tree has no such field in RunConfig -- and
    `from_params_yaml` filters params.yaml to known fields *silently*, so checking
    one out re-runs at the default brightness and looks like it worked. 18 paired
    runs, ~30h of compute, reproducible only from the archived config.json.
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


def stale_crop_stages() -> list[str]:
    """Crop stages that `dvc repro train` would regenerate before training.

    Deliberately *not* a check on overall `dvc status`, which is dirty by design
    here: set_fold() rewrites params.yaml precisely so the train stage re-runs, so
    "train is out of date" is the required state at launch, not a fault.

    The crop stages are different. If one is stale, `dvc repro train` regenerates
    the dataset first, and every fold then trains on different pixels than the
    reference runs it is about to be compared against -- a silent
    comparison-invalidating event. It is also the one failure git cannot see:
    `data/cropped/` is gitignored, so on-disk loss or corruption of the crops
    shows up in `dvc status` and nowhere else. An interrupted `dvc repro` deletes
    the stage's outs, which is exactly how this happens in practice.
    """
    stale = []
    for rig in RIGS:
        stage = f"crop@{Path(rig).name}"
        out = subprocess.check_output(["dvc", "status", "--json", stage], cwd=REPO_ROOT).decode()
        if json.loads(out or "{}"):
            stale.append(stage)
    return stale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--start-exp", type=int, required=True, help="experiment number of the first fold")
    p.add_argument("--only", help="run just this held-out rig (substring match)")
    p.add_argument("--force", action="store_true", help="re-run folds that are already archived")
    p.add_argument("--epochs", type=int, default=None,
                   help="epoch budget AND cosine T_max; both arms of a comparison must share it")
    p.add_argument("--tag", default="", help="slug suffix distinguishing this sweep, e.g. e80")
    p.add_argument("--seed", type=int, default=None, help="training seed; the replication axis")
    p.add_argument("--brightness-jitter", type=float, default=0.0,
                   help="sets brightness_jitter_strength; orthogonal to --arm, see module docstring. "
                        "Defaults to 0.0 (off) on *every* invocation -- unlike --epochs/--seed this is not "
                        "'leave whatever was there', because that silently carried a stale strength from one "
                        "sweep into the next 'reference' sweep once (see EXPERIMENTS_LOG.md Phase 13).")
    p.add_argument("--freeze-mode", default="none", choices=["none", "last_block", "full"],
                   help="how much of the backbone to fine-tune. Like --brightness-jitter this states its "
                        "value on EVERY run rather than inheriting whatever params.yaml held.")
    p.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="Beta(a,a) batch mixing. Phase 8 rejected 0.2 on in-distribution alone; the log "
                        "carries a standing note to re-test those augmentations against cross-rig.")
    p.add_argument("--mixstyle-p", type=float, default=0.0,
                   help="per-batch probability of MixStyle (domain-agnostic v1, resnet18 only). Like "
                        "--mixup-alpha, states its value on EVERY run rather than inheriting params.yaml.")
    p.add_argument("--no-commit", action="store_true",
                   help="skip the per-fold git commit (default is to commit each run)")
    p.add_argument("--allow-dirty", action="store_true",
                   help="start even with uncommitted source; the fact is recorded in each fold's note")
    args = p.parse_args()

    # Provenance gate. Cheap here, unrecoverable later: a sweep that runs on
    # uncommitted source produces experiment commits that cannot re-run it.
    dirty = dirty_provenance_paths()
    if dirty:
        print("Uncommitted changes outside params.yaml/dvc.lock/outputs/experiments:\n")
        for line in dirty:
            print(f"    {line}")
        if not args.allow_dirty:
            print("\nCommit these before starting the sweep. The per-fold commit stages only\n"
                  "params.yaml, dvc.lock, the metrics files and experiments/, so the changes\n"
                  "above would run for hours and land in no commit -- see EXPERIMENTS_LOG.md\n"
                  "Phase 13 for what that cost last time. Override with --allow-dirty.", flush=True)
            raise SystemExit(1)
        print("\n--allow-dirty: continuing. Each fold's note will record that the source was\n"
              "uncommitted at launch, so the gap is visible in the archived record.", flush=True)

    stale = stale_crop_stages()
    if stale:
        print(f"\nCrop stages out of date: {', '.join(stale)}\n")
        if not args.allow_dirty:
            print("`dvc repro train` would regenerate the dataset before training, so this sweep\n"
                  "would not be comparable to the reference runs it is meant to pair against.\n"
                  "Run `dvc repro crop` deliberately and re-baseline the references first, or\n"
                  "override with --allow-dirty.", flush=True)
            raise SystemExit(1)
        print("--allow-dirty: continuing on stale crops; recorded in each fold's note.", flush=True)

    # This script commits once per fold. A detached HEAD would put all of them
    # off-branch, which has already cost this project an entire phase of work once
    # (recovered from the reflog only because the commits still descended from
    # main). `git status` stays clean and reassuring the whole time, so the branch
    # name is the only thing that tells you.
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT).decode().strip()
    if branch == "HEAD" and not args.no_commit:
        print("HEAD is detached -- every per-fold commit would land off-branch and be\n"
              "invisible to `git log` on main. Check out a branch first, or pass\n"
              "--no-commit to run without committing.", flush=True)
        raise SystemExit(1)

    frac_min, frac_max, beans_min, beans_max = ARMS[args.arm]
    heldouts = [r for r in RIGS if not args.only or args.only in r]

    # Experiment numbers are assigned by hand via --start-exp, and the resume
    # check below keys on exp id *and* slug -- so reusing an id under a different
    # slug silently creates two exp<N>__* directories. index.csv is rebuilt from
    # those directories, so the collision surfaces as duplicate ids in the record
    # rather than as an error. Near-miss during Phase 13: an interrupted sweep and
    # the sweep that replaced it both started at 75.
    collisions = []
    for i, heldout in enumerate(heldouts):
        exp_id = args.start_exp + i
        tag = f"_{args.tag}" if args.tag else ""
        slug = f"lorio_{args.arm}{tag}_heldout_{Path(heldout).name.split('__')[-1]}"
        for existing in (REPO_ROOT / "experiments").glob(f"exp{exp_id}__*"):
            if existing.name != f"exp{exp_id}__{slug}":
                collisions.append(f"exp{exp_id}: would add '{slug}' beside existing '{existing.name}'")
    if collisions:
        print("Experiment id collision:\n")
        for c in collisions:
            print(f"    {c}")
        print("\nPick a --start-exp past the end of experiments/, or --force to re-run the\n"
              "existing runs under their own slug.", flush=True)
        if not args.force:
            raise SystemExit(1)
        print("--force given: continuing.", flush=True)

    for i, heldout in enumerate(heldouts):
        exp_id = args.start_exp + i
        short = Path(heldout).name
        tag = f"_{args.tag}" if args.tag else ""
        slug = f"lorio_{args.arm}{tag}_heldout_{short.split('__')[-1]}"

        # Resume: a fold that already archived a metrics.json is done. Two power
        # cuts during this sweep made restart-from-scratch the expensive default;
        # each fold is ~90 min, so re-running completed ones burns the budget the
        # outage already dented.
        done = REPO_ROOT / "experiments" / f"exp{exp_id}__{slug}" / "metrics.json"
        if done.exists() and not args.force:
            print(f"fold {i + 1}/{len(heldouts)}  exp{exp_id} ({short}) already archived, skipping",
                  flush=True)
            continue

        print(f"\n{'=' * 72}\nfold {i + 1}/{len(heldouts)}  exp{exp_id}  arm={args.arm}  "
              f"held out: {short}\n{'=' * 72}", flush=True)
        cfg = set_fold(heldout, frac_min, frac_max, beans_min, beans_max, args.epochs, args.seed,
                       args.brightness_jitter, args.freeze_mode, args.mixup_alpha, args.mixstyle_p)

        # Post-condition on the CLI contract, checked HERE rather than only inside
        # set_fold, because set_fold's own assertions are all guarded by
        # `if arg is not None` -- so an argument the caller forgets to forward
        # skips both the write and its check, in silence. That is exactly what
        # happened to exp100-105: --mixup-alpha and --freeze-mode were parsed,
        # never passed through, and six folds ran the plain adopted config for
        # ~15h while their slugs claimed otherwise. This check cannot be skipped
        # by a plumbing mistake, because it reads the config that was actually
        # loaded and compares it against what was asked for.
        for name, wanted, got in (
            ("freeze_mode", args.freeze_mode, cfg.freeze_mode),
            ("mixup_alpha", args.mixup_alpha, cfg.mixup_alpha),
            ("mixstyle_p", args.mixstyle_p, cfg.mixstyle_p),
            ("brightness_jitter_strength", args.brightness_jitter, cfg.brightness_jitter_strength),
            *(( ("epochs", args.epochs, cfg.epochs),) if args.epochs is not None else ()),
            *(( ("seed", args.seed, cfg.seed),) if args.seed is not None else ()),
        ):
            if got != wanted:
                raise SystemExit(
                    f"{name} is {got!r} in the config that will train, but {wanted!r} was requested. "
                    f"Refusing to start: the run would be archived under a slug describing a "
                    f"configuration it did not use."
                )

        t0 = time.time()
        if run(["dvc", "repro", "train"]) != 0:
            print(f"fold {short} FAILED; stopping so the failure is not buried")
            raise SystemExit(1)
        mins = (time.time() - t0) / 60
        print(f"fold {short} finished in {mins:.0f} min", flush=True)

        # Built from the config that was actually loaded, not from the CLI args.
        # `--epochs`/`--seed` mean "leave whatever params.yaml had" when omitted,
        # so reporting the arg wrote the literal string "default" into the record
        # -- naming a value without stating it. This is the general form of the
        # bug that produced three mislabelled reference runs (exp 81-83).
        note = (f"leave-one-rig-out, arm={args.arm}, held out {short}; "
                f"scale_frac={cfg.patch_scale_frac_min}-{cfg.patch_scale_frac_max}, "
                f"beans={cfg.patch_beans_min}-{cfg.patch_beans_max}, "
                f"epochs={cfg.epochs}, seed={cfg.seed}, "
                f"brightness_jitter={cfg.brightness_jitter_strength}, "
                f"freeze_mode={cfg.freeze_mode}, mixup_alpha={cfg.mixup_alpha}, "
                f"mixstyle_p={cfg.mixstyle_p}")
        if dirty:
            note += "; WARNING: uncommitted source at launch, not reproducible from this commit"
        if stale:
            note += f"; WARNING: crop stages stale at launch ({', '.join(stale)}), data may differ from references"
        # Archiving is the only durable record: outputs/ is gitignored and the next
        # fold overwrites it. If it fails, the run is already unrecoverable, so
        # stop rather than carry on producing folds that cannot be reported.
        if run(["python", "-m", "coffeecv.archive_experiment",
                "--id", str(exp_id), "--slug", slug, "--note", note]) != 0:
            print(f"archiving exp{exp_id} FAILED; stopping -- outputs/ is about to be "
                  f"overwritten by the next fold and this run would be lost", flush=True)
            raise SystemExit(1)

        if not args.no_commit:
            # One commit per fold, so each run is its own git revision carrying its
            # params.yaml, dvc.lock and outputs/summary.json. That is what lets the
            # VS Code DVC extension list them as separate experiments and plot
            # metrics across them; committing only at the end of a sweep would
            # collapse every fold into a single revision.
            run(["git", "add", "-A", "params.yaml", "dvc.lock", "outputs/metrics.json",
                 "outputs/summary.json", "experiments"])
            # A failed commit leaves the run archived but with no revision carrying
            # its params/lock, which is the same provenance hole the launch gate
            # exists to prevent -- so it stops rather than silently continuing.
            if run(["git", "commit", "-q", "-m",
                    f"exp{exp_id}: {slug}\n\n{note}\n\n"
                    f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"]) != 0:
                print(f"committing exp{exp_id} FAILED; stopping so later folds do not pile\n"
                      f"uncommitted state on top of it", flush=True)
                raise SystemExit(1)


if __name__ == "__main__":
    main()
