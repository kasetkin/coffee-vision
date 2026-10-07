# Running work on the VM

Read before starting, watching or syncing a run on the VM (SSH alias `powervpsssh`). The VM also serves the live site from its own release directory ([ADR 0008](../adr/0008-webapp-serves-immutable-releases.md)); ask the owner before starting work there.

## The machine

AMD EPYC, 4 physical cores (8 logical), 16 GB RAM plus a 16 GB swap file, no GPU. It has internet access but no GitHub key, so it cannot push to `origin`. The repo is `~/coffee-vision`; the training environment is `~/coffee-vision-venv`, and `dvc` is only in that venv. Refresh it after a code sync with `UV_PROJECT_ENVIRONMENT=$HOME/coffee-vision-venv UV_PYTHON_DOWNLOADS=never uv sync --locked` ([ADR 0007](../adr/0007-uv-for-every-environment.md)).

Threads: torch's default (the 4 physical cores) is right for training; 8 threads gave nothing there. For frozen-ViT inference 8 threads do help (36 ms against 47 ms per image for DINOv3-S/16).

## Launching and waiting

- Launch with `scripts/remote_launch.sh '<payload>'`, which detaches the job from the SSH session so a dropped connection cannot kill it. Pass **only the payload**: a `~/...` path argument is expanded by the local shell into this container's home before the script sees it, the remote `cd` fails, and the script still prints `LAUNCHED`. Its defaults are expanded on the VM, which is right.
- Get the completion notice by running `scripts/remote_wait.sh` as a background command here: it blocks on the job's status file and returns when the job ends. `scripts/remote_watch.sh` tails the filtered log; `scripts/remote_log.sh` fetches a readable copy.
- To confirm new code actually ran, set `PYTHONUNBUFFERED=1` and look early for a line only the new code prints: a stale checkout once ran a whole benchmark without the change, and block-buffered stdout hides that for a long time.

## Keeping the VM's history and local history together

- **Before a run**, sync the VM to local `main`; **after it**, bring the VM's experiment commits home promptly. The longer both sides commit, the harder the merge.
- Sync by simple merges only ([ADR 0009](../adr/0009-simple-merges-only.md)), the procedure in OPS-1 §7 (`docs/ticket_webapp_release_isolation.html`): `git fetch powervpsssh main:tmp/vm-sweep`, merge on a `tmp/merge-sweep` branch, check, `git merge --ff-only` into `main`; then push `main` to the VM's `tmp/from-local` and `git merge --ff-only` there.
- `git reset --hard` discards every uncommitted tracked change in the tree, not only the commit being undone; it once wiped a day's uncommitted code on the VM. Commit or stash first, or undo with `git checkout <commit> -- <paths>`.
- Before trusting a VM run, check `git log` there matches what you meant to run.
