# Running work on a remote machine

Read before starting, watching or syncing a run on a machine other than this one. There is none now: the VM (`powervpsssh`) that ran sweeps and served the live site was shut down for good on 2026-10-12, so runs happen here until the owner names a new machine. Ask the owner before starting work on one; the rest of this page holds for any of them.

## Launching and waiting

- Launch with `scripts/remote_launch.sh '<payload>' <ssh_host>`, which detaches the job from the SSH session so a dropped connection cannot kill it. Pass **only the payload and the host**: a `~/...` path argument is expanded by the local shell into this container's home before the script sees it, the remote `cd` fails, and the script still prints `LAUNCHED`. Its defaults are expanded on the remote machine, which is right.
- Get the completion notice by running `scripts/remote_wait.sh <ssh_host>` as a background command here: it blocks on the job's status file and returns when the job ends. `scripts/remote_watch.sh` tails the filtered log; `scripts/remote_log.sh` fetches a readable copy.
- To confirm new code actually ran, set `PYTHONUNBUFFERED=1` and look early for a line only the new code prints: a stale checkout once ran a whole benchmark without the change, and block-buffered stdout hides that for a long time.
- After a code sync, refresh the remote environment with `uv sync --locked` ([ADR 0007](../adr/0007-uv-for-every-environment.md)).

## Keeping the remote history and local history together

- **Before a run**, sync the remote checkout to local `main`; **after it**, bring its experiment commits home promptly. The longer both sides commit, the harder the merge.
- Sync by simple merges only ([ADR 0009](../adr/0009-simple-merges-only.md)), the procedure in OPS-1 §7 (`docs/ticket_webapp_release_isolation.html`): `git fetch <host> main:tmp/vm-sweep`, merge on a `tmp/merge-sweep` branch, check, `git merge --ff-only` into `main`; then push `main` to the remote's `tmp/from-local` and `git merge --ff-only` there.
- `git reset --hard` discards every uncommitted tracked change in the tree, not only the commit being undone; it once wiped a day's uncommitted code on the VM. Commit or stash first, or undo with `git checkout <commit> -- <paths>`.
- Before trusting a remote run, check `git log` there matches what you meant to run.
