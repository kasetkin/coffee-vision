# Development

How the repo is laid out, how to set up the dev environment, the checks every change passes, and how photos are
versioned. For what the project is, see the top-level [README.md](../README.md).

## Repo layout

- `dataset/` — labeled photo captures. Each capture session is its own folder, usually dated (e.g. `2026-09-24__pixel/`; `data/segcropped/pool/merge_manifest.json` lists the ones training reads), tracked with [DVC](#dataset--dvc) rather than committed directly to git. `classes.txt` is a plain git-tracked text file with one line per coffee (class folder), `id;Country[,Region[,Subregion]];Misc`, e.g. `011;Peru,Junin,Satipo;Minca` (ticket ML-3 D15). The model's class is the **country**: several folders of one country train as one class. The farm region and misc are facts about the photos, never labels (`GLOSSARY.md`).
- `coffeecv/` — the training, evaluation and inference code; `coffeecv_dino/` holds the DINOv3 screening tools.
- `models/` — shipped checkpoints (`.pt` via DVC, and the live one's LiteRT conversion, `<name>.litert/`) with their git-tracked cards and OOD sidecars; `models_pretrained/` — upstream backbone weights (untracked) and their checksums.
- `experiments/` — one archived directory per run, plus `experiments/index.csv`.
- `webapp/` — the public web service serving the shipped classifier (`webapp/README.md`).
- `scripts/` — the guardrail (`check.sh`), the deploy and its local rehearsal (`deploy_webapp.sh`, `rehearse_deploy_local.sh`) and the remote-run toolkit (`remote_launch.sh`, `remote_watch.sh`, `remote_wait.sh`, `remote_log.sh`), idle while there is no remote training machine.
- `docs/` — plans, investigations and their records. `docs/patch_samples.png`, the README's sample patches, is drawn by `python -m coffeecv.patch_samples` from the photos listed in `docs/patch_samples.txt`.
- `hardware/` — sensor datasheets (NIR: AS7263/AS7265x/AS7343, gas: BME688, LEDs) and a design-research writeup (`Computer vision models for coffee bean origin classification - Claude.pdf`) on which physical/chemical signals actually carry origin information. Reference material for a fallback sensor rig, only worth building if plain-image CV underperforms — nothing here is built or wired up yet.
- `.devcontainer/` — the dev environment (below).
- `.vscode/c_cpp_properties.json` — C/C++ IntelliSense config anticipating firmware work; unused while CV-only is the active path.

## Dev environment

Open in VS Code with the Dev Containers extension ("Reopen in Container"). It builds `Dockerfile.cpu`: Python 3.12, PyTorch (CPU wheels — no NVIDIA GPU on this machine), OpenCV, scikit-learn, DVC, etc. `--device=/dev/dri` passes through this machine's AMD iGPU for OpenCV's OpenCL path; as configured the devcontainer won't start on a host without that device (cloud VM, macOS, NVIDIA-only box) — there's no separate GPU/cloud variant. Training runs on this machine: the CPU VM that ran sweeps and served the web app was shut down for good on 2026-10-12, and there is no other training machine until the owner names one (`docs/agents/remote-sweeps.md`). The web app runs on its own host, `webappremote`, which trains nothing.

Persisted across rebuilds via named Docker volumes (not part of the repo — a `docker volume prune` or Docker reset would lose them): bash history, IPython/Jupyter history and the uv cache. Two gitignored folders in the workspace itself (a bind mount from the host) persist too: `.private_claude/`, mounted as Claude Code's config directory (auth, sessions, memory), and `.private_ssh/`, the devcontainer's SSH keys and config. `~/.ssh` is a link to `.private_ssh/`, so the SSH config with the host aliases (`webappremote`, `dvcnewremote`), the keys and `known_hosts` survive a rebuild too; `.devcontainer/link-private-ssh.sh` makes the link on container create, and also sets this clone's `git config core.sshCommand` to push with the devcontainer's key, `.private_ssh/coffee-vision-devcontainer`.

## Checks

`scripts/check.sh` is the guardrail: `coffeecv.leak_check` (no public IPs, emails, credentials or photo GPS in what git would publish), `coffeecv.ticket_check` (the HTML tickets, plans and ADRs keep the tracker's rules), `ruff check` (the lint rules in `pyproject.toml`'s `[tool.ruff]`), `coffeecv.coverage_report` and the unittest suite, in two tiers. `.githooks/pre-commit` runs the leak check and Ruff on the staged files before every commit. The fast tier (~45 s on 2026-10-18) skips the tests marked `@real_data` in `tests/_tiers.py` (real photos or real weights); `.githooks/pre-push` runs it before every push to origin, pushes to any other remote skip it, and `git push --no-verify` bypasses it once. `scripts/check.sh --full` (~11 min on 2026-10-18) runs every test; `scripts/deploy_webapp.sh` runs it on the commit it deploys, before touching the host. The devcontainer wires the hook (`git config core.hooksPath .githooks`); on a fresh clone outside it, run that command yourself.

## Dataset & DVC

New capture sessions get tracked with:

```bash
dvc add dataset/<session-name>
git add dataset/<session-name>.dvc dataset/.gitignore
```

This keeps the actual images out of git (only a small `.dvc` pointer + hash gets committed) while still versioning them alongside code.

The default remote is `dvcnewremote`, an SSH remote configured in `.dvc/config`: `/home/sun/dvc-store`, reached through the `dvcnewremote` SSH host alias (in `.private_ssh/config`). `dvc push` backs the cache up there. It replaced `powervpsssh`, the VM, which was shut down for good on 2026-10-12; the objects that lived only there are gone (`docs/agents/dvc.md`). Verify a push by checking that the objects `dvc.lock` names exist on the remote, not by `dvc status --cloud`'s summary, which can report "in sync" while objects are missing.
