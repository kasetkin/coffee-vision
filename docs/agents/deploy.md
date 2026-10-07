# Deploying the webapp

Read before a deploy, a rollback, or a claim that something is live. Usage is in `scripts/deploy_webapp.sh`'s header (`--help`); the design is [ADR 0008](../adr/0008-webapp-serves-immutable-releases.md) and OPS-1.

- **Only pushed commits deploy.** The script deploys commits on `origin/main`. Push when the owner asks: a plain `git push origin main` uses the devcontainer's key, `/workspace/.private_ssh/coffee-vision-devcontainer`, set as this clone's `core.sshCommand`. If a push fails with `Permission denied (publickey)`, that setting is missing (a fresh clone): `git config core.sshCommand "ssh -i /workspace/.private_ssh/coffee-vision-devcontainer -o IdentitiesOnly=yes"`.
- **A model ships with its OOD probe** (`models/<name>.ood_probe.json`, [ADR 0005](../adr/0005-linear-probe-ood-guard.md)), fitted to that checkpoint.
- **"Deployed" means verified on the live site**: `scripts/deploy_webapp.sh --verify` (needs `DOMAIN`) or a real request to the domain, not a successful restart. A restart succeeding once hid a service that crash-looped right after.
- **Latency**: quote the server-side `latency_ms` from the app's log: about 7 s idle and 12.5 s during a sweep (OPS-1). A `curl` from this container adds about 7 s of upload and is not server latency.
- **Rehearse locally** with `scripts/rehearse_deploy_local.sh` before changing the deploy script itself.
- Rollback is `--rollback`, which switches the VM to the `previous` release.
