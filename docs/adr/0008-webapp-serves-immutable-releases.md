# The webapp serves immutable releases, apart from the training checkout

The live site runs from `/opt/coffee-cv/releases/<id>` (only `current` and `previous` kept), each release with its own full uv environment and its model named in `release.env`, under its own system user, so training sweeps cannot change what it serves. `scripts/deploy_webapp.sh` deploys only commits on `origin/main`; rollback switches to `previous`.

Source: [OPS-1](../ticket_webapp_release_isolation.html), closed 2026-09-28; implementation in `docs/ops1_release_isolation_plan.html`.
