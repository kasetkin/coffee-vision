# Containers on the server

The VM runs the webapp, and anything added beside it, as native services: systemd units, nginx, and a uv environment per release ([ADR 0008](../docs/adr/0008-webapp-serves-immutable-releases.md)). Docker is not installed there and is not introduced, for the webapp or for additions such as visitor analytics.

## Why this is out of scope

Deployment stays one script plus config templates (`scripts/deploy_webapp.sh`, `webapp/deploy/`), and the release isolation it needs (its own directory, user, environment and systemd sandbox) is already there without a container runtime. A second way to run services would double what a deploy has to get right. The devcontainer is a separate matter: it is the development environment, not the server.

## Prior requests

- [OPS-1](../docs/ticket_webapp_release_isolation.html), `docs/ops1_release_isolation_plan.html` §10: "moving serving to another machine or into a container".
- The Umami analytics design (2026-08-31): native install with pnpm, no Docker.
