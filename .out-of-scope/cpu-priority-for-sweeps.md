# CPU priority between sweeps and the live site

The training VM also serves the live site. There is no scheme that ranks the webapp's CPU above a training sweep's, or the other way round (nice levels, cgroup weights, pausing a sweep for traffic).

## Why this is out of scope

When both are busy they share the CPU, and the site stays on during sweeps: server-side latency was measured at about 7 s idle and 12.5 s during a sweep, and the owner accepts that. Whether the site runs during a sweep is the owner's call, not something the deploy tooling decides. The deploy itself is capped (`CPUQuota`, `MemoryMax`, `--nice=19` in `scripts/deploy_webapp.sh`), which is a different matter: it keeps a deploy from disturbing a running sweep.

## Prior requests

- [OPS-1](../docs/ticket_webapp_release_isolation.html), `docs/ops1_release_isolation_plan.html` §10 (Out of scope); confirmed by the owner when OPS-1 closed.
