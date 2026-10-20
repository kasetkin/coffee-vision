---
status: accepted
---

# A deploy stops the live service for its smoke test

A deploy to `webappremote` stops the live service (`coffee-cv-web`) before it smokes the new release, and photo answers are down until the full deploy restarts the service after the flip, about 5-7 minutes; the page stays up. This is the normal way to deploy, not a workaround: the preflight's `MIN_MEM_GB` (5) stays, and is met with the service stopped.

- **Why**: the host's memory holds one GPU worker, not two. A worker costs about 3.3 GB of MemAvailable, its GPU buffers as shared memory included; beside the live one the host has about 3.6 GB of its 7.7 GB (OPS-10, measured in ML-8's deploy, 2026-10-20). A smoke worker beside it would leave about 0.3 GB and run on swap or meet the OOM killer, whose likeliest victim is the live worker. Lowering the bar to fit would trade a clean refusal for a crash mid-deploy.
- **Other ways, not taken**: the smoke test must run the release in production's sandbox on the GPU (ADR 0020), so it cannot move to the CPU; the host's other services (a desktop session, monitoring) hold only about 0.2-0.3 GB; a host with more memory would fit both workers.
- **Procedure**: `scripts/deploy_webapp.sh` does it (`make_room`, the owner's go 2026-10-20): it stops the service only once everything the release needs is on the host, so the outage covers only the release's run there (the smoke, `--compare`), the flip and the restart, and it starts the old release again on any failure before the flip. `docs/agents/deploy.md` has the details.

The immutable releases and rollback of ADR 0008 are unchanged. Supersedes the owner's per-deploy choice in OPS-7 and settles OPS-10's question of making room.

Source: the owner, 2026-10-20, after ML-8's deploy; [OPS-10](../ticket_deploy_memory_headroom.html).
