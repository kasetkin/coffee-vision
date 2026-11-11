---
status: withdrawn, with OPS-12 (D26): no JavaScript it was for was written
---

# pnpm installs every JavaScript dependency, never npm

The web page's TypeScript and its WASM codecs are installed with pnpm from a committed `pnpm-lock.yaml` (`pnpm install --frozen-lockfile`, the counterpart of `uv sync --locked` in [ADR 0007](0007-uv-for-every-environment.md)); npm is not used for any install or script, here or in a release build. The reason is supply-chain attacks: pnpm does not run a dependency's install scripts unless the project allows that package by name, so a compromised package cannot run code on the machine that installs it, which npm does by default.

Source: [OPS-12](../ticket_browser_prepared_photo.html) D7, owner decision 2026-10-28. Written as ADR 0031 on OPS-12's branch, which was not merged until 2026-11-11, when `main` had its own ADR 0031; renumbered 0038 and withdrawn with the ticket (D26). Should a JavaScript dependency ever come, its reasoning stands to be taken up again.
