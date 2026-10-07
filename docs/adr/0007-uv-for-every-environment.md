# uv manages every Python environment

Training (devcontainer and VM) and each webapp release are built with uv from a committed `uv.lock` (`uv sync --locked`), never pip. Moving a tool to uv first locks the versions already installed; upgrading them is a separate step the owner approves.

Sources: [OPS-1 §1](../ticket_webapp_release_isolation.html) (webapp, 2026-09-27); owner decision, 2026-09-30 (training).
