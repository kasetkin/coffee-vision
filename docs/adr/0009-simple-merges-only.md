# Git history is joined by simple merges only

Diverged histories (the VM's sweep commits and local work) are joined with fetch and merge, as a fast-forward or one merge commit, rehearsed on `tmp/*` branches. Rebase, cherry-pick, amend, `reset --hard` and force-push are not used: `experiments/index.csv` records the commit each run trained on, and rewritten hashes orphan those records ([ADR 0004](0004-deleted-provenance-commits-stay-deleted.md)).

Source: [OPS-1 §7](../ticket_webapp_release_isolation.html), owner decision 2026-09-27.
