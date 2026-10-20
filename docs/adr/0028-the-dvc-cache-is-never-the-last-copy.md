---
status: accepted
---

# A DVC object leaves this machine's cache only once the remote holds it

No DVC object is deleted from the local cache (`.dvc/cache`) until the same object, by md5, is checked on the remote `dvcnewremote`. That covers `dvc gc` with any flag, removing cache files by hand, and anything else that frees the cache. The check reads the remote's files: list every object the cleanup would remove (`dvc gc --dry`, each `.dir` expanded) and test each one on the remote, as `docs/agents/dvc.md` shows. DVC's own verdict is not the check. An object that is not on the remote is either pushed first or kept. The exception is one the owner names as not wanted on the remote (private data, ADR 0014), which may then be deleted.

- **Why**: on 2026-10-20 a `dvc gc --all-commits` deleted `data/seg_cache_ml5/pass1` and `pass2`, ML-5's encoder caches (7.5 GB). They had never been pushed, so no copy exists anywhere. The dry run had warned "missing version info. Cache for it will not be collected", which was read as "kept". It means the output is not counted as in use. DVC's reports have misled before: `dvc status --cloud` said "in sync" with objects missing, and `dvc push --all-commits` said "up to date" with an object missing (`dvc.md`). Only the remote's files settle it.
- **The remote is the last copy**: nothing here backs it up. Deleting an object from the remote needs the owner's go, for each named set, and a check that no commit's `dvc.lock` or `.dvc` file names it.
- **Cost**: a cleanup pushes first. On 2026-10-20, freeing the 26.5 GB that only older commits name would have meant pushing about 26 GB first (the remote had 109 GB free); it was left in the cache instead. A rebuildable output (a stage's cache) is pushed like any other: rebuilding it costs hours, and a mask may not come back bit for bit (ML-2 P0).

Source: the owner, 2026-10-20, after the loss above (`docs/agents/dvc.md`, "Cleaning the cache").
