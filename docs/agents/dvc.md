# DVC: moving data and trusting what it says

Read before moving data between machines, pushing to the remote, or committing after a DVC command. The remote's setup is in docs/development.md, "Dataset & DVC".

## Moving data

- **Use `dvc push` here and `dvc pull` (or `dvc fetch`) on the other machine.** Both verify hashes. Copying `.dvc/cache` objects with rsync or scp is retired: an interrupted `rsync --partial --ignore-existing` once left a truncated object under its final name, the retry skipped it, and a VM run failed with "image file is truncated".
- **Push explicit targets** (`dvc push models/<name>.pt.dvc`) and verify the objects. `dvc push --all-commits` has reported "Everything is up to date" while an object it collected was missing from the remote.
- **After the host reboots**, the remote (`dvcnewremote`, `webappremote`: yappy's Tailscale name) may not resolve here: Docker regenerates the container's `/etc/resolv.conf` with the LAN's resolver only, which knows no `*.ts.net` name ("Name or service not known"). Put Tailscale's resolver first, in place (the file is bind-mounted): `{ echo nameserver 100.100.100.100; cat /etc/resolv.conf; } > /tmp/r && sudo tee /etc/resolv.conf < /tmp/r > /dev/null`. Seen 2026-10-22 and 2026-10-27.
- `--all-commits` reads `.dvc/config` as of each commit. A remote declared in `.dvc/config.local` without a `url` makes every older commit fail with `ConfigError: expected 'url'`, logged as a WARNING with exit 0 (323 commits silently skipped once). Keep a local override self-sufficient: `dvc remote modify --local <remote> url ...`.

## Trusting the result

DVC's last line is not the verdict. `dvc status --cloud` has printed "Cache and remote are in sync" after three warnings that objects were missing, and exited 0. Read the warnings; to verify, expand every `.dir` in `dvc.lock` and test each object:

```python
import yaml, json, pathlib
cache = pathlib.Path('.dvc/cache/files/md5')   # or the remote store's files/md5
lock = yaml.safe_load(open('dvc.lock'))
want = set()
for st in lock['stages'].values():
    for o in st.get('outs') or []:
        h = o.get('md5')
        if not h: continue
        want.add(h)
        if h.endswith('.dir') and (p := cache / h[:2] / h[2:]).exists():
            want |= {e['md5'] for e in json.load(open(p))}
print(sorted(h for h in want if not (cache / h[:2] / h[2:]).exists()))
```

Only `cache: false` outputs may come back missing, today each fine-tune's `models/seg/<name>.json` (nine on 2026-10-27, `xl0_v3` to `xl0_v5_s7`): they live in git.

`dvc repro --dry` has said "cached, skipping" for a stage `dvc status` correctly reported as changed. Check `dvc status`, and after a run check that `outputs/config.json` holds the intended values.

## Keeping git and DVC apart

- Every DVC-tracked output stays gitignored (DVC writes a sibling `.gitignore`; if one goes missing, `git add -A` commits the data). Read what `git add -A` stages after DVC has touched the tree, and never `git add` a `dataset/` directory by name: that overrides `.gitignore`, and photos once landed in git history that way, permanently.
- `dvc exp run` can leave HEAD detached, and commits made afterwards land off-branch with no warning. Run `git rev-parse --abbrev-ref HEAD` after any `dvc exp` command; `HEAD` means check out the branch first. Recorded runs go through `dvc repro` (or the drivers that call it), not `dvc exp`.
- If a `.dir` hash differs between machines, compare the two listings entry by entry before assuming the data drifted; re-running the stage on one machine only mints a third hash. Precedent: an absolute path in a crop report (`5d0e9fd`).

## What `dvc status` shows on a clean checkout

Since 2026-10-20 the stages whose recipe moved on are frozen: `seg_ml6_finetune` (XL0-v3, ADR 0027) and `seg_ml5_predict@xl0_v3`. Since 2026-10-24 `seg_ml9_predict` too, ML-9 P4's record of a test split that has grown since. Since 2026-10-27 `seg_xl0v5_predict` too, the record of XL0-v5's blind paired sessions, for the same reason. `dvc status` lists a frozen stage as frozen without checking it. ML-2's segmenter stages and ML-5's XL0-v1 and v2 were removed on 2026-10-21, and the ResNet trainer's `train` stage on 2026-10-20, each with its code; they are in git history, and their outputs in the DVC cache but for the encoder caches lost before (`data/seg_cache`, `data/seg_cache_ml5`, below). A clean checkout reports "Data and pipelines are up to date": anything else it reports is real.

## Cleaning the cache

Nothing leaves the cache until the remote holds it ([ADR 0028](../adr/0028-the-dvc-cache-is-never-the-last-copy.md)): `dvc gc` deletes objects only this cache holds, and nothing warns about that. Use `dvc gc --all-commits` (never `--workspace`, which deletes what older commits' `dvc.lock` names: on 2026-10-20 that was 26.5 GB, 7,990 of 8,572 objects, most never pushed). Before it runs, read each warning with `--dry`:

- **"Output ... is missing version info. Cache for it will not be collected."** The output is *not* counted as in use, so its objects are deleted if nothing else names them. Here it was raised for outputs whose working copy was missing, of frozen stages. On 2026-10-20 it cost `data/seg_cache_ml5/pass1` and `pass2` (7.5 GB, never pushed), read as "kept". Push such an output, or `dvc checkout` it, before collecting.
- **A commit DVC cannot read** is skipped, and the objects only it uses are deleted.

Then test every object the dry run would remove on the remote, by its path under `files/md5/` (the listing of "Trusting the result" above, pointed at the remote's store), and push or keep each one it lacks.

## Objects lost with the VM

The VM (`powervpsssh`) was shut down for good on 2026-10-12 and took the objects only it held: `data/seg_cache` (ML-2's stage `seg_embed_cache`, removed on 2026-10-21, 5.8 GB) and the checkpoints, `history.json` and prediction CSVs of exp106-exp239 (their metrics stay in git). A `dvc pull` or `dvc checkout` that names them fails.

## Objects deleted on the owner's go

On 2026-10-21 the owner had the tray heuristic's `data/cropped` pools (retired in ML-3 P2b; nothing at HEAD reads them) deleted from the cache without a push, the exception ADR 0028 allows: the 790 objects (0.71 GB) that only `data/cropped` outputs name, in any commit. The objects they share with `dataset/` photos and `data/segcropped` stay. A `dvc checkout` of `data/cropped` at an older commit fails.
