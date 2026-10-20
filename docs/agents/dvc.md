# DVC: moving data and trusting what it says

Read before moving data between machines, pushing to the remote, or committing after a DVC command. The remote's setup is in README.md, "Dataset & DVC".

## Moving data

- **Use `dvc push` here and `dvc pull` (or `dvc fetch`) on the other machine.** Both verify hashes. Copying `.dvc/cache` objects with rsync or scp is retired: an interrupted `rsync --partial --ignore-existing` once left a truncated object under its final name, the retry skipped it, and a VM run failed with "image file is truncated".
- **Push explicit targets** (`dvc push models/<name>.pt.dvc`) and verify the objects. `dvc push --all-commits` has reported "Everything is up to date" while an object it collected was missing from the remote.
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

Only ML-2's `outputs/seg_eval_*.json` may come back missing: they are `cache: false` and live in git.

`dvc repro --dry` has said "cached, skipping" for a stage `dvc status` correctly reported as changed. Check `dvc status`, and after a run check that `outputs/config.json` holds the intended values.

## Keeping git and DVC apart

- Every DVC-tracked output stays gitignored (DVC writes a sibling `.gitignore`; if one goes missing, `git add -A` commits the data). Read what `git add -A` stages after DVC has touched the tree, and never `git add` a `dataset/` directory by name: that overrides `.gitignore`, and photos once landed in git history that way, permanently.
- `dvc exp run` can leave HEAD detached, and commits made afterwards land off-branch with no warning. Run `git rev-parse --abbrev-ref HEAD` after any `dvc exp` command; `HEAD` means check out the branch first. Recorded runs go through `dvc repro` (or the drivers that call it), not `dvc exp`.
- If a `.dir` hash differs between machines, compare the two listings entry by entry before assuming the data drifted; re-running the stage on one machine only mints a third hash. Precedent: an absolute path in a crop report (`5d0e9fd`).

## What `dvc status` shows on a clean checkout

Since 2026-10-20 the stages whose recipe moved on are frozen: ML-2's and ML-5's segmenter stages, `seg_ml5_predict`, and `seg_ml6_finetune` (XL0-v3, ADR 0027). The ResNet trainer's `train` stage was removed with its code. `dvc status` lists each as frozen without checking it. Three of them still report missing outputs, which is expected and not a fault: `seg_embed_cache` (`data/seg_cache`, lost with the VM, below) and `seg_ml5_cache@pass1` and `@pass2` (`data/seg_cache_ml5`, removed from the working tree by ML-6's plan and whole in the cache, 7.5 GB: `dvc checkout` brings them back). Anything else it reports is real.

## Cleaning the cache

`dvc gc --all-commits` only: it keeps every object any commit uses. Read its warnings first. "Missing version info" means the object is kept; a commit DVC cannot read is skipped, and the objects only it uses are deleted. `dvc gc --workspace` deletes what older commits' `dvc.lock` names. Most of that (old crop sets and masks) was never pushed, so it exists only here. On 2026-10-20 that was 26.5 GB, 7,990 of 8,572 objects. Push it first if it must go.

## Objects lost with the VM

The VM (`powervpsssh`) was shut down for good on 2026-10-12 and took the objects only it held: `data/seg_cache` (stage `seg_embed_cache`, 5.8 GB) and the checkpoints, `history.json` and prediction CSVs of exp106-exp239 (their metrics stay in git). A `dvc pull` or `dvc checkout` that names them fails until the stage runs again. Run `dvc repro -s <stage>` for `seg_eval` or `seg_predict_ft`: a plain `dvc repro` rebuilds the cache and retrains the segmenter.
