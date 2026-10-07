# DVC: moving data and trusting what it says

Read before moving data between machines, pushing to the remote, or committing after a DVC command. The setup (remote, `.dvc/config.local` on the VM) is in README.md, "Dataset & DVC".

## Moving data

- **Use `dvc push` here and `dvc pull` (or `dvc fetch`) on the VM.** Both verify hashes. Copying `.dvc/cache` objects with rsync or scp is retired: an interrupted `rsync --partial --ignore-existing` once left a truncated object under its final name, the retry skipped it, and a VM run failed with "image file is truncated".
- **Push explicit targets** (`dvc push models/<name>.pt.dvc`) and verify the objects. `dvc push --all-commits` has reported "Everything is up to date" while an object it collected was missing from the remote.
- `--all-commits` reads `.dvc/config` as of each commit. A remote declared in `.dvc/config.local` without a `url` makes every older commit fail with `ConfigError: expected 'url'`, logged as a WARNING with exit 0 (323 commits silently skipped once). Keep a local override self-sufficient: `dvc remote modify --local <remote> url ...`.

## Trusting the result

DVC's last line is not the verdict. `dvc status --cloud` has printed "Cache and remote are in sync" after three warnings that objects were missing, and exited 0. Read the warnings; to verify, expand every `.dir` in `dvc.lock` and test each object:

```python
import yaml, json, pathlib
cache = pathlib.Path('.dvc/cache/files/md5')   # on the VM: the store's files/md5
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

Only `outputs/metrics.json` and `outputs/summary.json` may come back missing: they are `cache: false` and live in git.

`dvc repro --dry` has said "cached, skipping" for a stage `dvc status` correctly reported as changed. Check `dvc status`, and after a run check that `outputs/config.json` holds the intended values.

## Keeping git and DVC apart

- Every DVC-tracked output stays gitignored (DVC writes a sibling `.gitignore`; if one goes missing, `git add -A` commits the data). Read what `git add -A` stages after DVC has touched the tree, and never `git add` a `dataset/` directory by name: that overrides `.gitignore`, and photos once landed in git history that way, permanently.
- `dvc exp run` can leave HEAD detached, and commits made afterwards land off-branch with no warning. Run `git rev-parse --abbrev-ref HEAD` after any `dvc exp` command; `HEAD` means check out the branch first. Recorded runs go through `dvc repro` (or the drivers that call it), not `dvc exp`.
- If a `.dir` hash differs between machines, compare the two listings entry by entry before assuming the data drifted; re-running the stage on one machine only mints a third hash. Precedent: an absolute path in a crop report (`5d0e9fd`).

## Stages that live on the VM

`data/seg_cache` (stage `seg_embed_cache`) exists only on the VM. Locally, run `dvc repro -s <stage>` for `seg_eval` or `seg_predict_ft`: a plain `dvc repro` rebuilds the cache and retrains the segmenter here.
