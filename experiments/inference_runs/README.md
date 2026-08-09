# Inference runs on unlabeled photos

Output of `coffeecv/infer.py` against `dataset/2026-08-06__box_pictures/` — the unlabeled
box-rig session whose ground truth only the user knows. Rescued from `outputs/inference/`,
which is gitignored and would have been lost on any `outputs/` clean.

**Provenance gap, read before trusting these**: each file records `crop_size`,
`patch_resize` and `n_patches_per_image`, but **not which checkpoint produced it** — no
model hash, no git commit, no config. They date from the 2026-08-06 / Phase 6 era, so
they were produced by a model trained on the old single-photo-per-class dataset at
`patch_crop_size=700` or smaller, i.e. two adopted-config generations behind current.
Treat them as a record that this analysis happened and roughly what it said, not as
numbers comparable to anything current.

Worth fixing if inference is re-run: `infer.py` should write the same `env` block
`train_baseline.py` already writes into `outputs/config.json` (torch version, git commit)
plus the checkpoint's hash, so a future reader can tell which model spoke.
