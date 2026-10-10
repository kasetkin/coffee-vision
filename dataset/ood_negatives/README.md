# OOD negatives

Photos that visually resemble a bean-tray capture (or, for the internet-proxy batch, at
least share the "small round-ish objects on a surface" framing) but do **not** contain
coffee beans. Used only to evaluate OOD/content-guard candidate methods
(`coffeecv/ood_eval.py`) — never for training.

## Isolation from the training pipeline (do not change)

- Scenario subfolders (`empty_tray/`, `confusable_grain/`, etc.) are free-text tags, **not**
  class directories. Never rename anything under here to the `class_NNN__Label` pattern
  that `coffeecv/dataset.py`'s `discover_classes_multi`/`find_class_dir` look for.
- Never add any path under `dataset/ood_negatives/` to `params.yaml`'s `train_rigs` or
  `heldout_rig`, or to `coffeecv/run_folds.py`'s `RIGS` list. Both are explicit, hardcoded
  lists (not globs), so this directory is invisible to training by construction — keep it
  that way rather than wiring it in "for convenience."
- These are raw photos, scored by `coffeecv/ood_eval.py` calling `patches_for_photo`/
  `classify_one` directly (the same live-crop code path a real upload takes) — they are
  never routed through the `data/cropped/` offline-crop tree.

## Batches

Each subdirectory is one capture/collection batch, DVC-tracked the same way as
`dataset/<session>/` (one `.dvc` pointer file per batch), with its own `manifest.csv`
(`filename, scenario_tag, camera, split, date, source_title, source_url, notes`).

- **`2026-09__internet_proxy/`** — 100 images downloaded from Wikimedia Commons (CC/public
  domain), one per row of `manifest.csv`, each visually vetted before inclusion. This is a
  **fast proxy pass, not the real hard-negative set**: images don't share the tray/
  background/lighting of an actual capture, so this is closer to a far-OOD test than the
  adversarial near-OOD test a real same-rig shoot would give. Treat results from this batch
  as directional screening, not a final comparison. `split` (`dev`/`holdout`, ~60/40 per
  scenario tag, seeded) was assigned before any scoring — holdout must stay untouched until
  a single final check, never iteratively re-tuned against.

  Scenario tags: `empty_tray`, `ground_coffee`, `confusable_grain`, `other_nuts_seeds`,
  `non_food_objects`, `wrong_bean_type`. `partial_beans` was planned but deliberately left
  empty — stock photos of "a few beans thinly scattered on a tray" don't really exist; that
  tag needs a real shoot. Green/unroasted coffee beans were collected and then **removed**:
  the training set already contains such photos as legitimate examples, so they are not
  negatives at all and would have scored as false failures.

- **`user_realworld/`** — general, random real-world photos from the user's own phones, no
  scenario subfolders (every row is tagged `real_world_negatives`). 451 photos, shot 2023–2026
  on a Pixel 9 Pro and a OnePlus 7 Pro (the rigs' phone models; `camera` is `pixel` or
  `oneplus`). Private metadata (GPS and the rest of ML-3 D13's deny list) stripped with
  `coffeecv.strip_metadata`; rows in `labels/ml3/strip_manifest.csv`.

  The first 8 (2026-09-10, Pixel, EXIF stripped that day) were the batch
  `2026-09__user_realworld/real_world_negatives/`, renamed by the owner on 2026-10-10 to
  `20260910_195601.jpg` … `20260910_195602-5.jpg` (`labels/photo_renames.csv` maps each old
  path; `coffeecv.rename_photos` re-pointed the label records). Outdoor/travel photos: rock,
  scree, lichen, glacier, a marmot, a yurt in a valley. **Three of the eight were
  misclassified by the guard shipped then**: it accepted them and named a bean origin, on
  photos containing no beans at all. Measured 2026-09-10 against `models/allrigs_cam_s123.pt`;
  each of those rows' `notes` records whether that photo was a `baseline_false_accept` or
  `baseline_refused`. All three failures are rocky/scree textures, a coherent failure mode
  rather than a fluke: small mid-brown fragments at roughly bean scale is exactly what the
  embedding space was trained to find interesting. Their dev/holdout split was assigned by
  hand, not randomly: with only three confirmed failures, a random split could have put all
  three on one side.

  The other 443 (added 2026-10-10) have a seeded split (seed 20261010, ~60/40 dev/holdout by
  photo), assigned before any scoring. Shots on one camera within 120 s of each other
  (`seg_lists.NEAR_DUP_SECONDS`) form one group and land on the same side. Their `date` is
  EXIF DateTimeOriginal's.
