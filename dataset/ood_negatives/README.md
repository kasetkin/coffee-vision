# OOD negatives

Photos that visually resemble a bean-tray capture (or, for the internet-proxy batch, at
least share the "small round-ish objects on a surface" framing) but do **not** contain
coffee beans. Used only to evaluate OOD/content-guard candidate methods
(`coffeecv/ood_eval.py`) — never for training.

## Isolation from the training pipeline (do not change)

- Subfolders (`empty_tray/`, `confusable_grain/`, etc.) only organise the files: they are **not**
  class directories, and since ticket ML-8 (D9, 2026-10-19) not tags either: nothing reads them. Never rename anything under here to the `class_NNN__Label` pattern
  that `coffeecv/dataset.py`'s `discover_classes_multi`/`find_class_dir` look for.
- Never add any path under `dataset/ood_negatives/` to `params.yaml`'s `train_capture_dirs`,
  `coffeecv/dataset.py`'s `CAPTURES` or the `merge_segpool` stage's sessions in `dvc.yaml`. All are
  explicit lists (not globs), so this directory is invisible to the classifier's training by
  construction — keep it that way rather than wiring it in "for convenience." (The segmenter
  dataset does fine-tune the segmenter on these negatives, and the OOD probe fits on its
  training and validation ones; `labels/ml5/seg_dataset.yaml`.)
- These are raw photos, scored by `coffeecv/ood_eval.py` calling `patches_for_photo`/
  `classify_one` directly (the same live-crop code path a real upload takes) — they are
  never routed through the `data/cropped/` offline-crop tree.

## Batches

Each subdirectory is one capture/collection batch, DVC-tracked the same way as
`dataset/<session>/` (one `.dvc` pointer file per batch), with its own `manifest.csv`
(`filename, split, date, source_title, source_url, notes`).

**No scenario tags, no camera column** (ticket ML-8 D2, D9, 2026-10-19): negatives are negatives. The
segmenter dataset (`labels/ml5/seg_dataset.yaml`) splits newly added negatives within their batch, the OOD
probe (`coffeecv.fit_ood_probe`) fits on every training and validation negative, and `coffeecv.ood_eval`
reports by batch. The manifests' `scenario_tag` and `camera` columns were dropped then; their values are in git
history (`2026-10-18`'s ingest, `554a2a4`, is the last commit with them). A new negative goes into
`user_realworld/` (D5) with a manifest row and no tag or camera. Below, "tag" and "camera" describe what the
columns held.

- **`2026-09__internet_proxy/`** — 100 images downloaded from Wikimedia Commons (CC/public
  domain), one per row of `manifest.csv`, each visually vetted before inclusion. This is a
  **fast proxy pass, not the real hard-negative set**: images don't share the tray/
  background/lighting of an actual capture, so this is closer to a far-OOD test than the
  adversarial near-OOD test a real same-rig shoot would give. Treat results from this batch
  as directional screening, not a final comparison. `split` (`dev`/`holdout`, ~60/40 per
  scenario tag, seeded) was assigned before any scoring — holdout must stay untouched until
  a single final check, never iteratively re-tuned against.

  Subfolders (the former tags): `empty_tray`, `ground_coffee`, `confusable_grain`, `other_nuts_seeds`,
  `non_food_objects`, `wrong_bean_type`. Despite its name, `wrong_bean_type` holds three coffee-cherry photos
  (dried cherries on a tray, cherry husk, cascara), no other bean species; since ML-8 D9 the probe fits on them
  like any other negative (until then it left them out). `partial_beans` was planned but deliberately left
  empty — stock photos of "a few beans thinly scattered on a tray" don't really exist; that
  tag needs a real shoot. Green/unroasted coffee beans were collected and then **removed**:
  the training set already contains such photos as legitimate examples, so they are not
  negatives at all and would have scored as false failures.

- **`user_realworld/`** — general, random real-world photos from the user's own phones, no
  scenario subfolders (every row is tagged `real_world_negatives`). 479 photos, shot 2023–2026,
  most on a Pixel 9 Pro and a OnePlus 7 Pro (the rigs' phone models; `camera` is `pixel` or
  `oneplus`), some on a Redmi Note 14 Pro (`xiaomi`) and five of unknown origin (`unknown`). Private metadata (GPS and the rest of ML-3 D13's deny list) stripped with
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

  27 more were added 2026-10-18 (2026-05 to 2026-10): 13 Pixel 9 Pro, 9 Redmi Note 14 Pro and 5
  without any EXIF (`IMG_20260505_*`, re-encoded at 756x1008, likely received via a messenger), whose
  camera is `unknown` and whose `date` comes from the file name. Four (`PXL_20261017_1143*`-`1144*`)
  show ground coffee in a V60 filter, no whole beans: a near-OOD case, noted in their rows. Same rules
  as the 443, with seed 20261018: 16 dev, 11 holdout in 12 groups.

  One more on 2026-10-21: `IMG_2625.JPG`, a roll-up banner in a hallway, no beans, with no EXIF at
  all (1200x1600) and no date in its name, so its `date` is the day the file reached the workspace,
  2026-10-19. Nothing to strip. Split with seed 20261021: dev.

  **This batch's holdout no longer means "unseen by every model"** (ticket ML-6 D1, from
  2026-10-12). The segmenter dataset (`labels/ml5/seg_dataset.yaml`) ignores the manifest's
  `split` column (ML-5 D16) and splits all 479 photos 60/15/25 itself, so about 120 holdout
  photos are in the segmenter's training split. The check no model has seen is that file's
  test split; no live code reads this manifest's holdout.
