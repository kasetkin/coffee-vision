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

Each dated subdirectory is one capture/collection batch, DVC-tracked the same way as
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

- **`2026-09__user_realworld/`** — 8 photos from the user's own phone (Pixel), EXIF stripped
  on 2026-09-10 (they carried GPS coordinates). Outdoor/travel photos: rock, scree, lichen,
  glacier, a marmot, a yurt in a valley. Nothing resembling a bean tray — and that is the
  point. **Three of the eight were already misclassified by the shipped guard**: it accepted
  them and named a bean origin, on photos containing no beans at all. Measured
  2026-09-10 against `models/allrigs_cam_s123.pt`; each row's `notes` records whether that
  photo was a `baseline_false_accept` or `baseline_refused`.

  This is the most valuable batch here, and the only one that is evidence rather than
  hypothesis: the internet batch is a guess about what *might* fool the guard, while these
  are photos that demonstrably *did*. All three failures are rocky/scree textures, which is
  a coherent failure mode rather than a fluke — small mid-brown fragments at roughly bean
  scale is exactly what the embedding space was trained to find interesting.

  The dev/holdout split here is assigned deliberately, not randomly: with only three
  confirmed failures, a random split could have put all three on one side and left the
  other with no genuinely hard case in it.
