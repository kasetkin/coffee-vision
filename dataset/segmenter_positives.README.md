# Segmenter positives

Positives the owner chose for the segmenter because their setup differs from the pools: framing,
container, distance (GLOSSARY.md, "Segmenter positive"). Ticket ML-5 (D1) repurposed and renamed this folder
on 2026-10-08; it was `dataset/ood_positives/`, the OOD guard's "never seen" positives, until then.
**There is no "never seen" claim any more**: 71 of the 132 photos are byte copies of photos in other
`dataset/` folders, 51 of them pool photos.

Used by the segmenter dataset (ticket ML-5 P5, `labels/ml5/`): every photo here is in it, split
train/validation/test within this source (D13), a copy in one group with its original. Nothing here trains
the classifier; a copy of a pool photo reaches it only as the original. The OOD guard no longer calibrates on
these photos (D2; the guard's code follows in ML-5 P3).

## Manifest

`segmenter_positives.manifest.csv`, one row per photo, sorted by file name:

| column | what |
|---|---|
| `filename` | the file in this folder |
| `camera` | `pixel`, `sony`, `oneplus` and `iphone` are the rigs; `iphone_15_pro_max` and `oneplus_kb2005` are other phones (EXIF make and model) |
| `date` | capture date (EXIF, local time) |
| `batch` | the date the photo arrived here; the older batch names are in the history below |
| `duplicate_of` | for a copy, the repo path of the photo it is a byte copy of (equal SHA-256); empty otherwise |
| `notes` | what the photo shows, where it is unusual |

The guard-era `split` (dev/holdout) and `scenario_tag` columns were dropped by ML-5 P2: the segmenter dataset
assigns its own split (D13), and the tags only meant something for the guard's calibration. The old values
stay in git history (`dataset/ood_positives.manifest.csv` before ML-5 P2), and ML-2's frozen
`labels/ml2/photo_lists.yaml` still lists the dev photos under their old paths.

## 2026-10-10: 4 Pixel photos

- **New photos** (batch `2026-10-10`): `PXL_20261008_074853981.jpg`, `PXL_20261008_074902440.jpg`,
  `PXL_20261008_074913048.jpg` and `PXL_20261010_041340477.jpg`, Pixel 9 Pro; the last is a byte copy of the
  `random_date_raccoon` pool photo of the same name, added the same day. Not yet in the segmenter dataset
  (`labels/ml5/seg_dataset.yaml`).
- **Metadata** (ticket ML-3, D13): stripped with `coffeecv.strip_metadata`; rows in `labels/ml3/strip_manifest.csv`.

## 2026-10-08: 70 copies, the rename

- **New photos** (batch `2026-10-08`): 70 photos the owner copied in from the pools and from
  `2026-07-24__first_pictures` and `2026-08-06__box_pictures`. Each is a
  byte copy of the photo its `duplicate_of` names (checked by SHA-256 against every DVC listing under
  `dataset/`; each matches exactly one photo). Per source folder: 22 `random_date_raccoon`, 17
  `2026-07-24__first_pictures`, 6 `2026-09-24__sony`, 4 each `2026-08-09__pixel_cam`, `2026-08-25__iphone` and
  `2026-09-24__oneplus`, 3 each `2026-08-06__box_pictures`, `2026-08-30__oneplus` and `2026-08-30__sony`, 2 each
  `2026-08-09__sony_cam` and `2026-08-27__oneplus_flash`.
- **Metadata** (ticket ML-3, D13): stripped with `coffeecv.strip_metadata`; rows in `labels/ml3/strip_manifest.csv`.
- **Committed** to DVC in ML-5 P1 (`62a4d3e`), then renamed with `dvc mv` in P2; the folder's `.dir` hash did
  not change.
- **Batch names** renamed to arrival dates: `ood_positives` became `2026-09-10`, `ood_positives_2026-09-11`
  became `2026-09-11`, and so on for 09-30, 10-05 and 10-06.

# History: the folder as the OOD guard's positives (to 2026-10-08)

The sections below are the README as it stood before ML-5, kept as the record of where the first 58 photos
came from. Paths and batch names in them are the old ones.

Genuine coffee-bean photos the shipped model has **never seen**. These are the population a
content guard must *not* refuse, and they exist because the checkpoint's own held-out split
turned out to be far too easy to stand in for them: photos from sessions the model trained
on score a median 1.02 on the shipped metric, while these score 1.21 and reach 1.57 — well
into the range the negatives occupy.

Used for evaluating guard metrics (`coffeecv/ood_eval.py`). Since 2026-09-30 (ticket ML-2, D26) the **dev** photos also train and evaluate the bean segmenter (`labels/ml2/photo_lists.yaml`, `pos_seg_*`); a dev photo from a burst that also holds a holdout photo is segmenter-eval only. The **holdout** is never used for any training, and the segmenter never reads it. Nothing here trains the classifier.

## Batches

58 photos, one flat directory since 2026-09-30, all in the manifest. The
manifest's `batch` column records which
collection each photo arrived in, because the conditions measured on them were defined per batch:

| `batch` | photos | what |
|---|---|---|
| `ood_positives` | 14 | the original set, below (2026-09-10) |
| `ood_positives_2026-09-11` | 30 | same-rig positives shot on 2026-09-11 with the Sony, Pixel and OnePlus rigs, the positive side of the `2026-09-11__user_samerig` negatives; was its own directory (`dataset/ood_positives_2026-09-11/`, files under `user_beans_independent/`) until the merge |
| `ood_positives_2026-09-30` | 10 | Pixel photos shot 2026-09-11..21, added at the merge |
| `ood_positives_2026-10-05` | 3 | Pixel photos shot 2026-10-05, all holdout (below) |
| `ood_positives_2026-10-06` | 1 | Pixel photo shot 2026-10-06, holdout (below) |

## Provenance and checks (2026-09-10)

- **Hash-checked against the whole training tree**: SHA-256 of every file compared against
  all 938 raw photos under `dataset/*/class_*__*`, plus a capture-timestamp comparison on the
  `PXL_<date>_<ms>` stems (millisecond precision, so a re-export of the same shot would
  collide even though its bytes differ). Zero matches on either — all genuinely new.
- **EXIF stripped in place.** Every file carried GPS coordinates. Stripped by rebuilding each
  image from raw pixel data, so nothing survives rather than only the tags we thought to
  delete. Orientation was 1 throughout, so no rotation was needed.
- Contents verified by eye: all genuinely show beans (one of green/unroasted beans in a
  container, the rest tight macro shots of roasted beans).

## The two tags are not interchangeable

- **`user_beans_independent`** (7 photos in the original set, 51 in all) — shot on days with no
  training session at all (2026-08-23, -24, -29, -31, 2026-09-03; the later batches on
  2026-09-11..21). These are the trustworthy ones.
- **`user_beans_same_day`** (7 photos) — six shot on 2026-08-09 *35–55 minutes before* the
  `2026-08-09__pixel_cam` session began, one 5.5 h before `2026-08-30__pixel`. Same beans,
  room and lighting almost certainly. Byte-distinct from training data but **not independent
  of it**, and they score visibly lower (median 1.12 vs 1.27) — which is the point: pooling
  them with the independent photos would flatter every metric measured here.

`split` (`dev`/`holdout`) was assigned deliberately rather than randomly. With only seven
independent-day photos, a random draw could have put them all on one side and left the other
with nothing load-bearing in it.

## 2026-09-30: merge, ten new photos, metadata pass

- **Merge.** The 2026-09-11 batch moved in from its own directory; its manifest rows came
  across unchanged apart from the flattened `filename`. Its DVC entry and manifest were removed.
- **New photos** (`ood_positives_2026-09-30`): all `user_beans_independent`, Pixel, shot on days
  with no training session. Checked by SHA-256 and by capture-timestamp stem against all 1,234
  other photos under `dataset/` (zero matches) and by eye: six top-down roasted, two green
  (unroasted), two wide shots (grinder hopper, open bag). Split 6 dev / 4 holdout, with photos
  shot within minutes of each other kept on the same side, so a near-duplicate pair never
  straddles dev and holdout: dev = 09-11, 09-15, the 09-17 green pair, the 09-21 06:41 pair;
  holdout = 09-12, 09-14, the 09-21 07:12 pair.
- **Six of the original fourteen** (the 2026-08-23..09-03 independent-day photos) were replaced
  by fresh copies from the phone during the merge. Same shots, but a different JPEG encoding than
  the 2026-09-10 files, so pixels differ slightly (mean |Δ| ≈ 0.4 of 255) from what earlier guard
  evaluations measured. The earlier bytes remain in the DVC cache under the previous
  `ood_positives.dvc` hash.
- **Metadata stripped in place, lossless** (JPEG segments edited, image data untouched; every
  file decodes to identical pixels, and the Ultra HDR gain maps are byte-identical and still
  reachable through MPF). Removed: GPS (16 files), C2PA content credentials (36 Pixel/OnePlus
  files: the signing certificate carries a device-stable ID shared by every photo), Google's
  opaque `HdrPlusMakernote` extended XMP (36), the Sony MakerNote (10) and EXIF thumbnails. Kept:
  Orientation (the Sony files need it), make/model, capture time, exposure tags, ICC.

## 2026-10-05: three new photos, metadata pass over everything

- **New photos** (`ood_positives_2026-10-05`): `PXL_20261005_054730718.jpg`, `PXL_20261005_054735253.jpg`
  (a near-duplicate pair 5 s apart: roasted beans in a grinder hopper) and `PXL_20261005_073955312.jpg`
  (top-down close-up, roasted). Pixel 9 Pro, shot 2026-10-05, a day with no training session, so
  `user_beans_independent` by the rule above. Checked by SHA-256 and capture-timestamp stem against all
  1,585 other photos under `dataset/` and `dataset_new_ignored/` (zero matches) and by eye. All three
  are **holdout** (owner, 2026-10-05): the holdout was spent by the 2026-10-02 check (ticket ML-3 R5) and
  needs fresh photos, and the near-duplicate pair stays on one side. Holdout photos are in no segmenter
  list (`labels/ml2/photo_lists.yaml` lists dev only).
- **Metadata** (ticket ML-3, D13): every photo here went through `coffeecv.strip_metadata`; the three new
  ones lost GPS, C2PA, the HDR+ maker note and the EXIF thumbnail, the 54 others were already clean. Rows in
  `labels/ml3/strip_manifest.csv`.

## 2026-10-06: one new photo

- **New photo** (`ood_positives_2026-10-06`): `PXL_20261006_053625750.jpg`, roasted beans in a grinder
  hopper, top-down (the setting of the 2026-10-05 pair). Pixel 9 Pro, shot 2026-10-06 10:36 local. Four
  training photos (`random_date_raccoon/class_005__Guatemala_Tata`) were shot about an hour later the same
  morning, so the rule above would make it `user_beans_same_day`; it is **`user_beans_independent`** by
  the owner's call (2026-10-06). **Holdout** (owner). Checked by SHA-256 and capture-timestamp stem against
  all 1,583 other photos under `dataset/` and `dataset_new_ignored/` (zero matches) and by eye.
- **Metadata** (ticket ML-3, D13): stripped with `coffeecv.strip_metadata` (GPS, C2PA, the HDR+ maker note,
  the EXIF thumbnail); row in `labels/ml3/strip_manifest.csv`.
- The `user_beans_independent` count above read 47 after the 2026-10-05 batch; the manifest had 50. It now
  reads 51.

## Sibling batch

`dataset/ood_positives_internet/` holds 12 Wikimedia-sourced photos of coffee beans under the
`internet_beans` tag — also genuine positives, but a *different* population (studio lighting,
varied roasts and framing), so they are a separate directory and a separate condition rather
than being mixed in here. Sparse arrangements (a handful of beans on a white surface) and
product/scene shots were rejected during vetting: the question is whether a guard refuses a
photo *of beans*, and a photo of four beans on a table is a different question.
