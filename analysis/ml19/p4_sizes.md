# ML-19 P4: the canonical store's sizes (D29)

2364 photos in 20 folders; written by `analysis/ml19/p4_sizes.py`.

## By the original's format

| Group | Photos | Originals MB | Canonical MB | Ratio | Canonical median / max MB |
|---|---|---|---|---|---|
| .jpg | 2210 | 7,150 | 4,790 | 0.67 | 1.96 / 13.74 |
| .heic | 138 | 291 | 390 | 1.34 | 1.91 / 9.51 |
| .png | 9 | 10 | 2 | 0.19 | 0.25 / 0.26 |
| .dng | 7 | 274 | 75 | 0.27 | 7.48 / 25.16 |

## By the canonical photo's size

| Group | Photos | Originals MB | Canonical MB | Ratio | Canonical median / max MB |
|---|---|---|---|---|---|
| under 8 MP | 183 | 217 | 145 | 0.66 | 0.28 / 3.21 |
| 8-16 MP | 1683 | 4,468 | 3,510 | 0.79 | 1.87 / 5.88 |
| 16-30 MP | 408 | 2,126 | 1,133 | 0.53 | 2.73 / 6.81 |
| 45-60 MP | 90 | 915 | 468 | 0.51 | 4.30 / 25.16 |

## By folder

| Group | Photos | Originals MB | Canonical MB | Ratio | Canonical median / max MB |
|---|---|---|---|---|---|
| 2026-08-07__box_pictures_all_classes | 180 | 369 | 294 | 0.79 | 1.61 / 1.93 |
| 2026-08-09__pixel_cam | 180 | 395 | 318 | 0.81 | 1.75 / 2.18 |
| 2026-08-09__sony_cam | 180 | 918 | 451 | 0.49 | 2.37 / 3.23 |
| 2026-08-25__iphone | 91 | 201 | 272 | 1.35 | 1.92 / 5.60 |
| 2026-08-25__oneplus | 90 | 151 | 114 | 0.76 | 1.27 / 1.39 |
| 2026-08-27__oneplus_flash | 90 | 247 | 200 | 0.81 | 2.23 / 2.48 |
| 2026-08-30__oneplus | 40 | 81 | 62 | 0.77 | 1.55 / 1.92 |
| 2026-08-30__pixel | 46 | 109 | 89 | 0.81 | 1.94 / 2.29 |
| 2026-08-30__sony | 39 | 134 | 44 | 0.33 | 1.13 / 1.47 |
| 2026-09-11__oneplus | 20 | 45 | 36 | 0.80 | 1.77 / 2.25 |
| 2026-09-11__pixel | 20 | 55 | 46 | 0.83 | 2.35 / 2.58 |
| 2026-09-11__sony | 20 | 131 | 75 | 0.57 | 4.18 / 4.74 |
| 2026-09-24__oneplus | 69 | 166 | 133 | 0.80 | 1.96 / 2.78 |
| 2026-09-24__pixel | 70 | 197 | 162 | 0.82 | 2.28 / 2.86 |
| 2026-09-24__sony | 70 | 469 | 271 | 0.58 | 4.26 / 6.81 |
| random_date_raccoon | 183 | 576 | 406 | 0.70 | 1.90 / 6.73 |
| 2026-10-21__new_green | 41 | 147 | 104 | 0.71 | 2.53 / 6.17 |
| segmenter_positives | 311 | 1,371 | 823 | 0.60 | 1.94 / 25.16 |
| ood_negatives | 613 | 1,958 | 1,353 | 0.69 | 2.20 / 5.88 |
| ood_positives_internet | 11 | 4 | 3 | 0.79 | 0.30 / 0.51 |

## The largest canonical photos

- 25.16 MB, 48.6 MP: `dataset/segmenter_positives/20261027_135714_Pixel_9_Pro_Guatemala.dng` (original 41.28 MB)
- 17.17 MB, 50.1 MP: `dataset/segmenter_positives/20261027_135600_Pixel_9_Pro_Guatemala.dng` (original 51.90 MB)
- 13.74 MB, 49.9 MP: `dataset/segmenter_positives/20261028_112720_Galaxy_Z_Fold6_Brazil.jpg` (original 16.34 MB)
- 12.39 MB, 50.1 MP: `dataset/segmenter_positives/20261027_135628_Pixel_9_Pro_Guatemala.dng` (original 50.70 MB)
- 9.51 MB, 48.8 MP: `dataset/segmenter_positives/IMG_6761.HEIC` (original 6.29 MB)
- 8.30 MB, 48.8 MP: `dataset/segmenter_positives/IMG_6762.HEIC` (original 5.52 MB)
- 7.75 MB, 49.9 MP: `dataset/segmenter_positives/20261028_112737_Galaxy_Z_Fold6_Brazil.jpg` (original 10.74 MB)
- 7.74 MB, 50.0 MP: `dataset/segmenter_positives/20261103_150531_Galaxy_Z_Fold8_Ultra_Colombia.jpg` (original 18.34 MB)

In all: originals 7.54 GB, canonical photos 5.13 GB (0.68). The largest canonical photo is 25.16 MB.
