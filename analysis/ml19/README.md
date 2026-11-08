# ML-19: measurements behind canonicalization's first decisions

Measured 2026-11-03 on the local checkout at `840bb87`, for [ML-19](../../docs/ticket_canonicalization.html). Scratch
inputs lived in `/tmp/ml16/q13/`; the scripts here are as run.

## The dataset's colour profiles, orientation and second images

`exiftool -r -fast2` over `dataset/` (2,463 photos: 2,288 JPEG, 148 HEIC, 20 PNG, 7 DNG):

- ICC profile: Display P3 951 JPEG + 148 HEIC, DCI-P3 (sRGB transfer) 50, Display P3 (sRGB transfer) 9; sRGB 487;
  Adobe RGB 7; other 14; none 797 (of these, EXIF ColorSpace sRGB 657, "Uncalibrated" 1, nothing 139).
- EXIF Orientation: normal 1,593; Rotate 90 CW 361; Rotate 180 281; Rotate 270 CW 20; Unknown (0) 35; none 173.
- A second image in the file (MPF image 2, HDR gain map or HEIF auxiliary image): 1,518 photos.

`coffeecv/photo_io.py:load_rgb_image` ignores all three: `Image.open(path).convert("RGB")`.

## Pixels outside sRGB (`gamut.py`)

The 1,152 readable Display-P3-tagged photos, decoded at about 800 px, linearised and taken from P3 (D65) to linear
sRGB by matrix: 0.86% of pixels fall outside sRGB, 0.29% by more than 2%. 120 photos have more than 1% outside, 18
more than 10%: mostly `user_realworld` negatives and two label cards; the highest bean photos are
`2026-09-24__pixel/.../PXL_20260924_054456514.jpg` (57.8%) and two segmenter positives (12.0%, 11.7%).

## Downscaling the photos over 50 MP (`spectrum.py`, `run_infer.py`)

The 7 photos over 50 MP (six 200 MP, one 84 MP, all Galaxy Z Fold8 Ultra, 19-33 MB) downscaled with Lanczos to at
most 50 MP.

- Power above the 50 MP Nyquist, in three 2048 px tiles each: 0.005% to 0.49%, 21 tiles of 21 under 0.5%.
- The live models (`dino3b16_segxl0v6_country_aug8_s42`, `xl0_v6_s42`, PyTorch, CPU), original against
  downscaled (`downscale_answers.json`): the same verdict on all 7 (5 predicted, 2 refused), the same top-1, its
  probability within 0.012; mask area the same to 3 decimals; bean pitch halved exactly with the scale; the slowest
  photo 34.3 s against 16.7 s.
