# Bean scale estimation — method comparison against manual ground truth

Measured 2026-08-11. The pipeline sizes patches as a fraction of the frame, which
only produces matched bean coverage because all three rigs happen to frame a
similar bean *count*. Nothing measures a bean at runtime, so a rig framed
differently would be mis-sized silently, and inference on an unknown camera has
no way to choose a crop size at all. This compares candidate estimators that
would remove that assumption.

## Ground truth

30 crops — 10 random photos per rig, centred, side = 0.40 × the photo's short
side — counted by hand. Ground truth is equivalent centre-to-centre spacing:

    spacing_px = crop_side_px / sqrt(n_beans_in_crop)

which is the quantity that decides how many beans land in a patch, and is
directly comparable to the FFT "period" used to characterise the rigs.

| rig | mean count | GT spacing | FFT value previously on record | error |
|---|---|---|---|---|
| old_box | 19.4 | **103.8 px** | 101.7 | −2% |
| pixel_cam | 26.6 | **238.7 px** | 208.9 | **−12.5%** |
| sony_cam | 22.2 | **323.1 px** | 323.6 | +0.2% |

**This corrected a figure used throughout Phase 11.** pixel_cam is 2.30× the old
rig, not 2.05×, and sony is 1.35× pixel_cam, not 1.55×. "Beans across the frame"
is 10.7 / 12.5 / 11.4 — tighter across rigs than the 10.9–14.3 previously quoted,
so fraction-of-frame sizing rests on firmer ground than assumed, though it still
rests on an unchecked assumption.

Counting error is roughly ±2 beans on ~20, i.e. ~5% on spacing. Within-rig GT
spread is only 4–7% CV, comparable to that, so **within-rig correlations in the
benchmark are noise against noise and carry no information**.

## Methods

| id | method | idea |
|---|---|---|
| M0 | FFT radial profile | incumbent; dominant period of the power spectrum |
| M1 | distance transform | Otsu mask → EDT → median local-maximum radius |
| M2 | granulometry | binary morphological opening spectrum; peak radius |
| M3 | autocorrelation | first zero crossing of the radial autocorrelation |
| M4 | MobileSAM | segment-everything → median mask equivalent diameter |

## Results

Raw estimate, with (ratio to ground truth):

| method | old_box | pixel_cam | sony_cam | bias spread | ms |
|---|---|---|---|---|---|
| M1 distance transform | 92 (0.89) | 161 (0.68) | 220 (0.68) | 1.32 | 26 |
| M2 granulometry | 94 (0.90) | 162 (0.68) | 235 (0.73) | 1.32 | 725 |
| M3 autocorrelation | 104 (1.00) | 292 (1.23) | 339 (1.05) | 1.22 | 54 |
| M4 MobileSAM | 79 (0.76) | 179 (0.75) | 247 (0.77) | **1.02** | 97026 |
| M0 FFT (incumbent) | 87 (0.84) | 199 (0.83) | 280 (0.87) | **1.04** | 48 |

After each method's single global constant k:

| method | k | per-image MAPE | session MAPE | worst rig | within ±40% |
|---|---|---|---|---|---|
| M1 | 1.33 | 16.5% | 9.3% | 11.6% | 93% |
| M2 | 1.30 | 19.5% | 16.4% | 22.5% | 97% |
| M3 | 0.92 | 26.9% | 14.8% | 19.5% | 83% |
| M4 | 1.32 | **11.1%** | **1.3%** | **2.4%** | 100% |
| M0 | 1.18 | 19.2% | 6.3% | 13.4% | 90% |

## What it says

**Three of the four candidates are worse than the estimator already in use.** M1
and M2 have the worst bias consistency (1.32); M3 is worst overall. Only M4 beats
the incumbent FFT, and only 1.02 vs 1.04.

**Every method needs k ≈ 1.3.** They measure inscribed radius or short axis, not
centre-to-centre spacing, and beans are elongated. That bias is structural, not a
bug.

**Session-level estimation beats per-image by 2–8×** (median over 10 photos):
M0 goes 19.2% → 6.3%, M4 goes 11.1% → 1.3%.

## The constraint that decides the design

The same estimator must run at **training and inference**. A two-tier scheme —
accurate calibration during training, cheap estimation live — would inflate every
reported metric, because the model would be trained and evaluated on well-sized
patches and then meet worse-sized ones in the field. Measured quality would be
real for the lab and fiction for the user.

Corollary: with one shared estimator, **absolute bias becomes irrelevant** — a
constant k redefines what "patch scale" means, identically on both sides. What
still matters is cross-rig consistency (bias spread) and per-image variance.

By that criterion the ranking is M4 (spread 1.02, CV 0.13), then M0 (1.04, 0.24),
then M3, M1, M2. M4's 97 s/image is the open problem; see whether reducing
`points_per_side` retains its accuracy.

## Reproducing

    python analysis/bean_scale/make_crops.py    # writes 30 crops for counting
    python analysis/bean_scale/benchmark.py     # ~50 min, MobileSAM dominates

`ground_truth.json` holds the manual counts and is the reusable artifact — the
counting does not need repeating unless the rigs change.

---

## Band sweep, 2026-09-03 — the pinning hypothesis is refuted

`band_sweep.py` sweeps `_K_LO` against these same 30 crops through the **production**
`estimate_bean_pitch_k`, not through `estimators.py::m0_fft_radial` (which is a frozen
private copy hardcoding `lo, hi = 4, 80`, so a sweep driven through it reports a null
no matter what the band does — that is why this is a separate script).

The hypothesis under test was: the argmax is pinned at the band's low edge on a large
fraction of photos, so the true period lies *past* that edge and is being clipped;
**lowering** `_K_LO` should therefore reduce pinning and improve accuracy.

It does the opposite.

| `_K_LO` | MAPE | MAPE (refit K) | refit K | pinned | rig-bias spread | corr(pred, GT) |
|---|---|---|---|---|---|---|
| 2 | 30.9% | 29.6% | 1.131 | 0% | 0.195 | 0.699 |
| **4 (shipped)** | **19.2%** | **20.0%** | **1.242** | **23%** | **0.041** | **0.838** |
| 5 | 14.7% | 15.3% | 1.348 | 33% | 0.041 | 0.877 |
| 6 | 17.1% | **9.7%** | 1.449 | 50% | **0.029** | **0.929** |

Every rig improves monotonically as the floor *rises* (old_box 23.5→7.4%, pixel_cam
42.4→9.0%, sony_cam 23.0→12.8% at refitted calibration), rig-bias spread narrows, and
correlation against ground truth improves. So the low-frequency floor is **suppressing
1/f spectral drag, not clipping bean signal** — at low `k` the `k^1.5` weighting still
loses to the spectrum's own falloff and the argmax runs to spurious long periods.
"Pinned at the floor" is the guard working, not a defect.

Checked for the obvious failure mode: a floor high enough to make `k` nearly constant
would degenerate into frame-fraction sizing, scoring well here only because these three
rigs happen to frame similar bean counts. It is not degenerating — correlation *rises*
with `_K_LO` and the rigs stay separated (means 86/202/263 px against GT 104/239/323 at
`_K_LO=6`), which is the opposite of what a collapse to a constant would show.

**Not adopted, and not a config change yet.** Three reasons:

1. Ground truth here is **3 rigs** (old_box, pixel_cam, sony_cam, 2026-08-11). It predates
   iPhone, oneplus and the 08-30 sessions entirely, so it cannot speak to the five newer
   rigs — and the rig-dependence of pinning is the whole reason this mattered.
2. The gain needs `CALIBRATION_K` refitted from 1.18 to ~1.45. That is two shipped
   constants changing together, and calibration and band are not separable.
3. Counting error is ~5% on spacing, so ~5% MAPE is this benchmark's floor; 9.7% is
   meaningfully above it but the margin should not be oversold.

Next step is extending hand-counted ground truth to the newer rigs, then validating
through `run_folds.py`'s cross-rig metric — per this project's rule that a config change
is validated on folds, never on an offline benchmark alone.
