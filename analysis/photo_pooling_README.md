# Photo-level pooling, measured at full-rig n — B3′, 2026-09-03

Phase 14 measured what pooling a photo's patches into one prediction buys, at **n=27 photos/fold**,
and said so honestly: "±3.7 points per photo; directional, not precise." One fold (box) came out at
−0.035, and the plan's §3h pointed out that this regression is **a single photo** — not enough to
justify engineering a confidence-weighted or outlier-rejecting aggregator against it. B3′ was
therefore "raise n first, then reconsider pooling".

Raised. `coffeecv/photo_pooling_eval.py` scores **every photo in each held-out rig** using that
fold's own archived checkpoint (fetched from DVC history), 40 patches per photo to match Phase 14,
pooling by mean of the per-patch probability vectors — the same thing `infer.py` does at inference.
Patch and photo accuracy are computed over identical patches, so the delta isolates pooling.

| exp | held-out rig | n | patch | photo | **delta** | Phase 14 (n=27) |
|---|---|---|---|---|---|---|
| 163 | box | 180 | 0.782 | 0.894 | **+0.113** | −0.035 |
| 164 | pixel_cam | 180 | 0.776 | 0.883 | **+0.107** | +0.079 |
| 165 | sony_cam | 180 | 0.729 | 0.806 | **+0.077** | +0.015 |
| 166 | oneplus | 90 | 0.855 | 0.922 | **+0.067** | — |
| 167 | iphone | 92 | 0.784 | 0.880 | **+0.097** | — |
| | **mean** | **722** | | | **+0.092** | +0.020 |

**5/5 folds positive, mean +0.092 over 722 photos** against Phase 14's +0.020 over 81.

**The box regression was noise.** At n=27 it read −0.035; at n=180 the same fold is **+0.113**, the
*largest* gain in the set. Nothing about the model changed — only the sample size. Phase 14's own
caveat was right, and building an aggregator to chase that −0.035 would have been tuning against a
single photo.

**So B3′ answers itself: do not build a cleverer aggregator.** Plain mean pooling is worth ~9 points
of photo-level accuracy, roughly 4.5× what Phase 14 could resolve, and it is uniformly positive
across all five rigs including the two phone rigs Phase 14 never covered. The remaining question is
not "can we pool better" but whether ~0.09 is already close to the ceiling that correlated
within-photo patch errors allow — Phase 14's mechanism argument (same beans, same rig, same lighting,
so 40 patches are not 40 independent votes) still stands and still caps how much pooling can buy.

Caveats: single seed per fold (these are the archived exp163-167 checkpoints, all seed 42); dihedral
TTA off, so these are not the shipped inference configuration; and exp166's rig is the *unmerged*
`2026-08-25__oneplus` (90 photos), since that is what that fold held out.

Reproduce: `python -m coffeecv.photo_pooling_eval --exp-id 163 164 165 166 167 --n-patches 40 --no-dihedral-tta`
