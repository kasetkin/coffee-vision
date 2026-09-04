# Dataset & training reorganization plan

Status: REVISED twice on 2026-09-02 — an adversarial re-verification pass (second review), then a
third pass that actually reached the `powervpsssh` remote box over SSH (its git log, DVC status, and
plain files) instead of inferring remote state from local evidence alone. The third pass changed §3d's
cost estimate materially; see the note at its end. A fourth pass on 2026-09-03 audited the plan for
implementation-readiness, rewrote C1 after finding its "cheap test" does not work, and **removed the
scale guard from this plan entirely** — see §3a and §0.
Scope: `dataset/`, `data/cropped/`, `coffeecv/` training + inference pipeline. Excludes
`webapp/` deploy/logging (handled in other recent work) — the remote-state check for this pass did
look at that excluded area (the box is shared), found nothing alarming, and the one real gap it did
find is noted at the end of §3d for completeness rather than folded into this plan's recommendations.

Every number below was re-derived from this working tree on 2026-09-02 by running code against the
data, not by reading prior notes or the previous draft. Claims that survived are marked; claims that
did not are corrected in place with the actual value. See "Verification record" at the end.

---

## 0. What changed in this revision

The first draft's dataset arithmetic was almost entirely correct — the §2 coverage table reproduces
cell-for-cell, and so do the MixStyle (+0.1400, 9/9), pooling (−0.035 box / +0.020 mean), TTA
(+0.0235), val-saturation and DVC-remote claims. Its analysis, however, had three real errors and
missed six issues that are more urgent than most of what it recommended:

**Errors corrected**

- **The iPhone shortfall is not "88 photos of class_008."** 88 is the *total* deficit across eight
  classes; class_008's share is 20. The old A1 would have sent you to shoot 88 photos of one bean.
  (§2c, A1)
- **The headline in-distribution range was inflated.** 0.89–0.95 mixed in 2-rig-era runs. At the
  current adopted config it is 0.884–0.933. (§1)
- **H1 was called a "confirmed finding."** It is a 1-seed arm compared against a 3-seed mean, with
  a matched-control set that reads null. Suggestive, not confirmed by this project's own standard. (§2a)

**Issues the first draft missed**

- The framing-distance scale guard is not "broken one way" — **both** of its branches are
  mathematically unreachable and it is a total no-op. (§3a)
- The bean-pitch estimator is **pinned at its search-band floor on 39% of photos**, rig-dependently.
  This is new, it is measurable today, and it attacks the same cross-rig gap the whole plan is
  about. (§3b)
- `experiments/index.csv` has **two duplicate rows** and no longer matches what `rebuild_index()`
  produces. (§3c)
- **Eight archived experiments cite git commits that do not exist in this repository** — including
  exp171, the run behind the currently shipped model. (§3d)
- **exp174 is a void run** that silently trained 9 classes, not 10, and is not marked as such. (§3e)
- The per-epoch and final cross-rig macro-F1 are computed on **different denominators**. (§3f)

**Third pass — actually reached the remote box**

The second draft inferred remote state from local evidence only (`argv` paths in archived configs).
This pass SSH'd into `powervpsssh` directly and changed one conclusion materially:

- **P1's premise holds but its cost estimate does not.** The 8 commits genuinely exist and are
  fetchable from the remote box — confirmed by adding it as a real git remote and fetching. But local
  and remote have **mutually diverged**, not "remote is 8 commits ahead." Fetching is the easy part;
  reconciling two independently-advanced histories onto one line is not a 15-minute task. (§3d)
- §3i (no DVC remote) is now confirmed on **both** machines, and the remote's own DVC cache is missing
  several historical artifacts that local's cache is the sole copy of — a stronger case for A4, not a
  new one.
- Checked, out of scope, and turned out fine: the remote box's dirty webapp files were first read as a
  risk (uncommitted logging code live in production). A byte-for-byte diff against local's committed
  files disproved that — see the correction at the end of §3d.

**Fourth pass (2026-09-03) — readiness audit, and the scale guard leaves this plan**

- **C1's "cheap test" did not work.** `analysis/bean_scale/estimators.py` never imports `coffeecv`, so
  editing the estimator and re-running the benchmark returns byte-identical output. C1 is now three
  steps (C1a/C1b/C1c) at ~1 day, not one command at 2-3 hours. (§3b, §4)
- **The scale guard is being deleted, not repaired.** It cannot fire and never has, so removing it
  changes zero behavior — but the docstring advertising it is exactly where someone would check
  whether framing is handled. Removal is S1 (Tier 0). What replaces it is *measurement*: S2 logs
  `beans_across` per request, which is the only item in this cluster with real time-value. (§3a)
- **B2 moves out of this plan** into `docs/scale_guard_plan.md`. Re-introducing a guard is a design
  problem blocked on two inputs that do not exist yet (C1c, plus logged framing data). Keeping it here
  as a half-day table row misrepresented it.
- Four smaller corrections: P2's `rebuild_index()` invocation (§3c), §5's stale P1 line, the DVC
  crop-stage staleness prerequisite (§5), and the plan doc itself now tracked in git.

---

## 1. Where the project actually stands

- **9 core classes** (001–009) captured since project start; 20 photos/class on box/pixel/sony,
  10–20 on the two phones.
- **1 new class**, 010 (Indonesia, Java), added 2026-08-30. Its coverage shape differs from 001–009
  and this matters — see §2.
- **Shipped model**: `models/allrigs_oneplusmerged_s17.pt`, from exp171 — 9 classes, all 5 rigs in
  training, none held out (so it has no cross-rig number by construction). Verified: `webapp/app.py:36`
  loads it, and `config_for_checkpoint` correctly redirects `classes_file` to the frozen 9-row
  sidecar `models/allrigs_oneplusmerged_s17.classes.txt`, so the 10-row `dataset/classes.txt` cannot
  desync the head. Confirmed by execution, not by reading.
- **class_010 has been trained exactly once, for 5 epochs** (exp175). exp174, its apparent twin, did
  not train class_010 at all — see §3e.
- **Adopted levers**: bean-unit patch sizing, MixStyle p=0.5 (agnostic), TTA at inference (+0.0235),
  eta_min=1e-5 (adopted on 1-seed evidence, flagged as such in `params.yaml`). Closed: seed
  ensembling, brightness jitter, mixup, freeze modes, cross-rig MixStyle v2.
- **Headline numbers, recomputed over the 20 runs at the current adopted config** (mixstyle_p=0.5,
  epochs=100, patience=20, eta_min=1e-05):

  | metric | range | n |
  |---|---|---|
  | in-distribution test macro-F1 | **0.884 – 0.933** | 20 |
  | cross-rig (leave-one-rig-out) macro-F1 | **0.674 – 0.851** | 14 |

  The previous draft's "0.89–0.95 / 0.66–0.85" drew its top end from exp39/44/46/47/65 — 2-rig-era
  runs scored on an easier split, max 0.9694. Not comparable; don't quote them together.

  The ~20-point in-distribution/cross-rig gap is still the single biggest lever. Recipe tuning
  (MixStyle, eta_min, TTA, mixup, freeze modes, brightness jitter) is largely screened out, and this
  plan's premise remains that the next gains come from data rather than another training trick —
  **with one revision**: §3b identifies a measurement defect in the patch-sizing pipeline that is
  neither data nor recipe, is rig-correlated, and should be tested before more capture work is
  planned around the transfer gap.

  Note on sourcing: `EXPERIMENTS_LOG.md` — which `README.md:51` calls "authoritative" — stops at
  Phase 14 / exp100-105. MixStyle, eta_min, H1 and exp106–175 exist only in `experiments/index.csv`
  and `params.yaml` comments. Confirmed by heading scan; the log has no Phase 15/16 content.

---

## 2. Rig × class coverage

Photo counts per class per raw capture session. **Reproduced exactly from disk; every cell verified.**

| class | box (08-07) | pixel (08-09) | sony (08-09) | oneplus_combined (08-25 + 08-27) | iphone (08-25) | 08-30 sessions |
|---|---|---|---|---|---|---|
| 001 Ethiopia Sidamo | 20 | 20 | 20 | 20 | 10 | — |
| 002 Kenya AA | 20 | 20 | 20 | 20 | 10 | — |
| 003 Colombia PinkBourbon | 20 | 20 | 20 | 20 | 10 | — |
| 004 CostaRica LaPastora | 20 | 20 | 20 | 20 | 10 | — |
| 005 Guatemala Tata | 20 | 20 | 20 | 20 | 12 | — |
| 006 Brazil Cerrado | 20 | 20 | 20 | 20 | 20 | — |
| 007 Brazil MonteCristo | 20 | 20 | 20 | 20 | 10 | — |
| 008 Ethiopia Kochere | 20 | 20 | 20 | 20 | **0** | — |
| 009 Vietnam Robusta | 20\* | 20 | 20 | 20 | 10 | — |
| **010 Indonesia Java** | **0** | 0 | 0 | **0** | **0** | pixel 46, sony 40, oneplus 40 |

\* box's directory is named `class_009__Vietnam`, every other rig's is `class_009__Vietnam_Robusta`.
Harmless — `CLASS_DIR_RE = ^class_(\d+)__` keys on the numeric id — but it is exactly the sort of
drift A5 should assert against.

Not shown, and deliberately outside the pipeline: `2026-07-24__first_pictures`,
`2026-08-06__box_pictures`, `classes_labels_only`. All three are flat directories with no `class_*`
subdirectories and none appears in `dvc.yaml`'s `crop` foreach. A5's script needs an explicit
exclusion list or it will flag them forever.

### 2a. class_010 has 3-rig coverage, not 5 — but the supporting evidence is weaker than claimed

Classes 001–009 appear on 5 independent rigs. class_010 appears on 3 (pixel, sony, oneplus) and is
**absent from box and iPhone**. It has more raw photos than any other class (126 total vs 90 for
class_001) but less rig diversity.

The first draft justified prioritizing this with H1 — "a class transfers better when multiple rigs
reinforce it in training" — and called it a confirmed finding backed by exp166/167 at +0.070/+0.099
"outside noise." **The numbers reproduce; the epistemic label does not.** What the record actually
contains:

| held-out rig | 3 training rigs | 4 training rigs | delta |
|---|---|---|---|
| oneplus | 0.7859 / 0.7584 / 0.7983 (s42/123/7, mean 0.7809) | 0.8508 (s42 only, exp166) | **+0.0699** |
| iphone | 0.7282 / 0.7245 / 0.7094 (s42/123/7, mean 0.7207) | 0.8193 (s42 only, exp167) | **+0.0986** |

Three problems the first draft did not disclose:

1. **It is unpaired.** One seed on the treatment side against a three-seed mean on the control side.
   `feedback-experiment-evidence-standard` requires paired multi-seed and records that single-seed
   wins on this project have three times been favourable draws.
2. **It is confounded with training-set size.** exp166/167 add a fourth rig; the control has three.
   "One more rig of data" and "a *phone* rig that shares framing style" are not separated.
3. **The matched control reads null.** exp163/164/165 go 2→4 training rigs on box/pixel/sony and
   deliver +0.029 / **−0.075** / +0.030. Adding rigs is not reliably good; the phone pairing may
   genuinely be special, or these five numbers may be five draws.

**What this does and doesn't justify.** It does not justify "class_010 will transfer worse, therefore
capture box+iPhone" as a confident prediction. It does justify capturing class_010 on box and iPhone
for a much plainer reason that needs no hypothesis at all: **class_010 currently cannot be evaluated
the way every other class is.** It has no box or iPhone photos, so every leave-one-rig-out fold that
holds out box or iPhone scores class_010 on nothing. That is a measurement gap, not a prediction, and
it is a sufficient reason on its own.

### 2b. Holding out an 08-30 session produces a one-class macro-F1

The 08-30 sessions were deliberately kept unmerged — `run_folds.py`'s own comment says so ("unlike
the oneplus_flash merge, these carry no overlapping class coverage to reconcile"), and that call is
defensible. The first draft accepted this and reduced the complaint to "a different eval semantic."
That understates it. Traced through the code:

- The 08-30 sessions contain **only class_010**.
- `train_baseline.py:345-347` computes the final cross-rig metric with
  `macro_labels=xrig_ds.present_class_idxs`, restricting the macro average to classes present in the
  held-out rig.
- Therefore a `2026-08-30__pixel`-heldout fold's `xrig_macro_f1` is **class_010's F1 alone**, a
  one-class average against a 10-way head.
- `RIGS` in `run_folds.py` now has **8 entries**, so a plain `python -m coffeecv.run_folds` sweeps 8
  folds, three of which are these degenerate one-class evaluations — at roughly 90 min each.

Meanwhile the other five folds have their own mismatched denominators: box/pixel/sony/oneplus hold-outs
average over 9 classes (no class_010 in the held-out rig), iPhone over 8 (no 008 either). So
`xrig_macro_f1` is already a column with four different denominators in it, and `index.csv` records
none of them.

This is no longer a naming nitpick. It is why B-fix moves up and gains a hard requirement: **the fold
metric must record its own denominator**, and the sweep must not silently run three one-class folds.

### 2c. iPhone is missing class_008 and is uneven elsewhere — 88 photos across *eight* classes

**Correcting the previous draft**, which said "+88 photos needed, all of class_008." The 88 is
correct; the attribution is not. (The underlying `project_iphone_photo_shortfall` note is *right* —
it carries the correct per-class table below. The previous draft compressed it wrongly on the way in,
which is its own lesson about citing a summary instead of the source.) Recomputed from disk
(iPhone has 92 photos, target 20 × 9 = 180, deficit 88):

| class | have | need | shortfall |
|---|---|---|---|
| 001 | 10 | 20 | +10 |
| 002 | 10 | 20 | +10 |
| 003 | 10 | 20 | +10 |
| 004 | 10 | 20 | +10 |
| 005 | 12 | 20 | +8 |
| 006 | 20 | 20 | 0 |
| 007 | 10 | 20 | +10 |
| 008 | 0 | 20 | **+20** |
| 009 | 10 | 20 | +10 |
| | | | **88** |

class_008 is the only class that is *absent* rather than merely thin, and it is the one that makes
every iPhone-heldout cross-rig number in the log an 8-class average silently reported next to 9-class
ones. If only part of this gets shot, shoot class_008 first.

### 2d. Confusable classes — **recomputed 2026-09-03; one of the two claimed pairs is wrong**

This was the one claim the previous pass's verification record admitted it had *not* checked against
confusion matrices, carrying it forward from the experiment log and spot-check notes instead. Now
computed (`coffeecv/confusion_report.py`), aggregating the **44 cross-rig confusion matrices** from
exp106-167 — cross-rig deliberately, since an in-distribution matrix is near-diagonal and says little
about class similarity. Rates are symmetrised over the pair's combined support.

| rank | pair | rate | per-direction |
|---|---|---|---|
| **1** | **006 Cerrado ↔ 007 MonteCristo** | **21.9%** | 006→007 836, 007→006 1476 |
| 2 | **002 Kenya AA ↔ 005 Guatemala Tata** | 10.2% | 579 / 501 |
| 3 | 002 Kenya AA ↔ 006 Cerrado | 7.1% | 530 / 218 |
| 4 | 002 Kenya AA ↔ 003 Colombia PinkBourbon | 6.9% | 426 / 304 |
| … | | | |
| 15 | ~~001 Sidamo ↔ 008 Kochere~~ | **3.6%** | 288 / 70 |

**006/007 is confirmed, emphatically** — rank 1 of 36 pairs at more than double the runner-up, and
both are Brazils, so the physical story is plausible. The direction is lopsided: 007 is misread as 006
almost twice as often as the reverse.

**001/008 is refuted.** It ranks 15th of 36 at 3.6%, below average, and per-class **008 Kochere is the
second-*cleanest* class in the set** (7.2% total off-diagonal, behind only 009 Robusta at 4.4%). It
should not have been named alongside 006/007, and B4 should not spend a screen on it.

**What the previous drafts missed entirely: 002 Kenya AA.** It appears in ranks 2, 3 and 4 — confused
with Guatemala Tata, Cerrado *and* PinkBourbon. Per-class off-diagonal rates make the real shape clear:

    007 MonteCristo 52.5%   002 Kenya AA 43.1%   005 Guatemala Tata 41.7%   006 Cerrado 35.8%
    003 PinkBourbon 27.5%   001 Sidamo   23.4%   004 LaPastora     22.4%
    008 Kochere      7.2%   009 Robusta   4.4%

So this is not "two confusable pairs" but **one dominant pair (006/007) plus a diffusely confusable
cluster (002/005/006/007)**, against two classes that are nearly free. That is a different problem
shape than a pairwise fix addresses, and it reframes B4.

### 2e. "Fresh-scoop" validation has never happened

Every cross-rig and cross-camera number in this project photographs the **same physical bags of beans**
used in training, with a different camera or framing. Nothing in `dataset/` is a session from an
independently-purchased lot of an existing origin. So the entire cross-rig literature here answers
"does this generalize across cameras" and has never answered "does this generalize across a new batch
of the same bean" — roast variation, moisture, bag-to-bag drift. Unchanged and still the single
largest unmeasured axis.

---

## 3. Pipeline defects (no new data required)

### 3a. The scale guard was a complete no-op, in both directions — REMOVED 2026-09-03

The first draft said the too-close refusal was unreachable and too-far had no check. Half right; the
reality is worse. `scale_verdict` (`coffeecv/infer.py:302`) *is* wired in at line 346, and has three
branches:

```
across < patch_beans_min (4.0)  -> REFUSE, "move the camera back or zoom out"
across < patch_beans_max (7.0)  -> WARN,   "usable, but framing wider would help"
otherwise                       -> OK
```

`beans_across = min(region.w, region.h) / estimate_bean_pitch(...)`. Substituting the estimator's own
constants — `ANALYSIS_FRAC = 0.40`, `CALIBRATION_K = 1.18`, FFT search band `_K_LO, _K_HI = 4, 80`,
`safety_margin = 0.97`:

```
pitch    = (n/k) · (0.40·short / n) · 1.18  =  0.472 · short / k
across   = 0.97·short / pitch               =  2.055 · k,     k ∈ [4, 79]
         => beans_across ∈ [8.22, 162.3]
```

**The hard floor is 8.22.** Both thresholds — 4.0 and 7.0 — sit below it. Neither branch can ever
fire. And there is no upper-bound check at all, so a photo taken from across the room passes too.

Measured, not just derived: across 15 photos spanning five rigs, `beans_across` ranged 8.22–14.39,
minimum **exactly** 8.22, and `scale_verdict` returned `(True, "…covering the trained 4-7 range")`
for every one. The guard the module docstring advertises as one of "two independent refusals" has
never refused anything and cannot.

**Resolution 2026-09-03: deleted, not repaired.** The earlier draft said "two things must both happen:
pick thresholds inside the reachable range, and add the missing too-far bound." Both are still true of
any *future* guard, but neither is worth doing now, for three reasons:

1. **Removing it changes nothing observable.** The guard always returns the third branch. Deleting it
   is removing dead code, not removing a safety feature.
2. **Leaving it is a lie with a compounding cost.** `infer.py`'s module docstring leads with *"Two
   independent refusals, because a wrong answer stated confidently is worse than no answer"* and lists
   Scale as #1, complete with the "move the camera back" rationale. That is precisely where a reader
   checks whether framing is handled. One refusal exists.
3. **Repairing it now would repeat the original mistake.** The thresholds are unreachable *because*
   they were picked from theory (mirror the trained 4.0-7.0 range) rather than from any measurement of
   what real frames produce. Setting new ones before C1c fixes the estimator and before any production
   framing data exists would be theory twice.

So: **S1** deletes it (Tier 0). **S2** starts logging `beans_across`, which is already computed at
`infer.py:292` and discarded, so the replacement is built on measurement instead. Designing an actual
guard is out of scope here and lives in `docs/scale_guard_plan.md`.

**What this leaves unprotected, stated plainly:** a genuinely too-far photo now gets a confident answer
with no framing warning. That is already true today — the guard never fires — so nothing regresses, and
the OOD guard plus the tray-crop detector are partial backstops. But it is "no worse than today," not
"handled."


### 3b. The bean-pitch estimator is pinned at its band floor on 39% of photos — **interpretation REFUTED 2026-09-03, see the result at the end**

This is the finding that most changes the plan's premise, and it is one measurement away from being
actionable.

`estimate_bean_pitch` takes the argmax of a `k^1.5`-weighted radial power spectrum over
`k ∈ [_K_LO=4, 80)` cycles across the analysis window. Sampling 24 cropped photos from each of the 8
rigs (192 total):

| rig | k histogram | pinned at k=4 |
|---|---|---|
| box (08-07) | 4:13, 5:7, 6:3, 7:1 | **54%** |
| pixel_cam (08-09) | 4:7, 5:3, 6:2, 7:7, 8:4, 9:1 | 29% |
| sony_cam (08-09) | 4:5, 5:3, 6:9, 7:5, 8:1, 9:1 | **21%** |
| iphone (08-25) | 4:11, 5:3, 6:6, 7:2, 8:1, 9:1 | 46% |
| oneplus_combined | 4:6, 5:7, 6:1, 7:2, 8:3, 9:3, 10:1, 11:1 | 25% |
| 08-30 pixel | 4:14, 5:4, 6:4, 7:2 | **58%** |
| 08-30 sony | 4:7, 5:7, 6:6, 7:4 | 29% |
| 08-30 oneplus | 4:12, 5:8, 6:2, 7:1, 8:1 | 50% |
| **overall** | | **39% of 192** |

Inspecting the weighted band profile on pinned photos shows it **decreasing monotonically from the
first bin**. The draft read that as: the true dominant period lies past the band's low-frequency edge
and is being clipped to k=4, so pitch is under-estimated. **That reading was tested on 2026-09-03 and
is wrong** — see the result at the end of this section. The measurement (39%, rig-dependent) stands;
the causal story does not.

Why this matters more than it looks:

1. **It is rig-correlated** — 21% (sony_cam) to 58% (08-30 pixel). Bean-unit sizing exists precisely
   to make patch content rig-invariant. A rig-dependent clipping bias inside the estimator partially
   defeats that, and it sits directly on the cross-rig gap this whole plan is organized around.
2. **It is symmetric across train and inference** (`dataset.py:473` and `infer.py:269` call the same
   function), so it does not violate the parity rule — but that also means it has been baked into
   every cross-rig number in the log rather than showing up as a train/test discrepancy.
3. **It explains the "~24% pitch noise" already noted in `run_folds.py`'s `ARMS` comment.** At k≈4
   the integer quantization step is 1/4 = 25% of pitch. That comment treats the noise as inherent and
   deliberately kept; it is substantially an artifact of operating at the band edge.
4. ~~**It set §3a's dead floor**~~ (§3a's guard is now deleted; retained for the record): `beans_across = 2.055 · k`, so the floor was `2.055 × _K_LO = 8.22`.
   With the guard now deleted (§3a) this is no longer a reason to fix the band — but it does mean any
   future guard's reachable range is defined by whatever C1c settles on, which is why
   `docs/scale_guard_plan.md` takes C1c as an entry criterion rather than a nice-to-have.

**RESULT, 2026-09-03 — executed, and it inverts the hypothesis above.**

Getting here needed two things first. `analysis/bean_scale/estimators.py` never imports `coffeecv`:
its `m0_fft_radial` is a private copy hardcoding `lo, hi = 4, 80`, so editing
`coffeecv/bean_scale.py` and re-running `benchmark.py` returns byte-identical output — a sweep driven
through it would have reported a confident null whatever the band did. And neither MAPE nor a pinned
fraction was computed by any committed code. So `bean_scale.py` gained `estimate_bean_pitch_k`
(returns `(pitch, k)`, accepts `k_lo`/`k_hi`/`analysis_frac` overrides; the plain
`estimate_bean_pitch` forwards to it and is bit-identical for every production caller, verified on
real photos), and `analysis/bean_scale/band_sweep.py` scores the sweep.

*One correction to an earlier draft of this section*: the benchmark does **not** have an
`ANALYSIS_FRAC` parity defect. `make_crops.py` cuts its crops at `FRAC = 0.40`, exactly what
`ANALYSIS_FRAC` does internally, so passing a pre-cut crop is correct by construction — confirmed
empirically, production/benchmark = 1.180 on every crop, precisely `CALIBRATION_K`. The hardcoded
band was the real blocker, and the only one.

| `_K_LO` | MAPE (refit K) | refit K | pinned | rig-bias spread | corr(pred, GT) |
|---|---|---|---|---|---|
| 2 | 29.6% | 1.131 | 0% | 0.195 | 0.699 |
| **4 (shipped)** | **20.0%** | **1.242** | **23%** | **0.041** | **0.838** |
| 6 | **9.7%** | 1.449 | 50% | **0.029** | **0.929** |

**Lowering `_K_LO` makes everything worse; raising it makes everything better**, monotonically and on
every rig individually. The low-frequency floor is *suppressing 1/f spectral drag*, not clipping bean
signal — at low `k` the `k^1.5` weighting still loses to the spectrum's own falloff and the argmax
runs to spurious long periods. "Pinned at the floor" is the guard working. Checked the obvious
failure mode too: a floor high enough to make `k` nearly constant would degenerate into
frame-fraction sizing and score well here only because these rigs frame similar bean counts. It is
not degenerating — correlation *rises* and the rigs stay separated (86/202/263 px against GT
104/239/323 at `_K_LO=6`).

**Deliberately not adopted.** Ground truth is 3 rigs from 2026-08-11, predating iPhone/oneplus/08-30
entirely — and rig-dependence was the whole reason this mattered. The gain also needs `CALIBRATION_K`
refit from 1.18 to ~1.45, so two shipped constants move together, and ~5% is the benchmark's own
noise floor. Next step is hand-counted ground truth on the newer rigs, then validation through
`run_folds.py`'s cross-rig metric — never on an offline benchmark alone. Full detail:
`analysis/bean_scale/README.md`.

### 3k. NEW (2026-09-03) — the pitch estimator's geometry was not part of any run config — FIXED

Found while preparing C1-ship, which needs a paired `_K_LO=4` vs `_K_LO=5` fold comparison.

`_K_LO`, `_K_HI`, `ANALYSIS_FRAC` and `CALIBRATION_K` lived as **module constants** in
`coffeecv/bean_scale.py`. Nothing recorded them: they are absent from `params.yaml`, from `RunConfig`,
and therefore from every archived `config.json`. Three consequences, in rising order of seriousness:

1. No run can be reproduced at its own estimator geometry, because no run states what geometry it used.
2. `config_for_checkpoint` — whose entire purpose is that "patch geometry is the one thing inference
   must get right" — **cannot restore them**, so it silently does not.
3. Changing one is a globally-breaking edit: every checkpoint ever trained, the deployed model
   included, would be sized by the new constant at inference while having been trained under the old
   one. A train/inference parity break with no error, no warning, and no record anywhere.

That last point also blocked C1-ship outright — a per-arm band comparison is impossible when the value
is not per-run, and adopting a new band would retroactively re-size every archived checkpoint rather
than applying only to runs that opted in.

**Fixed**: promoted to `RunConfig` as `bean_k_lo` / `bean_k_hi` / `bean_analysis_frac` /
`bean_calibration_k`, defaulting to exactly the old constants so an unset config reproduces every
existing run. `bean_scale.pitch_kwargs(cfg)` is the single mapping, threaded through the dataset,
`infer.py`, `xrig_eval`, `build_ood_reference` and `eval_legacy_lens_photos`. Parity verified rather
than assumed: bit-identical pitch on 8 real photos, and the full `infer.py` CLI returns identical
verdicts, top-1 classes, probabilities and `bean_pitch_px` over the same 6 photos before and after.
`run_folds` gained `--bean-k-lo`/`--bean-calibration-k`, forwarded and covered by the post-condition
assertion, and refuses one without the other since the calibration is fitted to the band.

### 3c. NEW — `experiments/index.csv` has duplicate rows

142 data rows, **140 unique experiment ids**; exp169 and exp170 each appear twice, byte-identical.
There are 140 `experiments/exp*__*` directories, and `archive_experiment.rebuild_index()` regenerates
the file *from those directories* rather than appending — so the committed file is not what the tool
produces. Almost certainly a bad CSV merge when the remote and local sweeps were folded together
(commits `6ea794a`, `b33bbb9`). Any analysis that reads `index.csv` — including the ranges in §1 —
double-counts those two runs unless it dedupes first. One `rebuild_index()` run fixes it — but call
it **directly**:

```
python -c "from coffeecv.archive_experiment import rebuild_index; print(rebuild_index())"
```

Not via the CLI. `--replot-all` is the only flag that reaches `rebuild_index()`, and it regenerates
matplotlib charts for all 140 archived experiments first — hundreds of rewritten PNGs and a large
spurious diff, for a fix that touches one CSV.

### 3d. NEW — eight archived experiments cite commits that don't exist in this repo

Checking each `experiments/*/config.json`'s `env.git_commit` with `git cat-file`: 132 resolve, **8 do
not**.

| exp | commit | note |
|---|---|---|
| 168 | `942f577d…` | |
| 169 | `6d1b5977…` | |
| 170 | `56311a0d…` | |
| **171** | `4bdf0d33…` | **the currently shipped model** |
| 172 | `70f7ccb3…` | |
| 173 | `a7376855…` | |
| 174 | `9a09fad2…` | |
| 175 | `830dfd8c…` | |

All eight have `argv: /home/alioth/coffee-vision/…` — they ran on the remote box, and their commits
were never pushed here. Their archives (config.json, metrics.json) came across; the revisions did not.

This is a direct violation of the project's own provenance rule, and it lands on the worst possible
run: **exp171 is the model in production at coffee.kasetkin.com, and this repository cannot reproduce
it.** The archived `config.json` is the only record of how it was fitted. Fetch those commits from
the remote before that machine changes — this is time-sensitive in a way none of the capture work is.

**Verified 2026-09-02 (third pass) by actually reaching the remote box.** `ssh powervpsssh` works; the
repo at `~/coffee-vision` there is a live, ordinary git checkout, currently at `895d424` — **101**
commits ahead of `origin/main` (local is only 12 ahead). Added it as a real local git remote
(`powervpsssh:coffee-vision`, over the existing SSH alias — no new host config needed) and ran
`git fetch`, which succeeded. Two things changed from the second draft's picture:

1. **All 8 commits are confirmed to exist and are fetchable** — this part of §3d's premise holds.
2. **The fetch is not the ~15-minute task §4 estimated, and "fetch" is not the whole job.** The pack
   transferred is unexpectedly large (~250-300MB for what should be small metadata commits) because
   remote history contains a fixed-forward mistake: commit `14f15a0` ("Untrack 2026-08-30 raw session
   photos from git -- DVC-tracked, not git") shows an earlier sync committed those sessions' raw JPEGs
   *directly to git* (a `git add` naming the directories explicitly overrode `.gitignore`), then
   reverted the tracking. The mistake is fixed forward and disk content is fine, but the blobs are
   permanent in history and every fetch pays for them. Budget real time for this, not 15 minutes.

   More importantly: **local and remote have mutually diverged**, not "remote is ahead." Remote's most
   recent sync-from-local point is its own commit `830dfd8` ("Sync working tree to local main @
   96d6a9a") — and local's `96d6a9a` is now well behind local's current tip.

   **2026-09-03 update — superseding the cherry-pick recommendation below with something simpler and
   safer, after actually diffing both tips instead of reasoning from the commit list alone.** The real
   merge-base is `632d576` (exp162), much further back than either "sync" point suggests. Rather than
   assume the 13 remote-only commits need replaying, ran `git diff main powervpsssh/main --stat`
   directly: **only 18 files differ between the two tips, total**, and every one traces to something
   already accounted for — each side's own transient run-state (`outputs/*.json`, `dvc.lock`,
   `params.yaml`'s resting config), the already-known `index.csv` duplicate rows (confirmed introduced
   on local's side; remote's own copy never had them), local-only additions remote's tip predates
   (`scripts/remote_log.sh`, the `.pt.dvc` pointer), and the webapp files (remote's *committed* tip
   predates local's logging work, but remote's *uncommitted* working copy already matches it byte for
   byte — same false alarm as before, visible again from the commit side). The one asymmetry running
   the other way: `docs/inference_path.html`/`docs/logging_plan.html` are committed on remote (swept up
   by an over-eager `git add` during a sync) but only sit untracked locally — worth `git add`-ing
   locally too, unrelated to this reconciliation.

   **Nothing on remote is at risk of loss, and cherry-picking would cost more than it buys**: it would
   replay two bugs local already fixed independently (`b6f5b4f`/`c86c800`, both confirmed superseded by
   local's fuller `5b7ada0`/`c4b2660`), resurrect the self-corrected raw-photo mistake as phantom
   conflicts, and fight over files that are scratch state, not authored content. The only genuine
   remote-only thing is **atomic, individually-attributed commits for exp168-175** — their *data*
   (config.json, metrics.json, model card) is already safely committed on local via `c4b2660`,
   `6ea794a`, and `b33bbb9`; what's missing is provenance, not content.

   **Fix, done 2026-09-03**: froze remote's entire line under
   `git tag -a archive/powervpsssh-2026-09-02 powervpsssh/main`, created locally in this repo. This is
   pure addition — can't conflict with anything, and preserves exp171's exact reproducing commit
   (`4bdf0d33`, see point 3 below) forever once pushed. **Not yet pushed to `origin`**: this environment
   has no GitHub push credentials (no SSH agent, no `gh` CLI) — that's also why local `main` has sat
   12 commits ahead of `origin/main` unpushed already, an existing gap, not a new one. Push both from
   wherever push access actually exists:
   ```
   git push origin archive/powervpsssh-2026-09-02
   git push origin main
   ```
   Until that push happens, the tag is exactly as fragile as everything else living only in this one
   repo — it needs to reach `origin` to deliver the durability it's for.

   **DVC-specific check, same pass**: tagging/pushing git history never touches `.dvc/cache` — no risk
   there, but it also doesn't *carry* the actual cached bytes a `.dvc` pointer names, only the pointer.
   Checked what that means concretely: `dvc status models/allrigs_oneplusmerged_s17.pt.dvc` →
   *"Data and pipelines are up to date"* — the shipped model's actual 44.8MB blob (md5 `c646750d...`)
   is verified intact in local's cache, not just a dangling pointer. Traced `outputs/checkpoints/best.pt`'s
   md5 across every relevant commit (`6ea794a`, `c4b2660`, `b33bbb9`, `96d6a9a`, `9db4e76`): identical
   in all of them — confirming exp168/169/170/172/173/174's individual checkpoint weights were never
   separately preserved (each got overwritten by the next seed's run before ever being committed; only
   exp171 was promoted to `models/` because it was the one chosen to ship). That's the project's
   existing archive convention working as intended, not a gap this pass opened or one a DVC remote
   could retroactively close — only exp171's weights ever needed to survive, and they have. **A4 (DVC
   remote) is still the right next step for the raw dataset and future checkpoints, which have no
   backup at all right now** — raised with the user and deliberately deferred to a later session by
   their own choice, not forgotten.

3. **Clarifying, not a defect**: `git_commit` in an archived `config.json` names the code state at the
   *start* of that run — which, by construction, is always the *previous* experiment's own archival
   commit (a commit cannot record its own hash). Confirmed against purely local history, no fetch
   needed: exp167's config cites `b27c5ab`, which is exp166's own commit message ("exp166: ..."); the
   same pattern holds for exp160/163/166. So exp171's cited commit `4bdf0d33` is exp170's own archival
   commit — correct and sufficient for reproduction once fetched, but do not go looking for a commit
   *named* "exp171" to check out; that commit (`70f7ccb`) is where exp171's own results were archived
   one step later, and its tree is exp171's *output*, not the code that produced it.

4. **Out of this plan's scope; initially read as a risk, corrected after checking.** The remote box's
   working tree is dirty — uncommitted changes to `coffeecv/infer.py`, `webapp/app.py` (+181 lines),
   `webapp/README.md`, the systemd unit, the nginx template, and `setup_server.sh`, plus an untracked
   logrotate file — and `coffee-cv-web` is confirmed `active`, restarted 2026-09-02 06:15 UTC, after
   `app.py`'s last edit (Aug 31). The first pass over this read as "production running code with no git
   record anywhere" and treated it as time-sensitive. **That was wrong, caught by actually diffing the
   content instead of stopping at `git diff --stat`.** Pulling remote's live files and comparing them
   byte-for-byte against local's committed versions: `webapp/app.py` and `coffeecv/infer.py` are
   **identical** to what's already committed locally (`3f08b4b`..`2b5e718`); so is `webapp/README.md`.
   `setup_server.sh` and the service/logrotate files differ from local by exactly one commit —
   `427d174`, the username-parameterization rename to `.template` + `envsubst` (`APP_USER=alioth`
   hardcoded on remote vs `APP_USER="${APP_USER:-alioth}"` locally). So this is not lost or at-risk
   work: it is the ordinary residue of this project's documented deploy process (manual rsync + restart,
   no `git commit` step on the remote box — see `project-webapp-production`), and every line of it
   already lives safely in local's git history bar one cosmetic commit. No action needed beyond folding
   a normal rsync of `427d174`'s delta into the next deploy; not worth a numbered action here since nothing is missing or at risk.

### 3e. NEW — exp174 is a void run and is not labelled as one

exp174 and exp175 share a slug (`allrigs_class010_smoketest_s42`), a seed (42), and **byte-identical
`config.json` apart from `git_commit`** — yet:

- exp174's `metrics.json` has **9** `class_ids`.
- exp175's has **10**.

Commit `96d6a9a` explains it: before that fix, `class_ids` came from
`discover_classes_multi(train_rigs[0])` — box, which has no class_010 — so exp174 trained a 9-way head
and class_010 never appeared, with no crash and no warning. The bug is found and fixed; the
consequences are not cleaned up:

- **class_010 has been trained once, not twice.** §1 of the previous draft said "smoke-tested,
  exp174/175," treating them as a pair. Only exp175 is a class_010 test.
- exp174 is still in `index.csv` under a slug that claims otherwise, with no `VOID_` marker — unlike
  exp100-105, which the project did rename `VOID_flagbug_…`. Its own convention is not being applied.
- **`config.json` does not record the class list.** Two archived runs can have identical configs and
  different output spaces. That is a hole in the archive format, not just in one run — add `class_ids`
  (or a `classes.txt` hash) to the archived config.
- `run_folds.py`'s collision guard only catches *same id, different slug*. This pair is *different id,
  same slug*, which slips through.

### 3f. NEW — per-epoch and final cross-rig F1 use different denominators

`train_baseline.py:296` computes the per-epoch `xrig_macro_f1` for `history.json` **without**
`macro_labels`; line 345-347 computes the final one **with**
`macro_labels=xrig_ds.present_class_idxs`. When the held-out rig lacks a class, absent classes score
F1=0 and drag the epoch curve while being excluded from the reported number. For an iPhone hold-out
that is /9-with-a-phantom-zero versus /8. For an 08-30 hold-out it would be /10 with **nine** phantom
zeros versus /1 — the curve and the headline number would differ by roughly 10×. The Phase 11 chart
that `dvc.yaml` plots is therefore not the metric that gets reported. One-line fix.

### 3g. Val-set saturation (unchanged, verified)

`README.md:53`, verbatim: 5 of 9 classes sit at or near val F1 = 1.000 and one run hit val macro-F1
0.9917. Since `best.pt` is selected on peak val macro-F1, this degrades *which checkpoint ships*, not
just reporting.

### 3h. Photo-level pooling — **resolved 2026-09-03: the box regression was noise**

Verified from Phase 14: box −0.035, pixel +0.079, sony +0.015, mean +0.020, with the log's own next
sentence being "n=27 photos/fold, so ±3.7 points per photo; directional, not precise." The box fold's
−0.035 is *one photo*, so chasing it with a confidence-weighted or outlier-rejecting aggregator would
have been tuning against noise. The prerequisite was a bigger photo-level eval set, not a cleverer
aggregator — which is what B3′ did.

**Done.** `coffeecv/photo_pooling_eval.py` scored every photo of each held-out rig using that fold's
own archived checkpoint, 40 patches per photo to match Phase 14:

| fold | n | patch | photo | delta | Phase 14 (n=27) |
|---|---|---|---|---|---|
| box | 180 | 0.782 | 0.894 | **+0.113** | −0.035 |
| pixel_cam | 180 | 0.776 | 0.883 | **+0.107** | +0.079 |
| sony_cam | 180 | 0.729 | 0.806 | **+0.077** | +0.015 |
| oneplus | 90 | 0.855 | 0.922 | **+0.067** | — |
| iphone | 92 | 0.784 | 0.880 | **+0.097** | — |
| **mean** | **722** | | | **+0.092** | +0.020 |

**5/5 positive; the box fold's −0.035 became +0.113 — the largest gain in the set — with nothing
changed but the sample size.** Plain mean pooling is worth ~9 points, about 4.5× what Phase 14 could
resolve, uniformly across all five rigs including the two phone rigs it never covered. Phase 14's
mechanism argument still caps the ceiling (within-photo patch errors are correlated, so 40 patches are
not 40 independent votes), but the "one fold goes backwards" finding does not survive a real sample.
Full write-up: `analysis/photo_pooling_README.md`.

### 3i. No DVC remote (unchanged, verified — and now confirmed on both machines)

`.dvc/config` is empty; `dvc remote list` returns nothing. Every capture session exists only in the
local `.dvc/cache`. `experiments/README.md:24` additionally documents ~4.7 GB of historical
checkpoints (90 of them) reachable only through old `dvc.lock` commits in that same local cache — a
`dvc gc` would silently delete them.

**Verified 2026-09-02 on `powervpsssh` too**: `.dvc/config` is equally empty there — this was never a
local-only gap, so A4 helps both machines, not just this one. More pointedly, remote's own `dvc status`
shows `models/phase8_best_random_erasing_0.5.pt`, `models/phase12_beans47_s7.pt`,
`models/phase14_allrigs_s42.pt`, and `dataset/2026-07-24__first_pictures` all **"not in cache"** on that
box (expected — only the working set was rsynced there, per `project-remote-compute`). Right now
**local's `.dvc/cache` is the sole surviving copy of those four artifacts on either machine.** A4 is not
just insurance against a future `dvc gc`; it is the only thing standing between those four artifacts
and total loss today.

### 3j. Docs drifting from code (unchanged, verified)

- `webapp/README.md:35` says `app.py` hardcodes `allrigs_mixstyle05_e100p20_s17.pt`;
  `webapp/app.py:36` loads `allrigs_oneplusmerged_s17.pt`. Documentation-only — the frozen sidecar
  mechanism was verified by execution and does protect the head size.
- `EXPERIMENTS_LOG.md` stops at Phase 14 while `README.md:51` calls it authoritative (§1).
- `params.yaml`'s resting `train_rigs` lists 4 rigs including the *unmerged* `2026-08-25__oneplus`,
  with `heldout_rig: 2026-08-25__iphone`, and a comment reading "rotate heldout_rig across the three
  rigs." `RIGS` has 8. Harmless — both sweep drivers rewrite the file — but it is the first file a
  reader opens.

---

## 4. Recommendations, re-prioritized

Ranked by (evidence strength) × (cost). The ordering changed materially from the first draft: two
provenance items and one measurement item now outrank the capture work.

### Tier 0 — do this week; two are time-sensitive, two are nearly free

| # | Action | Why | Cost |
|---|---|---|---|
| **P1** | **Tag and push, don't merge**: `git tag -a archive/powervpsssh-2026-09-02 powervpsssh/main` (done locally 2026-09-03), then `git push origin` that tag plus local `main` from wherever GitHub push access exists (not available in this environment) | §3d: confirmed by full-tree diff — nothing on remote is genuinely at risk or unmerged; the only real gap is exp168-175's atomic commits, which a tag preserves without the conflict risk of cherry-picking. Blocked only on push access, not analysis | ~5 min once you have push access |
| ~~**A4**~~ **DONE 2026-09-04** | Remote `remoteconfig` (`ssh://coffdvc@dvcremote:49081/...`, 611G free) now holds the **entire** local cache: 2812 objects both sides, 14G, and checkpoint-sized objects went 18 -> 224. §3i's ~4.7 GB of archived fold checkpoints are finally off a single machine | **Two traps found on the way, both worth knowing.** (1) `.dvc/config.local` declared the remote with a `keyfile` and no `url`; harmless at HEAD, but `--all-commits` reads `.dvc/config` *as of each commit*, where it was empty, so the merge produced a url-less remote and **323 historical commits failed to collect** with only a WARNING -- a push that looked clean while protecting almost none of the history it was for. (2) Even after fixing that, `dvc push --all-commits` still reported "Everything is up to date" while demonstrably missing objects it had collected, so the cache was mirrored with `rsync` instead -- valid because an SSH remote is content-addressed with the identical `files/md5/<2>/<rest>` layout | done |
| ~~**P2**~~ **DONE 2026-09-03** | `rebuild_index()`; rename exp174 to `exp174__VOID_9class_dup_of_exp175`; add `class_ids` to the archived config | §3c + §3e: the record the whole plan reads from is wrong in two places, both one command or one field away from fixed | ~30 min |
| ~~**C2**~~ **DONE 2026-09-03** | Fix the epoch-vs-final macro denominator (§3f) and record the denominator in `summary.json`/`index.csv` | One-line correctness fix, plus it is a precondition for §2b being interpretable | ~1 hour |
| ~~**S1**~~ **DONE 2026-09-03** | **Delete the dead scale guard**: remove `scale_verdict` (`infer.py:302`) and the `REFUSED (scale)` branch in `classify_one`; drop `webapp/app.py:214-215`'s now-unreachable `refused_scale` mapping (it reads `entry["scale_note"]`, which would no longer exist — a latent `KeyError`); rewrite `infer.py`'s module docstring, which leads with "Two independent refusals". Keep `beans_across` in the diag — only the verdict goes. The frontend's `refused_scale` string (`index.html:386`) can stay; it is an OR-chain still serving `refused_ood`/`refused_unreadable` | §3a: it cannot fire and never has, so deletion changes zero behavior — but the docstring is where someone checks whether framing is handled. Same record-cleaning category as P2 | ~30 min |
| ~~**S2**~~ **DONE 2026-09-03 (not yet deployed)** | **Log `beans_across` per request** — add it to `webapp/app.py`'s structured log fields. It is already computed at `infer.py:292` and thrown away | The one item in this cluster with real time-value: production framing data cannot be recovered retroactively, so every day unshipped is a day lost. It is also the entry criterion for `docs/scale_guard_plan.md`. *Scope note*: this plan excludes webapp work, but this is one field added to a log that already ships, not a feature | one line |

### Tier 1 — cheap, high-information, no new data

| # | Action | Why | Cost |
|---|---|---|---|
| ~~**C1**~~ **DONE 2026-09-03** | Built the harness (`estimate_bean_pitch_k` with band overrides; `analysis/bean_scale/band_sweep.py` for MAPE + pinned fraction) and swept. **Result inverts the hypothesis**: raising `_K_LO` improves MAPE (20.0% -> 9.7%), rig-bias spread (0.041 -> 0.029) and GT correlation (0.838 -> 0.929); lowering it degrades all three. The floor suppresses 1/f drag rather than clipping signal | Answered "is the ruler straight?" -- the ruler is not obviously bent in the direction assumed, and the cross-rig numbers are not distorted the way §3b feared | done (~4h, not the ~1 day estimated) |
| ~~**C1-next**~~ **DONE 2026-09-03** | Generated 30 crops (6 each) across iPhone, oneplus_combined and the three 08-30 sessions, VLM-counted, re-swept. **Direction replicates across all 8 rigs** — `_K_LO=2` bad on both sets (29.6/31.0%), shipped `_K_LO=4` identical (20.0/20.1%), raising it improves both. **Optimum does not**: old rigs peak sharply at 6 (9.7%), new rigs are flat 5-7 (13.7/13.7/13.0%) with non-monotone correlation (peaks at 5, 0.918; drops at 6, 0.854) | `_K_LO=5` is the defensible move if one is made — good on both sets. `_K_LO=6` is excellent on the old three but buys nothing over 5 on the new five | done (~3h, no human labour) |
| **C1-ship** *(RUNNING, launched 2026-09-03)* | 5-fold sweep at `bean_k_lo=5` + refitted `bean_calibration_k=1.334`, seed 42, tag `klo5`, exp176-180. Unblocked by §3k (the constants are now config fields, so the arm is recorded per run and asserted by `run_folds`' post-condition). Paired against exp163-167 | Offline reconstruction accuracy is not classification accuracy, and this project validates config changes on folds. **Pairing caveat**: exp166 held out the *unmerged* `2026-08-25__oneplus` (90 photos) while exp179 holds out `oneplus_combined` (180), so 4 of 5 folds pair cleanly and the oneplus one does not — a k_lo=4 baseline on `oneplus_combined` would be needed to complete it | ~7.5h, in progress |
| ~~**B2**~~ | **Moved out of this plan → `docs/scale_guard_plan.md`.** Re-introducing a framing guard is a design problem (refuse vs warn; what a too-far bound is even based on; whether scale and OOD are actually independent; whether the tray-crop detector makes it partly moot; pre- vs post-capture advice), blocked on two inputs that do not exist yet: C1c's corrected estimator, and enough logged `beans_across` (S2) to see the real distribution | Setting thresholds today would repeat the original mistake — deriving them from theory rather than measurement. S1 deletes the broken one so the scale plan starts from no guard rather than a lying one | — (tracked in the scale plan) |
| ~~**A5**~~ **DONE 2026-09-03** | **Coverage-check script** (`coffeecv/coverage_report.py`): print the §2 table, assert class-dir naming consistency, carry an explicit exclusion list for the three non-pipeline sessions, warn on rig×class imbalance | §2 took manual cross-referencing of eight directories to produce, and will need it again on every new session. Also catches the `class_009__Vietnam` drift | 1-2 hours |
| ~~**B-fix**~~ **DONE 2026-09-03** | **Make fold semantics explicit**: label the 08-30 entries as class-transfer folds distinct from rig-transfer folds, exclude them from the default `run_folds` rotation, and surface the macro denominator per fold | §2b: three of eight default folds are degenerate one-class evaluations at ~90 min each, sitting in the same column as 9-class rows | Half a day (naming + a `RIGS` split, no DVC stage needed) |

### Tier 2 — capture work

| # | Action | Why | Cost |
|---|---|---|---|
| **A1** | **iPhone: +88 photos across 8 classes** — class_008 (+20) first, then 001-004/007/009 (+10 each), 005 (+8). 006 needs none | §2c: class_008's absence makes every iPhone-heldout number an 8-class average reported beside 9-class ones. **Not** 88 photos of class_008 | 1 session |
| **A2** | **class_010 on box and iPhone** (~20 each) | §2a: not because H1 predicts it, but because class_010 is currently unscored in any box- or iPhone-heldout fold. Also consider trimming the existing 40-46 down toward the 20/class convention rather than leaving it oversized | 1-2 sessions |
| **B5** | Once A1/A2 land (and C2/B-fix make the metric readable), run a proper multi-seed fold sweep including class_010 and ship a real 10-class model | Nothing has cleared the project's evidence bar at 10 classes; the only 10-class run in existence is a single 5-epoch smoke test | ~1-2 days compute |

### Tier 3 — exploratory

| # | Action | Why | Cost |
|---|---|---|---|
| **A3** | **Fresh-scoop session** from newly-purchased bags of an existing origin, held out from all training | §2e: the one generalization axis never measured, and the closest thing to real deployment | 1 session + bean purchase |
| ~~**B3′**~~ **DONE 2026-09-03** | Scored every photo of all five held-out rigs (722 photos vs Phase 14's 81) with each fold's archived checkpoint. **Pooling is worth +0.092, 5/5 folds positive** — and the box fold's −0.035 became **+0.113** on sample size alone | §3h: answers itself. Do not build a confidence-weighted or outlier-rejecting aggregator — there is no regression to chase, and plain mean pooling already buys ~4.5× what Phase 14 could measure | done (~40 min scoring) |
| **B4** | **Target the confusion cluster, not the two pairs the old §2d named.** 006/007 is confirmed rank 1 (21.9%) and worth a direct screen; 001/008 is refuted (rank 15, 3.6%, and 008 is the 2nd-cleanest class) and should be dropped from the target list. The bigger prize may be 002 Kenya AA, which the plan never mentioned and which sits in ranks 2-4 | §2d, recomputed from 44 cross-rig confusion matrices. Per-class rates say the problem is a 002/005/006/007 cluster, not isolated pairs — a coarse/hierarchical auxiliary signal is a better fit for that shape than a pairwise fix | One fold-sweep-scale screen, paired multi-seed |
| **B1** | Address val saturation — grow the val split, or change checkpoint selection | §3g: affects which checkpoint ships | Design + a fold-scale screen |
| **D1** | Reconcile `EXPERIMENTS_LOG.md` with `index.csv`/`params.yaml`; fix `webapp/README.md:35`; refresh `params.yaml`'s stale resting state and comments | §3j: the doc `README.md` calls authoritative is ~70 runs behind, including both largest adopted levers | Half a day |

---

## 5. Sequencing

```
P1 (tag + push)            ── DONE locally 2026-09-03; only `git push origin` remains (needs access)
A4 (DVC remote)            ── independent, pure insurance; deferred by choice 2026-09-03
P2 (index + VOID + class_ids in config)
C2 (macro denominator)     ──┐
                             ├──> B-fix (fold semantics) ──┐
A5 (coverage script)       ──┘                             │
                                                           │
S1 (delete dead scale guard) ── independent, ~30 min
S2 (log beans_across)        ── independent, one line; starts the scale plan's clock
                                                           │
C1a/b/c    DONE 2026-09-03 -- harness built, band swept, hypothesis refuted
  └─> C1-next  DONE 2026-09-03 -- replicates on all 8 rigs; direction yes, optimum no
        └─> C1-ship (fold-sweep _K_LO=5 + refitted CALIBRATION_K)  ── not started
                                                           │
A1 (iPhone +88 across 8 classes) ──┐                       │
A2 (class_010 on box + iPhone)  ───┴───────────────────────┴──> B5 (10-class fold sweep + ship)

A3 (fresh-scoop)   ── independent, parallel with everything
B3′ (grow photo eval set), B4 (confusable pairs), B1 (val saturation), D1 (docs) ── whenever free
```

C1 is internally ordered — C1a/C1b build the measurement, and only C1c produces a number anyone can act
on. The old "C1c before B2" edge has left this plan with B2: it is now an *entry criterion* of
`docs/scale_guard_plan.md`, along with S2's accumulated logging, rather than a dependency inside this
one. Nothing in this plan is downstream of C1c any more.

**Prerequisite before touching the pipeline at all** (P2, C2 and the doc items are exempt; anything
that runs `dvc repro` is not): all six `crop` stages currently report `changed deps: modified
coffeecv/crop_tray.py`, so a `dvc repro` — including B5's fold sweep — will re-crop all six sessions
before doing anything else. Decide deliberately which you want: `dvc commit` the crop stages if the
existing outputs are still valid, or accept and budget the re-crop. Do not discover this mid-sweep.

---

## 6. Immediate next actions

1. **P1 — done, except the final push.** A full-tree diff (not just the commit list) showed nothing on
   remote is genuinely at risk: its data is already on local `main` via `c4b2660`/`6ea794a`/`b33bbb9`,
   and the only real gap — atomic, individually-attributed commits for exp168-175 — is fixed by tagging,
   not merging. `archive/powervpsssh-2026-09-02` (pointing at remote's `895d424`) exists locally as of
   2026-09-03. It still needs `git push origin` (both the tag and local `main`) from wherever GitHub
   push access actually exists — this environment has none. DVC-checked too: the shipped model's actual
   weights are verified intact in local's cache; the other archived experiments' weights were never
   individually preserved by design (only the shipped one gets promoted), which a DVC remote couldn't
   have changed retroactively. (The box's dirty webapp files looked like a related risk at first
   glance; they check out fine — see §3d's correction — nothing to do there beyond an ordinary future
   deploy.)
2. **A4 — DVC remote.** Zero risk, ~30 minutes, and it protects both the raw captures and the 4.7 GB
   of archived checkpoints that a stray `dvc gc` would take.
3. **S1 + S2 — delete the dead guard, start measuring instead.** ~30 minutes and one line. S1 removes
   a refusal the docstring promises and the code cannot deliver (§3a); S2 begins logging `beans_across`
   on every request. S2 is the only thing in this cluster that gets worse by waiting — framing data
   from production cannot be backfilled, and it is what lets a future guard be set from the real
   distribution instead of from theory, which is what made the original unreachable.
4. **P2 + C2 — clean the record.** Rebuild the index (call `rebuild_index()` directly, **not** via
   `--replot-all` — see §3c), mark exp174 VOID, add `class_ids` to the archived config, fix the
   epoch/final denominator. Under two hours total, and it stops the experiment log from lying about
   two things it currently lies about. These touch no DVC stage, so the crop-staleness prerequisite
   in §5 does not apply — this is the cleanest place to start.
5. **C1 — DONE 2026-09-03, and it came back the other way round.** Built the missing harness
   (the benchmark measured a private copy of the estimator, so a band sweep through it was a
   guaranteed null) and swept. Raising `_K_LO` improves accuracy, rig-bias spread and correlation
   against ground truth; lowering it — the thing §3b proposed — makes all three worse. The band floor
   is suppressing 1/f spectral drag, not clipping bean signal. **Not adopted**: the ground truth is 3
   rigs and predates the 5 newer ones, and the gain needs `CALIBRATION_K` refit alongside. Follow-up
   is C1-next, not a config change.
6. **A5 + B-fix — make coverage and fold semantics legible**, so the next class or rig added doesn't
   require re-deriving §2 by hand and doesn't silently add degenerate folds to the default sweep.

Capture work (A1, A2) stays high-value but drops below these: it is a day of physical work whose
payoff is measured through a metric that §2b, §3b and §3f are currently distorting. Fix the ruler,
then measure.

---

## Verification record

Method: re-derived from the working tree on 2026-09-02 by execution, not by reading the previous
draft or prior session notes.

- **Coverage table** — enumerated `dataset/*/class_*/` directly. All cells reproduce.
- **MixStyle +0.1400, 9/9** — recomputed the paired deltas from `index.csv` (exp115-123 vs the plain
  beans_e80 arm exp60/61/62, 66/67/68, 84/85/86). Mean **+0.14001**, 9/9 positive. Exact.
- **H1 +0.070 / +0.099** — reproduce as exp166 − mean(exp157/158/159) = +0.0699 and
  exp167 − mean(exp160/161/162) = +0.0986. The design weaknesses in §2a were found by pulling the
  `train_rigs` list out of each run's `config.json`.
- **Pooling −0.035 / +0.020, TTA +0.0235, val saturation, 4.7 GB checkpoints, empty `.dvc/config`,
  `webapp/README.md` vs `app.py`** — all verified verbatim at source.
- **Frozen `.classes.txt` sidecar** — verified by *running* `config_for_checkpoint` on the shipped
  checkpoint: it resolves `classes_file` to the 9-row sidecar. The first draft's claim holds.
- **Scale guard dead** — derived from the estimator constants, then confirmed by running
  `scale_verdict` over 15 photos across 5 rigs. Minimum `beans_across` = 8.22, exactly the predicted
  `2.055 × _K_LO` floor.
- **39% band-floor pinning** — 192 photos, 24 per rig across all 8 rigs, seeded sample; confirmed as
  boundary-clipping by inspecting the weighted band profile on pinned cases.
- **index.csv duplicates** — 142 rows / 140 unique ids against 140 archive directories.
- **Missing commits** — `git cat-file -e` on all 140 archived `env.git_commit` values; 8 fail.
- **exp174 void** — `metrics.json` class counts (9 vs 10) against byte-identical `config.json`; root
  cause read from commit `96d6a9a`.
- **Fold denominators** — traced `macro_labels` from `train_baseline.py` into
  `metrics.compute_split_metrics` and `dataset.present_class_idxs`.

Not re-verified in this pass: the confusable-pair claims in §2d (taken from the experiment log and
spot-check notes, not recomputed from confusion matrices) and the capture-cost estimates in §4.

**Third pass (2026-09-02, remote state)** — everything below was checked by actually reaching
`powervpsssh` over SSH, not by reading local evidence about it:

- **Remote git log/status/branches** — `ssh powervpsssh 'cd ~/coffee-vision && git log/status/branch
  -a'`. Confirmed HEAD at `895d424`, 101 ahead of `origin/main`, and a dirty working tree (webapp files
  + `infer.py`).
- **The 8 commits are fetchable** — added `powervpsssh` as a real git remote
  (`powervpsssh:coffee-vision`) and ran `git fetch`; confirmed transferring real objects (~250-300MB).
- **Mutual divergence** — diffed remote's log against local's; remote's last sync point (`830dfd8`,
  "@ 96d6a9a") is behind local's current tip, while remote separately has 13 commits local lacks.
- **`git_commit` field semantics** — checked exp160/163/166/167's config.json commits against *local*
  git log (no fetch needed): each names the previous experiment's own commit, not its own. Pattern
  holds; not a defect.
- **DVC remote absence on remote** — `ssh powervpsssh 'cd ~/coffee-vision && cat .dvc/config; dvc
  remote list'` (after activating the remote's venv, since `dvc` isn't on its bare `PATH`): empty, same
  as local.
- **Remote DVC cache gaps** — `dvc status` on remote lists `phase8_best_random_erasing_0.5.pt`,
  `phase12_beans47_s7.pt`, `phase14_allrigs_s42.pt`, and `2026-07-24__first_pictures` as "not in cache."
- **Remote dataset/experiments inventory** — `ls dataset/`, `ls experiments/ | grep -c '^exp'`: same
  sessions as local's §2 table, 140 experiment directories, matching local exactly.
- **Uncommitted webapp state + live-production check** — `git diff` on the remote's dirty files;
  `systemctl is-active coffee-cv-web nginx` (both `active`); `systemctl show -p ActiveEnterTimestamp`
  (`2026-09-02 06:15 UTC`) against `stat` on `app.py` (last modified Aug 31) — the active process
  postdates the edit, which first read as risky. **Caught and corrected within this same pass**: pulled
  remote's actual file content (`ssh powervpsssh cat ...`) and ran a byte-for-byte local `diff` against
  the committed versions, rather than trusting `git diff --stat` alone. `app.py`, `infer.py`, and
  `README.md` came back identical; `setup_server.sh`/service/logrotate differ by exactly one already-
  committed local commit (`427d174`). No risk; see §3d point 4.
- **Photos-committed-to-git root cause** — `git show --stat 14f15a0` on remote: confirms a prior sync
  committed raw JPEGs directly (explains the fetch's size), already fixed forward in a later commit.
- **Fetch completion** — the backgrounded `git fetch powervpsssh` finished after this pass's other
  checks; all 8 commits (and their full surrounding history back to the shared base) now resolve with
  `git cat-file -e` in this local repo. They are fetched, not yet reconciled onto local `main` — see
  the reconciliation approach above.

**Fourth pass (2026-09-03, implementation-readiness audit)**

- **All 11 file:line citations re-checked** by `sed -n` on each — `infer.py:302`/`346`,
  `train_baseline.py:296`/`347`, `dataset.py:473`, `infer.py:269`, `webapp/app.py:36`,
  `webapp/README.md:35`, `README.md:51`/`53`, `experiments/README.md:24`. All resolve to the claimed
  content; no bit-rot across four revisions.
- **C1's blocker** — `grep` for `coffeecv` across `analysis/bean_scale/*.py` returns nothing, and
  `m0_fft_radial` returns `float(n / k * scale)` against production's `float(n / k * scale * CALIBRATION_K)`.
  Confirmed the benchmark cannot observe changes to `coffeecv/bean_scale.py`.
- **C2 confirmed genuinely one line** — `train_baseline.py:296` calls `compute_split_metrics(...)`
  without the `macro_labels=` kwarg that line 347 passes.
- **P2's invocation trap** — read `archive_experiment.main()`: `rebuild_index()` is reachable only via
  `--replot-all`, which regenerates charts for all 140 archives first.
- **S1's blast radius** — `grep` for `scale_verdict`/`beans_across`/`scale_ok`/`scale_note` across the
  repo: consumed only by `infer.py` itself, `webapp/app.py:214-215`, and the frontend's OR-chain at
  `index.html:386`. `eval_legacy_lens_photos.py` computes its own `beans_across` independently and is
  unaffected.
- **Crop staleness** — `dvc status`: all six `crop` stages report `changed deps: modified
  coffeecv/crop_tray.py`.
