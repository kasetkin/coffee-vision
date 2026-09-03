# Framing / scale guard — design plan

**Status: NOT READY TO EXECUTE.** This is a design stub, deliberately not a task list. Both of its
inputs are missing (see Entry criteria). It exists so the reasoning from 2026-09-03 is not lost and so
the next person does not re-derive it — not because work should start.

Scope: inference-time framing feedback in `coffeecv/infer.py` and the webapp. Split out of
`docs/dataset_training_reorg_plan.md`, where it lived as "B2" and was misrepresented as a half-day fix.

---

## Why this document exists

The original guard (`scale_verdict`) was **deleted on 2026-09-03**, not repaired. It had three
branches: refuse below `patch_beans_min` (4.0), warn below `patch_beans_max` (7.0), otherwise OK. But
`beans_across = 2.055 · k` with `k ∈ [4, 79]`, giving a hard floor of **8.22** — both thresholds sat
below it, so neither branch could ever fire, and there was no upper bound at all. Measured across 15
photos on 5 rigs: minimum exactly 8.22, guard returned OK every time. Full derivation in the reorg
plan's §3a.

The root cause matters more than the arithmetic: **the thresholds were picked from theory** — mirror
the trained 4.0-7.0 bean range — rather than from any measurement of what real frames actually produce.
Any replacement built the same way would fail the same way. That is why this plan's first entry
criterion is data, not code.

---

## Entry criteria — do not start before both hold

1. **C1c is done** (reorg plan §4): the bean-pitch estimator's search band has been re-benchmarked
   against the production code path, and `_K_LO`/`ANALYSIS_FRAC` are settled. `beans_across`'s
   reachable range is a direct function of `_K_LO` (floor `= 2.055 × _K_LO`), so any threshold picked
   before this would have to be picked again.
2. **Enough logged `beans_across` to see a distribution** (reorg plan S2). Not a fixed number of weeks
   — enough that the histogram has a visible shape, including its tails, across more than one device
   type. The point is to learn what real users' framing looks like, which no QA sweep over known rigs
   has ever told us (see `feedback-validate-against-real-distribution`).

---

## Open design questions

These are genuinely open. None should be answered from priors.

1. **Refuse, warn, or neither?** The original justified a hard refusal on "a wrong answer stated
   confidently is worse than no answer" (Phase 9: p=0.755 on a wrong class for an out-of-rig photo).
   Does that argument still hold now that the OOD guard exists and is the one refusal that actually
   fires?
2. **Are scale and OOD actually independent?** `infer.py`'s docstring claimed "two independent
   refusals." Plausibly OOD already catches most of what a scale guard would — a badly-framed photo is
   likely embedding-distant too. If the overlap is near-total, the honest answer is one guard, not two.
   **This is measurable against logged data once S2 accumulates**: correlate `beans_across` against
   `ood_median` per request. Do that before designing anything.
3. **What is a "too far" bound based on?** Two different answers: *mechanical* (the frame can no longer
   supply patches in the trained range) or *empirical* (accuracy measurably degrades past some
   framing distance). The second is the useful one and needs held-out data at varying distances, which
   does not currently exist in `dataset/`.
4. **Does the tray-crop detector make this partly moot?** `crop_to_bean_region` (shipped 2026-08-28)
   crops to the bean region before sizing, which normalizes framing. The guard was designed before
   that existed and has never been re-evaluated against it.
5. **Pre-capture or post-capture?** "Move the camera back" is far more useful *before* the shot than as
   a rejection after it. A live framing hint in the web UI is a different product than a refusal, and
   arguably the one the original was reaching for — its own rationale was that this is "the
   interpretable form of the guard — advice a user can act on."

---

## What is deliberately NOT here

- **Threshold numbers.** Picking them is the output of this plan, not an input to it.
- **Re-adding `scale_verdict` as it was.** If a guard returns, it should be designed from questions 1-5,
  not restored from git history.
- **Anything blocking the reorg plan.** Nothing in `docs/dataset_training_reorg_plan.md` is downstream
  of this document. C1 stands on its own (is bean-unit sizing rig-invariant?), and the capture work
  does not touch inference.

---

## Current state, for whoever picks this up

- No framing guard exists at inference. A too-far photo gets a confident answer with no warning. This
  is unchanged from before the deletion, since the old guard never fired — but it is now *honest* about
  it rather than advertised in a docstring.
- `beans_across` is still computed per photo (`infer.py:292`) and carried in the diag dict.
- The OOD guard is the only refusal, plus the "unmeasurable" path when pitch estimation throws.
- The frontend still has a `refused_scale` branch (`index.html:386`) in an OR-chain with
  `refused_ood`/`refused_unreadable`. Harmless and left in place, so re-enabling needs no frontend work.
