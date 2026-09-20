# Next release model: plan

Written 2026-09-08, from the working tree (`git status` clean, local `main` in sync with
`origin/main`, `dvc status` shows no stale crop stages) plus `docs/dataset_training_reorg_plan.md`
and `experiments/index.csv` through exp203. Scope: get a shipped model that reflects the two
structural changes since the current production checkpoint was trained — the rig redefinition
(session → camera model) and the class_010 addition — not a re-litigation of the reorg plan itself.

## 1. Where things actually stand

- **Production (`webapp/app.py:36`) still serves `allrigs_oneplusmerged_s17.pt`** — exp171, shipped
  2026-08-30 (`c4b2660`). Trained on the **old session-rig definition** (5 sessions, `oneplus_combined`
  merge) with **9 classes**. It predates both structural changes below by construction.
- **Rig redefinition** (2026-09-03, `62df477`): a rig is now a camera model. Nine capture sessions
  fold into four `cam_*` rigs (`pixel`/`sony`/`oneplus`/`iphone`) via `merge_cam_*` DVC stages.
  `run_folds.RIGS` and `run_all_rigs` both already point at the four `cam_*` rigs — the tooling is
  ready, nothing else needs to change to use it.
- **class_010** (Indonesia, Java) added 2026-08-30. Present on `cam_pixel`/`cam_sony`/`cam_oneplus`
  (all 10 classes), **absent from `cam_iphone`** (still 8 classes — no class_008, no class_010; iPhone
  was never re-shot after the phone-photo shortfall or the class_010 addition).
- **The camera-rig leave-one-rig-out baseline is done**: exp200–203 (2026-09-04/05), single seed 42.
  Cross-rig mean **0.8025** over the three 10-class-denominator folds (pixel/sony/oneplus); iPhone's
  0.8537 is an 8-class number and not comparable to the other three. `cam_sony` transfers worst by
  ~2x (generalization gap 0.158 vs 0.048–0.082 for the others). exp200 (`cam_pixel`) and exp202
  (`cam_oneplus`) never converged inside the 100-epoch budget (best epoch 87/100 and 99/100).
- **What has not happened yet**: an all-rigs, no-held-out training run under the new rig definition
  and class list — the actual release-candidate weights, analogous to what exp168–173 were for the
  9-class/session-rig model. The only 10-class all-rigs run in existence is exp175, a 5-epoch smoke
  test predating the rig redefinition entirely (session rigs, `oneplus_combined`). **This is the
  single missing step between where the project is and a shippable 10-class model.**
- Nothing has been captured, trained, or shipped since exp203 (git log and `dataset/` are unchanged
  since 2026-09-05; the only commit since is a `crop_report.json` path fix).
- No blockers: git is clean, `dvc status` shows the six `crop@*` stages current (the staleness noted
  in the reorg plan is resolved), `run_all_rigs.py`'s own dirty/stale/branch guards would pass.

## 2. The next experiment: ship the 10-class camera-rig model

This is the direct answer to "new release model after restructuring + new class." One command,
mirroring exactly how exp171 was produced:

```
python -m coffeecv.run_all_rigs --seeds 42 123 7 17 99 256 --start-exp 204 \
    --epochs 100 --mixstyle-p 0.5 --mixstyle-mode agnostic --freeze-mode none --eta-min 1e-05 \
    --tag allrigs_cam
```

Every flag states the already-adopted config explicitly (`run_all_rigs` does not inherit
`params.yaml`'s resting value — see `feedback-cli-defaults-overwrite-config`). `RIGS` inside the
script already resolves to the four `cam_*` rigs, so this trains on all four with all 10 classes,
no held-out rig, exactly the config exp200–203 already validated for transfer.

**Cost**: exp168–173 (same shape, previous rig definition) each ran ~7.5–9.5h. Budget the same per
seed here; six seeds is ~2 days of wall clock. Options, in order of preference:
1. Run all 6 seeds, as precedent — pick the best by val macro-F1 once they land, same as exp171
   was chosen (`c4b2660`: picked from the first 4 of 6 on explicit instruction to ship best-so-far,
   never re-shipped once 172/173 landed since they were within noise).
2. If 2 days is too much right now, 3 seeds (42/123/7) is a defensible floor — halves the cost and
   still gives a same-precedent "pick the best" choice, just a shallower pool. Not a lever decision,
   so it doesn't need the project's usual 3-seed *paired* bar — this is a final fit, not a config
   claim (see `run_all_rigs.py`'s own docstring on this point).
3. Run on `powervpsssh` in the background rather than this environment, per
   `project-remote-compute` — confirm the remote is synced to local `main` first (it was 66 commits
   stale before exp200–203 and had to be caught up; check again before launching, not assumed).

**After training lands**, promote the chosen seed the same way exp171 was:
- Copy the winning `outputs/checkpoints/best.pt` → `models/allrigs_cam_s<seed>.pt` (+ `.pt.dvc`).
- Rebuild the OOD reference: `python -m coffeecv.build_ood_reference --checkpoint models/allrigs_cam_s<seed>.pt`.
- The frozen `.classes.txt` sidecar mechanism (`config_for_checkpoint`) should pick up all 10 classes
  automatically now that `discover_classes_multi`'s single-training-rig bug (`96d6a9a`) is fixed —
  verify by running `config_for_checkpoint` on the new checkpoint before trusting it, the same way the
  reorg plan verified it for exp171 rather than assuming.
- Spot-check through the real `infer.py` CLI path before committing (same precedent: 9/9 on the fixed
  legacy-lens set for exp171). Re-run `eval_legacy_lens_photos.py` against the new checkpoint.
- Update `webapp/app.py:36`'s `CHECKPOINT` constant, redeploy per the documented rsync process
  (`project-webapp-production`), and update `webapp/README.md` alongside it — the previous ship left
  that file naming a different checkpoint than `app.py` actually loaded; don't repeat it.

## 3. Firm up the headline cross-rig number before it goes in release notes

exp200–203 is single-seed and two of its four folds hit the epoch cap without converging. Neither
issue blocks shipping (the all-rigs fit in §2 doesn't need a cross-rig score by construction — the
folds already validated the *config*, not this specific fit), but both should be fixed before quoting
"0.8025" anywhere as *the* number for this release:

- **Re-run exp200 (`cam_pixel`) and exp202 (`cam_oneplus`) at a higher epoch budget** (e.g.
  `--epochs 150`) — both were still improving at cutoff (exp202's best landed at epoch 99/100 with
  val_loss also still falling). Cheapest way to know whether 0.8025 is a real number or an
  underestimate. ~2 runs, same cost per run as above.
- Optionally, once budget allows: repeat the camera-rig LORO sweep at 2–3 seeds to bring the
  headline number up to this project's usual evidence bar. Not required to ship — it would be
  characterizing the release, not deciding its config.

## 4. Known gap the release should state plainly, not hide

`cam_iphone` has no class_008 or class_010 photos. The shipped model will have **zero iPhone-framing
training signal for two of its ten classes** — those classes are represented only through the other
three cameras. This isn't fixable by anything short of a camera:

- **A1**: +88 iPhone photos across 8 classes (class_008 first, +20 — it's the only fully-absent one,
  not just thin; the rest need +8 to +10 each). See `project-iphone-photo-shortfall` for the exact
  per-class list.
- **A2**: class_010 on box and iPhone (~20 each) — currently unscored in any box- or iPhone-heldout
  fold at all.

Neither blocks this release; both are the natural input to the *next* one. Flag the iPhone/008/010
gap in whatever accompanies this release (model card, README) rather than letting it surface later
as a silent failure mode, the way class_008's absence silently made every iPhone cross-rig number in
the log an 8-class average next to 9-class ones.

## 5. Explicitly out of scope for this release

- **C1-ship** (`bean_k_lo=5` pitch-estimator retune) — was mid-flight when the rig redefinition
  landed and needs rethinking under the new rigs before it's worth relaunching. Ship at the currently
  adopted `bean_k_lo=4`; revisit as its own experiment.
- **B4** (006/007 confusion, the 002/005/006/007 diffuse cluster) — real, but a model-architecture
  question, not a blocker to shipping the restructured data.
- **A3** (fresh-scoop validation) — still the single largest unmeasured generalization axis in the
  project, independent of this release.
- **Provenance housekeeping** (the `archive/powervpsssh-2026-09-02` tag referenced in
  `project-experiment-record-gaps` doesn't currently exist in this repo, locally or on `origin` —
  worth recreating and pushing at some point, but unrelated to shipping this model).

## 6. Sequencing

```
(ready now — no blockers)
run_all_rigs, 4 cam rigs x 10 classes, seeds [42,123,7,(17,99,256)]  ── exp204+, ~2 days for 6 seeds
        │
        ├──> pick best seed by val macro-F1 ──> promote to models/ ──> OOD reference ──> legacy-lens
        │    spot-check ──> update webapp CHECKPOINT ──> deploy                                (§2)
        │
        └──> in parallel: re-run exp200/202 at --epochs 150 to firm up the 0.8025 headline    (§3)

A1 (iPhone +88) + A2 (class_010 on box/iPhone)  ── needs a camera; next release's input, not this one
C1-ship, B4, A3, tag/push housekeeping          ── independent, unblocked, lower priority
```
