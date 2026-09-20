# LR scheduler: what the last 30 runs say, and a screening plan

Written 2026-09-20. Data: `history.json` + `config.json` + `metrics.json` of exp154–173 and exp200–209
(the last 30 real runs; exp174 is VOID and exp175 is a 5-epoch smoke test, both excluded). Every one of
them ran the same recipe: `CosineAnnealingLR(T_max=100, eta_min=1e-5)`, `early_stop_patience=20`,
AdamW, head lr 1e-3 / backbone lr 1e-5, batch 32, MixStyle 0.5, `freeze_mode=none`. exp160–162 are the
*same training runs* as exp157–159 evaluated on a different extra-held-out rig (byte-identical train/val
curves), so the window holds **27 unique train/val curves** and **15 independent fold runs** with
per-epoch cross-rig curves. Two data regimes: exp154–173 are 9-class session-rig, exp200–209 are
10-class camera-rig. Helpers are in `analysis/lr_scheduler/` (`load.py`: window + dedupe loader,
`plateau.py`: PyTorch-semantics trigger replay, `two_group_lr.py`: the F7 check); the other numbers are
one-off queries over `experiments/exp*/history.json`.

Supersedes the "ReduceLROnPlateau deferred until the dataset is bigger" note from 2026-08-24: that was
deferred because it breaks the `epochs`-is-`T_max` coupling. Section 4 handles that coupling instead of
avoiding it.

## Decisions (2026-09-20)

- **Data frozen at the 10-class commit.** Baselines exp200–203 stay as the paired A0; class_011 is
  integrated afterwards with whichever scheduler wins. Launch checklist in §5.
- **Stage 1 screens A3 (plateau) only.** A1, A2, A4 were offered and not selected; they stay documented as
  follow-ups and are **not built**. A3's floors are pinned to A0's effective per-group behaviour (2.1) so
  the single-arm comparison stays interpretable without A1.
- **Adoption bar: to be decided later.** Stage 1 only prunes, so nothing blocks it; §4's strict bar is a
  proposal, not an agreement.
- **Warm restart deferred**, "launch a different experiment" is the default at the floor (2.3).
- **Stage 0 is done** (2026-09-20, commit `a4023f7` on main): implemented, 15 unit tests, and smoke-verified
  on `powervpsssh` — a legacy cosine run reproduces exp200's epoch-1 row bit-for-bit, and a forced-plateau
  run drops the head LR after epochs 2/3/4, lands on the floor after 5 and stops after 7 with the backbone
  flat at 1e-5. Nothing has been launched for a result. Stage 1 awaits a separate go (§5 lists what must be
  done first).

## 1. What the curves show

**F1. Train loss falls forever; val loss goes flat by epoch ~50, it does not turn up.**
Binned means over all runs: train loss 0.176 (ep 31–40) → 0.096 (51–60) → 0.040 (91–100); val loss
0.284 → 0.266 → 0.263. Train starts *above* val (1.8–2.0 vs 1.1–1.6: augmentation, MixStyle and dropout
are on for train only), crosses below at epoch 11–33 (median 20), and ends 2.4–13.6x below (median 5.9x).
So the generalization gap widens 6x while val loss stays flat: memorisation of the finite patch set, not
classic val-loss overfitting.

**F2. The apparent val-loss "rise" is mostly noise.** Raw val loss ends a median 1.15x above its minimum.
A flat curve with the measured 5.5%/epoch jitter predicts 1.16x from noise alone (simulated). On
7-epoch-smoothed curves the real tail drift is a median +7% (IQR 1.03–1.11).

**F3. One fixed `T_max`, two convergence regimes.** 13/27 runs were stopped by patience (epochs 41–86);
14/27 hit the 100-epoch cap. Head LR at the best epoch is bimodal: ≤5e-5 in 10 runs, >2e-4 in 13.
The stopping rule is close to a coin flip: 13 of the 14 capped runs survived a no-new-best stretch of
13–20 epochs against a patience of 20, and two (exp200, exp205) were **one epoch** from being stopped —
exp205 then set its best at epoch 100. The 13 early stops land at a median head LR of 1.1e-4 (11% of
peak; only 4/13 at ≥25% of peak), so they miss the low-LR tail (≤5e-5, epochs 86–100) where 13 of the 14
capped runs set their best: a noise-driven cut of the tail, not a cut before annealing starts. Consistent
with the 2026-08 relaunch (epochs 80→100, patience 8→20 beat the old config 6/6 val).

**F4. Annealing is where the gain is — for val F1 reliably, for cross-rig F1 partly.** In the 14 capped
runs, best val F1 in epochs 61–100 beats epochs 1–60 by +0.006…+0.024 (mean +0.0125, **14/14**); the
non-max statistic (mean F1, ep 90–100 vs 50–60) is also higher in 14/14. Caveat: capped runs are a
survivor-biased set. On the headline cross-rig F1 (9 capped fold curves; exp157/160 share a training run)
the mean is +0.0100 but only 5/9 are real gains (+0.009…+0.028); 4/9 are flat (|Δ|≤0.0014).

**F5. Val loss is the wrong signal to steer by; val F1 is the right one, but noisy.**
- Within a run, val F1 and cross-rig F1 correlate r = 0.87–0.97 across epochs.
- On the recorded cross-rig curves, checkpointing at min val loss would have *lost* 0.0063 cross-rig F1
  vs the current raw-val-F1 peak (1 run better, 9 worse, 5 picking the same epoch; 15 independent runs).
- Cross-rig loss (7-smoothed) ends a median 1.13x above its minimum, up to 1.45x; on Sony and iPhone its
  raw minimum sits at epoch 3–18 while cross-rig F1 keeps climbing.
- The current val-peak pick is far from the best available epoch: mean oracle gap **+0.036** (0.010–0.089).
  Simply taking the **last epoch** beats it by +0.0167 (10+/5−, 15 independent runs); trailing-9 F1
  selection +0.0066 (9+/5−). Hypothesis-generating only — but it says peak-picking a jittery curve is
  mostly noise-chasing, and that the annealed end state is a better deliverable than a lucky epoch.

**F6. Noise floor.** Epoch-to-epoch val-F1 jitter σ≈0.005 (same at high and low LR), cross-rig F1 σ≈0.013.
Val is 1120–1520 patches from ~3 photos per class per rig, so patches are heavily correlated. PyTorch's
default plateau threshold (rel 1e-4) is ~50x below that jitter, and its `best` is a raw running max, so a
single lucky spike (exp164's epoch-21 F1, chosen as `best.pt`, run stopped at epoch 41 at 66% of peak LR)
makes every later epoch count as "bad". Any plateau logic needs a smoothed monitor and an absolute
threshold near the noise.

**F7. A side effect nobody designed: the adopted `eta_min=1e-5` flattens the backbone LR.** The optimizer
has two groups (head 1e-3, backbone 1e-5, `train_baseline.py:246-248`), and `CosineAnnealingLR`'s
`eta_min` is one *absolute* floor for all groups. Verified with the real scheduler:

| epoch | head LR (eta_min=0 / 1e-5) | backbone LR (eta_min=0) | backbone LR (eta_min=1e-5) |
|---|---|---|---|
| 1 | 1.0e-3 / 1.0e-3 | 1.0e-5 | 1.0e-5 |
| 50 | 5.2e-4 / 5.2e-4 | 5.2e-6 | **1.0e-5** |
| 80 | 1.0e-4 / 1.1e-4 | 1.0e-6 | **1.0e-5** |
| 100 | 2.5e-7 / 1.0e-5 | 2.5e-9 | **1.0e-5** |

Since 2026-08-25 the backbone has trained at a constant 1e-5 for all 100 epochs and the head:backbone
ratio collapses from 100:1 to 1:1. `history.json` logged only `param_groups[0]`, so no record showed it
(Stage 0 now logs `lr_head` and `lr_backbone`). This reframes the eta_min screen (exp151–156, +0.0175 mean, sony flat): its
"mechanism check" found the tail-noise theory unsupported, and this is a candidate explanation — the
backbone stopped being annealed to zero — but the data cannot separate head-floor from backbone-floor.
It also means a scheduler that decays the backbone is **not** comparable to the current baseline without a
control that decays it too (arm A1 below). A3 sidesteps this by pinning its floor to A0's per-group
behaviour (2.1).

**F8. What is not a problem.** Warmup: train loss is monotone through epoch 10 in 26/27 runs and the worst
val-loss spike is 1.09x. Compute waste: epochs after the best epoch are 15% of all epochs, the patience
tail on early-stopped runs 11% — a plateau scheduler's case rests on adaptivity and end-state quality, not
savings. Cost reference: ~233–258 s/epoch on the camera folds (6.5–7 h per 100-epoch fold run),
~300 s/epoch all-rigs (~8.3 h).

## 2. Hypotheses

The four arms are ordered by cost-of-implementation; **only A3 is selected** (see Decisions). A0 is the existing baseline (exp200–203 at seed 42;
cross-rig F1 0.8184 / 0.7587 / 0.8304 / 0.8537, iPhone is an 8-class number).

| Arm | Schedule | Tests | Why it is here |
|---|---|---|---|
| **A0** | current: cosine, `eta_min=1e-5` absolute | — | control; exists |
| **A1** | cosine, per-group floor (`floor ratio 0.01`: head→1e-5, backbone→1e-7) | F7 | the *intended* eta_min; the fair control for any scheduler that decays both groups. `LambdaLR` factor r+(1−r)(1+cos)/2 on each group's base LR |
| **A2** | A0 unchanged, `early_stop_patience=100` (no early stop) | F3 | zero-code arm: does truncation alone explain the gap? All runs go 100 epochs (~+15% compute). Report last-epoch too |
| **A3** | **ReduceLROnPlateau** on smoothed val macro-F1 (your proposal) | F1–F6 | adaptive decay; see 2.1 |
| **A4** | hold LR, one plateau-triggered cooldown, stop | F3, F4, F5 | see 2.2; my pick as most principled |

### 2.1 A3 — ReduceLROnPlateau, configured for this noise level

- Monitor: `val_macro_f1`, `mode=max`, **trailing mean of 5 epochs**; `threshold_mode=abs`, `threshold=0.003`.
  Not val loss (F5). Not the PyTorch defaults (F6).
- `patience=6` (PyTorch reduces on the 7th non-improving epoch), `cooldown=3`, `factor=0.3`.
- **Floor = `min(eta_min, base_lr)` per param group, chosen on purpose.** It is the same absolute floor
  `CosineAnnealingLR` applies to every group, so A3 shares A0's floor by construction and stays in sync if
  either changes (no separate `min_lr` knobs). The head decays to 1e-5; the backbone (base 1e-5) can never go
  below its own base, so it stays at 1e-5 exactly as it does under A0 (F7). That makes A3-vs-A0 differ in
  the head's schedule and the stop rule only, and needs no A1 to interpret. The list is built from the
  actual param groups, so `freeze_mode=full` (one group) works, and a non-positive `eta_min` is rejected
  (the floor is what starts the stop countdown). Letting the backbone decay to 1e-7 (A3-pg) is the deferred
  A1 question, not this arm.
- Stopping is owned by the scheduler and is a fixed budget, not another patience counter: after the step that
  lands the head on its floor, train exactly `floor_epochs=15` more epochs, then stop. (A draft used "stop
  when the smoothed F1 has not gained for 12 epochs at the floor"; on the cosine tails 39% of 12-epoch
  windows show no smoothed gain >0.003, so that rule would have been a second noise-triggered stop.) The old
  raw-F1 patience-20 stop is off on this path — the widely reported early-stop-vs-plateau patience conflict.
  `epochs` becomes a cap (150), not `T_max`.
- Offline replay on the 27 recorded curves (first trigger only — cosine's LR is still ~7e-4–1e-3 there, so
  it is faithful to a constant-LR run; later triggers are *not* replayable): with exactly these settings
  the first LR drop lands at epoch median 45 (range 30–65), none before epoch 20. For contrast, raw val
  loss with PyTorch defaults (patience 10, rel 1e-4) fires at median 59 (range 32–84).
- Budget arithmetic (not a simulation): 1e-3→1e-5 takes 4 drops at factor 0.3; each later drop needs ≥
  patience+1+cooldown = 10 epochs, so ≥ 45 + 3×10 + `floor_epochs` 15 ≈ 90 in the best case; plan on
  90–120 epochs. Hence the 150 cap.
- The knobs are **untuned guesses** from noise arithmetic, checked against the first-trigger replay above
  (factor 0.3, cooldown 3, `floor_epochs` 15 and the 150 cap have no evidence behind them beyond that).
  **Do not retune them on cross-rig results** from the screen.
- **Attribution limit.** A3 also drops the raw-F1 patience-20 stop, so a win cannot be split between
  "adaptive decay" and "no truncation". If A3 wins, A2 (zero-code: current cosine, patience off) is the
  cheap follow-up that splits them; if A3 loses, A2 is the first thing to check before abandoning plateau.

### 2.2 A4 — hold, then one cooldown (warmup-stable-decay style)

Constant LR until the same smoothed-F1 detector fires once, then a fixed 40-epoch cosine (or linear)
decay to the per-group floor, then stop; take the last epoch (also record the raw-peak epoch). Post-trigger
length is deterministic, so **every run gets the whole anneal to the floor** — today 13/27 stop at a median
head LR of 1.1e-4, before the ≤5e-5 tail — and the noisy patience counter can no longer truncate one. K=40 mirrors the current cosine's tail (LR < 3.5e-4 for the last 40
epochs, where the F4 gains occur). Expected total ≈ 45 + 40 ≈ 85–90 epochs, cap 150. Literature grounding:
WSD/"cooldown" schedules exist for exactly the unknown-horizon problem, and budgeted-training results say
decay to ≈0 by the end of the budget matters more than the decay's shape.

### 2.3 Deferred / rejected (with the trigger that would revive them)

| Idea | Verdict |
|---|---|
| **Plateau + warm restart at the floor** (your idea) | **Defer.** No evidence yet of a plateau *at* the floor: 13 of the 14 capped runs set their best in the last 15 epochs (head LR ≤5e-5), i.e. still improving there. A restart also needs multi-snapshot selection (the published restart gains come with snapshot ensembling) and doubles cost. Revive only if A3/A4 show a real share of runs sitting ≥0.01 below their own peak at the floor. |
| **"Launch a different experiment instead"** | **Agree, and it is the default at the floor**: stop, and spend the saved epochs on another seed — the repo already runs 6-seed all-rigs sweeps. |
| EMA / weight averaging at constant LR | Adjacent lever, not a scheduler. The anytime-training literature says it can substitute for annealing; needs BN-buffer handling and doubles eval cost. Screen after A1–A4 if none is adoptable. |
| Linear-to-zero instead of cosine | Shape-only variant; differences are small. Not worth a slot before A1–A4 resolve. |
| Schedule-Free AdamW | Documented BatchNorm caveat (running stats need recomputing at the averaged point); ResNet18 + MixStyle makes that risky. Low priority. |
| Warmup, OneCycle | Not supported by the curves (F8). |
| Val loss as plateau monitor; reload-best-weights on each drop | Val loss: contradicted by F5. Reload: would restore a noise-spike checkpoint. |

## 3. Implementation

Follows the project's lever pattern (`set_fold` in `run_folds.py`, `eta_min` as the template). Defaults
must reproduce every existing run bit-for-bit; `scheduler: cosine` keeps the legacy path untouched.

As implemented in Stage 0 (2026-09-20):

1. **`coffeecv/lr_schedules.py` (new)**: `build_scheduler(cfg, optimizer)` returning `CosineLR` (the legacy
   `CosineAnnealingLR`, verbatim; the raw-F1 patience stop stays in the loop) or `PlateauLR` (5-epoch
   smoothing + torch `ReduceLROnPlateau` + the floor countdown). Both expose `step(epoch, val_f1)`,
   `lrs()` (all groups), `should_stop()`, `owns_stopping`. `cosine_pg` (A1) and `hold_decay` (A4) are not
   built; add them only if their arms are picked up.
2. **`coffeecv/config.py`** (`RunConfig`, after `eta_min`): `scheduler` (default `cosine`),
   `plateau_smooth/threshold/patience/cooldown/factor`, `floor_epochs`. The knobs are inert under cosine.
3. **`coffeecv/train_baseline.py`**: builds via `build_scheduler`, prints the effective scheduler at start,
   logs `lr_head`, `lr_backbone` (keeping `lr` = head), `val_f1_smooth` and `lr_event` (the LR change that
   takes effect from the *next* epoch) into `history.json`, TensorBoard `lr/head` and `lr/backbone`; the raw-F1
   patience stop runs only when the scheduler does not own stopping.
4. **`coffeecv/plotting.py`**: `plot_training_curves` gets a third panel, learning rate vs epoch on a log
   axis with head and backbone lines. Old histories (only `lr`) draw the head line alone; the same function
   regenerates archived charts, so nothing is reconstructed. `dvc.yaml` `plots:` gains `lr` and
   `lr_backbone` entries (`dvc status` unchanged by them).
5. **`coffeecv/run_folds.py`**: `--scheduler` only (no knob flags: the knobs are fixed in `params.yaml`);
   default `None` = inherit (`feedback-cli-defaults-overwrite-config`), so state it on every invocation and
   restore `params.yaml` to `cosine` after a plateau sweep. Written with `\S+`, asserted in `set_fold`, in the
   post-condition guard, and in the archive note. **`run_all_rigs.py`**: `--scheduler` defaults to `cosine`
   and is written on every run, so a leftover `plateau` can never reach a shipping run.
6. **`params.yaml`**: new keys, `scheduler: cosine`, commented like `eta_min`.
7. **`tests/test_lr_schedules.py`** (plain `unittest`, 15 tests): cosine wrapper equals raw
   `CosineAnnealingLR` with the backbone flat at 1e-5 (pins F7); `PlateauLR` equals a hand-driven torch
   `ReduceLROnPlateau` on synthetic and recorded curves; first-drop epoch equals an independent pure-Python
   reference; head reaches 1e-5 while the backbone never moves; single-group optimizer builds; the stop fires
   exactly `floor_epochs` after the floor; bad settings are rejected; params.yaml knobs equal the dataclass
   defaults.
8. **Smoke** (`feedback-test-the-fix`) on `powervpsssh`, the real deploy sandbox: a legacy 2-epoch run that
   must reproduce exp200's epoch-1 row, and a forced-plateau run that exercises every drop, the floor clamp
   and the stop. Nothing is launched until the code is committed (`run_folds` refuses dirty source).

Optional saving, only if ≥3 plateau-family variants survive: A3 and A4 (and future variants) share an
identical constant-LR prefix until the first trigger, so snapshot full state (model, optimizer, RNG,
history) at the trigger and branch. Needs `--resume-from` and DVC provenance handling; roughly halves the
cost of each additional variant.

## 4. Evaluation protocol

Lever validation runs through `run_folds.py` on the four camera folds, never all-rigs
(`feedback-validate-levers-via-folds`). Primary metric: **cross-rig macro-F1 at the raw-val-peak
checkpoint** (the deployed rule, so history stays comparable), paired against A0 at the same fold and seed.
Recorded for free from `history.json`: cross-rig F1 at the last epoch and at trailing-9-selected epoch,
val F1, epochs used, fraction of runs reaching the floor.

| Stage | Runs | Single-machine cost | Gate |
|---|---|---|---|
| 0 | code, tests, smoke (§3) | ~0 h compute | data freeze done (§5 checklist) |
| 1 — mechanism check | **A3** × 4 folds × seed 42 (A0 = exp200–203 exists) | ≈ **26–32 h** (100-epoch fold runs are 6.5–7 h; A3 is expected at 90–120 epochs; worst case ≈ 43 h) | the pre-declared mechanism and safety checks below; Δ reported with its SE, **not gated on** |
| 2 — confirm | A0 at seeds 123, 7 (8 runs ≈ 52 h) + A3 at both seeds (≈ 52–64 h) | ≈ 104–116 h | **adoption bar undecided** (user, 2026-09-20). Proposal: ≥9/12 paired positive, mean Δ ≥ +0.005, no fold-mean below −0.01 |
| 3 — ship | 6-seed all-rigs with the winner | ~50 h | only if adopted, and on the then-current (11-class) data |

**Why Stage 1 is not a significance test.** Seed-to-seed sd of cross-rig F1 on the same fold and recipe is a
median 0.029 (17 fold×recipe groups with 3 seeds each; IQR 0.016–0.038; Sony 0.035–0.066), so a paired
per-fold delta has sd ≈ 0.02–0.04 and the SE of a 4-fold mean is 0.010–0.021. A simulated gate of "mean Δ ≥
+0.005, ≥3/4 folds positive, none below −0.02" passes 15–20% of the time at a true Δ of 0 and only 27–52% at a
true +0.01; effects below ~0.02–0.04 are invisible at 4 folds. A plateau-vs-cosine difference should be a
fraction of the +0.010 annealing effect (F4) — an inference, not a measurement. Stage 1's 4 runs are the
seed-42 third of Stage 2's 12 pairs (SE ≈ 0.006–0.012), so nothing is wasted.

**Stage 1 acceptance (pre-declared):**
1. first LR drop between epochs 20 and 80 in every fold (replay predicted median 45);
2. the head reaches the floor before the cap in ≥3 of 4 folds — a run that hits 150 un-annealed is a
   *mechanism* failure that points to A4, not a verdict on plateau;
3. epochs used 60–150, no stop before epoch 50;
4. no fold more than 0.06 below its A0 (≈2 paired sd): a tripwire for breakage, not a test.

Reported beside them, per fold and mean, with the SE stated: Δ cross-rig F1 at the val-peak pick, Δ mean
cross-rig F1 over the last 10 epochs, cross-rig F1 at the last epoch. Whether to run Stage 2, and the
adoption bar, stay with the user; it may end inconclusive, in which case the case for plateau would rest on
adaptivity and on dropping the `epochs`-is-`T_max` coupling. One machine runs one fold at a time; a second
machine halves wall-clock, not compute.

## 5. Risks and open items

- **Data freeze — mostly done, one item left.** The class_011 work (`011;Peru,Minka` in `classes.txt`, plus
  the three `dataset/2026-09-11__{oneplus,pixel,sony}.dvc` pointers, 20 photos each, none for iPhone) is
  committed on branch `class-011-peru-minka` (`1a611c0`), off main's `78083c7`. Main's `classes.txt` is
  back to 10 lines, so the tracked tree is frozen at 10 classes and exp200–203 stay paired. The user then
  moved the three photo folders to `dataset_new_ignored/` at the repo root (20 files each, intact,
  referenced by no code). Git did not ignore that folder, and
  `run_folds.dirty_provenance_paths()` (`:202`) refuses any dirty file outside `params.yaml`, `dvc.lock`,
  `outputs/`, `experiments/`. **Done 2026-09-20:** `/dataset_new_ignored/` is in `.git/info/exclude` (local,
  not committed) and `git status` no longer lists it. **Still to do:** `--exclude /dataset_new_ignored/` on
  the deploy rsync (236 MB the VM never reads), and the other untracked entries (this doc,
  `analysis/lr_scheduler/`, `docs/release_model_plan.md`) still trip the gate until committed or excluded.
  `classes.txt` and the cropped data must agree: exp174 once silently trained 9 of its 10 classes. On
  `class-011-peru-minka` the `.dvc` pointers now name folders that no longer exist; `dvc checkout` there
  restores them from the local DVC cache. The branch is not pushed and the photos are not `dvc push`ed.
- **The VM's git state must be reconciled before Stage 1 (provenance).** `powervpsssh:~/coffee-vision` sits at
  `26f1497` with seven uncommitted files (`coffeecv/infer.py`, `webapp/app.py`, `webapp/README.md`, two
  `models/allrigs_cam_s123.ood_*.json`, untracked `coffeecv/{fit_ood_probe,ood_eval}.py`). Checked
  2026-09-20: all seven are **byte-for-byte local HEAD's content** — the OOD-guard commits (`3cbc790`,
  `82d9264`, `78083c7`) were rsync-deployed, never committed there. Consequences if left as is: `run_folds`
  needs `--allow-dirty`, every fold's archive note carries "uncommitted source at launch", and each
  `config.json` records `26f1497`, a commit that contains neither the OOD guard nor the scheduler code
  (`feedback-experiment-provenance`). Safe fix: fetch local main into the VM and fast-forward it. The dirt is
  identical to the target, so discarding it loses nothing, but the three untracked files must be removed
  first (git refuses to overwrite them) and production imports those files, so do it while the site is
  stopped (authorized during experiments). Not done yet: it is Stage 1 preparation, after the Stage 0 commit.
- **The backbone-flat question stays open.** A3 keeps A0's backbone behaviour on purpose (2.1), so this
  screen says nothing about whether annealing the backbone would help. If A3 is ambiguous or wins, A1
  (`cosine_pg`, backbone → 1e-7) is what tells you whether the deployed recipe's real lever is the head
  floor or the backbone.
- **Selection rule is a separate lever.** Kept fixed for the primary comparison; F5 says it is worth its own
  screen (last-epoch / smoothed pick) and it costs no training on fold runs — the curves are already there.
- **`epochs` no longer means the same thing across arms.** Compare on outcome plus epochs used.
- **Plateau knobs are extra degrees of freedom on a σ≈0.005 metric.** Fixed before launch and untuned (2.1).
- **`params.yaml` rests at `scheduler: cosine`.** `run_folds --scheduler plateau` rewrites it, so restore it in
  its own commit after the sweep; `run_all_rigs` re-states `cosine` on every run as a second guard.

## Sources

- PyTorch [ReduceLROnPlateau](https://docs.pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) (defaults, `threshold_mode`, per-group `min_lr`); off-by-one and patience behaviour: [#11305](https://github.com/pytorch/pytorch/issues/11305), [#119763](https://github.com/pytorch/pytorch/issues/119763)
- Smoothing a noisy monitored metric for plateau/early-stop callbacks: [tensorflow/addons#2498](https://github.com/tensorflow/addons/issues/2498); patience conflict: [keras#9541](https://github.com/keras-team/keras/issues/9541)
- WSD / cooldown: [River Valley perspective](https://arxiv.org/abs/2410.05192), [cooldown dynamics](https://arxiv.org/abs/2508.01483); horizon-free + averaging: [Anytime Pretraining](https://arxiv.org/abs/2602.03702)
- [Budgeted Training](https://arxiv.org/abs/1905.04753) (linear decay to ~0, budgeted convergence); [Cosine and exponential step sizes vs plateau](https://arxiv.org/abs/2002.05273)
- [SGDR](https://arxiv.org/abs/1608.03983) (restarts, snapshot ensembles); [Schedule-Free](https://arxiv.org/abs/2405.15682) (BatchNorm caveat)
