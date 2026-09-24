# Adding DINOv2 to `coffeecv`: integration, comparison protocol, and experiment plan

Status: PLAN, written 2026-09-21. Nothing here is implemented. Supersedes nothing; it is the
concrete follow-up to `docs/architecture_screen_plan.md` §8 R1, which is the screen that
motivated it (`analysis/architecture_screen/`).

Prerequisite that gated the whole document when it was written: Stage 1 (exp210-213, the plateau
LR screen) was still running on `powervpsssh`, and per `docs/lr_scheduler_plan.md` nothing could be
committed locally or pushed to the VM until it finished. **Lifted 2026-09-24:** Stage 1 finished on
2026-09-21 and the scheduler task is closed (`docs/lr_scheduler_plan.md` §6), so every "once Stage 1
frees the repo" condition below is met. Every step in Phase 0 and Phase 1 below runs on the
workstation. Phase 3 is the first step that needs the VM.

---

## 0. What this is, and what the evidence for it currently is

The frozen-feature screen measured, on identical patches with a frozen backbone and one logistic
regression, leave-one-camera-out cross-rig macro-F1:

| backbone | 4-fold mean | 3 ten-class folds |
|---|---|---|
| resnet18_in1k (current) | 0.4645 | 0.4938 |
| convnext_tiny_in1k | 0.6691 | 0.7013 |
| **dinov2_vits14** | **0.8324** | **0.8511** |

and attributed the gain to the DINO self-supervised objective rather than the recipe
(`resnet18.a1_in1k` is *worse*, 0.4496) or the transformer (supervised ViT-S/16 reaches only
0.5534). The deployed fine-tuned ResNet18 + MixStyle + TTA pipeline scores 0.8025 on the same
three ten-class folds.

**That screen is not admissible evidence for a change, and this plan does not treat it as such.**
It deviates from a real fold in three ways, all of which inflate it: it trains on `split="all"`
of the three training rigs rather than the 70% photo split; it does not exclude an absent class
from the macro average, so cam_iphone carries a phantom f1=0; and it scores patch-wise with no
photo pooling. Phase 0 exists to remove all three before a single line of `coffeecv/` is edited.

---

## 0b. Which generation of backbone — checked 2026-09-21

**DINOv2 is from April 2023.** It is three and a half years old, which is worth knowing before
building a plan on it. The landscape as of today:

| model | released | smallest variant | embed dim | licence |
|---|---|---|---|---|
| DINOv2 | Apr 2023 | ViT-S/**14**, 22.1M | 384 | **Apache-2.0** |
| **DINOv3** | Aug 2025 | ViT-S/**16**, 21.6M | 384 | custom `dinov3-license`, **gated** |
| SigLIP 2 | Feb 2025 | ViT-B/16, **86M** | 768 | Apache-2.0 |
| AIMv2 | Nov 2024 | ViT-L class | — | Apple sample-code terms |
| Perception Encoder | Apr 2025 | large | — | Meta custom |

There is **no DINOv4**. DINOv3 remains the current state of the art for self-supervised vision
features: trained on 1.7B images with a 7B-parameter teacher (against DINOv2's 142M / ~1B),
distilled down to ViT-S/B/L/H+ and ConvNeXt variants, and reported to match or exceed SigLIP 2
and Perception Encoder on classification while widening the gap on dense tasks.

### Why "the most powerful ViT" is the wrong question here

The headline models — DINOv3 ViT-7B, ViT-H+, PE-Core-G, SigLIP 2 g — are 300M to 7B parameters.
This project has no GPU: a frozen ViT-S/14 already costs 47 ms/img and the deployed path runs 40
patches per photo. A ViT-L would be roughly 15x that. The only question that matters is **which
~20M-parameter frozen encoder is best**, and on that question the field is narrow: DINOv3 ViT-S/16
and DINOv2 ViT-S/14 are the two serious entries. SigLIP 2's smallest is 86M — four times the
budget — and AIMv2 is reported to underperform on spatial tasks, which is what bean texture is.

### DINOv3 ViT-S/16 is a near drop-in, and probably *faster*

Same parameter class (21.6M vs 22.1M) and, decisively for this plan, **the same 384-d embedding
width** — so every OOD sidecar rebuild, the `embedding_dim_of` check and the
`DinoV2Classifier` head shape are all unchanged. The differences are small and mostly favourable:

- **Patch 16, not 14.** At 224 px that is 14x14 = **196 tokens instead of 256**, and DINOv3-S is
  indeed faster — but by less than the token count suggests. **Measured 2026-09-21** (batch 32,
  224 px, 6 threads): dinov2_vits14 **48.8 ms/img**, dinov3_vits16 **44.4 ms/img** — a **9%**
  saving, not the ~23% a quadratic-attention argument predicts. The FFN cost is linear in tokens
  and dominates at this model size, so the quadratic term is diluted. Still the right direction:
  newer, stronger, and slightly cheaper.
- The input guard becomes `patch_resize % 16 == 0`. 224 satisfies both, so the adopted config
  needs no change either way.
- It uses RoPE and 4 register tokens; `AutoModel`/`AutoImageProcessor` load it, so the vendoring
  story in §1-A.1 changes from a `torch.hub` zipball to an HF snapshot.

### What this does and does not change about the plan

It does **not** reopen the architecture question. The screen's finding was that the *pretrained
representation* is the dominant lever (+0.368 over ResNet18, 4/4 folds, and +0.279 over a
same-size supervised ViT-S). Which generation of DINO supplies it is a second-order question.

It does add **one arm to Phase 0**, which costs about 40 minutes of probe time and is exactly the
kind of question this project settles by measuring rather than arguing. If DINOv3 ViT-S/16 wins
on the corrected protocol and the licence is acceptable, use it; if the margin over DINOv2 is
inside the noise, take the Apache-2.0 model and never think about the licence again.

### The licence, after actually reading it (2026-09-21)

Source: `LICENSE.md` in `facebookresearch/dinov3`, "Last Updated: August 19, 2025". Meta's
downloads page carries no terms — it is site navigation. This is a reading of the text, not legal
advice; anything consequential wants a real opinion.

**Two earlier drafts of this document overstated the restriction.** The first said DINOv3 "must
not enter this repo"; the second softened that but still treated the licence as a significant
cost. Having read it: **it is not a meaningful blocker for this project.** Recording that plainly
here so it is not re-argued a third time.

What the licence permits, explicitly: a worldwide, royalty-free licence to **use, reproduce,
distribute, copy, create derivative works of, and modify** the weights (§1a) — with no commercial
carve-out. §5a says you **own** the derivative works and modifications you make.

What it notably does **not** contain, in contrast to Llama-family licences: no Acceptable Use
Policy document, no monthly-active-user threshold, no "Built with…" naming requirement, no
restriction on what you may do with model *outputs*, and no clause forbidding using outputs to
train other models.

The five real differences from Apache-2.0, in rough order of relevance here:

1. **Sharing a derivative carries the licence forward** (§1b-i). A fine-tuned DINOv3 checkpoint
   made public must be distributed under this same Agreement, with a copy attached. Running a
   service on it is *not* distribution — §1b-i is about making the materials available to a third
   party, and `/classify` returns predictions, not weights.
2. **Meta may modify the terms unilaterally** (§8), effective immediately, with continued use
   counting as acceptance. Apache-2.0 cannot change under you.
3. **California law and exclusive California jurisdiction** (§7) — worth a glance for a
   Helsinki-based project, though it is theoretical at this scale.
4. **Patent-style termination plus a broad indemnity** (§5b): suing Meta over IP in the materials
   terminates the licence, and you indemnify Meta against third-party claims arising from your
   use or distribution. Wider than Apache-2.0's equivalent.
5. **Gated download** — accepting terms and sharing contact information with Meta, which is a
   distribution mechanism rather than a licence term.

Prohibited uses are narrow and irrelevant to coffee beans: military/warfare, nuclear, espionage,
weapons, anything under ITAR or Trade Controls, plus no reverse-engineering of the materials.

**The one genuine grey area is §7's fallback plan.** Distilling DINOv3 into a serving ResNet18
is not prohibited by any clause — there is no outputs restriction. But whether the student's
weights are a "derivative work of the DINO Materials" (and so inherit §1b-i if published) is
unsettled: the student is trained against the teacher's outputs, not derived from its weights.
If the distillation path is taken *and* the student is published, that question needs answering
rather than assuming. Using the student privately raises it either way.

### Will it be relicensed? Checked the primary sources, 2026-09-21

Short answer: **no plans announced, and Meta has not commented in 13 months.** Do not plan around
a relicense.

The paper trail, from the GitHub API rather than secondhand reporting:

| | |
|---|---|
| `LICENSE.md` commits | **2, both August 2025** (initial import 2025-08-14, a section-reference fix 2025-08-19). Unchanged since. |
| Issue #31, "[request] Release under a more standard license (e.g. Apache 2.0)" | opened 2025-08-15, **closed after 2 days**, 29 reactions. Meta's reply: *"Noted and thanks for the feedback. I will close though as we are more generally collecting feedback on the license in #52"* |
| Issue #52, "Feedback / support on license" | **opened by Meta**, 2025-08-17, still **open**, 26 reactions, last activity 2026-09-04 |
| Meta comments in #52 | **zero.** All 9 comments are from outside contributors, spanning Oct 2025 to Sep 2026. |

(An earlier draft of this document said issue #31 was "still open". It is closed; #52 is the open
one.)

The community side of #52 is a steady, unanswered drumbeat — "could we maybe get a comment on if
the release under a standard license is something that is being considered?" (Oct 2025), "any
updates on this?" (Dec 2025), "any update on this topic?" (Jul 2026) — plus a detailed five-point
commercial-compliance question from a Korean company in Aug 2026 that also went unanswered. The
most-reacted comment (18) is simply "+1 … it would be great to see DINOv3 released under the
Apache 2.0 license as well".

**The DINOv2 precedent exists but is not repeating.** The same person who filed DINOv3 #31
(`giacomov`) filed the equivalent DINOv2 request (#128) on 2023-06-23; DINOv2's LICENSE was
changed on 2023-08-31 — **69 days**. The DINOv3 request is now at **402 days** with no movement.
So relicensing is demonstrably possible at Meta, and demonstrably not happening here.

**Meta's one substantive clarification** (issue #28, 2025-08-21, patricklabatut) is worth quoting
because it settles the question this project actually cares about:

> The DINOv3 License allows for commercial use. You are only required to attach the DINOv3
> License if you redistribute copies or derivatives of DINOv3 model weights or code. You do not
> need to include the license if you are incorporating DINO into a commercial offering like an
> API or SaaS offering, but such offering must be in compliance with the DINOv3 license.

That is an on-record Meta statement that **serving it behind an API — the `coffee.kasetkin.com`
case — does not require attaching the licence.** It confirms the reading in the previous section
from the horse's mouth.

**What researchers object to** is not permission but *standardisation*. The recurring complaint,
put best in #52: *"Overall the licence it is under is reasonable but any unique licence makes it
difficult to use without getting a legal department involved."* The release also sits inside the
broader "open-washing" argument — Meta's blog calls it an open-source release, but the licence is
not OSI-approved and would not meet the OSI's Open Source AI Definition. A concrete downstream
cost showed up in #52 on 2026-09-04: Microsoft's TRELLIS2 is "billed as MIT Licenced, but it
actually uses DINOv3 for a lot of its architecture, so it's not really usable under a MiT licence
at all". That is §1b-i's copyleft propagating exactly as predicted.

For a solo project serving predictions from a private checkpoint, none of this bites. It bites
libraries and vendors who redistribute.

**Revised verdict.** The licence does not block using DINOv3 ViT-S/16 here, serving it at
`coffee.kasetkin.com`, or keeping weights in the private DVC remote. Apache-2.0 remains
marginally cleaner — no unilateral amendment, no California forum, no copyleft on a published
checkpoint — so the tie-breaker rule stands: **if DINOv2 and DINOv3 measure within noise of each
other, take DINOv2.** But choose on the measurement, not on the licence.

---

## Phase 0 — Protocol-matched re-measurement. **Gate. ~2 hours, workstation only, no code surgery.**

Modify `analysis/architecture_screen/xrig_probe.py` only — nothing under `coffeecv/`:

1. Build the training-rig patch sets with `split="train"` (and the photo_frac from `params.yaml`)
   instead of `split="all"`, so the arm sees the same 70% of photos a real fold sees.
2. Score with the project's own macro handling. `MultiPhotoPatchDataset` already exposes
   `present_class_idxs`; pass those labels to `f1_score(..., labels=...)` so cam_iphone is a true
   9-class average rather than a 10-class one with a phantom zero.
3. Add photo-level pooling alongside the patch-level number, mirroring
   `coffeecv/photo_pooling_eval.py`, so the reported figure is comparable to what `/classify`
   returns.
4. Run the same protocol on the **ImageNet ResNet18 control** in the same script, so Phase 0
   produces a paired frozen-vs-frozen table, not a DINOv2 number in isolation.
4b. Add **DINOv3 ViT-S/16** as a fourth arm (§0b). Same 384-d width, same parameter class, and
   196 tokens at 224 against DINOv2's 256 — so it may be both stronger and cheaper. Needs an HF
   account to accept the gated terms. If the margin over DINOv2 is inside the noise, stay on
   Apache-2.0 DINOv2 and record the decision.
5. Sweep seeds 42/123/7/17/99/256. This is free — a frozen backbone has no backward pass — and
   it is the first time in this project's history a lever can be given a six-seed answer in an
   afternoon instead of a week.

**Gate condition.** Proceed to Phase 1 only if, under the corrected protocol, frozen DINOv2 still
clears frozen ResNet18 by a margin that is sign-consistent across all folds and seeds *and* the
absolute figure is within ~0.05 of the deployed 0.8025. If the corrected number collapses toward,
say, 0.65, DINOv2 is still interesting but is no longer a candidate to replace the classifier,
and this plan should be rewritten around distillation (§7) instead of replacement.

Record the result in `analysis/architecture_screen/README.md` whichever way it goes. A negative
here is worth as much as a positive and costs two hours.

---

## Phase 1 — Code layout: a separate path first, integration only if it earns it

**Decision (2026-09-21): the DINOv2 arm starts as its own top-level package, `coffeecv_dino/`,
and touches nothing under `coffeecv/`.** Integration into `build_model` is deferred to Phase 1-B
and happens only if Screen A passes. Rationale below, because the seam matters more than the
decision does.

### Why separate, concretely

- **`dvc.yaml`'s `train` stage lists `coffeecv` as a dep.** Any edit anywhere under that directory
  invalidates the stage and marks every ResNet18 baseline stale. Iterating on a DINOv2 branch
  inside `coffeecv/` would repeatedly put the comparison's own baseline into a "needs re-run"
  state. This alone settles the location question.
- **`params.yaml` is shared, rewritten state.** Both sweep drivers rewrite it per fold, and it
  rests at the shipping config by design — it is documentation as much as configuration. Three
  inert `dinov2_*` keys sitting in it during a screen are noise in the one file a reader consults
  to learn what the project currently does. And `RunConfig.from_params_yaml` raises on unknown
  keys by design, so those keys and the dataclass must move in lockstep forever after.
- **The frozen regime is genuinely different work**, not a variant: no backward pass through the
  backbone, an embedding cache, a head that converges in a handful of epochs. Threading that
  through `train_baseline.py`'s epoch/scheduler/early-stop loop means `if dinov2` branches in code
  that currently has none.

### The seam — what `coffeecv_dino/` may own, and what it must import

This is the part that decides whether the separation helps or quietly destroys the experiment.

**Must import from `coffeecv/`, never reimplement:**

| module | why it cannot be copied |
|---|---|
| `dataset.MultiPhotoPatchDataset` | the comparison's entire value is that both arms see **byte-identical patches** from the same seeded RNG. A private patch sampler makes this two pipelines being compared, not two backbones. |
| `transforms.build_{train,eval}_transform` | DINOv2's normalisation is already identical; a second copy is a place for them to drift apart |
| `bean_scale` | patch sizing must match the baseline exactly, and train/inference parity is a standing project rule |
| `metrics.compute_split_metrics` | `macro_labels` is what stops cam_iphone's absent class becoming a phantom f1=0. A hand-rolled macro will not match, and the numbers become incomparable. |
| `config.RunConfig` | for the shared knobs (patch geometry, splits, seed). Extend by composition — a `DinoConfig` holding a `RunConfig` — not by adding fields to it. |
| `archive_experiment.archive` | it is a file contract: write `outputs/metrics.json` and `outputs/config.json` in the usual shape and the run lands in `experiments/index.csv` like any other. A screen whose results are not in the project record is a screen that will be re-argued later. |

**May own:**

backbone construction and weight vendoring; the embedding cache; the fit loop; its own CLI driver
and its own small config file.

### The failure mode this seam exists to prevent — it has already happened here, twice

`analysis/bean_scale/estimators.py` does not import `coffeecv`. The consequence, recorded in the
reorg plan's fourth pass: editing the production estimator and re-running the benchmark returned
**byte-identical output**, because the benchmark was scoring a private copy. The same directory's
`m0_fft_radial` was "a frozen private copy hardcoding `lo, hi = 4, 80`" that had silently drifted
from production.

So "separate folder" is not free in this repo — it has a specific, twice-realised failure mode.
The rule that avoids it: **a separate path may own behaviour that is genuinely different; it must
never own a second copy of behaviour that has to stay identical.** Enforce it mechanically — a
test asserting `coffeecv_dino` defines no patch-sampling or metric function of its own is cheap
and would have caught both prior incidents.

### You already have this, and it already got the seam right

`analysis/architecture_screen/xrig_probe.py` imports `MultiPhotoPatchDataset` and
`build_eval_transform` from `coffeecv` and owns only its classifier and scoring loop. That is
exactly the shape described above. Phase 0 is therefore not new construction — it is promoting a
working script into a package and adding the archive contract.

### Switching between arms

During the screen, **do not add a flag to `run_folds.py`.** That script owns params.yaml
rewriting, per-fold git commits and the provenance gate, none of which the frozen arm uses; a
separate driver (`python -m coffeecv_dino.run_folds`) is cleaner and cannot destabilise the
baseline sweeps. The `--model-name` pass-through in §1-B.4 belongs to Phase 1-B,
when there is something worth integrating.

### Proposed layout

```
coffeecv_dino/
  __init__.py
  arm.py          # DinoV2Arm implementing coffeecv.arms.Arm -- moves to coffeecv/arms/dinov2.py at 1-B.2
  backbone.py     # vendored-weight loading, sha256 check, patch_resize % 14 assert
  cache.py        # embedding cache; key = (weights sha, resize, seed, rig, split, fold, unfreeze_n)
  evaluate.py     # thin: calls coffeecv.metrics.compute_split_metrics, writes outputs/metrics.json
  run_folds.py    # own driver: leave-one-camera-out rotation, archives via coffeecv.archive_experiment
  config.py       # DinoConfig, HOLDING a coffeecv RunConfig rather than extending it
  dino_params.yaml

Write `arm.py` against the `Arm` protocol (§1-B.1) from day one even though nothing enforces it
during the screen: it is the difference between Phase 1-B.2 being a file move and being a
rewrite.
```

Nothing above writes to `params.yaml`, `coffeecv/`, or `dvc.yaml`. The only shared mutable
surface is `outputs/`, which the archive contract already expects a run to populate — so run
Screen A when no ResNet18 sweep is mid-flight, or point it at its own outputs directory.

### 1-A.1. Vendor the weights; do not depend on the network mid-sweep

`torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` fetches a GitHub zipball *and* an
84 MB checkpoint from `dl.fbaipublicfiles.com`. A sweep that resolves either at run time is not
reproducible and will fail on a box without egress.

- Download once, then save the backbone `state_dict()` to `models/dinov2_vits14_pretrain.pth`,
  `dvc add` it, and commit the `.dvc` pointer. DINOv2 is Apache-2.0, so vendoring, serving and
  redistributing are all unambiguously permitted.

  On DINOv3 the licence was read in full on 2026-09-21; see §0b. Short version: it permits use,
  modification, derivative works and commercial use, you own your derivatives, and serving
  predictions is not distribution. It is not a blocker. Apache-2.0 is still marginally cleaner,
  so it breaks a tie — it does not decide the question.
- Record the file's sha256 in the run config so a swapped weight file is visible in the archive,
  the same way `bean_k_lo`/`bean_calibration_k` were promoted from module constants to config
  after they turned out to be silently unrecorded.
- Pin the hub repo checkout too (`torch.hub.load(..., source=...)` against a local clone at a
  recorded commit), or vendor the ~15 ViT source files directly. The architecture definition is
  part of what makes a checkpoint loadable; a floating `main` is the same class of provenance
  hole Phase 13 fell into.

Do **not** substitute `timm`'s `vit_small_patch14_dinov2.lvd142m` as a shortcut. It defaults to
`img_size=518` with a different positional-encoding configuration, so at this project's 224 px it
would not reproduce the embeddings the screen measured. If it is ever used, it needs its own
equivalence check first.

### 1-A.1b. Loading DINOv3: what `facebookresearch/dinov3` actually requires

**Yes, that repo is sufficient** — verified on 2026-09-21 by cloning it and loading
`models_pretrained/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` end to end. Three
findings, all measured.

#### `hubconf.py` does not work, and the failure is misleading

The obvious entry point fails:

```python
torch.hub.load(REPO, "dinov3_vits16", source="local", weights=W)
# ModuleNotFoundError: No module named 'torchmetrics'
```

`hubconf.py` imports the segmentors, which import `dinov3.eval`, which imports `torchmetrics`.
None of that is a backbone dependency — it is the evaluation stack. Importing the backbone module
directly skips it entirely:

```python
sys.path.insert(0, DINOV3_REPO)
from dinov3.hub.backbones import dinov3_vits16     # not via hubconf.py
```

Worth knowing before someone concludes DINOv3 "needs torchmetrics" and installs a dependency tree
to satisfy a module the inference path never touches.

#### The footprint is small: 19 modules, no third-party beyond torch

Building the backbone imports exactly four subpackages — `dinov3/hub`, `dinov3/layers`,
`dinov3/models`, `dinov3/utils` — 19 modules, **26 `.py` files, ~400 KB of the 2.3 MB package**,
and **nothing outside torch**. (`requirements.txt` lists ftfy, omegaconf, submitit, torchmetrics
and friends; those belong to training and evaluation, not to loading a backbone.)

#### Load the state dict yourself; do not pass `weights=<path>`

`dinov3_vits16(weights="/path/to.pth")` treats the local path as a URL and **copies the file into
`~/.cache/torch/hub/checkpoints/`** — 82.5 MB duplicated, and a cache dependency reintroduced in
the one place this plan is trying to remove it. Build unweighted and load explicitly:

```python
model = dinov3_vits16(pretrained=False)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=True)
# verified: "All keys matched successfully"
```

`strict=True` is deliberate — a silently partial load is the failure mode that would produce
plausible-looking garbage features for a whole sweep.

#### Vendor or pin? The repo is public, so this is a licensing question

Unlike the weights — which are gitignored and pushed to the private DVC remote, so `dvc add` is
not publication — **copying Meta's source files into `github.com/kasetkin/coffee-vision` is
redistribution to third parties**, and §1b-i of the DINOv3 License then applies: the copy must be
distributed under that Agreement with a copy of it attached.

| option | reproducible? | redistributes Meta code? | cost |
|---|---|---|---|
| **A.** vendor the 26 files into `coffeecv_dino/vendor/dinov3/` | yes, fully offline | **yes** — public repo gains a non-Apache subtree that must carry the DINOv3 License | licence hygiene, and a fork to maintain |
| **B.** pin as a dependency (submodule, or `pip install git+…@<sha>`) | yes, if the SHA is pinned | **no** | needs network at setup; breaks if upstream moves |
| C. `torch.hub.load(..., source="local")` | yes | no | same as B, plus the broken-`hubconf.py` trap above |

**Recommendation: B.** A pinned commit SHA gives the same reproducibility as vendoring without
publishing Meta's code from a public repo, and it keeps the licence boundary clean — the repo
stays entirely your own code. Record the SHA in `RunConfig` next to the weights' sha256, so an
archived `config.json` identifies both the weights and the architecture definition that produced
a run.

The one real risk of B is upstream disappearing or force-pushing. Mitigate by keeping a private
mirror (a bare clone pushed to the same box as the DVC remote) — that is storage for your own use,
not redistribution, so it raises none of the §1b-i questions that option A does.

If option A is ever chosen anyway, keep the vendored files in one clearly-named directory with
`LICENSE.md` copied in beside them, and never edit them — a modified vendored copy is both a
licence question and the private-copy drift trap from Phase 1.

### 1-A.1c. Obtaining the DINOv3 weights on a fresh clone

The repository cannot ship these files: §1b-i of the DINOv3 License makes redistributing weights
an act that carries the Agreement with it, and `github.com/kasetkin/coffee-vision` is public.
So reproducibility here means **a documented, verifiable acquisition procedure**, not a download
in the repo. There are two audiences and they are served differently:

- **You, on another machine** — `dvc pull`. Once the weights are `dvc add`ed they live in the
  private `ssh://coffdvc@dvcremote` cache; fetching them is your own storage, not redistribution.
  This is the normal path and needs nothing from Meta a second time.
- **Anyone else cloning the public repo** — they must obtain their own copy from Meta, using the
  procedure below, and verify it against `models_pretrained/manifest.json`.

#### Prerequisite: `models_pretrained/README.md` and `manifest.json` must be tracked

They were not. `.gitignore` carried a blanket `/models_pretrained/`, which hid the manifest and
the instructions along with the weights — so a fresh clone had no way to learn what to download or
what it should hash to. Fixed 2026-09-21 by ignoring the binaries rather than the directory:

```gitignore
/models_pretrained/**/*.pth
/models_pretrained/**/*.pt
/models_pretrained/**/*.safetensors
```

Without this the rest of this section cannot work, because the checksums would not exist in a
clone.

#### Two channels, serving two different files — do not mix them

This is the trap. Meta distributes DINOv3 twice, in incompatible formats:

| | Meta direct | Hugging Face |
|---|---|---|
| where | request access, then `https://dl.fbaipublicfiles.com/dinov3/<model_dir>/<file>.pth` | `facebook/dinov3-vits16-pretrain-lvd1689m` (gated: **manual** approval) |
| format | torch `.pth` state dict, filename carries a sha256 prefix | `model.safetensors` + `config.json` + `preprocessor_config.json` |
| loads with | `dinov3.hub.backbones` (§1-A.1b) | `transformers.AutoModel` |
| matches `manifest.json`? | **yes** | **no** — different file, different hash |

The files already in `models_pretrained/dinov3/` are the **Meta-direct `.pth`** form. Use that
channel. The HF safetensors will not load into `dinov3.hub.backbones`, and its hash will never
match the manifest, so a contributor who grabs the HF copy will hit a checksum failure that looks
like corruption but is actually the wrong format.

#### The procedure

1. **Request access** at Meta's DINOv3 downloads page and accept the licence. (That page is a
   form; it carries no licence text — the terms are `LICENSE.md` in `facebookresearch/dinov3`,
   pinned here at `docs/reference_dinov3_LICENSE.md`.) Access arrives as a signed URL.
2. **Download the variants this project uses**, into `models_pretrained/dinov3/`, keeping the
   upstream filenames — the `-08c60483` suffix is torch's sha256 prefix and is load-bearing:

   | file | bytes | sha256 (first 16) |
   |---|---|---|
   | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | 86,531,063 | `08c60483bc63c04f` |
   | `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` | 114,924,783 | `4057cbaaad8c1665` |
   | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 342,860,279 | `73cec8be7427c865` |

   Only `vits16` is needed for Screens A-C; the other two are a teacher and an upper bound.
3. **Verify against the manifest before using them.** Two independent checks, both cheap:

   ```bash
   python -m coffeecv_dino.verify_weights      # full sha256 vs models_pretrained/manifest.json
   ```

   and upstream's own convention, which `dinov3.hub.backbones` supports natively — the filename
   prefix must equal the leading sha256 hex, so `check_hash=True` validates a download without
   any project-specific metadata at all.
4. **Never rename the files.** The hash suffix is the integrity mechanism; renaming discards it.

#### When a hash does not match

Do not proceed and do not "fix" the manifest. In order of likelihood: the HF safetensors were
downloaded instead of the Meta `.pth` (see the table above); the download truncated (check the
byte count first — it is the fastest discriminator); or Meta re-issued the file, in which case
the discrepancy needs recording in `manifest.json` with a date and a note, because it means every
prior run used different weights.

#### Record which weights a run used

The sha256 belongs in `RunConfig` (§1-B.3) so it lands in every archived `config.json`. That is
what makes an experiment reproducible rather than merely repeatable: a run that does not state
which weights it loaded cannot be paired against one that used a different copy. The same applies
to the pinned `facebookresearch/dinov3` commit SHA from §1-A.1b — weights and architecture
definition together, or neither.

### 1-A.2. The embedding cache — what makes this affordable

Measured on the workstation, batch 32 at 224 px:

| arm | trainable | train step | vs resnet18 (1.33 s) |
|---|---|---|---|
| head only, frozen backbone (naive) | 0.004M | 1.47 s | 1.11x |
| last 1 block unfrozen | 1.78M | 1.71 s | 1.29x |
| last 2 blocks unfrozen | 3.55M | 1.94 s | 1.46x |
| last 4 blocks unfrozen | 7.10M | 2.49 s | 1.87x |
| last 6 blocks unfrozen | 10.66M | 2.99 s | 2.25x |
| frozen, forward only | — | 47.0 ms/img | — |

Full fine-tuning of the backbone is **out of scope by decision (2026-09-21)**: this project will
use head-only or last-N-blocks and nothing deeper. Recorded here with its measured cost — 21.3M
trainable, 4.37 s/step, 3.3x resnet18 — only so it is not re-proposed later. §1-B.2 turns the
decision into an enforced invariant rather than a convention.

The naive frozen arm already costs about what resnet18 costs, because it re-runs the backbone
every epoch on data the backbone's weights never see. Caching removes that entirely:
`train_patches_per_class=150` x 10 classes x 3 rigs = 4500 train patches; at 47 ms that is
3.5 minutes to embed once, after which each epoch is a linear layer over a 4500x384 matrix —
milliseconds.

The complication is augmentation: the train transform applies a random dihedral rotation, flips,
colour jitter and random erasing *per epoch*, so a single cached embedding per patch would
silently delete all of it. Resolve it the way this project already reasons about TTA: **cache the
8 dihedral views per patch and sample among them** (8 x 4500 = 36 000 embeddings, 55 MB at
384 floats, ~28 min to build), because the dihedral group is exact and lossless, and drop the
photometric augmentations for frozen arms. The probe scored 0.8511 with no augmentation at all,
so this is a superset of what was measured, not a gamble.

Caveat to write into the code: **the cache is valid only while the backbone is frozen.** Key it
on (weights sha256, patch_resize, seed, rig, split, fold) and have it refuse to load when any of
those changed. A stale cache silently training a head on another fold's features is the worst
failure mode in this plan.

---

## Phase 1-B — Integration, gated on Screen A (except 1-B.0)

Do §1-B.1 onward **only if Screen A passes**, and specifically when Screen C (unfreeze depth) is
next. That arm needs the real epoch loop, the LR schedule, early stopping and
checkpoint-selection-on-best-val — and reimplementing early stopping and checkpoint selection in
a second trainer is precisely how two arms silently stop being comparable. At that point the cost
of integration is lower than the cost of duplication, which is the trigger to move.

**§1-B.0 is the exception and is not gated on anything.** Pruning two rejected architectures is
correct whether or not DINOv2 ever ships, and it should land as its own commit as soon as Stage 1
frees the repo — ideally before the DINOv2 branch is written, so that branch arrives in a
two-architecture file.

Nothing in §1-B.1–1-B.4 should be built during Phase 0 or Screen A.


### 1-B.0. Prune `mobilenet_v3_small` and `efficientnet_b0` first

**Decision (2026-09-21): `build_model` supports `resnet18` and `dinov2_vits14`, and nothing else.**

Do this **before** adding the DINOv2 branch, so the new branch lands in a two-architecture file
rather than making it four. It is also **independent of whether DINOv2 succeeds** — both
architectures are already-rejected dead code either way — so it can land on its own as soon as
Stage 1 frees the repo, without waiting on Screen A.

#### The audit says this is free

| surface | mobilenet_v3_small | efficientnet_b0 |
|---|---|---|
| archived experiments (`experiments/*/config.json`) | **0 of 150** | **0 of 150** |
| shipped checkpoints (`models/*.json`) | **0 of 7** | **0 of 7** |
| live code references | `config.py:77`, `model.py:136`, `train_baseline.py:59` | `model.py:151`, `train_baseline.py:59` |

Every archived run and every shipped model is resnet18. Nothing that exists depends on either
branch, so deletion breaks no provenance: the runs that *did* use them are prose records in
`EXPERIMENTS_LOG.md`, and are reproducible from their own commits as always.

#### Both were screened and explicitly rejected — this makes the code match a decision already taken

- **exp3** — efficientnet_b0 frozen: "clear reject", val −0.10, test −0.26.
- **exp29** — efficientnet_b0 full fine-tune: "clear reject", val −0.0225 and test −0.0374, both
  outside their bands, and ~1.5x slower per batch. "Worse *and* more expensive — unambiguous."
- **exp31** — mobilenet_v3_small: val 0.8490 against resnet18's 0.9579, recorded as the largest
  gap of the alternative architectures.

The log already summarises both as rejected alternatives. Deleting the branches removes code that
the project decided against three separate times.

#### The `RunConfig` default is a live trap, not just dead weight

```python
model_name: str = "mobilenet_v3_small"   # coffeecv/config.py:77
```

`RunConfig()` — which `from_params_yaml` returns verbatim when `params.yaml` is absent — hands
back the **worst architecture this project ever measured**, and one no run has ever used. Change
the default to `resnet18` in the same commit. This is the kind of silent-wrong-default that
`feedback-cli-defaults-overwrite-config` is about.

#### It also deletes the head-slice bug instead of repairing it

The `head_module` defect documented in Phase 4 (`model.classifier[2:4]` is a slice, so the
embedding pre-hook never fires) is a mobilenet-only bug. Deleting the branch resolves it
outright, which is strictly better than fixing code nothing runs. The efficientnet branch has no
such defect — it is simply unused.

#### The edits

1. `coffeecv/model.py` — delete the `mobilenet_v3_small` and `efficientnet_b0` branches. The
   trailing `else: raise ValueError(f"Unknown model_name: {name!r}")` already gives the right
   behaviour for an old config, but widen its message so it is actionable:

   ```python
   raise ValueError(
       f"Unknown model_name: {name!r}. Supported: 'resnet18', 'dinov2_vits14'. "
       f"'mobilenet_v3_small' and 'efficientnet_b0' were removed on 2026-09-21 after being "
       f"screened and rejected (EXPERIMENTS_LOG.md exp3/exp29/exp31); to re-run an experiment "
       f"that used one, check out that run's own git_commit."
   )
   ```

   This matters because `config_for_checkpoint` faithfully restores `model_name` from an archived
   card. An old config must fail loudly with a pointer, never fall back to a substitute — a
   silent substitution would mislabel every number it produced.

2. `coffeecv/config.py:77` — `model_name: str = "resnet18"`.
3. `coffeecv/train_baseline.py:59` — `choices=["resnet18", "dinov2_vits14"]`.
4. **Do not edit `EXPERIMENTS_LOG.md`.** It is contemporaneous prose and the justification for
   this deletion; erasing the mentions would destroy the reason the code can go. Add a Phase 0
   style line recording the removal and its date instead.
5. `analysis/architecture_screen/bench.py` keeps its `efficientnet_b0` row — it is a record of a
   measurement taken on a specific date, not a code path, and rerunning it should reproduce the
   published table.

#### A test worth adding alongside

```python
@pytest.mark.parametrize("name", ["resnet18", "dinov2_vits14"])
def test_head_module_pre_hook_actually_fires(name):
    """Every supported architecture must return a head_module the forward pass
    really calls -- infer.forward_with_embeddings captures the embedding with a
    pre-hook on it, and a slice-copy (mobilenet's old bug) silently captures
    nothing, leaving torch.cat([]) to raise at inference time."""
    model, head = build_model(name, num_classes=10, freeze_mode="none", dropout=0.2)
    seen = []
    h = head.register_forward_pre_hook(lambda _m, i: seen.append(i[0].shape))
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 3, 224, 224))
    h.remove()
    assert seen, f"{name}: head_module pre-hook never fired -- is head_module a slice?"
    assert seen[0][1] == embedding_dim_of(head)
```

Parametrised over the supported list, so a future architecture cannot be added without proving
its embedding contract works.

### 1-B.1. Split the per-architecture code into one file per arm

**Decision (2026-09-21): each architecture owns its own file — model, optimizer param groups,
scheduler choice and per-batch specifics — behind one small protocol.** `train_baseline.py` keeps
the epoch loop and stops being an `if resnet / if dino` file.

Today those concerns are smeared across three files. In `train_baseline.py` alone, five separate
places know something architecture-specific: the `build_model` call, the
`cross_domain_mixstyle` flag (which changes the *DataLoader*, via `return_domain_id`), the
head/backbone param-group split, the `build_scheduler` call, and `train_one_epoch`'s MixStyle
`domain_ids` assignment. Adding DINOv2 without this refactor means five new conditionals in a
449-line function that currently has none.

#### The protocol

```python
# coffeecv/arms/__init__.py
class Arm(Protocol):
    """One architecture plus the training specifics that belong to it.

    Deliberately small. Everything NOT here -- the epoch loop, evaluate(),
    compute_split_metrics, checkpoint-on-best-val-macro-F1, the patience rule,
    history/tensorboard, the outputs/ contract -- stays shared in
    train_baseline.py, because two arms that select checkpoints or compute
    metrics differently are not comparable, and comparability is the entire
    point of adding a second arm.
    """
    name: str

    def validate_config(self, cfg: RunConfig) -> None:
        """Raise on a config this arm cannot honour, at construction time."""

    def build_model(self, cfg, num_classes: int) -> tuple[nn.Module, nn.Module]:
        """(model, head_module). head_module MUST be an object the forward pass
        really calls -- see the mobilenet slice bug in Phase 4."""

    def build_optimizer(self, cfg, model, head) -> Optimizer: ...
    def build_scheduler(self, cfg, optimizer) -> Scheduler: ...

    def needs_domain_ids(self, cfg) -> bool:
        """True if the train DataLoader must yield (x, y, domain_id)."""

    def set_batch_context(self, model, domain_ids) -> None:
        """Called before each forward. No-op unless the arm needs per-sample rig ids."""

    def describe(self, cfg) -> str:
        """One line printed at run start, alongside the scheduler's describe()."""


ARMS: dict[str, Arm] = {}          # populated by the modules below

def get_arm(name: str) -> Arm:
    if name not in ARMS:
        raise ValueError(
            f"Unknown model_name: {name!r}. Supported: {sorted(ARMS)}. "
            f"'mobilenet_v3_small' and 'efficientnet_b0' were removed on 2026-09-21 "
            f"(EXPERIMENTS_LOG.md exp3/exp29/exp31); to re-run an experiment that used "
            f"one, check out that run's own git_commit."
        )
    return ARMS[name]
```

#### What lands in each file

`coffeecv/arms/resnet18.py` — the current adopted recipe, moved verbatim: the torchvision
ResNet18 + dropout/linear head, `MixStyle` and `_install_mixstyle` (hooks after `layer1`/`layer2`),
`_apply_freeze_mode`, the head-lr/backbone-lr param groups, `cosine | plateau` selection, and the
`domain_ids` assignment in `set_batch_context`.

`coffeecv/arms/dinov2.py` — vendored-weight loading and the sha check, `DinoV2Classifier`, the
`patch_resize % 14` assert, `dinov2_unfreeze_blocks`, its own param groups (at depth 0 there is
only a head group at all), its own schedule, `needs_domain_ids` returning `False` and
`set_batch_context` a no-op, and `validate_config` raising on `mixstyle_p > 0` and on any
`freeze_mode` but `full`.

#### "Its own scheduler" means its own *policy*, not its own copy of the mechanism

This is the one place the split can go wrong. `lr_schedules.py` already defines `CosineLR` and
`PlateauLR` behind a duck-typed interface (`lrs()`, `step(epoch, val_f1)`, `should_stop()`,
`owns_stopping`, `describe()`). Those are **mechanisms** and must stay shared:

- **Shared, imported by both arms:** `CosineLR`, `PlateauLR`. Forking `CosineAnnealingLR` into
  two files is exactly the private-copy drift that bit `analysis/bean_scale` twice — and it would
  be worse here, because a scheduler difference between arms is indistinguishable from an
  architecture difference in the final number.
- **Owned by the arm:** which scheduler, with which hyperparameters, and any mechanism that is
  genuinely new. A frozen head converging in a handful of epochs does not want a 100-epoch
  `T_max`; and if the unfreeze arm needs warmup or layer-wise LR decay, that is new mechanism and
  belongs in `arms/dinov2.py` — implemented against the same protocol, so the loop cannot tell
  the difference.

```python
# coffeecv/arms/dinov2.py
    def build_scheduler(self, cfg, optimizer):
        # Cosine over the head's own (much shorter) budget, not the resnet recipe's 100.
        # Same CosineLR class the resnet arm uses -- only T_max and the floor differ.
        return CosineLR(optimizer, epochs=cfg.epochs, eta_min=cfg.eta_min)
```

#### What `train_baseline.py` becomes

```python
arm = get_arm(cfg.model_name)
arm.validate_config(cfg)                      # fails in ~1 ms, not 40 min into patch building
model, head = arm.build_model(cfg, len(class_ids))
optimizer = arm.build_optimizer(cfg, model, head)
sched     = arm.build_scheduler(cfg, optimizer)
print(f"arm: {arm.describe(cfg)}")
print(f"scheduler: {sched.describe()}; epoch cap {cfg.epochs}")
```

and `train_one_epoch` loses its `cross_domain_mixstyle` parameter in favour of
`arm.set_batch_context(model, domain_ids)`. Mixup stays in the shared loop: it is a config lever
that applies to any architecture, not an architecture property, and moving it into the arms would
let two arms silently compute the loss differently.

#### Sequencing and how to prove it changed nothing

Order matters: **§1-B.0 (prune) first, then this refactor, then the DINOv2 arm.** Pruning first
means the refactor moves one architecture instead of three.

This is a **pure refactor — no behaviour change** — and that claim has to be demonstrated, not
asserted, because every subsequent comparison rests on the baseline being untouched:

```bash
# before the refactor, on a clean tree
python -m coffeecv.train_baseline --epochs 3 --seed 42   # -> outputs/history.json
cp outputs/history.json /tmp/history_before.json
# after
python -m coffeecv.train_baseline --epochs 3 --seed 42
diff /tmp/history_before.json outputs/history.json       # MUST be empty
```

`train_loss` and `val_macro_f1` must match to the last digit on every epoch. Anything else means
an RNG draw moved, and an RNG draw moving means every paired comparison against exp200–203 is
invalid. Three epochs is enough to catch it and costs minutes.

Two further notes. This touches `coffeecv/`, which `dvc.yaml`'s `train` stage depends on, so it
will mark the stage stale once — fine as a single deliberate commit, which is why it must not be
done while a sweep is running (Stage 1). And `coffeecv_dino/` from Phase 1-A should implement
this same `Arm` protocol from the start, importing it from `coffeecv.arms`; then Phase 1-B.2 is
a file move rather than a rewrite.

### 1-B.2. `coffeecv/arms/dinov2.py` — the DINOv2 arm

The arm's `build_model` returns `(model, head_module)` like every other, where `head_module` is
used for two things: its own LR group, and the forward pre-hook in
`infer.py::forward_with_embeddings` that captures the embedding. A thin wrapper satisfies both:

```python
class DinoV2Classifier(nn.Module):
    """Frozen-or-partially-frozen DINOv2 ViT-S/14 + linear head.

    The head is a real submodule (not a functional call) so the pre-hook in
    infer.forward_with_embeddings captures the 384-d CLS feature -- the same
    "embedding is the head's input" contract resnet18 already honours, which is
    what keeps the OOD guard measuring what the classifier actually sees.
    """
    def __init__(self, backbone, num_classes, dropout):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(384, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))
```

Three things the arm must get right, all of them in `validate_config` or `build_model` rather
than scattered through the training loop:

- **`freeze_mode` already does the work — and must be pinned to `full`.**
  `_apply_freeze_mode(model, "full", ...)` sets `requires_grad=False` on everything, and the head
  is constructed afterwards with fresh `requires_grad=True` params. So
  `model_name=dinov2_vits14, freeze_mode=full` *is* the head-only arm, and `train_baseline.py`'s
  existing head/backbone param-group split already handles it with no change.

  `validate_config` must **raise on any other `freeze_mode`**, including `none`. `none` is a full
  backbone fine-tune, which is out of scope by decision; `last_block` is meaningless for a ViT
  and would collide with the knob in 1-B.3. Depth is then controlled by exactly one parameter,
  `dinov2_unfreeze_blocks` (0 = head only, N = re-enable grad on the last N transformer blocks),
  carving exceptions out of the full freeze. Two knobs steering the same quantity is the config
  trap this project has been bitten by before — one source of truth, and it fails loudly when
  something sets the other.
- **Raise on a MixStyle request**, the way `build_model`'s resnet18-only guard does today:
  `mixstyle_p > 0` with a ViT must fail loudly, because MixStyle on token statistics is a
  different formulation that nothing here has validated. Silently ignoring the flag would
  compare an arm that lost the project's +0.1400 lever against one that kept it, with nothing in
  the log saying so.
- **Guard the input size early.** ViT-S/14 requires a multiple of 14; `patch_resize=224` is
  exactly 16 patches. Measured behaviour: 224/238/252 all work, and 256/300 raise
  `AssertionError: Input image height 256 is not a multiple of patch height 14`. So this fails
  *loudly* — but at the first forward pass, which is after patch materialisation, i.e. minutes
  into a run. Assert it in `validate_config` instead, which the loop calls before any data is
  built, so a bad `patch_resize` costs a millisecond rather than the whole dataset build. (An
  earlier draft of this document claimed the mismatch silently interpolated positional
  encodings. That is wrong; it asserts.)

**Normalization needs no change.** DINOv2's own transform uses
`IMAGENET_DEFAULT_MEAN/STD = (0.485,0.456,0.406)/(0.229,0.224,0.225)` — byte-identical to
`coffeecv/transforms.py`'s `IMAGENET_MEAN`/`IMAGENET_STD`. Verified in the vendored source. This
is the single luckiest fact in the integration: `build_train_transform` and
`build_eval_transform` work unmodified, so train/inference parity is preserved for free.

### 1-B.3. `coffeecv/config.py` — new fields

Add `dinov2_weights: str` (path to the vendored checkpoint), `dinov2_unfreeze_blocks: int = 0`
(0 = head only; the only depth knob, see 1-B.2), and `dinov2_weights_sha256: str`. Bound
`dinov2_unfreeze_blocks` to `0 <= N <= 6` in validation — ViT-S/14 has 12 blocks, and permitting
12 would reintroduce the full fine-tune this plan excludes.

Remember `from_params_yaml` raises on unknown keys by design — so params.yaml and RunConfig must
be changed in the same commit, or every run in between is mislabelled.

### 1-B.4. `coffeecv/run_folds.py` — the lever pass-through

`set_fold()` has **no `model_name` parameter at all** today. Add one following the existing
pattern exactly, which already implements the rule that matters:

```python
if model_name is not None:                       # None => INHERIT what params.yaml holds
    text = re.sub(r"^model_name: \S+", f"model_name: {model_name}", text, count=1, flags=re.M)
...
if model_name is not None:
    assert cfg.model_name == model_name, f"model_name is {cfg.model_name!r}, wanted {model_name!r}"
```

The `is not None` guard is not a style preference. A flag that defaults to a concrete
architecture instead of inheriting is precisely the failure that cost this project 6.5 h of
invalid training once already. The read-back assertion is the other half: a regex that silently
fails to match would otherwise run the wrong architecture and look like a result.

Add `--model-name` to the CLI with `default=None`, and extend the per-fold note so the archived
record names the architecture.

### 1-B.5. `dvc.yaml`

Add `models/dinov2_vits14_pretrain.pth` as a dep of the `train` stage. Same reasoning as the
comment already there about listing every rig regardless of the current fold: if the weights file
is not a tracked dep, swapping it reuses a stale run.

---

## Phase 2 — How to compare ResNet18 and DINOv2 honestly

This is the part most likely to go wrong, because the two arms are not a single-variable change
and pretending otherwise would produce a confident wrong answer.

### The comparison is paired at the patch level, and that is the strongest asset available

Patch boxes are drawn from a seeded RNG keyed on `(seed, rig_idx, class_idx, photo_idx, split)`
and materialised at dataset construction, so **two arms run at the same seed and fold see
byte-identical patches**. This is much stronger than the usual "same dataset" pairing: it removes
sampling variance from the difference entirely. Every comparison below must therefore be reported
as a *per-pair delta* (`dinov2 − resnet18` on the same fold and seed), never as a difference of
two independently-averaged means. The project's adoption standard already reflects this — the
MixStyle adoption rests on 9/9 sign-consistent pairs, not on a gap between two averages — and
that is the bar to hit here.

### The headline metric, and the two controls that must travel with it

The headline is `xrig_macro_f1`: the held-out camera's patches, from a rig never trained on.
Two controls have to be reported next to it every time, or a regression hides:

- `test_macro_f1`, the in-distribution control. An arm that wins cross-rig while losing
  in-distribution is making a trade, and that trade is a product decision, not a free win. The
  camera-rig baseline sits at 0.9002–0.9164 in-distribution against 0.8025 cross-rig.
- The **photo-pooled** number, via `photo_pooling_eval.py`. `/classify` returns a photo-level
  verdict, so a patch-level win that does not survive pooling is not a user-visible win. Pooling
  has previously measured −0.035 on box and +0.020 mean, so it is not a formality.

Never quote a mean that mixes the ten-class folds with cam_iphone. cam_iphone has no class_008,
so exp203's cross-rig figure is an 8-class macro (`xrig_macro_n=8` in `index.csv`) and is not
comparable. Report the 3-ten-class-fold mean as the headline and the 4-fold mean beside it,
always both, always labelled.

### The baseline to pair against does not fully exist yet, and that is a real constraint

The camera-rig era has exactly **four** cosine fold runs — exp200–203, all seed 42. There is no
seed-123 or seed-7 camera-rig fold baseline. So a first comparison can only be 4 pairs at one
seed, which is a screen, not an adoption. Two honest ways forward, and the choice should be made
deliberately rather than by default:

- **Cheap and sufficient to decide whether to continue:** run DINOv2 at seed 42 on the same four
  folds and compare 4 paired deltas against exp200–203. If the deltas are large and
  sign-consistent 4/4, that justifies spending the compute on a full baseline. If they are mixed,
  stop — and note that a 4/4 sign-consistent result at one seed has *three times* in this
  project's history turned out to be a favourable draw.
- **Complete but expensive:** add resnet18 camera-rig folds at seeds 123 and 7 (~6 runs, roughly
  3 days of VM time) to build a 12-pair matrix. This is what an adoption actually needs.

### The seed-variance asymmetry — the subtlest trap here

A fine-tuned ResNet18's run-to-run spread comes mostly from training stochasticity: init, batch
order, augmentation draws. This project measured seed sd = 0.029 and a ±0.048 noise band *for
that kind of arm*. **A frozen DINOv2 arm has almost none of that** — the backbone is fixed, and
only the head init and patch draw vary. Its own seed sd will be far smaller.

The consequence: **you cannot judge a DINOv2 delta against ResNet18's noise band.** Doing so
would overstate significance. Measure DINOv2's own seed sd first (free, per Phase 0 step 5), and
use the larger of the two arms' spreads when deciding whether a delta is real. Conversely, do not
let DINOv2's tiny spread be read as "more reliable" — a frozen model is not more reliable, it is
less stochastic, which is a different property.

### It is a design comparison, not an architecture ablation — label it as such

The deployed ResNet18 recipe includes MixStyle p=0.5 (+0.1400 cross-rig) and dihedral TTA
(+0.0235). A frozen DINOv2 can use neither: MixStyle has no validated ViT formulation and does
not apply to a frozen backbone at all, and the screen showed TTA was not needed to reach 0.8511.

So the comparison is **"best available ResNet18 design" vs "best available DINOv2 design"** — a
legitimate and decision-relevant question, but not a controlled single-variable change. Write
that sentence into the experiment note so nobody later reads the result as "ViT beats ResNet".
Resist the temptation to "fix" it by disabling MixStyle on the baseline: that would compare two
handicapped designs and answer a question nobody is asking.

### Also compare the things that are not macro-F1

- **Latency**, measured end to end, not inferred. The deployed path is 40 patches x 8 TTA = 320
  passes ≈ 4.45 s at resnet18's 13.9 ms/img. A frozen DINOv2 needing no TTA is 40 passes x
  47 ms ≈ 1.9 s. If that holds, the change is a *speedup*, which materially changes its
  cost/benefit — so measure it rather than citing this estimate.
- **OOD separability in the new 384-d space.** The guard is the thing standing between a user and
  a confident wrong answer on a photo of lentils. A classifier win paired with an OOD regression
  is not a win. See Phase 4.
- **Per-class confusion**, via `confusion_report.py`. class_006 (Cerrado) / class_007
  (MonteCristo) confusion is confirmed on three rigs; whether a different representation breaks
  that specific pair is more informative than the macro average alone.

---

## Phase 3 — The experiments to run

Numbering starts at **exp220**, leaving 214–219 clear of Stage 1's exp210–213. Each phase gates
the next; do not queue them all.

### Screen A — frozen DINOv2, 4 folds, seed 42. `exp220–223`

Config: `model_name=dinov2_vits14`, `freeze_mode=full`, `mixstyle_p=0.0`, `epochs`/`patience` as
the head needs (a linear head on frozen features converges in far fewer than 100 epochs — set the
cap low and let early stopping decide), everything else at the adopted recipe.

Runs through `python -m coffeecv_dino.run_folds` (Phase 1-A), not `coffeecv.run_folds` — no
`params.yaml` rewrite, no `dvc repro`, no edits under `coffeecv/`. Pairs against exp200–203
directly, because the patches are the same objects from the same seeded sampler. With the
embedding cache this is roughly **30–45 minutes per fold on the workstation**, so Screen A does
not need the VM at all and can run before Stage 1 finishes — as long as nothing is committed
until it does.

Decision rule: continue only on 4/4 sign-consistent positive cross-rig deltas with the
in-distribution control not regressing by more than the seed sd.

### Screen B — seed replication. `exp224–235`

The same four folds at seeds 123, 7 and 17. Free for the DINOv2 arm; the cost is the *ResNet18*
side, which needs matching camera-rig folds at those seeds (~3 days VM). Run the DINOv2 seeds
immediately and regardless — they cost minutes and give the arm's own seed sd, which Phase 2 says
is needed before any delta can be called real.

Adoption bar: 12 paired deltas, sign-consistent, in the project's usual form.

### Screen C — unfreeze depth. `exp236–243`

With full fine-tuning excluded, depth is the only remaining architecture lever, and it is worth
treating as a proper one-dimensional sweep rather than a single guess. The ladder is measured:

| `dinov2_unfreeze_blocks` | trainable | s/step | vs resnet18 | 4 folds, 1 seed, VM |
|---|---|---|---|---|
| 0 (Screen A) | 0.004M | 1.47 | 1.11x | cached — minutes |
| 1 | 1.78M | 1.71 | 1.29x | ~0.9 day |
| **2** | 3.55M | 1.94 | 1.46x | ~1.0 day |
| 4 | 7.10M | 2.49 | 1.87x | ~1.3 day |
| 6 | 10.66M | 2.99 | 2.25x | ~1.5 day |

Note the cost cliff is *not* where intuition puts it: going from head-only to last-2 costs only
+32% per step, because the backward pass still skips ten of twelve blocks. The expensive part of
Screen A was never the gradient — it is the forward pass, and the embedding cache removes that
for depth 0 only. **Every depth ≥ 1 loses the cache** (the backbone's weights now change every
epoch, so yesterday's embeddings are wrong), which is the real discontinuity: minutes at depth 0,
about a day at depth 1. Budget accordingly.

Run **N = 2 first**, one seed, four folds, paired against Screen A — not against exp200–203,
since this is a separate lever layered on a decision Screen A already made. Only if N = 2 beats
depth 0 sign-consistently is it worth spending another day probing N = 4; if N = 2 is flat or
negative, stop at depth 0 and keep the cache, which is worth more than a marginal gain.

The mechanism to expect: frozen features may be slightly off-distribution for top-down bean
texture, and a few unfrozen blocks let the representation adapt without enough capacity to
memorise 938 photos. If that story is right, the gain should appear at small N and flatten;
a curve that keeps climbing to N = 6 is a signal to re-examine for overfitting against the
in-distribution control, not a reason to unfreeze further.

### Screen D — TTA necessity. Free, no training

Re-score Screen A's best checkpoints with and without dihedral TTA. The screen suggested TTA is
unnecessary for DINOv2; if that is wrong, the latency argument in Phase 2 collapses and the
deployment story changes. This is a scoring-time question, so it costs nothing but must be asked
explicitly rather than assumed.

### Screen E — shipping candidate. `exp244–249`

Only after an adoption decision: six seeds via `run_all_rigs.py`, no held-out rig, to produce
weights. Note the standing rule — `run_all_rigs.py` is for shipping weights only, and never for
validating a config change. Every adoption decision above must already have been made on
`run_folds.py` numbers.

---

## Phase 4 — The image path: what migrates, and what does not

### First, a correction to a common assumption

**512 and 384 are embedding dimensions, not input sizes.** Measured on this machine:

```
input (2, 3, 224, 224) -> resnet18 feature (2, 512)
input (2, 3, 224, 224) -> dinov2   feature (2, 384)
```

Both models consume the **same 224x224 tensor**. What differs is the width of the vector that
comes out the far end. That distinction decides where the work is: nothing changes in how images
are read, and everything that changes is downstream of the classifier, in the OOD artifacts that
live in the embedding space.

### The image path needs no migration at all — and that is checkable, not asserted

Both arms run the identical chain, because `coffeecv_dino` imports it rather than copying it
(Phase 1's seam):

| stage | function | differs between arms? |
|---|---|---|
| decode (HEIF/AVIF/JXL/RAW) | `dataset.load_rgb_image` | no |
| tray/bean-region crop | `infer.crop_to_bean_region` -> `crop_tray.locate_bean_crop` | no |
| bean pitch | `bean_scale.estimate_bean_pitch` | no |
| patch boxes | `geometry.sample_bean_unit_patch_boxes` | no |
| resize + tensor + normalise | `transforms.build_eval_transform(224)` | no |
| **forward** | `resnet18` / `DinoV2Classifier` | **yes — 512-d vs 384-d out** |
| pool over patches / TTA | `infer.forward_with_embeddings` | no (generic) |

The normalisation constants are the reason this works out: DINOv2's own transform uses
`IMAGENET_DEFAULT_MEAN/STD = (0.485,0.456,0.406)/(0.229,0.224,0.225)`, byte-identical to
`coffeecv/transforms.py`. Verified in the vendored source. Had they differed, `build_eval_transform`
would have needed a per-arm branch and train/inference parity would have become a live risk.

Make the "no migration" claim enforceable rather than a comment — a test is three lines:

```python
def test_dino_arm_reuses_the_image_path():
    """coffeecv_dino must not own a second copy of decode/crop/pitch/patch/transform.
    The paired comparison is only meaningful while both arms read images identically;
    analysis/bean_scale has twice grown a private copy that silently drifted."""
    import coffeecv_dino.backbone as b
    src = Path(b.__file__).parent.rglob("*.py")
    banned = ("def load_rgb_image", "def estimate_bean_pitch", "def sample_bean_unit_patch_boxes",
              "def build_eval_transform", "def compute_split_metrics")
    for f in src:
        text = f.read_text()
        for name in banned:
            assert name not in text, f"{f.name} defines its own {name} -- import it from coffeecv"
```

### The one input-size constraint that is real

```python
# coffeecv_dino/backbone.py
DINOV2_PATCH = 14          # ViT-S/**14**: the "14" is the patch size, not the feature width
DINOV2_EMBED_DIM = 384

def assert_input_size(patch_resize: int) -> None:
    """DINOv2 requires a side that is a whole number of patches.

    It does raise on its own -- `AssertionError: Input image height 256 is not a
    multiple of patch height 14` -- but only at the first forward pass, which is
    after MultiPhotoPatchDataset has materialised every patch. On this dataset
    that is minutes of decode work thrown away to learn something knowable at
    construction time. 224 = 14*16 (the adopted value), 238 and 252 also valid;
    256 and 300 are not.
    """
    if patch_resize % DINOV2_PATCH:
        raise ValueError(
            f"patch_resize={patch_resize} is not a multiple of {DINOV2_PATCH}, which DINOv2 "
            f"ViT-S/14 requires. Nearest valid values: "
            f"{patch_resize // DINOV2_PATCH * DINOV2_PATCH} or "
            f"{(patch_resize // DINOV2_PATCH + 1) * DINOV2_PATCH}. The adopted config is 224."
        )
```

Call it from `DinoConfig.__post_init__`, not from the forward pass.

### What genuinely migrates: the 384-d embedding space

`forward_with_embeddings` captures the head's *input*, so it yields 384-d vectors for DINOv2 with
no change — the contract is already generic. The **code** downstream is dimension-agnostic too
(`build_ood_reference.py` reads `embedding_dim` off the array shape rather than hardcoding 512).

What is *not* interchangeable is the **artifacts**. Every sidecar beside a checkpoint is a
description of one specific embedding space:

| file | current content | after a DINOv2 swap |
|---|---|---|
| `<ckpt>.ood_reference.json` | `embedding_dim: 512`, per-class centroids | must be rebuilt at 384 |
| `<ckpt>.ood_embeddings.npz` | 512-wide kNN bank | must be rebuilt at 384 |
| `<ckpt>.ood_probe.json` | `mu`/`sd` length 512, `w` length 513 | must be refit at 384/385 |

Rebuild with `build_ood_reference.py` then `fit_ood_probe.py`, and re-validate against the
green-legume negatives (see below) — the 0.9681 threshold is a property of the old space.

### Why `DinoV2Classifier` must hold its head as a called attribute — evidenced, not assumed

The embedding contract is "the head's input, captured by a forward pre-hook". That only works if
`head_module` is the object the forward pass actually calls. Checking the three architectures
`build_model` returns today:

| architecture | `head_module` | pre-hook fires? |
|---|---|---|
| resnet18 | `model.fc` — the real submodule | yes, 512-d |
| efficientnet_b0 | `model.classifier` — the real submodule | yes, 1280-d |
| **mobilenet_v3_small** | `model.classifier[2:4]` — **a slice** | **no** |

Slicing an `nn.Sequential` builds a *new* wrapper around the same children. `head[1] is
model.classifier[3]` is `True`, but `head is model.classifier` is `False`, and `model.forward()`
never calls the wrapper — so the pre-hook never fires. Measured:

```
pre-hook on head_module fired during model forward? False
pre-hook on the REAL submodule fired?               True  [torch.Size([2, 1024])]
```

**This is a latent bug in the current codebase**, not a DINOv2 issue: `forward_with_embeddings`
would collect nothing and `torch.cat([])` would raise. It is dormant only because nothing ships
mobilenet — `params.yaml` sets resnet18 and every `models/*.pt` is resnet18 — while
`RunConfig.model_name` still *defaults* to `mobilenet_v3_small`. **§1-B.0 resolves this by deleting the branch
rather than repairing it**, which is the better outcome for code no run has ever used; the
parametrised pre-hook test proposed there is what stops it recurring.

For this plan the consequence is narrow and already handled: `DinoV2Classifier` must expose
`self.head` as a real registered attribute that `forward` calls — which the sketch in §1-B.2
does. Do not be tempted to return `self.head[1]` or a slice of it as `head_module`.

### A gap this exposes, which exists today independently of DINOv2

Three places load the OOD reference. Two check that it belongs to the checkpoint:

- `coffeecv/infer.py` main — checks `checkpoint_sha`, `raise SystemExit` on mismatch
- `coffeecv/ood_eval.py` — checks `checkpoint_sha`, exits on mismatch
- **`webapp/app.py:56` — `ref = json.loads(ref_path.read_text())`, no check at all**

So the production service is the *only* consumer that will accept a reference built from other
weights. Today that yields wrong-but-plausible refusal behaviour; after a DINOv2 swap it becomes
a 512-vs-384 shape error raised per request inside `/classify` rather than once at import.

Fix it before the swap, as its own small commit, because it is a live defect in the deployed
service regardless of what happens to this plan. Mirror the probe's existing loader:

```python
# coffeecv/infer.py -- beside load_ood_probe, which already does exactly this for the probe

def embedding_dim_of(head: torch.nn.Module) -> int:
    """The width of the vector this model feeds its classifier.

    Reads the final Linear's in_features rather than being told, so it cannot
    disagree with the model actually loaded. Uniform across every architecture
    build_model returns: resnet18/efficientnet/mobilenet heads and
    DinoV2Classifier.head are all Sequential(Dropout, Linear).
    """
    *_, last = (m for m in head.modules() if isinstance(m, torch.nn.Linear))
    return last.in_features


def load_ood_reference(checkpoint: Path, head: torch.nn.Module | None = None,
                       explicit: Path | None = None) -> dict | None:
    """Load the reference beside a checkpoint, refusing a mismatched pairing.

    `load_ood_probe` has refused mismatched probes since probe v2; the reference
    had no equivalent, and webapp/app.py was loading it unchecked. A reference
    describes one embedding space: against other weights its centroids are not
    wrong so much as meaningless, and against a different *width* they are a
    shape error deferred to request time.
    """
    path = explicit or reference_path_for(checkpoint)
    if not path.exists():
        return None
    ref = json.loads(path.read_text())
    if ref.get("checkpoint_sha") and ref["checkpoint_sha"] != _sha(checkpoint):
        raise SystemExit(
            f"OOD reference {path} was built from a different checkpoint "
            f"({ref['checkpoint_sha'][:12]} vs {_sha(checkpoint)[:12]}); rebuild it with "
            f"`python -m coffeecv.build_ood_reference --checkpoint {checkpoint}`.")
    if head is not None and ref.get("embedding_dim") not in (None, embedding_dim_of(head)):
        raise SystemExit(
            f"OOD reference {path} is {ref['embedding_dim']}-d but this checkpoint emits "
            f"{embedding_dim_of(head)}-d embeddings -- the reference belongs to a different "
            f"architecture. Rebuild it for these weights.")
    ref["_path"] = str(path)
    return ref
```

and in `webapp/app.py`, replacing the bare load at line 55-58:

```python
from coffeecv.infer import load_ood_reference        # added to the existing import

ref = load_ood_reference(CHECKPOINT, head)           # `head` comes from load_model above
if ref is None:
    logger.warning("OOD guard unavailable: no reference beside %s -- "
                   "predictions below will be unguarded", CHECKPOINT.name)
```

Both mismatches now raise at import, which for a gunicorn worker means the deploy fails loudly
instead of serving 500s per request. Point `coffeecv/infer.py` main and `ood_eval.py` at the same
helper too, so there is one implementation of this rule rather than three.

### Swapping the model in the webapp

Given the above, the actual deployment change is small — the sidecars are addressed by
`<checkpoint>.<suffix>`, so repointing `CHECKPOINT` swaps the whole set atomically:

```python
CHECKPOINT = REPO_ROOT / "models" / "dino_frozen_s17.pt"   # + .json .classes.txt
                                                           #   .ood_reference.json
                                                           #   .ood_embeddings.npz
                                                           #   .ood_probe.json
```

`config_for_checkpoint` reads `model_name` from the card's `training_config`, and `load_model`
already takes `model_name`, so nothing else in the service needs to know which architecture it is
running. Ship every sidecar together — a checkpoint whose reference was left behind either runs
unguarded (if absent) or now refuses to boot (if stale), and the second is the behaviour worth
having.

Remaining deployment notes: N_PATCHES stays 40; whether TTA stays on is Screen D's question, not
an assumption; static/nginx need manual re-copy; a dirty remote working tree is the normal steady
state there; health check is `GET /` -> 404 and `GET /classify` -> 405.

---

## Phase 4-B — OOD guard rebuild and deployment checklist

Phase 4 covered the code changes. This is the operational sequence, in order.

1. **Rebuild the OOD reference and refit probe v2** in the 384-d space
   (`build_ood_reference.py`, then `fit_ood_probe.py`). The threshold 0.9681 and its α ≤ 4.3%
   are properties of the ResNet18 embedding space and carry no meaning in the new one.
2. **Re-validate against the same negatives, including the green legumes.** Probe v1 scored
   0.970 AUROC that turned out to be 0.849 once source confounding was removed, and green legumes
   defeated it entirely. A new embedding space must be tested against that same adversarial set,
   using same-rig negatives, or the 0.970 mistake gets made a second time. **Blocking gate** — a
   classifier win paired with an OOD regression is not a win.
3. **Land the `load_ood_reference` fix separately and first.** It is a live defect in the
   deployed service today (Phase 4), so it should not arrive tangled in an architecture swap.
4. **Checkpoint size.** A DINOv2 checkpoint is ~88 MB against resnet18's 44 MB. It is DVC-tracked,
   so this is a storage note, not a problem — but the DVC remote's three known traps apply, and
   the push must be verified by expanding `dvc.lock`'s `.dir` trees rather than trusting
   `dvc status --cloud`.
5. **Stop `coffee-cv-web.service` while experiments run** on the VM, and restart and verify it
   afterwards.
6. **Deploy**: repoint `CHECKPOINT`, rsync, re-copy static/nginx by hand, then health check
   (`GET /` → 404, `GET /classify` → 405) and one real photo end to end with the latency timed.

---

## 5. Risks, and how each one announces itself

| risk | how it shows up | mitigation |
|---|---|---|
| Phase 0 collapses the screen's margin | corrected cross-rig figure far below 0.8511 | that is the point of Phase 0; it costs 2 h and precedes all code |
| Stale embedding cache trains a head on another fold's features | absurdly good val, cross-rig unchanged | key the cache on weights sha + resize + seed + rig + split + fold + `dinov2_unfreeze_blocks`; refuse on mismatch |
| Cache silently used at unfreeze depth >= 1 | training loss falls while the backbone never actually updates | the cache is valid only at depth 0 — `build_model` and the cache loader must both assert it, not just document it |
| MixStyle silently dropped | DINOv2 arm looks good for the wrong reason, or bad for the wrong reason | `build_model` raises on `mixstyle_p > 0`; the note records the design difference |
| `patch_resize` changed away from a multiple of 14 | DINOv2 raises at the first forward — loud, but minutes into the run, after patch materialisation | assert `patch_resize % 14 == 0` at config construction |
| OOD guard regresses unnoticed | classifier metrics improve, refusals get worse in the field | Phase 4 step 2 is a blocking gate, tested on the green-legume set |
| Uncommitted source during a sweep | archived `git_commit` whose tree lacks the DINOv2 branch | `run_folds.py`'s dirty-provenance gate already catches this — do not pass `--allow-dirty` |
| Single-seed win is a favourable draw | 4/4 at seed 42, mixed at seed 123 | Screen B; the DINOv2 seeds are free, so there is no excuse to skip them |
| `coffeecv_dino` grows a private copy of patch sampling or the macro metric | both arms look fine in isolation, deltas are meaningless | the import table in Phase 1; a test asserting the package defines no sampler/metric of its own — this has bitten `analysis/bean_scale` twice |
| An old archived config names a pruned architecture and is silently substituted | a re-run of an early experiment reports numbers under the wrong backbone | `build_model`'s `else` raises with the removal date and a pointer to check out that run's own commit (§1-B.0); never fall back to a default |
| Excluded full fine-tune creeps back in | a run at `freeze_mode=none`, or `dinov2_unfreeze_blocks` near 12, quietly costing 3.3x | `build_model` raises on any `freeze_mode` but `full`; config bounds N to 0..6 (§1-B.2, §1-B.3) |

---

## 6. What "done" looks like

An adoption decision recorded in `EXPERIMENTS_LOG.md` resting on: 12 paired fold×seed deltas with
the in-distribution and photo-pooled controls beside them; DINOv2's own measured seed sd; a
rebuilt and re-validated OOD guard including the green-legume negatives; an end-to-end measured
`/classify` latency; and a plainly written note that this was a comparison of two *designs*, not
an architecture ablation.

Or a recorded negative with the same rigour, which is equally valuable and much cheaper.

---

## 7. The fallback if Phase 0 or Screen A fails

If DINOv2 cannot be the classifier — because the corrected protocol shrinks the margin, or
because the OOD guard cannot be made to work in 384-d — the screen's finding does not evaporate.
It says a representation exists that separates these beans across cameras far better than
anything ResNet18 reaches on its own. The way to use it then is **feature distillation**: keep
the serving ResNet18, add a loss pulling its 512-d embedding toward a projection of DINOv2's
384-d embedding on the same patch, and train as usual. The teacher runs offline only, so serving
cost, MixStyle, TTA and the existing OOD guard are all preserved unchanged.

That is the same structural move as the SAM→U-Net distillation in
`docs/architecture_screen_plan.md` R2, and for the same reason: on this hardware every strong
model is affordable offline and unaffordable live.
