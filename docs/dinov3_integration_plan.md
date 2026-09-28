# DINOv3 in `coffeecv`: implementation, first experiment, and integration plan

Status: Step 0 and Experiment 1 DONE (2026-09-25); **the owner selected DINOv3 ViT-B/16 on
2026-09-26** (§0.4), so every step from 1b on is run with `dinov3_vitb16 × cls_mean`. **2026-09-27:**
the depth-0 integration (§8.6) and a deploy-ready candidate, `models/allrigs_dino3b16_s123.pt`, with a
rebuilt and holdout-verified OOD guard (§9.3), are on branch `dino-integration`. **2026-09-27: deployed.**
The owner chose it as the first release of the isolated `/opt/coffee-cv` service (OPS-1,
`docs/ops1_release_isolation_plan.html`); it passed the smoke test in the real systemd sandbox (§9.4) and
has been live since 14:25 UTC. That was the owner's call ahead of the §7.2 comparison, which is still
open: exp232-239 finished 2026-09-28 and are merged (`b224982`), so its 12 paired deltas can now be
computed. This file
replaces `docs/dinov2_integration_plan.md` (written 2026-09-21; its last version is at commit
`1caf2f1`). On 2026-09-24 the owner decided that **DINOv3 is the family this project builds first**,
and that the first experiment is implementing it and testing it, with ViT-S/16 as the pre-registered
primary cell. DINOv2 stays in that experiment only as a reference measurement. §0.2 lists what changed
and why.

This is still the follow-up to `docs/architecture_screen_plan.md` §8 R1 and the frozen-feature
screen in `analysis/architecture_screen/`. Everything up to and including Experiment 1b runs on
the workstation. The first step that needs the VM is the ResNet18 seed baseline (§7.1).

---

## 0. Where this stands

### 0.1 The evidence, and why it is not admissible yet

The screen froze each backbone, embedded identical patches, fitted one logistic regression, and
scored leave-one-camera-out macro-F1 (2026-09-21):

| backbone | 4-fold mean | 3 ten-class folds |
|---|---|---|
| resnet18_in1k (current family) | 0.4645 | 0.4938 |
| convnext_tiny_in1k | 0.6691 | 0.7013 |
| vit_small supervised in1k | 0.5534 | 0.5548 |
| **dinov2_vits14** | **0.8324** | **0.8511** |

Attribution is settled: the gain is not the recipe (`resnet18.a1_in1k` scores *worse*, 0.4496)
and only marginally the transformer (a same-size supervised ViT-S gains +0.09). DINOv2 beats that
supervised ViT-S by +0.279, so the credit belongs to DINO self-supervision at scale. For scale,
the fine-tuned ResNet18 + MixStyle recipe scores **0.8025** on the same three folds (exp200–203,
patch-level, val-peak checkpoint, **no TTA**; `train_baseline.evaluate` is a plain loop). The
previous version of this plan called that figure "with TTA". It is not: deployed TTA adds about
+0.0235 on top.

**The screen is not admissible evidence for a change.** It differs from a real fold in four ways.
The first version of this plan listed three; the second was found while writing this one:

1. It trains on `split="all"` of each training rig (every photo), not the 70% train split.
2. **It builds each rig as its own one-rig dataset, so `rig_idx` is 0 for every rig.** `rig_idx`
   is the rig's position in the fold's `train_rigs` list. It seeds the photo split
   (`split_photos_by_class`, `dataset.py:291`) and every patch box (`_extract_photo`,
   `dataset.py:469`). So switching the probe to `split="train"` would still draw a different
   70% of photos and different boxes than exp200–203 trained on. The held-out side is not
   affected: a real fold also builds the held-out rig as a one-rig `split="all"` dataset with
   `rig_idx` 0 and 120 patches per class. That is why the screen's *evaluation* patches already
   match a real fold. Only its training side differs.
3. It does not restrict the macro average to present classes. cam_iphone has **8** of the 10
   classes (no class_008, no class_010; `xrig_macro_n=8` in `index.csv`). Any prediction of an
   absent class enters the average as a zero-F1 class.
4. It scores patches, not photos.

Experiment 1 (§5) removes all four.

### 0.2 What changed in this rewrite (2026-09-24)

| | DINOv2 version (2026-09-21) | this version |
|---|---|---|
| primary arm | `dinov2_vits14` | **`dinov3_vits16`** (owner's decision) |
| DINOv2's role | the candidate | reference arm in Experiment 1 only |
| tie-break rule | within noise → DINOv2, for its Apache-2.0 licence | retired. If DINOv2 wins on every pair, stop and ask the owner (§5.6 G3) |
| loader | Meta's repo pinned at a SHA | **timm**, verified bit-identical to Meta's reference after one fix (§2) |
| readout | CLS token (what the screen measured) | CLS vs **CLS ⊕ mean of patch tokens** (Meta's own linear-eval input), both measured |
| input size | 224 | **224 only** (owner's decision, 2026-09-24). 256, DINOv3's native size, was considered and dropped |
| head at depth 0 | SGD-trained Linear over a dihedral embedding cache | **L2 logistic regression**, C chosen on val, exported as an `nn.Linear`. Depth 0 needs no training loop and no cache |
| fold pairing | not addressed | new prerequisite: extract `build_fold_datasets` (§4.1) |
| cost of the first measurement | "~2 hours, six seeds free" | **~4 hours for three seeds**, unattended. Real-fold patches change with every seed *and* every fold |
| OOD guard | checked at the end | **feasibility check moved to right after Experiment 1**, dev split only (§6.1) |
| VM baseline | "choose deliberately" | **launch ResNet18 seeds 123/7 as soon as Experiment 1 passes**, because it is the long pole (§7.1) |
| order of the final steps | Screen C (unfreezing) before shipping | **ship frozen first**, Screen C afterwards (§3) |

### 0.3 Decisions already taken, not reopened here

- **Full fine-tuning is out of scope.** Head-only or last-N blocks only, with N bounded to 0..6
  in config (§8.3).
- **The DINO work starts as a separate package, `coffeecv_dino/`**, which imports from
  `coffeecv` and never copies (§4.2). The exception is the two small pure refactors in §4.1.
  They exist *so that* nothing has to be copied.
- **`mobilenet_v3_small` and `efficientnet_b0` get deleted** (§8.0).
- **One file per architecture behind an `Arm` protocol**, with schedulers shared (§8.1).
- **The DINOv3 licence is not a blocker** (Appendix A).
- **DINOv3 runs at 224 px only.** 256, its native size, was considered and dropped by the owner
  on 2026-09-24. §1.5 shows why 224 is sound.
- **The backbone is DINOv3 ViT-B/16, readout cls_mean** (owner, 2026-09-26; §0.4).

### 0.4 Experiment 1's result, and the owner's choice of ViT-B/16 (2026-09-26)

Experiment 1 ran on the VM (2026-09-24 17:00 to 2026-09-25 23:28 UTC, commit `8460a40`); the owner
extended the optional arms to all three seeds and added photo-level scoring for them. Cross-rig
macro-F1, mean of the three ten-class folds, 3 seeds, patch level, no TTA:

| backbone | readout | 3-fold mean | seed sd | photo-pooled |
|---|---|---|---|---|
| resnet18, frozen (control) | avgpool | 0.5225 | 0.005 | |
| dinov2_vits14 | cls / cls_mean | 0.8563 / 0.8522 | | |
| dinov3_vits16 (**primary cell**, exp220–231) | cls_mean | **0.8487** | 0.003 | 0.9136 |
| dinov3_vits16 | cls | 0.8596 | 0.005 | |
| dinov3_vits16plus | cls_mean | 0.8512 | | |
| **dinov3_vitb16 (selected, exp240–251)** | **cls_mean** | **0.8873** | 0.023 | see §5.8 |
| dinov3_vitb16 | cls | 0.8921 | 0.019 | |
| fine-tuned resnet18 + MixStyle (exp200–202, seed 42 only) | | 0.8025 (last-10 0.8092) | | |

Gates (§5.6), on the primary cell: **G1 PASS** (12/12 over frozen R18, +0.352), **G2 PASS** (0.8487),
**G3 not triggered** (V2 wins 8/12, mean +0.008: a tie), readout switch not triggered (CLS-only 9/12).

**Why B/16:** it is the only arm that moved. Paired against the primary cell it wins **11/12**
(+0.0354 mean), and photo-pooled 7/8 (+0.037); S+/16 is a null (5/12, −0.0003). The owner chose it
over S/16 knowing the price: ~3× S/16's per-image cost and a wider seed spread (0.023 vs 0.003).
**The readout stays cls_mean** by the same pre-registered rule applied to the primary cell: CLS-only
beats it on 9/12 B/16 pairs (+0.005), not 12/12.

What this changes downstream, and nothing else:

- **Every later step uses `dinov3_vitb16 × cls_mean`**: the OOD feasibility check (§6.1), the latency
  gate (§6.2, re-measured), the adoption comparison (§7.2, against exp240–251), Screen D (§7.3),
  integration (§8) and shipping (§9).
- **Widths:** the B/16 embedding is 768-d; cls_mean is **1536-d**, so the head is `Linear(1536, 10)`
  and the OOD probe reads 1536-d patch embeddings.
- **The weight file is 343 MB** (`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`, 85.7 M params), not
  86 MB. It is already on the VM and verified. The unfreeze bound 0..6 still means half the blocks:
  ViT-B/16 has 12, like S/16.
- **Experiment ids:** exp232–239 stay the ResNet18 seed 123/7 folds (§7.1); the selected cell's 12
  runs are archived as **exp240–251** (seed 42 → 240–243, 123 → 244–247, 7 → 248–251) by
  `python -m coffeecv_dino.screen --archive selected`.
- ViT-S/16 stays in the record as the pre-registered primary cell and as the cheaper fallback if B/16's
  latency or OOD space fails a gate.

---

## 1. The candidate, and what to expect from it

### 1.1 Facts, measured on the workstation

| | dinov2_vits14 (reference) | **dinov3_vits16** | dinov3_vits16plus | dinov3_vitb16 |
|---|---|---|---|---|
| params | 22.1M | 21.6M | 28.7M | 85.7M |
| embed dim | 384 | 384 | 384 | 768 |
| patch size | 14 | 16 | 16 | 16 |
| tokens at 224 px | 256 | 196 | 196 | 196 |
| extra tokens | CLS | CLS + 4 registers | CLS + 4 registers | CLS + 4 registers |
| position encoding | learned absolute, interpolated to the input grid | RoPE, so any multiple of 16 works with no interpolation | same | same |
| CPU ms/img at 224 | 48.8 | 44.4 (Meta's code, 09-21) / 46.2 (timm, 09-24) | not measured | not measured |
| licence | Apache-2.0 | DINOv3 License | same | same |
| role here | reference | **candidate** | optional single-seed arm | optional upper bound and teacher |

Timings: batch 32, 6 threads, Ryzen 5 5600U. The two dinov3_vits16 figures at 224 were taken on
different days with implementations that produce bit-identical outputs (§2.1), so treat them as
equal until a same-session benchmark says otherwise. For reference, resnet18 costs 13.9 ms/img.

### 1.2 Expectation: at ViT-S, v3 is not a guaranteed win on a global task

Published ImageNet-1k linear-probe accuracy:

- DINOv2 ViT-S/14: **81.1%** (DINOv2 README).
- DINOv3 ViT-S/16: **81.4%**. Meta reports only ImageNet-ReaL for the distilled models. The 81.4%
  comes from third-party heads trained to Meta's protocol at 512 px, which also reproduce Meta's
  ReaL figure (87.08%; `yberreby/dinov3-in1k-probes`).

That is a 0.3-point gap, at a resolution this project will not use. DINOv3's headline gains are on
dense tasks (segmentation, depth), which is what its Gram-anchoring objective targets. So
"DINOv3 beats DINOv2 on beans" is a hypothesis for Experiment 1 to test, and testing it costs
only one reference arm.

One property might favour DINOv3 here specifically. A bean patch is texture: 36–81 beans, with no
single object in it. DINOv3's patch tokens are its strongest output, and the patch-mean readout
(§5.2) is how a linear head gets access to them.

### 1.3 What carries over unchanged (verified)

- **Normalisation.** The LVD-1689M models use ImageNet mean/std. Both timm's `_dinov3_cfg` and
  Meta's README `make_transform` say so, and both match `coffeecv/transforms.py`'s
  `IMAGENET_MEAN`/`IMAGENET_STD` exactly. So `build_{train,eval}_transform` work unmodified, and
  train/inference parity comes for free. **Never use the SAT-493M satellite checkpoints**: they
  use different statistics.
- **224 px is valid** (14 × 16).
- **The embedding contract is generic.** `forward_with_embeddings` captures the head's *input*
  with a pre-hook, whatever its width.

### 1.4 What is different, and what each difference costs

| difference | consequence |
|---|---|
| patch size 16, not 14 | input guard is `patch_resize % 16 == 0`. Derive it from the loaded weights, don't hardcode it (§2.2). **Meta's code does not enforce it:** its `PatchEmbed` has the divisibility asserts commented out, so a 250 px input silently becomes its top-left 240×240 (verified: output identical). timm raises, and DINOv2 asserts |
| native input size is 256 (timm's `input_size`, Meta's `make_transform` default) | **Not used.** DINOv3 runs at 224 only (§0.3), which gives 196 tokens instead of 256. For the record, 256 would cost +34% (46.2 → 61.7 ms/img). §1.5 explains why 224 is sound |
| three readout conventions | calling the model upstream returns the **CLS** token. Upstream's linear-classification eval feeds **CLS ⊕ mean(patch tokens)** (`dinov3/eval/linear.py`: `setup_linear_classifiers` only builds `avgpool=True` heads). **timm defaults to `global_pool="avg"`**, the patch mean alone, which is a third convention. Always set the readout explicitly |
| 4 register tokens | token layout is `[CLS, R1..R4, P1..PN]`. Code that slices patch tokens must skip `num_prefix_tokens` (= 5), not 1 |
| RoPE periods stored in bfloat16 | see §2 |

### 1.5 How each model handles input size (measured 2026-09-24)

Neither model has a fixed input size. Both take any side that is a whole number of patches. They
differ in how they encode where a patch sits, and in the sizes they were trained at:

| | DINOv2 ViT-S/14 | DINOv3 ViT-S/16 |
|---|---|---|
| position encoding | learned table of **37 × 37** (+ CLS). 518 / 14 = 37, from the final high-res phase | **none stored**. RoPE angles are computed per forward pass from coordinates normalised to [−1, 1] on each axis |
| what 224 px does | 16 × 16 grid. The 37 × 37 table is **bicubically resized** to 16 × 16 every forward pass | 14 × 14 grid. Nothing is resized: the coordinates are just computed on a 14 × 14 grid (neighbour step 0.143 vs 0.125 at 256) |
| non-multiple of the patch size | `AssertionError` (256 fails) | Meta's code **silently crops**; timm raises (§1.4) |
| training sizes | 224 global / 96–98 local crops, then a short phase at 518 | 256 global / 112 local crops, with RoPE coordinates randomly rescaled ×[0.5, 2] in training (`rescale_coords: 2`). The 7B teacher was then adapted at 512–768. The public distilled-model config (ViT-L) uses 256 / 112 |

**Token grid, not pixels, is what differs at the same input size.** At 224 px, DINOv2 sees a
16 × 16 grid and DINOv3 a 14 × 14 grid, so each DINOv3 token covers more of a bean. DINOv3
would need 256 px to match DINOv2's 16 × 16 grid. Experiment 1 runs both models at 224 only
(§0.3), so V2 vs V3 is a same-pixels comparison. If V2 wins, part of that could be DINOv3's
coarser grid rather than its generation; record that caveat with G3 (§5.6) if it comes up.

**How far features move when the size changes.** 40 real bean-unit patches from cam_sony (4 per
class, seed 42, the project's `build_eval_transform`), centred cosine similarity, CLS readout:

| | another patch, same class (224) | another class (224) | same patch, one grid step up | same patch, ~2.3× the pixels |
|---|---|---|---|---|
| DINOv2 | 0.30 | 0.01 | 224→252: **0.90**, finds itself 100% | 224→518: 0.61, finds itself 95% |
| DINOv3 | 0.31 | 0.00 | 224→256: **0.88**, finds itself 100% | 224→512: 0.62, finds itself 100% |

The CLS ⊕ patch-mean readout gives the same picture (within ±0.04). Two consequences:

- **One grid step is a small move.** Every patch stays its own nearest neighbour, far closer than
  another patch of the same bean class. Running DINOv3 at 224 rather than its native 256 doesn't
  give it a different view of the bean, which is what makes 224-only a sound choice. It doesn't
  show whether 256 would classify better; that is deliberately not tested (§0.3).
- **A large change is not neutral** (cosine ≈ 0.6). A head fitted at one size must be served at
  the same size. `patch_resize` travels in the checkpoint card, and `config_for_checkpoint`
  restores it, which is the existing train/inference parity rule.

This is illustrative only: one rig, one seed, 40 patches.

---

## 2. The loader: timm, verified bit-identical to Meta's reference

### 2.1 What was measured on 2026-09-24

- **Meta reference:** `facebookresearch/dinov3` at `6876159a11b4df116f30f667f8c9888617df0751`
  (2026-07-15), cloned to scratch. Built with `dinov3.hub.backbones.dinov3_vits16(pretrained=False)`
  and `load_state_dict(strict=True)`.
- **timm:** version 1.0.29, already in the devcontainer venv. It was installed by hand for probe2
  and is not in any requirements file. Built with
  `timm.create_model("vit_small_patch16_dinov3", pretrained=False, num_classes=0, global_pool="token")`,
  then timm's `checkpoint_filter_fn`, then a strict load.
- Same weights file (`...-08c60483.pth`), same random input, eval mode.

| check | result |
|---|---|
| strict load, both implementations | all keys matched, no missing or unexpected keys |
| checkpoint `qkv.bias` (12 blocks) | exactly 0, and `bias_mask` all 0, so timm's no-bias variant loses nothing |
| CLS token, timm as shipped vs Meta | max abs diff **1.2–1.6e-3** (mean \|CLS\| ≈ 0.41), at both 224 and 256 |
| patch tokens, same comparison | max abs diff up to **1.4e-2** |
| cause | the checkpoint stores `rope_embed.periods` as **bfloat16** (max relative error 1.9e-3 vs fp32). Meta loads it as a persistent buffer. timm discards it and recomputes the periods in fp32 |
| Meta's code with fp32 periods vs timm | **0.0**, so the periods are the only difference |
| timm with the checkpoint's periods copied in | **0.0 at 224 and at 256: bit-identical** |

Which is correct? The checkpoint's values. The stored dtype shows the model carried
bf16-rounded periods (`DinoVisionTransformer` defaults `pos_embed_rope_dtype` to bf16), and
Meta's reference loads exactly those values. timm's fp32 recompute is a small perturbation of a
trained network. It very likely makes no difference to a linear probe, but it breaks bit-exact
testing, and that property is worth keeping.

Not the cause: Meta's `norm_layer="layernormbf16"` is a plain fp32 `nn.LayerNorm(eps=1e-5)`.
Meta's `rescale_coords=2` only applies in train mode (§10).

### 2.2 The code (sketch)

```python
# coffeecv_dino/backbone.py
ARCHS = {  # arm name -> (timm model name, file under models_pretrained/)
    "dinov3_vits16":     ("vit_small_patch16_dinov3",      "dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    "dinov3_vits16plus": ("vit_small_plus_patch16_dinov3", "dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
    "dinov3_vitb16":     ("vit_base_patch16_dinov3",       "dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
}

def build_dinov3(arm: str, root: Path = MODELS_PRETRAINED) -> nn.Module:
    timm_name, rel = ARCHS[arm]
    path = root / rel
    verify_against_manifest(path)          # full sha256 vs models_pretrained/manifest.json
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # timm's no-bias variants drop qkv.bias. That is lossless only if the biases are zero, so check it.
    assert all(not v.any() for k, v in sd.items() if k.endswith("qkv.bias")), f"{arm}: non-zero qkv bias"
    model = timm.create_model(timm_name, pretrained=False, num_classes=0,
                              global_pool="token")        # upstream's readout; timm's default is "avg"
    with contextlib.redirect_stdout(io.StringIO()):       # the filter prints one line per zero bias
        converted = checkpoint_filter_fn(dict(sd), model)
    model.load_state_dict(converted, strict=True)         # a partial load gives plausible garbage for a whole sweep
    # timm recomputes the RoPE periods in fp32; the checkpoint holds bf16-rounded ones. Copy them in
    # so the output is bit-identical to facebookresearch/dinov3 (plan §2.1). The copy works because
    # dynamic_img_size=True leaves no cached sin/cos table. Assert that, so a timm release that starts
    # caching fails here instead of silently ignoring the copy.
    assert model.rope.feat_shape is None
    model.rope.periods.copy_(sd["rope_embed.periods"].float())
    return model.eval()

def assert_input_size(model: nn.Module, patch_resize: int) -> None:
    p = model.patch_embed.patch_size[0]                   # read from the model, never hardcoded
    if patch_resize % p:
        lo, hi = patch_resize // p * p, (patch_resize // p + 1) * p
        raise ValueError(f"patch_resize={patch_resize} is not a multiple of {p}; nearest valid: {lo} or {hi}")

def readout(model: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    t = model.forward_features(x)                         # [B, 1+4+N, D], final norm already applied
    cls = t[:, 0]
    if mode == "cls":
        return cls                                        # D (384 for S/16)
    if mode == "cls_mean":                                # upstream linear-eval input, 2D (768)
        return torch.cat([cls, t[:, model.num_prefix_tokens:].mean(1)], dim=1)
    raise ValueError(mode)
```

Call `assert_input_size` when the config is built, not at the first forward pass. Otherwise a bad
`patch_resize` is only discovered after every patch has been decoded.

### 2.3 Why timm rather than Meta's repository

1. **Licence boundary.** The DINOv3 License's "DINO Materials" explicitly include "machine-learning
   model code … inference-enabling code", so Meta's repository is under the same terms as the
   weights. timm's implementation is timm's own Apache-2.0 code. With timm, the only
   DINOv3-licensed thing this project touches is the weight file, which is gitignored and kept in
   the private DVC remote. The previous version's whole vendor-vs-pin analysis becomes
   unnecessary: which of three ways to ship Meta's code, plus a private mirror.
2. **It can reach production.** The webapp installs with `pip install --require-hashes`
   (`webapp/requirements.txt`, compiled with pip-compile), and pip cannot hash a VCS requirement.
   So the previous plan's "pin Meta's repo as a git dependency" could never have been deployed
   without vendoring or building a private wheel. `timm==1.0.29` pins with hashes like every other
   dependency. Its transitive dependencies: torch and torchvision (already present), pyyaml,
   huggingface_hub, safetensors.
3. **Two traps disappear.** `hubconf.py`'s torchmetrics failure, and `weights=<path>` copying
   82 MB into the torch hub cache, only exist in Meta's loading path (Appendix B).
4. **Tooling for Screen C:** `forward_intermediates`, gradient checkpointing, and layer-decay
   parameter groups.

What it costs:

- timm is a reimplementation, so equivalence has to be enforced by a test (§2.4) rather than
  assumed. **Pin timm's exact version and treat any upgrade as an architecture change** that must
  pass the test again.
- Record `timm.__version__` in every run's `config.json`, beside the weights' sha256. Weights and
  architecture definition go together, or the record means nothing.

### 2.4 Tests (`tests/test_dino_backbone.py`)

Skip with a pointer to `models_pretrained/verify.py` when the weights are absent, since they are
gitignored.

1. **Equivalence with Meta's reference, against a stored fingerprint.** A one-off script
   (`scripts/make_dinov3_fixture.py`) clones `facebookresearch/dinov3` at `6876159` into a temp
   directory. It runs the reference on a fixed seeded input at 224 and saves CLS and
   patch-mean (a few KB) to `tests/fixtures/`. The test compares timm's output against it with
   `atol=1e-4`. Meta's code is never committed; the fixture is model output, which the licence does
   not restrict. **Use a tolerance, not `torch.equal`:** bit-exactness holds on one machine, but a
   different CPU reorders floating-point sums. Measured 2026-09-24: the same weights on the
   workstation (Zen 3, AVX2) and the VM (EPYC Genoa, AVX-512) differ by up to **3.2e-5** on real
   bean patches (cosine ≥ 0.9999998), so 1e-5 would fail on the VM. 1e-4 clears that and is still
   more than 10× below the periods bug (1.2–1.6e-3).
2. The strict load has no missing or unexpected keys, and every `qkv.bias` is zero.
3. `model.rope.periods` equals the checkpoint's values, and `rope.feat_shape is None`.
4. The input guard accepts 224 and rejects 230 with a message naming 224 and 240.
5. Readout shapes: `cls` is 384, `cls_mean` is 768, and `num_prefix_tokens == 5`.
6. **No private copies:** nothing under `coffeecv_dino/` defines `load_rgb_image`,
   `estimate_bean_pitch`, `sample_bean_unit_patch_boxes`, `build_eval_transform`,
   `compute_split_metrics` or `build_fold_datasets`. `analysis/bean_scale` has twice grown a
   private copy that silently drifted from production (§4.2).

### 2.5 Weights: obtaining and verifying them (DVC still pending)

The files in `models_pretrained/dinov3/` are the **Meta-direct `.pth`** checkpoints, verified
against `models_pretrained/manifest.json` by `python models_pretrained/verify.py`. That script
exits non-zero on any missing, truncated or wrong file.

- **Never use `pretrained=True`.** timm's `vit_small_patch16_dinov3.lvd1689m` tag downloads a
  different file from the Hugging Face hub (`timm/` org), and so does Meta's HF repo
  (`facebook/dinov3-vits16-pretrain-lvd1689m`, safetensors). Neither matches the manifest.
  Always load the manifest-verified `.pth` as in §2.2. A download at run time is also exactly
  the network dependency `models_pretrained/` exists to remove.
- **On a fresh clone:** `dvc pull`, once the weights are `dvc add`ed. That is your own private
  storage, not redistribution. Anyone else has to request the files from Meta (gated). The public
  repo cannot ship them (licence §1b-i).
- **Never rename the files.** The `-08c60483` suffix is torch's sha256 prefix, and it is what lets
  the file verify itself.
- **On a hash mismatch, do not "fix" the manifest.** In order of likelihood: HF safetensors were
  downloaded instead of the `.pth`; the download was truncated (compare byte counts first, it is
  the fastest check); or Meta re-issued the file. In the last case, record it in `manifest.json`
  with a date, because every earlier run used different weights.
- **`dvc add` is still pending.** It needs no remote, but the push does, and the remote was down
  for maintenance on 2026-09-24. Verify the push by expanding `dvc.lock`'s `.dir` trees, never by
  trusting `dvc status --cloud`.
- **The VM already has them.** All five files were copied to `powervpsssh:~/coffee-vision/models_pretrained/`
  by rsync on 2026-09-24 (679 MB) and pass `verify.py` there. DINOv2's hub checkout was copied to
  the VM's `~/.cache/torch/hub/` as well, so the V2 reference arm loads offline there too.

---

## 3. Order of work

Each step gates the next unless it is marked parallel.

| # | step | where | cost | gate / output |
|---|---|---|---|---|
| 0 | **Prerequisites:** pin timm; `coffeecv_dino/backbone.py` and its tests; extract `build_fold_datasets` and the photo-pooling core (§4) | workstation | ½ day | equivalence test passes; refactor proven behaviour-neutral (history diff empty) |
| 1 | **DONE 2026-09-25 — Experiment 1: frozen DINOv3 under the real fold protocol** (§5, result §0.4) | workstation and VM, split by seed (§5.5) | ~4 h on one machine, ~2.5 h on both | gates G1–G3; primary cell archived as exp220–231 |
| 1b | **DONE 2026-09-26, both pass for B/16 (§6.1, §6.2)** — **Kill-switch:** OOD feasibility in the frozen space, dev split only (§6.1). VM latency is already measured and passes (§6.2) | workstation | ~1–2 h | OOD no worse than today's guard on dev |
| 2 | **ResNet18 camera-rig folds at seeds 123 and 7** (§7.1) | VM | ~60 h | the multi-seed baseline any adoption needs |
| 2′ | parallel: Screen D (TTA and dihedral-augmented head); 6-seed extension if a lever is borderline; housekeeping H1–H3 | workstation | hours | |
| 3 | **Adoption decision** (§7.2) | — | — | 12 paired (fold, seed) deltas |
| 4 | **Integration into `coffeecv`:** prune, `Arm` refactor, `arms/dinov3.py` (§8) | workstation, on a branch | 1–2 days | resnet18 3-epoch history diff empty |
| 5 | **Ship frozen DINOv3:** all-rigs head, OOD rebuild, holdout verified once, webapp deploy (§9) | workstation + VM | ~1 day | health check plus one timed real photo |
| 6 | Screen C: unfreeze depth, N = 2 first (§10) | VM | ~1 day per 4 folds per seed | only if more accuracy is wanted after shipping |

**Why this order:**

- **The cheapest decisive measurement comes first.** Experiment 1 costs about four unattended
  hours on the workstation. It decides whether anything after it is worth doing, and it settles
  the readout choice along the way.
- **Kill-switches come before long VM runs.** A frozen backbone fixes the OOD embedding space
  before any head exists, so the OOD check can end the plan for about an hour's work. Latency, the
  other kill-switch, was measured on the VM with the real weights on 2026-09-24 and passes (§6.2).
- **The long pole starts as early as it is justified.** Once G1–G3 pass, a multi-seed ResNet18
  baseline is certain to be needed. It is also useful if DINOv3 later fails, because every
  future camera-rig lever needs it. At ~60 h, the only way to keep it off the critical path is to
  start it immediately and do everything else while it runs.
- **Ship frozen, then consider unfreezing.** A frozen model that passes the adoption bar is
  already better than production and cheaper to serve. Shipping it needs only inference-side
  integration. Screen C costs VM days per seed and gives up the frozen advantages (seconds-long
  refits, tiny checkpoints, a fixed OOD space). Its question, "how much more does unfreezing
  buy?", is best answered paired against a shipped frozen baseline.

**Housekeeping, in parallel and not gating anything:**

- **H1.** Fix the unchecked OOD-reference load in the webapp (§9.2). It is a live defect today,
  whatever happens to this plan.
- **H2.** Prune `mobilenet_v3_small` and `efficientnet_b0` (§8.0).
- **H3.** `dvc add` the pretrained weights, and push when the remote is back (§2.5).

---

## 4. Step 0: prerequisites

### 4.1 Extract the fold-dataset construction (the pairing fix)

`train_baseline.main()` builds its four datasets inline (lines 148–235): rig resolution,
`patches_per_class`, `photo_frac`, `common_kwargs`, and four `MultiPhotoPatchDataset` calls.
Importing `MultiPhotoPatchDataset` is therefore not enough to reproduce a fold. Rig order, split
budgets and geometry kwargs would all have to be re-typed, and the probe's `rig_idx` bug (§0.1
item 2) is exactly what re-typing produces.

Three small pure refactors, each its own commit:

1. **`build_fold_datasets(cfg, train_transform, eval_transform)`**, moved out of
   `train_baseline.main()` (into `coffeecv/dataset.py` or a small new module). It returns
   `train/val/test/xrig` plus `class_ids`/`class_labels`, and `main()` calls it. The DINO path
   calls it with `eval_transform` for the training split too, because frozen arms get no
   photometric augmentation (§5.3).
2. **`run_folds.train_rigs_for(heldout)`**, the one-line `[r for r in RIGS if r != heldout]`
   that is currently inline in `set_fold`. That gives the DINO driver the fold's rig *order*
   (and so its `rig_idx` values) without rewriting `params.yaml`.
3. **A photo-pooling core.** `photo_pooling_eval.evaluate(exp_id, ...)` loads a model from an
   archived experiment id. Split it into a loader (exp id → model, head, cfg) and a scorer
   (model, head, cfg → pooled metrics). Then both arms are pooled by one implementation, with the
   same inference-time seed keys.

**As built (2026-09-24, commits `ca3c6b0` and `279cd15`).** Item 3 turned out smaller than
planned: the scorer already existed as `xrig_eval.run_photowise(model, head, cfg, rig, ...)`.
Only the pooling rule was extracted, as `photo_pooling_eval.pool_photos()`, and checked against
the inline code it replaced. `archive_experiment.archive()` also gained `src_dir` (default
`outputs/`), so screen runs archive from their own directories.

These touch `coffeecv/`, which `dvc.yaml`'s train stage depends on, so each must be proven
behaviour-neutral:

```bash
python -m coffeecv.train_baseline --epochs 3 --seed 42 && cp outputs/history.json /tmp/before.json
# ...refactor...
python -m coffeecv.train_baseline --epochs 3 --seed 42 && diff /tmp/before.json outputs/history.json   # MUST be empty
```

Every epoch's `train_loss` and `val_macro_f1` must match to the last digit. A moved RNG draw
would invalidate every pairing against exp200–203. None of this may happen while a sweep is
running on the same checkout.

### 4.2 The `coffeecv_dino` package: what it imports, and what it may own

**It must import these from `coffeecv`, never reimplement them:** `MultiPhotoPatchDataset` and
`build_fold_datasets` (byte-identical patches are the whole value of a paired comparison);
`transforms.build_{train,eval}_transform`; `bean_scale`; `metrics.compute_split_metrics` and
`build_metrics_json` (with `macro_labels=present_class_idxs`); `RunConfig`, extended by
composition (`DinoConfig` holds a `RunConfig`); `archive_experiment.archive`;
`infer.forward_with_embeddings` and `patches_for_photo`; and the photo-pooling core.

**It may own:** building the backbone and verifying its weights; the readout; fitting and
exporting the depth-0 head; its own driver; and its own small config.

**Why the rule matters here:** `analysis/bean_scale/estimators.py` does not import `coffeecv`.
Editing the production estimator once returned byte-identical benchmark output, because the
benchmark was scoring a private copy. Its `m0_fft_radial` was a second private copy that had
drifted. A separate folder is not free in this repo. It may own behaviour that is genuinely
different, and never a second copy of behaviour that has to stay identical. Test 6 in §2.4
enforces that.

Layout for step 0 and Experiment 1:

```
coffeecv_dino/
  __init__.py
  backbone.py   # §2.2: timm build, manifest sha256 check, periods copy, input guard, readout
  head.py       # §5.3: L2 logistic regression, C chosen on val, exported to nn.Linear
  model.py      # DinoClassifier(backbone, readout, head); the head is a real, called submodule
  screen.py     # Experiment 1 driver: per (seed, fold), build the datasets once and run every arm
  reference.py  # a fixed set of real cam_iphone patches for the fixture and the tests
```

As built there is no `config.py`. Each run's `config.json` is `RunConfig` (via `replace`, with
the fields that describe a frozen backbone and a convex head set to what actually ran) plus a
`dino` block: weights sha256, timm version, readout, chosen C, host, CPU. The screen is
resumable (`outputs/dino_screen/results.json`), and it imports its provenance and stale-data
preflight from `run_folds`. `--summary` prints the tables and gates, including for a partial
run, and `--archive` files the primary cell as exp220–231.

Compared with the previous version, the embedding cache, the SGD fit loop, and a second
`run_folds` are gone. At depth 0 the head is a convex fit, and one dataset construction per
(seed, fold) serves every arm in memory, so nothing needs caching across processes. There is
therefore no stale-cache failure mode to guard against.

---

## 5. Experiment 1: frozen DINOv3 under the real fold protocol

### 5.1 The question

Measured exactly the way a real fold is measured, does frozen DINOv3 ViT-S/16 with a linear head:

- (a) still beat frozen ResNet18 on every fold and seed?
- (b) come within 0.05 of the fine-tuned recipe's 0.8025?
- (c) at least match DINOv2?

And which readout should it use?

### 5.2 Arms

One dataset construction per (seed, fold) serves every arm.

| arm | backbone | input | readouts | seeds | role |
|---|---|---|---|---|---|
| R18 | torchvision resnet18 IMAGENET1K_V1, `fc = Identity` | 224 | avgpool (512) | 42, 123, 7 | control; ties the corrected protocol to the screen |
| V2 | dinov2_vits14, loaded exactly as the screen did (torch.hub, cached checkout) | 224 | cls, cls_mean | 42, 123, 7 | reference: v3 vs v2, and continuity with 0.8511 |
| **V3** | dinov3_vits16 (§2) | 224 | **cls_mean (primary)**, cls | 42, 123, 7 | **candidate** |
| V3+ *(optional)* | dinov3_vits16plus | 224 | both | 42 | is extra capacity at the same 384-d width worth ~30% more time? |
| V3-B *(optional)* | dinov3_vitb16 | 224 | both | 42 | upper bound; sizes the distillation teacher (§13) |

**The primary cell is pre-registered: V3 × cls_mean**, which is Meta's own linear-eval input
(§1.4). Pre-registering matters because several DINOv3 cells are scored on the same folds (two
readouts, plus the optional arms). Picking the best one on cross-rig numbers afterwards would
inflate the headline figure. The other cells are measured and reported, and can replace the
primary only by the rule in §5.6.


### 5.3 Protocol: what makes it a real fold

- **Datasets:** `build_fold_datasets(cfg)`, with `train_rigs = train_rigs_for(heldout)` in
  `run_folds.RIGS` order (§4.1). At seed 42 that gives the same photos and boxes as exp200–203.
  `params.yaml` is read, never written.
- **Transforms:** `build_eval_transform` for every split. Frozen arms get no photometric
  augmentation; the screen reached 0.8511 without any. Dihedral augmentation is Screen D's
  question (§7.3).
- **Head:** L2-regularised multinomial logistic regression (lbfgs) on standardised features. C is
  chosen from an 11-point log grid over 1e-3…1e2 by **val** macro-F1, with ties going to the
  smaller C (stronger regularisation). This is the depth-0 analogue of the ResNet arm's
  checkpoint-on-best-val rule: selection never sees test or cross-rig data. The fit is
  deterministic, takes seconds, and has one hyperparameter.
- **Export:** fold the scaler into one `nn.Linear` (`W' = W/σ`, `b' = b − W·(μ/σ)`), assert
  `clf.classes_` is `arange(10)`, and test that the Linear reproduces `decision_function` to
  1e-6. The result is an ordinary `DinoClassifier` that `forward_with_embeddings` and the pooling
  core accept unchanged. It can later ship without refitting.
- **Metrics:** `compute_split_metrics` for val, test and cross-rig, with
  `macro_labels=present_class_idxs`, so the iPhone fold is an honest 8-class macro. Also record
  MCC, per-class F1 (watch the class_006 Cerrado / class_007 MonteCristo pair), the chosen C, and
  ms/img.
- **Photo pooling:** cross-rig photo-level macro-F1 through the pooling core (§4.1), using the
  exported head and the same seed keys as the ResNet side.
- **Memory:** build and embed one split at a time. Peak is the training split, about
  4,500 × 448² × 3 bytes ≈ 2.7 GB. `torch.set_num_threads(6)`, as in the screen.

### 5.4 How it is compared

- **Against frozen R18:** paired per (fold, seed).
- **Against fine-tuned exp200–203:** seed 42 only, since that seed is all that exists, so 4
  paired deltas at patch level. Quote both the val-peak mean (0.8025) and the last-10-epoch mean
  (0.8092). The fine-tuned number carries up to ~0.03 per fold of checkpoint-selection noise that
  a convex probe does not.
- **Headline:** the 3-ten-class-fold mean, with the 4-fold mean beside it. Always report both,
  always labelled. Never average cam_iphone's 8-class fold together with the others.
- **Noise:** measure the frozen arm's own seed sd here. A frozen arm is much less stochastic than
  a fine-tuned one, so judge any DINO delta against the **larger** of the two arms' spreads. A
  small sd means less stochastic, not more reliable (§7.2).

### 5.5 Cost: about four hours, not two

Each fold decodes all 938 photos once (406 pixel, 220 sony, 220 oneplus, 92 iphone), which is
about 6 minutes judging by the screen's timings. Each fold has about 7,700 patches, so about 31k
per seed. Embedding time per seed:

- R18: ~7 min
- V2: ~25 min
- V3: ~24 min

That is about 56 min of embedding plus about 25 min of decoding, roughly **1.4 h per seed and ~4 h
for three**. The optional arms add about 30 min (V3+) and about 1.5 h (V3-B) at seed 42.
Logistic-regression fits take seconds.

The previous "~2 hours, and six seeds are free" was wrong twice. Each seed draws new patches, so
everything is re-embedded per seed. And real-fold construction gives the same rig a different
`rig_idx` in different folds, so nothing is shared across folds either. The screen embedded
4,680 patches in total; this embeds about 31k per seed. Extending to seeds 17, 99 and 256 costs
another ~4 hours. Do it only if the readout decision is borderline at three seeds.

**The VM can run it too, and is faster on the real models.** Measured 2026-09-24, real weights,
batch 32 at 224 px, same session:

| ms/img | ResNet18 | DINOv2-S/14 | DINOv3-S/16 | photo decode (cam_iphone) |
|---|---|---|---|---|
| workstation, Ryzen 5 5600U, 6 threads | 15.6 | 53.2 | 44.3 | 0.13 s/photo |
| VM, EPYC Genoa 4c/8t, 8 threads | 14.3 | 49.9 | **36.0** | 0.13 s/photo |
| VM, 4 threads (production default) | 16.5 | 64.8 | 46.8 | — |

Running the seeds on both machines at once cuts Experiment 1 to about 2.5 h of wall-clock time.
Rules for the split:

- **Split by seed, never by arm or by fold within a seed.** Each machine runs every arm for its
  seeds, so every paired delta is computed on one machine. The two machines agree only to float32
  rounding (≤ 3.2e-5 on the same patches, §2.4), not bit-for-bit.
- **Same git commit on both.** Record the hostname and CPU model in the results JSON.
- **Pre-assign experiment ids per seed** so the machines never write the same `expNNN`: seed 42
  → exp220–223, seed 123 → exp224–227, seed 7 → exp228–231.
- **The VM needs timm first.** Its only venv, `~/coffee-vision-venv`, is also the one the webapp's
  gunicorn runs from, so installing timm there changes production's environment (§6.2).

### 5.6 Gates

- **G1: the finding survives the correction.** V3 × cls_mean beats R18 on cross-rig
  macro-F1 on all 12 (fold, seed) pairs. If it fails, the screen was an artefact of its protocol:
  record that and stop.
- **G2: it is a candidate to replace the classifier.** The primary cell's 3-ten-class-fold mean
  cross-rig macro-F1 (patch level, averaged over seeds) is **≥ 0.75**, within about 0.05 of
  0.8025. If it falls short, DINOv3 is still valuable as a teacher, and the plan should be
  rewritten around distillation (§13).
- **G3: the direction check for the owner's decision.** If V2 beats V3 at the same readout on
  **all 12** pairs, stop and bring the result to the owner, rather than switching silently or
  pressing on. Report it with the §1.5 caveat: at 224, DINOv3 sees a coarser token grid than
  DINOv2. Anything short of that means continue with DINOv3. The old licence tie-break is
  retired.
- **The readout lever:** the readout stays at the pre-registered cls_mean. CLS-only replaces it
  only if it wins on all 12 pairs, which is the project's usual sign-consistency bar.

### 5.7 Where the result goes

- **Every arm:** a results JSON and a new section in `analysis/architecture_screen/README.md`.
- **The primary cell's 12 runs:** archived as **exp220–231** through `archive_experiment.archive`,
  with `metrics.json` and `config.json` in the usual shape, so they land in
  `experiments/index.csv` beside exp200–203. The config records the arm, readout, input size,
  weights sha256, timm version and chosen C. If `archive()` insists on a `history.json`, write
  the C sweep as the history rather than editing `archive_experiment`.
- **`EXPERIMENTS_LOG.md`:** an entry either way. A negative result is worth as much as a positive
  one.

---

## 6. Experiment 1b: two kill-switches that need no trained model

### 6.1 OOD feasibility in the frozen DINOv3 space (dev split only)

**RESULT 2026-09-26: PASS, in the selected ViT-B/16 × cls_mean space** (`python -m
coffeecv_dino.ood_feasibility`, VM, commit `7cd4c7e`; JSON in `outputs/dino_ood_feasibility/`). The
deployed method (`linear_probe`, photo-level 5-fold CV from `ood_eval`, imported) in both spaces,
same day, same 264 dev photos, same 40 patches per photo. AUROC (negatives caught at a 5%
false-refusal threshold):

| population | pos / neg | R18 (deployed space, 512-d) | B/16 cls_mean (1536-d) |
|---|---|---|---|
| **same rig** (2026-09-11 positives vs same-rig negatives) | 18 / 25 | 1.000 (1.00) | **1.000 (1.00)** |
| **same rig, green legumes only** | 18 / 7 | 1.000 (1.00) | **1.000 (1.00)** |
| user positives vs user negatives | 27 / 30 | 1.000 (1.00) | 1.000 (1.00) |
| internet vs internet | 7 / 56 | 0.936 (0.81) | 1.000 (1.00) |
| all unseen positives vs all negatives | 34 / 86 | 0.983 (0.95) | 1.000 (1.00) |

Read it with two cautions. **The gate is saturated:** both spaces sit at 1.000 on the same-rig rows
that decide it, so "no worse" is all it can say. And **the B/16 margins are extreme** (median
P(not-beans) 0.000 on genuine photos vs R18's 0.334, 1.000 on negatives), in a 1536-d space fitted
from ~230 photos. Photo-level CV guards against memorising a photo, not against a space where a
few hundred examples are trivially separable. The one-shot holdout verification at ship time
(§9.3), which includes 4 green legumes and 12 independent-day positives, stays the blocking test.
2 internet photos were unmeasurable in both spaces (the pitch estimate declines them before any
guard runs).

**Why now:** with a frozen backbone, the embedding space the OOD guard lives in is fixed before
any head exists. The guard blocks shipping, because a classifier win paired with an OOD
regression is not a win. So check it now, cheaply, rather than after days of VM time.

**How:** run `ood_eval`'s photo-level cross-validated comparison on the **dev** split, which is
the protocol that chose probe v2. Use the DINOv3 backbone and readout in place of the deployed
ResNet18. `ood_eval.main` loads a `coffeecv` checkpoint, so do it through a thin `coffeecv_dino`
driver that *imports* `ood_eval`'s scoring. The embedding-space methods (probe, centroid, kNN)
are independent of the head. Skip the logit methods (energy, MSP) until a head exists. Use the
same negatives, including `2026-09-11__user_samerig` (the green legumes that defeated probe v1)
and the internet proxies, and the same positives.

**Never read the holdout here.** `fit_ood_probe --verify` spends the holdout exactly once, on
the checkpoint that ships (§9.3). A feasibility check that reads it would turn it into a second
dev set.

**Pass:** in the pre-registered readout's space, the probe's dev-CV AUROC on same-rig negatives,
and specifically on the green legumes, is no worse than the deployed ResNet18 space's on the
identical protocol, re-run the same day. **On a fail:** the classifier screens continue, but
shipping (§9) is blocked until the OOD guard has had its own work.

### 6.2 Latency on the VM (S/16 measured 2026-09-24, passes; B/16 re-measured 2026-09-26, parity)

**The selected ViT-B/16, measured 2026-09-26** (`python -m coffeecv_dino.bench`, real weights, 32
real cam_iphone patches, batch 32 at 224 px, VM idle):

| ms/img | ResNet18 | DINOv3-S/16 | **DINOv3-B/16** |
|---|---|---|---|
| 4 threads (production default) | 15.7 | 45.0 | **145.1** |
| 8 threads | 14.2 | 33.0 | 108.0 |

Model time per photo at 4 threads: today's ResNet18 with 8-view TTA ≈ **5.0 s**; frozen B/16 without
TTA ≈ **5.8 s** (40 × 145 ms), about 16% *slower*, not the 2.8× saving S/16 offered (1.8 s). At 8
threads B/16 is 4.3 s. So B/16 is roughly latency parity with production, not a saving: it passes
only as "no worse than a few seconds per photo", and **B/16 can never take TTA** (46 s/photo at 4
threads), whatever Screen D finds. That was the known price of the owner's choice (§0.4). The
end-to-end timed photo at deploy (§9.5) still decides it.

The S/16 measurement of 2026-09-24, kept for reference:

Measured with the real weights (§5.5 table). The webapp sets no thread count, so torch uses its
default of 4 on the VM:

- **Today:** ResNet18 at 16.5 ms/img × 320 passes (40 patches × 8 TTA views) ≈ **5.3 s** of model
  time per photo.
- **Frozen DINOv3 without TTA:** 46.8 ms/img × 40 passes ≈ **1.9 s**, about 2.8× cheaper.

Decode, crop and pitch overheads are unchanged and excluded from both. The end-to-end timed photo
at deploy (§9.5) remains; it confirms this, it doesn't decide it. If Screen D shows DINOv3 needs
TTA after all, the projection becomes 8 × 1.9 s and this gate fails.

Two facts about the VM that matter later:

- **It has one venv, `~/coffee-vision-venv`, shared by training and the webapp** (gunicorn's
  `ExecStart` runs from it). There is no training-only environment to install timm into. timm
  needs only torch, torchvision and pyyaml when `pretrained=False`; that was verified with
  huggingface_hub and safetensors blocked. So the options are `pip install --no-deps
  timm==1.0.29` into the shared venv (one pure-Python package), or a separate directory on
  `PYTHONPATH` for experiment runs only until §9.4 adds it properly. That is the owner's call.
- **Production runs 4 of the VM's 8 hardware threads.** DINOv3 at 8 threads is 36.0 vs
  46.8 ms/img. This is a deploy-time knob, not part of this plan's decisions.

`coffee-cv-web.service` is the owner's to manage: check its state and don't restart it.

---

## 7. Experiment 2: the baseline that doesn't exist yet, and seed replication

### 7.1 Launch ResNet18 camera-rig folds at seeds 123 and 7 once G1–G3 pass

**LAUNCHED 2026-09-26 15:50 UTC on the VM** from commit `7cd4c7e` (clean tree; the recipe check
below passed: exp200–203's archived configs differ from `params.yaml` only in the fold fields and in
scheduler fields that postdate them and sit at their cosine defaults). Chained after the §6.1 run:
`run_folds --arm beans --start-exp 232 --seed 123 --tag s123`, then `--start-exp 236 --seed 7
--tag s7`; torch's default thread count, as exp200–203 used. Log `~/exp2_r18.log`, status
`~/exp2_r18_status.log`.

**FINISHED 2026-09-28 15:44 UTC** (`SWEEP_DONE_EXIT=0`), merged into `main` at `b224982`. Cross-camera
macro-F1 over the three ten-class folds (pixel/sony/oneplus): seed 123 0.7908, seed 7 0.7965 (seed 42,
exp200-203: 0.8025). Per fold: pixel 0.7939/0.7949, sony 0.7319/0.7240, oneplus 0.8465/0.8705, iphone
(8 classes) 0.8049/0.8164. The §7.2 comparison has not been run yet.

The camera-rig era has exactly four fine-tuned fold runs, exp200–203, all at seed 42. An
adoption needs a multi-seed baseline, and it is the slowest item in this plan: about 7.5 h per
fold run, so about 60 h for 8 runs (4 folds × 2 seeds). Suggested numbering: exp232–239.

Before launching:

- **Check the recipe.** Confirm exp200–203's archived `config.json` matches `params.yaml`'s
  current resting recipe (cosine, 100 epochs, patience 20, `eta_min` 1e-5, MixStyle 0.5
  agnostic). Pass no flag that could silently reset a lever; that CLI-defaults trap cost 6.5 h
  once.
- **Check provenance:** `git status` must be clean.
- **Track it properly:** launch with background tracking, not nohup.
- **Keep the VM's checkout still:** don't push `coffeecv/` changes to the VM mid-sweep, because
  its per-fold provenance commits would pick them up.

### 7.2 The comparison protocol

- **Paired per (fold, seed).** Same seed and same held-out rig give the same cross-rig patches;
  §4.1 makes the training photos identical too. Report per-pair deltas, never a difference of two
  separately averaged means.
- **Controls travel with the headline:** `test_macro_f1` (in-distribution; the camera-rig
  baseline is 0.9002–0.9164), photo-pooled cross-rig, and per-class confusion. An arm that wins
  cross-rig while losing in-distribution is making a trade, and that trade is a product decision.
- **Seed-variance asymmetry:** use the larger of the two arms' spreads (§5.4).
- **It is a design comparison, not an architecture ablation.** Best ResNet18 design (MixStyle,
  plus TTA at deploy) vs best frozen-DINOv3 design. Write that sentence into the experiment note.
  Do not "fix" it by disabling MixStyle on the baseline; that compares two handicapped designs.
- **Also compare the things that aren't macro-F1:** measured end-to-end latency, OOD separability
  (§6.1, then §9.3), and the class_006/007 confusion.

**Adoption bar:** 12 paired deltas, sign-consistent. The in-distribution control must not regress
by more than the seed sd, and the photo-pooled figure must agree in direction.

### 7.3 Screen D: TTA and dihedral augmentation for the frozen head

This runs on the workstation while the VM works: seed 42, 4 folds, about 2 hours.

- **Evaluation side:** 8-view dihedral TTA (average probabilities) on the cross-rig set. The
  screen suggested DINO doesn't need it. If it does, the latency advantage shrinks eightfold.
- **Training side:** fit the logistic regression on all 8 dihedral views of every training patch
  (36k rows). This is the frozen equivalent of the ResNet arm's rotation augmentation. DINO
  features are not rotation-invariant, and the standing rule is to prefer invariance the model
  learns over normalising orientation per rig.

---

## 8. Integration into `coffeecv` (after the adoption decision)

**Trigger:** a decision to ship (§9) or to run Screen C (§10). Work on a branch.

### 8.0 Prune `mobilenet_v3_small` and `efficientnet_b0` (H2, any time)

The audit says it is free: **0 of 150** archived experiments and **0 of 7** shipped checkpoints
use either one. Live references are `config.py:77`, `model.py:136/151` and
`train_baseline.py:59`. Both were screened and rejected three times: exp3 (efficientnet frozen,
"clear reject"), exp29 (efficientnet full fine-tune, worse *and* ~1.5× slower) and exp31
(mobilenet at val 0.8490 vs 0.9579).

- **`RunConfig.model_name` defaults to `"mobilenet_v3_small"`.** That is the worst architecture
  this project ever measured, and it is what `from_params_yaml` returns when `params.yaml` is
  absent. Change the default to `"resnet18"` in the same commit.
- **Deleting the mobilenet branch also deletes the head-slice bug.** `model.classifier[2:4]` is a
  slice, so the embedding pre-hook never fires, and `forward_with_embeddings` would reach
  `torch.cat([])`. It is better to delete code nothing runs than to repair it.

The edits:

- `model.py`: delete both branches. Make the `else` actionable: "Supported: 'resnet18',
  'dinov3_vitb16'. mobilenet_v3_small and efficientnet_b0 were removed on <date> (EXPERIMENTS_LOG
  exp3/exp29/exp31); to re-run an old experiment, check out its own git_commit."
  `config_for_checkpoint` restores `model_name` faithfully, so an old card must fail loudly, never
  fall back to something else.
- `train_baseline.py:59`: `choices` updated to match.
- **Do not edit `EXPERIMENTS_LOG.md`'s old mentions.** They are the justification for the
  deletion; add a dated line recording the removal instead.
- **Keep `analysis/architecture_screen/bench.py`'s efficientnet row.** It records a measurement,
  not a code path.

Add a test parametrised over the supported architectures that proves `head_module`'s pre-hook
fires and captures `embedding_dim_of(head)` features. A future arm then cannot land without
proving its embedding contract.

### 8.1 One file per architecture behind the `Arm` protocol (pure refactor)

`train_baseline.py` has five architecture-specific touch points: `build_model`, the
`cross_domain_mixstyle` DataLoader flag, the head/backbone parameter groups, `build_scheduler`,
and the MixStyle `domain_ids` in `train_one_epoch`. Move them behind:

```python
class Arm(Protocol):
    """One architecture and the training specifics that belong to it. Everything else -- the epoch
    loop, evaluate(), compute_split_metrics, checkpoint-on-best-val, patience, history, outputs/ --
    stays shared, because two arms that select checkpoints or compute metrics differently are not
    comparable."""
    name: str
    def validate_config(self, cfg) -> None: ...            # raise at construction, not 40 min in
    def build_model(self, cfg, num_classes) -> tuple[nn.Module, nn.Module]: ...  # head must be CALLED by forward
    def build_optimizer(self, cfg, model, head): ...
    def build_scheduler(self, cfg, optimizer): ...         # policy is owned by the arm; mechanism is shared
    def needs_domain_ids(self, cfg) -> bool: ...
    def set_batch_context(self, model, domain_ids) -> None: ...
    def describe(self, cfg) -> str: ...
```

- **`arms/resnet18.py`:** the adopted recipe, moved verbatim (MixStyle hooks, freeze modes,
  parameter groups, `cosine | plateau`, `domain_ids`).
- **Schedulers stay shared.** `CosineLR` and `PlateauLR` in `lr_schedules.py` are mechanisms. A
  forked copy would make a scheduler difference indistinguishable from an architecture difference.
  An arm owns only which scheduler it uses and with what settings; new mechanism (warmup, layer
  decay) is written against the same interface.
- **Mixup stays in the shared loop.** It is a config lever, not a property of an architecture.
- **Proof:** the 3-epoch `history.json` diff from §4.1 must be empty. Order: prune first, then
  this refactor, then the DINOv3 arm.

### 8.2 `coffeecv/arms/dinov3.py`

- **It is a move, not a copy.** `backbone.py`, `head.py` and `model.py` move from
  `coffeecv_dino` into `coffeecv` in the same commit that deletes the originals. `coffeecv_dino`
  keeps only its screen driver. That also avoids a package-dependency cycle (`coffeecv` →
  `coffeecv_dino` → `coffeecv`).
- **`DinoClassifier` = backbone → readout → head.** The head is an `nn.Linear` whose width
  follows the readout (768 or 1536 for the selected ViT-B/16), and it is a real, called submodule, so the pre-hook contract
  holds. Never return a slice or an inner layer of it as `head_module`.
- **`validate_config`** raises on:
  - any `freeze_mode` but `full`;
  - `mixstyle_p > 0`: token-statistics MixStyle is unvalidated, and silently ignoring the flag
    would hide the +0.1400 lever's absence;
  - `patch_resize % 16`;
  - `dino_unfreeze_blocks` outside 0..6;
  - **depth 0 inside `train_baseline`.** A depth-0 head is fitted by the convex path (§5.3), and
    two ways of producing the same thing is how arms stop being comparable. The SGD loop is for
    depth ≥ 1.
- **A checkpoint is the head plus the backbone's sha256.** The frozen backbone never changes at
  depth 0. It is loaded from `models_pretrained/` and verified at load time, never written into
  each checkpoint: 343 MB saved per run, and no derivative-checkpoint copies scattered through
  `experiments/`. At depth ≥ 1, the unfrozen blocks are part of the checkpoint.

### 8.3 Config fields

`dino_weights` (path), `dino_weights_sha256`, `dino_readout: cls | cls_mean`, and
`dino_unfreeze_blocks: int = 0`, bounded to 0..6 (ViT-B/16, like S/16, has 12 blocks; allowing 12 would bring
back the excluded full fine-tune). `timm_version` is recorded automatically; it is not a knob.
`from_params_yaml` raises on unknown keys, so `params.yaml` and `RunConfig` change in the same
commit.

### 8.4 `run_folds.py --model-name` (needed for Screen C only)

```python
if model_name is not None:        # None => INHERIT what params.yaml holds
    text = re.sub(r"^model_name: \S+", f"model_name: {model_name}", text, count=1, flags=re.M)
...
if model_name is not None:
    assert cfg.model_name == model_name, f"model_name is {cfg.model_name!r}, wanted {model_name!r}"
```

The `is not None` default is the rule that a CLI flag must inherit, never zero. The read-back
assertion catches a regex that silently fails to match.

### 8.5 `dvc.yaml`

After `dvc add`, make the DINOv3 weight file a dependency of the train stage. A weight swap must
not reuse a stale run.

### 8.6 What was built on 2026-09-27 (depth 0), and what was deliberately not

Built on branch `dino-integration` (commit `6bdfdb1`, on top of H1/H2), ahead of the adoption
decision at the owner's request, so a deploy-ready B/16 model exists when the 12 pairs land:

- **The move (§8.2), with flatter names:** `coffeecv_dino/{backbone,head,model}.py` became
  `coffeecv/backbones.py`, `coffeecv/linear_head.py` and `coffeecv/dino_classifier.py`, in the commit
  that deleted the originals. The banned-definitions test now also bans `build_backbone`,
  `verify_weights`, `fit_head`, `fit_head_at`, `export_linear`, `DinoClassifier` and
  `FrozenBackbone` inside `coffeecv_dino`.
- **The checkpoint (§8.2):** `{format: coffeecv.frozen_head/1, backbone, weights, weights_sha256,
  timm_version, readout, class_ids, C, head}`, about 60 KB. `load_frozen_checkpoint` refuses a
  wrong `model_name`, class count, backbone digest, timm version or format.
- **One loader:** `infer.load_model` dispatches on `model_name`. That means the webapp, the infer
  CLI, `build_ood_reference`, `fit_ood_probe` and `ood_eval` load a frozen model with no change of
  their own. `build_model` refuses frozen names: depth 0 is the convex fit, never SGD.
- **TTA is the model's property:** `infer.inference_tta_for` reads the card's
  `inference_defaults.tta`. With no card, TTA is on for a fine-tuned net and off for a frozen one.
  All four inference callers use it, so B/16 is never served with 8 views by accident.
- **Webapp:** `COFFEE_CV_CHECKPOINT` overrides the checkpoint path, for a sandbox smoke test
  before repointing. The default is unchanged.
- **`fold_data.build_fold_datasets(only=...)`** builds a subset of splits. The result is
  byte-identical to building them all (tested), and it bounds the memory the shipping fit needs.
- **`coffeecv.fit_frozen_head`** is the §9.1 fit, and `--ship` copies a run to `models/`.
- **Dependencies (§9.4):** `timm==1.0.29` is in `webapp/pyproject.toml` and `webapp/uv.lock` since
  OPS-1 (2026-09-27); `[[tool.uv.dependency-metadata]]` declares it without huggingface_hub and
  safetensors. The pip files that held it (`webapp/requirements*.txt`, `--no-deps` via
  `setup_server.sh`) were deleted on 2026-09-28.
  - `huggingface_hub` and `safetensors` are left out on purpose. timm imports them only to
    download weights, and the production venv has run timm without them since 2026-09-24.

**Not built, because depth 0 does not need it.** Each of these belongs to Screen C (§10), if it
ever runs:

- **The `Arm` refactor (§8.1).** A frozen head never enters `train_baseline`.
- **The `dino_*` config fields (§8.3).** The checkpoint and the run's `config.json` `dino` block
  carry the backbone's identity; `params.yaml` is untouched.
- **`run_folds --model-name` (§8.4) and the `dvc.yaml` dependency (§8.5).**

---

## 9. Shipping a frozen DINOv3 model

### 9.1 The model

`run_all_rigs.py` is for shipping weights only, and never for validating a config; that rule
stands. For a frozen arm, "all rigs" means fitting the head on the train split of all four rigs
at the C the folds selected (their median). With a convex head, seeds only vary the patch draw.
Report the spread across seeds as a sanity check and ship one.

### 9.2 Fix the unchecked OOD-reference load first (H1, a live defect today)

Three places load the OOD reference. `coffeecv/infer.py` main and `ood_eval.py` check its
`checkpoint_sha`. **`webapp/app.py:56` does not**:
`ref = json.loads(ref_path.read_text()) if ref_path.exists() else None`. Today that gives
wrong-but-plausible refusals if a stale reference is ever deployed. After a swap to DINOv3, it
becomes a 512-vs-1536 shape error raised on every request. Land the fix as its own commit,
mirroring `load_ood_probe`:

```python
def embedding_dim_of(head: torch.nn.Module) -> int:
    """Width of the vector this model feeds its classifier: the final Linear's in_features."""
    *_, last = (m for m in head.modules() if isinstance(m, torch.nn.Linear))
    return last.in_features

def load_ood_reference(checkpoint: Path, head: torch.nn.Module | None = None,
                       explicit: Path | None = None) -> dict | None:
    path = explicit or reference_path_for(checkpoint)
    if not path.exists():
        return None
    ref = json.loads(path.read_text())
    if ref.get("checkpoint_sha") and ref["checkpoint_sha"] != _sha(checkpoint):
        raise SystemExit(f"OOD reference {path} was built from a different checkpoint; rebuild it.")
    if head is not None and ref.get("embedding_dim") not in (None, embedding_dim_of(head)):
        raise SystemExit(f"OOD reference {path} is {ref['embedding_dim']}-d but this model emits "
                         f"{embedding_dim_of(head)}-d embeddings; rebuild it for these weights.")
    return ref
```

Point `webapp/app.py`, `infer.py` main and `ood_eval.py` at this one helper. A mismatch then
fails at import, so a gunicorn deploy fails loudly instead of returning 500s per request.

### 9.3 OOD rebuild and re-validation (blocking)

**Done 2026-09-27 for `models/allrigs_dino3b16_s123.pt`** (exp253). Full numbers are in EXPERIMENTS_LOG
exp252-254 and in the card's `ood_guard` block.

- **Probe:** threshold 0.0459 at certified α ≤ 4.3%.
- **Holdout:**

  | | B/16 probe | deployed ResNet18 probe, same photos |
  |---|---|---|
  | negatives caught | 56/56, green legumes 4/4 (gate 3 passes) | 51/56 |
  | genuine photos falsely refused | **4/22** | 0/22 |

  - Three of B/16's four false refusals are internet-framed photos, which the calibration never
    covered.
  - On the population the threshold certifies it refuses 1/15, which is consistent with 4.3% at
    this n.
- **Owner override, same day:** the threshold is now **0.5**, the probe's decision boundary. At 0.5
  it refuses 0/22 genuine holdout photos and still catches 56/56 negatives; every negative anywhere
  scores ≥ 0.9998. The α ≤ 4.3% bound still holds, conservatively. Because the holdout informed the
  choice, it is no longer an unbiased estimate. See `threshold_override` in the probe file.


1. Run `build_ood_reference.py`, then `fit_ood_probe.py`, on the shipped checkpoint. The 0.9681
   threshold and its α ≤ 4.3% are properties of the ResNet18 space and mean nothing in the new
   one.
2. Run `fit_ood_probe --verify` **once**. This spends the holdout.
3. Same-rig negatives, including the green legumes, are a blocking gate. Probe v1's 0.970 AUROC
   became 0.849 once source confounding was removed; don't repeat that.

### 9.4 Dependencies and the deploy sandbox

**Done through OPS-1 (2026-09-27).** timm is locked in `webapp/uv.lock` without its download-only
dependencies, and every deploy smoke-tests the release in the production unit's own sandbox before
the flip (`scripts/deploy_webapp.sh`). The bullets below are the original plan.

- **Add `timm==1.0.29` to `webapp/requirements.txt` with hashes** via pip-compile, which also
  brings pyyaml, huggingface_hub and safetensors. timm itself runs without the last two when
  `pretrained=False` (§6.2), so leaving them out is an option, but it has to be a deliberate
  choice, because `--require-hashes` resolves declared dependencies. Add timm to
  `.devcontainer/requirements.txt` in step 0; that file says not to add torch, and timm is not
  torch.
- **Smoke-test inside the real systemd sandbox** before calling it deployed. `ProtectSystem`
  blocking a path that local testing couldn't see has happened before. With `pretrained=False`
  nothing should touch the network or the HF cache, but `HF_HUB_OFFLINE=1` in the unit file is
  free insurance.
- **The backbone file is already on the VM** (copied and verified 2026-09-24, §2.5). Once the DVC
  remote is back, `dvc pull` is the normal path. Never commit it.

### 9.5 Deploy checklist

**Superseded by OPS-1.** A deploy is now `DOMAIN=... scripts/deploy_webapp.sh <sha> <model>`: the model
is chosen in the release's `release.env`, the manifest ships the sidecars together, nginx serves the
page from the release, and the health checks are its `--verify` step. See `webapp/README.md`. The
original checklist:

1. Repoint `CHECKPOINT`. The sidecars are addressed as `<checkpoint>.<suffix>`, so repointing
   swaps the whole set: `.json`, `.classes.txt`, `.ood_reference.json`, `.ood_embeddings.npz`,
   `.ood_probe.json`. Ship them together. With §9.2 in place, a stale reference refuses to boot,
   which is the behaviour worth having.
2. Keep `N_PATCHES` at 40. Whether TTA stays on is Screen D's answer.
3. Rsync, then re-copy static files and nginx config by hand. A dirty remote working tree is the
   normal steady state.
4. Health check: `GET /` → 404 and `GET /classify` → 405. Then send one real photo end to end
   and time it.

---

## 10. Screen C: unfreeze depth (after shipping, optional)

- **Re-measure the step-cost ladder for DINOv3.** The previous numbers are DINOv2 at 256 tokens,
  in s/step at batch 32: 1.47 for head only, 1.71 / 1.94 / 2.49 / 2.99 for 1 / 2 / 4 / 6 blocks,
  against 1.33 for resnet18. DINOv3-S at 196 tokens should be cheaper.
- **Every depth ≥ 1 loses what made the frozen arm cheap:** refits in seconds, head-only
  checkpoints, a fixed OOD space, and possibly the no-TTA result.
- **Start with N = 2**, one seed, four folds, paired against the shipped frozen arm. Probe N = 4
  only if N = 2 wins sign-consistently. A curve that keeps climbing to N = 6 is a signal to check
  for overfitting against the in-distribution control, not a reason to unfreeze further.
- **Recipe, to be re-checked against current literature before launching** (standing rule):
  - **LP-FT:** initialise the head from the depth-0 logistic-regression solution, not at random.
    Kumar et al. (ICLR 2022, "Fine-Tuning can Distort Pretrained Features and Underperform
    Out-of-Distribution") show that fine-tuning with a random head distorts pretrained features
    and hurts out-of-distribution accuracy, and cross-rig accuracy is exactly an
    out-of-distribution metric here. The depth-0 head already exists by then, so this costs
    nothing.
  - **Layer-wise LR decay and a short warmup.** These are new mechanisms, so they live in
    `arms/dinov3.py`, against the shared scheduler interface.
  - **MixStyle stays off.**
  - **RoPE coordinate augmentation is a choice, not a default.** Meta's model applies RoPE
    coordinate rescaling (×[0.5, 2]) whenever it is in train mode; timm does not implement it.
    Under Meta's code, `model.train()` would switch it on silently.
- **Budget:** about 1 VM day per 4-fold seed at N = 2 (DINOv2-era estimate).

---

## 11. Risks, and how each one shows up

| risk | how it shows up | mitigation |
|---|---|---|
| the corrected protocol collapses the margin | primary cell far below 0.8511 | that is Experiment 1's purpose; G1/G2 (§5.6) |
| probe not paired to real folds | training photos differ from exp200–203; deltas noisier than they should be | `build_fold_datasets` + `train_rigs_for` (§4.1) |
| timm upgrade silently changes the architecture | features drift; results stop reproducing | exact version pin; fixture test (§2.4) fails loudly |
| RoPE periods not copied, or timm starts caching | ~1e-3 feature drift; fixture test fails | periods copy + `feat_shape is None` assert (§2.2) |
| timm's default `global_pool="avg"` used by accident | a third readout silently in play | `global_pool="token"` plus explicit `readout()`; shape tests |
| register tokens counted as patch tokens | patch mean contaminated by 4 register tokens | slice by `num_prefix_tokens`; test |
| readout chosen on cross-rig numbers | headline inflated by picking a winner | pre-registered primary cell (§5.2) |
| `pretrained=True` or HF weights used | hash mismatch, or a run-time download | always load the manifest-verified `.pth` (§2.5) |
| holdout read during the feasibility check | holdout becomes a second dev set | dev split only (§6.1); `--verify` once, at ship time (§9.3) |
| OOD regresses unnoticed | better classifier metrics, worse refusals in the field | §6.1 early, §9.3 blocking |
| MixStyle silently dropped | DINO arm looks good or bad for the wrong reason | `validate_config` raises; the note records the design difference |
| `patch_resize` not a multiple of 16 | timm: failure minutes in, after decoding. Meta's code: **no failure at all**, the edge pixels are silently dropped | `assert_input_size` at config construction (§2.2); timm raises regardless |
| a head fitted at one input size, served at another | features move (cosine ≈ 0.88 for one grid step, ≈ 0.6 at 2.3× the pixels, §1.5); predictions degrade with no error | `patch_resize` recorded in the checkpoint card and restored by `config_for_checkpoint` |
| a single-seed win is a favourable draw (this has happened 3 times here) | 4/4 at seed 42, mixed at 123 | 3 seeds in Experiment 1; ResNet18 seeds before adoption (§7.1) |
| `coffeecv_dino` grows a private copy | both arms fine in isolation, deltas meaningless | import table (§4.2) plus the banned-definitions test |
| an old config names a pruned architecture | re-run silently uses a substitute | `build_model` raises with the removal date (§8.0) |
| full fine-tune creeps back in | 3.3× cost, and against a decision | `freeze_mode` must be `full`; N bounded to 0..6 |
| DINOv3 weights reach the public repo or `experiments/` | licence §1b-i distribution | binaries gitignored; checkpoints store the head plus a sha only (§8.2) |
| uncommitted source during a sweep | archived `git_commit` lacks the code that ran | `run_folds`' dirty-provenance gate; no `--allow-dirty` |

---

## 12. What "done" looks like

An adoption decision recorded in `EXPERIMENTS_LOG.md`, resting on:

- 12 paired (fold × seed) deltas, with the in-distribution and photo-pooled controls beside them;
- the frozen arm's own measured seed sd;
- the readout choice, with its paired evidence;
- an OOD guard rebuilt in the new space and verified once on the holdout, including the green
  legumes;
- measured end-to-end `/classify` latency on the VM;
- a plain sentence saying this compared two *designs*, not two architectures.

Or a recorded negative with the same rigour, which is equally valuable and much cheaper.

---

## 13. Fallback: distillation

If DINOv3 cannot be the classifier (because G1/G2 fail, or the OOD guard can't be made to work
in its space), the screen's finding still stands: a representation exists that separates these
beans across cameras far better than ResNet18 does on its own. In that case, use it as a
**teacher**:

- Keep the serving ResNet18.
- Add a loss pulling its 512-d embedding toward a projection of DINOv3's embedding on the same
  patch, and train as usual.
- The teacher runs offline only, so serving cost, MixStyle, TTA and the existing OOD guard are
  all unchanged.

Experiment 1's optional V3-B arm tells you whether ViT-B/16 is a better teacher than S/16. This
is the same structural move as `docs/architecture_screen_plan.md` R2 (SAM → U-Net): on this
hardware, every strong model is affordable offline and unaffordable live.

The one licence question sits here. No clause prohibits distillation, and there is no
restriction on outputs. But whether a student trained against the teacher's outputs is a
"derivative work" matters only if the student is **published**. For private serving it doesn't
come up.

---

## Appendix A: the licence in brief

The full reading, with the relicensing history, is in this file's previous version (commit
`1caf2f1`, §0b). The text is pinned at `docs/reference_dinov3_LICENSE.md`. This is not legal
advice.

- **Permitted:** use, reproduction, modification, derivative works and commercial use (§1a).
  You own your derivatives (§5a). There is no acceptable-use policy document, no user threshold,
  no naming requirement, and no restriction on outputs.
- **Meta's on-record clarification** (issue #28, 2025-08-21): *"You do not need to include the
  license if you are incorporating DINO into a commercial offering like an API or SaaS offering,
  but such offering must be in compliance with the DINOv3 license."* Serving predictions at
  `coffee.kasetkin.com` is that case.
- **What does bite:** redistributing the weights, or a derivative checkpoint, carries the
  Agreement with it (§1b-i). The public repo therefore never contains the weights, and checkpoints
  stay in the private DVC remote.
- **Differences from Apache-2.0:** Meta may amend the terms unilaterally (§8); California law and
  forum (§7); a broader indemnity and patent-style termination (§5b); a gated download.
- **Relicensing:** as of 2026-09-21, request #31 was closed in favour of feedback thread #52,
  which Meta opened and has not commented in for 13 months. Do not plan around a relicense.
- **Since this rewrite,** the code dependency is timm (Apache-2.0), so the licence attaches only
  to the weight file (§2.3).

---

## Appendix B: loading through Meta's repository (superseded, kept for reference)

Verified on 2026-09-21, and again on 2026-09-24 at `6876159` to produce the §2.1 comparison.

- **`hubconf.py` fails** with `ModuleNotFoundError: torchmetrics`, because it imports the
  segmentors and so the eval stack. The fix was `from dinov3.hub.backbones import dinov3_vits16`
  directly: 4 subpackages, 26 files, about 400 KB, nothing beyond torch.
- **`dinov3_vits16(weights="/local/path.pth")`** treats the path as a URL and copies the file
  into `~/.cache/torch/hub/checkpoints/`. That copy still exists on the workstation. The fix was
  to build with `pretrained=False` and `load_state_dict(torch.load(W), strict=True)`.
- **Its `PatchEmbed` does not check the input size.** The divisibility asserts are commented out,
  so the stride-16 convolution silently drops the right and bottom remainder. At 250 px the output
  is identical to feeding the top-left 240×240 (verified 2026-09-24).
- **The hub builder's own settings** for vits16: `img_size=224`, RoPE dtype fp32 (while loading
  bf16-rounded periods from the checkpoint), `rescale_coords=2` (train mode only),
  `qkv_bias=True` (zero-valued, with a zero `bias_mask`), and `norm_layer="layernormbf16"` (a
  plain fp32 LayerNorm).
- **Why it was dropped:** the code is under the DINOv3 License, so vendoring it would put a
  non-Apache subtree in the public repo. A git-pinned dependency cannot pass the webapp's
  `--require-hashes` install. timm reproduces its output exactly (§2.1).
