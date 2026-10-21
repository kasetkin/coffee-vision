# Vendored pretrained backbones

Upstream pretrained weights, kept in the repo tree rather than resolved from the network at run
time. Rationale in `docs/dinov3_integration_plan.md` §2.5: a sweep that downloads its backbone
mid-run is not reproducible and fails on a box without egress.

The weight binaries are gitignored; **this README, `manifest.json` and `verify.py` are tracked
on purpose** — they are the acquisition instructions, the checksums and the checker, i.e. the only
way a fresh clone can obtain and validate the right files. (Until 2026-09-21 `.gitignore` excluded
the whole directory and hid all three.) Every weight file is DVC-tracked, its `.dvc` pointer beside it, and
its object is on the DVC remote (`dvcnewremote`, `docs/development.md`).

## Getting these files on a fresh clone

```bash
python models_pretrained/verify.py     # what you have, and what is missing or wrong
```

- **You, on another machine:** `dvc pull -R models_pretrained`, then `verify.py`. The weights live in the
  private DVC remote, so this needs nothing from Meta a second time. (Before they were DVC-tracked this folder
  was rsynced, last to the VM `powervpsssh`, retired 2026-10-12; `docs/agents/dvc.md` retires copying by hand.)
- **The web service** never reads this folder: since ticket ML-7 a release ships no pretrained weights. The
  backbone reaches it inside the model's LiteRT conversion (`models/<name>.litert/`, the `litert_export` stage),
  which is built from this folder's copy on the machine that runs the export.
- **Anyone else:** DINOv3 cannot be redistributed from a public repo (licence §1b-i), so a copy
  has to be requested from Meta directly. Full procedure, including the two-incompatible-formats
  trap, is `docs/dinov3_integration_plan.md` **§2.5**. Short version: take the **Meta-direct
  `.pth`** files, not the Hugging Face `safetensors` — they are different files, the hashes here
  match only the former, and only the former is what the loader (timm, plan §2.2) is verified against.

DINOv2 needs no gate: it re-downloads automatically from `torch.hub`
if absent, which is exactly the run-time network dependency this folder exists to
remove — so verify it too rather than assuming.

## Contents

| file | params | embed dim | patch | tokens @224 | source / licence |
|---|---|---|---|---|---|
| `dinov2/dinov2_vits14_pretrain.pth` | 22.1M | 384 | 14 | 256 | `facebookresearch/dinov2` · **Apache-2.0** |
| `dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | 21.6M | 384 | 16 | 196 | `facebookresearch/dinov3` · DINOv3 License (gated) |
| `dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` | 28.7M | 384 | 16 | 196 | same |
| `dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 85.7M | 768 | 16 | 196 | same |
| `efficientvit_sam/efficientvit_sam_l0.pt` | 34.8M | — | — | — | HF `mit-han-lab/efficientvit-sam` @ `a2f0c592` (URLs in `manifest.json`) · Apache-2.0 · ticket ML-2 |
| `efficientvit_sam/efficientvit_sam_l1.pt` | 47.7M | — | — | — | same |
| `efficientvit_sam/efficientvit_sam_l2.pt` | 61.3M | — | — | — | same |
| `efficientvit_sam/efficientvit_sam_xl0.pt` | 117.0M | — | — | — | same |
| `efficientvit_sam/efficientvit_sam_xl1.pt` | 203.3M | — | — | — | same |

The EfficientViT-SAM files are the bean segmenter's base weights, built by `coffeecv/sam_loader.py` from the
vendored `third_party/efficientvit`. XL0 is the live segmenter's, under its fine-tuned decoder
`models/seg/xl0_v3.pt` (ticket ML-6); L0 was ML-2's; all five were screened in ML-2's P0.

`verify.py` checks every file's sha256 against `manifest.json` (all nine OK on 2026-10-21). The three DINOv3
files' names embed a sha256 prefix (torch.hub's convention), so they also verify against their own names: the
downloads are intact and authentic. The other files' names carry no hash, so their sha256 is recorded in
`manifest.json` only.

## What the numbers mean for this project

**`dinov3_vits16` and `dinov3_vits16plus` are both 384-d**, the same width as DINOv2 ViT-S/14.
That makes either a drop-in as far as the embedding-space work is concerned: the OOD reference,
the probe's `mu`/`sd`/`w` vectors, `embedding_dim_of` and the classifier head shape are all
unchanged when swapping between them. Only the weights and the patch size differ.

**`dinov3_vitb16` is 768-d and ~4x the parameters.** It is not a drop-in for the two above — it needs
its own OOD sidecar — and it is the backbone every shipped model has used since 2026-09-27 (selected on
exp240-251). Since ticket ML-7 the live site runs it as a LiteRT graph on the host's GPU, not in torch.

**Patch 16 vs 14 makes DINOv3-S slightly cheaper as well as stronger.** At 224 px the DINOv3
models see 196 tokens against DINOv2's 256. Measured 2026-09-21 (batch 32, 224 px, 6 threads):

| backbone | tokens | ms/img |
|---|---|---|
| dinov2_vits14 | 256 | 48.8 |
| dinov3_vits16 | 196 | **44.4** |

A 9% saving — real, but well short of the ~23% a quadratic-attention argument predicts, because
the FFN cost is linear in tokens and dominates at this size.

**The input-size guard differs per family**: DINOv2 needs `patch_resize % 14 == 0`, DINOv3 needs
`% 16 == 0`. The adopted 224 satisfies both, so the guard must be derived from the loaded
checkpoint's `patch_embed.proj.weight` shape rather than hardcoded.

## The ResNet18 file

Removed on 2026-10-20 with the fine-tuned ResNet18 path (`train_baseline`, `coffeecv/model.py`) and
the frozen screen's ResNet18 control; its manifest entry went with it.

`dinov2_vits14_pretrain.pth` was **copied** from `~/.cache/torch/hub/checkpoints/`, not moved: that cache
is where `torch.hub` looks, so deleting it there would make the DINOv2 reference arm re-download it mid-run.

## What is still missing for true offline reproducibility

Weights alone are not enough — the *architecture definition* has to be pinned too:

- **DINOv2** currently builds from the `torch.hub` checkout cached at
  `~/.cache/torch/hub/facebookresearch_dinov2_main/`, fetched from a floating `main`. Since
  2026-09-24 it is only a reference arm in Experiment 1 (plan §5.2), loaded exactly as the
  2026-09-21 screen loaded it; pin it only if it ever becomes a candidate again.
- **DINOv3** loads through **timm** (pin `timm==1.0.30`, in both locks; Apache-2.0 code), not through Meta's
  repository: `timm.create_model("vit_small_patch16_dinov3", pretrained=False, num_classes=0,
  global_pool="token")`, timm's `checkpoint_filter_fn`, a strict load of the `.pth` above, then
  copy the checkpoint's `rope_embed.periods` into `model.rope.periods`. The checkpoint stores those
  periods in bfloat16 and timm otherwise recomputes them in fp32 (a ~1e-3 difference on the CLS
  token); with the copy, timm is **bit-identical** to `facebookresearch/dinov3` at `6876159` at
  224 and 256 px (verified 2026-09-24). Never use `pretrained=True` -- it downloads a different
  file from the Hugging Face hub. Full write-up: `docs/dinov3_integration_plan.md` §2; the
  superseded Meta-repo loading notes (the `hubconf.py`/torchmetrics trap, `weights=<path>`
  copying into the torch hub cache) are its Appendix B.

## Steps toward reproducibility

1. Done: every file is `dvc add`ed and on the remote. The EfficientViT-SAM files were added on 2026-09-30 (ticket
   ML-2), the DINOv2 and DINOv3 files on 2026-10-21 (ticket OPS-11); each object was checked on the remote by its
   path, as `docs/agents/dvc.md` says, not by `dvc status --cloud`.
1b. Done 2026-09-24, ahead of step 1: `README.md`, `manifest.json` and `verify.py` were committed on
   their own so the tree is clean for the DINO work.
2. Dropped with the `train` stage (2026-10-20): the frozen fit is no DVC stage, and `litert_export` lists the
   weights it reads as deps.
3. Done for the frozen head: every run's `config.json` records `dino.weights_sha256`, and the checkpoint is
   checked against it on load.

## Licence

DINOv2 (Apache-2.0) is unrestricted for this use.
The DINOv3 weights are under the DINOv3 License, a copy of which is pinned at
`docs/reference_dinov3_LICENSE.md`. In brief: commercial use is permitted, you own your
derivatives, and serving predictions through an API does not require attaching the licence —
but redistributing the weights *or a derivative checkpoint* does. See
`docs/dinov3_integration_plan.md` Appendix A (full text at commit `1caf2f1`, §0b) for the full reading and the relicensing outlook.
