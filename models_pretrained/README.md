# Vendored pretrained backbones

Upstream pretrained weights, kept in the repo tree rather than resolved from the network at run
time. Rationale in `docs/dinov3_integration_plan.md` §2.5: a sweep that downloads its backbone
mid-run is not reproducible and fails on a box without egress.

The weight binaries are gitignored; **this README, `manifest.json` and `verify.py` are tracked
on purpose** — they are the acquisition instructions, the checksums and the checker, i.e. the only
way a fresh clone can obtain and validate the right files. (Until 2026-09-21 `.gitignore` excluded
the whole directory and hid all three.) **Not yet DVC-tracked**; see "Next steps".

## Getting these files on a fresh clone

```bash
python models_pretrained/verify.py     # what you have, and what is missing or wrong
```

- **You, on another machine:** `dvc pull` once the weights are DVC-tracked. They live in the
  private DVC remote, so this needs nothing from Meta a second time.
  Until then, rsync this folder from a machine that has it and run `verify.py` on the receiving
  side -- done for `powervpsssh:~/coffee-vision/models_pretrained/` on 2026-09-24 (all 5 files
  sha256 OK there).
- **Anyone else:** DINOv3 cannot be redistributed from a public repo (licence §1b-i), so a copy
  has to be requested from Meta directly. Full procedure, including the two-incompatible-formats
  trap, is `docs/dinov3_integration_plan.md` **§2.5**. Short version: take the **Meta-direct
  `.pth`** files, not the Hugging Face `safetensors` — they are different files, the hashes here
  match only the former, and only the former is what the loader (timm, plan §2.2) is verified against.

DINOv2 and ResNet18 need no gate: they re-download automatically from `torch.hub` /
`torchvision` if absent, which is exactly the run-time network dependency this folder exists to
remove — so verify them too rather than assuming.

## Contents

| file | params | embed dim | patch | tokens @224 | source / licence |
|---|---|---|---|---|---|
| `resnet18/resnet18-f37072fd.pth` | 11.7M | 512 | — | — | torchvision `ResNet18_Weights.IMAGENET1K_V1` (= `DEFAULT`) · BSD-3-Clause |
| `dinov2/dinov2_vits14_pretrain.pth` | 22.1M | 384 | 14 | 256 | `facebookresearch/dinov2` · **Apache-2.0** |
| `dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | 21.6M | 384 | 16 | 196 | `facebookresearch/dinov3` · DINOv3 License (gated) |
| `dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` | 28.7M | 384 | 16 | 196 | same |
| `dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 85.7M | 768 | 16 | 196 | same |

All five load cleanly. The four whose filenames embed a sha256 prefix (torch.hub's convention)
were **verified against their contents** — the downloads are intact and authentic. DINOv2's
upstream filename carries no hash, so its sha256 is recorded in `manifest.json` only.

## What the numbers mean for this project

**`dinov3_vits16` and `dinov3_vits16plus` are both 384-d**, the same width as DINOv2 ViT-S/14.
That makes either a drop-in as far as the embedding-space work is concerned: the OOD reference,
the probe's `mu`/`sd`/`w` vectors, `embedding_dim_of` and the classifier head shape are all
unchanged when swapping between them. Only the weights and the patch size differ.

**`dinov3_vitb16` is 768-d and ~4x the parameters.** It is not a drop-in — it needs its own OOD
sidecar rebuild — and on a CPU-only box it is very likely too slow for the deployed 40-patch path.
Useful as a distillation teacher or as an upper bound on what the representation can do; not a
serving candidate here.

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

## Note on the ResNet18 file

It was **copied** from `~/.cache/torch/hub/checkpoints/`, not moved. The torch cache is outside
the repo and is where `torchvision.models.resnet18(weights=DEFAULT)` looks; deleting it there
would make the next `build_model` call re-download 47 MB, which is exactly the mid-run network
dependency this folder exists to remove. Keeping both copies costs 47 MB and removes that risk.
Delete the cache copy only once `build_model` loads from this folder instead of from torchvision's
downloader.

The same applies to `dinov2_vits14_pretrain.pth`.

## What is still missing for true offline reproducibility

Weights alone are not enough — the *architecture definition* has to be pinned too:

- **DINOv2** currently builds from the `torch.hub` checkout cached at
  `~/.cache/torch/hub/facebookresearch_dinov2_main/`, fetched from a floating `main`. Since
  2026-09-24 it is only a reference arm in Experiment 1 (plan §5.2), loaded exactly as the
  2026-09-21 screen loaded it; pin it only if it ever becomes a candidate again.
- **DINOv3** loads through **timm** (pin `timm==1.0.29`; Apache-2.0 code), not through Meta's
  repository: `timm.create_model("vit_small_patch16_dinov3", pretrained=False, num_classes=0,
  global_pool="token")`, timm's `checkpoint_filter_fn`, a strict load of the `.pth` above, then
  copy the checkpoint's `rope_embed.periods` into `model.rope.periods`. The checkpoint stores those
  periods in bfloat16 and timm otherwise recomputes them in fp32 (a ~1e-3 difference on the CLS
  token); with the copy, timm is **bit-identical** to `facebookresearch/dinov3` at `6876159` at
  224 and 256 px (verified 2026-09-24). Never use `pretrained=True` -- it downloads a different
  file from the Hugging Face hub. Full write-up: `docs/dinov3_integration_plan.md` §2; the
  superseded Meta-repo loading notes (the `hubconf.py`/torchmetrics trap, `weights=<path>`
  copying into the torch hub cache) are its Appendix B.

## Next steps

1. `dvc add` each file and commit the `.dvc` pointers, so the weights reach the DVC remote and the
   train stage can depend on them. **Not done yet** — Stage 1 no longer blocks it (it finished
   2026-09-21), but the DVC remote is down for maintenance as of 2026-09-24. Verify the DVC push by
   expanding `dvc.lock`'s `.dir` trees, not by trusting `dvc status --cloud`.
1b. Done 2026-09-24, ahead of step 1: `README.md`, `manifest.json` and `verify.py` were committed on
   their own so the tree is clean for the DINO work; the `.dvc` pointers follow with step 1.
2. Add `models_pretrained/` entries as deps of the `train` stage in `dvc.yaml` (plan §8.5).
3. Record the chosen file's sha256 in `RunConfig` so it lands in every archived `config.json`.

## Licence

DINOv2 (Apache-2.0) and torchvision's ResNet18 (BSD-3-Clause) are unrestricted for this use.
The DINOv3 weights are under the DINOv3 License, a copy of which is pinned at
`docs/reference_dinov3_LICENSE.md`. In brief: commercial use is permitted, you own your
derivatives, and serving predictions through an API does not require attaching the licence —
but redistributing the weights *or a derivative checkpoint* does. See
`docs/dinov3_integration_plan.md` Appendix A (full text at commit `1caf2f1`, §0b) for the full reading and the relicensing outlook.
