# OPS-23: DINOv3 in 16-bit floating point (BF16 first, then FP16)

Research note for [OPS-23](../../docs/ticket_pi5_inference_host.html), written 2026-11-10. Sources are primary: source code
at a named tag or commit, official docs, specs, and the model authors' paper and repository. "Measured" means a run on
this devcontainer's CPU, an AMD Ryzen 5 5600U (Zen 3: AVX2, FMA, F16C; no AVX-512, no BF16 instructions), with
LiteRT 2.3.0 (`ai-edge-litert`), torch 2.14.0+cpu and timm 1.0.30 from the training environment, and litert-torch 0.9.4
from `convert/.venv` for the test conversions. The inputs are ML-7's 13 photos and their PyTorch fp32 readouts
(`analysis/ml7_gpu/artifacts/inputs.npz`, `refs.npz`). The scripts lived in `/tmp` and are not committed.
"Unverified" marks claims that could not be checked here.

Terms used throughout:

- **fp16** (IEEE binary16): 5 exponent bits and 11 significant bits. The largest finite value is 65,504, and relative
  rounding is about 2^-11 (5e-4).
- **bf16** (bfloat16): 8 exponent bits, the same range as fp32, and 8 significant bits. Relative rounding is about
  2^-8 (4e-3), 8 times coarser than fp16.
- **Accumulation precision**: the format in which a matrix multiply keeps its running sum of products. "fp16 GEMM with
  fp16 accumulation" rounds the running sum to fp16 after every product.
- **The strict bar**: D7's fp32 bar, least readout cosine ≥ 0.99999 over every patch.
- **The fp16 bar**: ADR 0020's bar for DINOv3 in fp16, which the owner accepted for `yappy`'s GPU. No NaN or Inf; least
  patch cosine ≥ 0.999 on served photos; no guard verdict changed; top-1 changed on at most 1% of served photos; p95 of
  the largest class-probability change ≤ 0.05.

## Answer

**The Pi's fp16 failure is a bug in XNNPACK, not a precision limit, and a one-line exact rewrite of the readout avoids
it.** Under `TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16`, XNNPACK rewrites `mul(reduce_sum(x), 1/N)` to run in fp32
(`widen_fp16_accumulators`). The helper that converts the `1/N` constant to fp32 reads it from a buffer it has just
allocated, not from the constant [S3 L2550-2560]. The patch mean comes out about 0, and the CLS half of the readout is
untouched. That is the whole 0.809.

- Measured on x86 with the release's `dino_b8.tflite` and the same flag: photo 0's least cosine is 0.813010, against
  0.8130068 on the Pi. The CLS half has cosine ≥ 0.999998 and the mean half about 0 (norm 1e-4 against 12).
- The defect is still on XNNPACK's `master` (a5e4de31) [S4 L2635-2645].
- **The fix: write the patch mean as `sum(x * (1/N))` instead of `sum(x) * (1/N)`.** It is the same function in real
  arithmetic (ADR 0022). With only that change, the release's graph under FORCE_FP16 gives these results on all
  13 photos × 40 patches on x86 (measured):
  - least cosine **0.9999982**;
  - no top-1 change;
  - largest class-probability change 3.6e-4;
  - before the change: 0.809, 11 of 13 top-1 changed, and 0.997.

**BF16 is not viable on any host we run or have named.**

- **Hardware.** Neither the Pi's Cortex-A76 (no FEAT_BF16, an Armv8.6 feature) nor the Intel N100 (no AVX512-BF16,
  no AMX) has BF16 arithmetic. Neither has a Core Ultra Meteor Lake, CPU or iGPU. Of the hosts discussed in D2, only an
  AMD Zen 4 CPU (Ryzen 7040/8040) has AVX512-BF16, and none of their iGPUs does: RDNA2 and RDNA3 have no BF16 in Mesa
  by default.
- **Runtimes.** WebGPU and WGSL have no bf16 type. LiteRT's XNNPACK delegate has no bf16 tensor path. XNNPACK's bf16
  GEMM runs only on x86-64 with AVX512-BF16. ONNX Runtime's bf16 "fastmath" needs FEAT_BF16 on arm64.
- **Precision.** BF16 is also less exact than FP16 for this model. Measured on 104 patches: full bf16 least cosine
  0.99987, PyTorch CPU autocast bf16 0.99994, and bf16 weights alone 0.99993. Each fails the strict bar and passes the
  fp16 bar's cosine. fp16 with fp32 accumulation gives 0.999996.
- **Speed.** Without hardware support, bf16 is slower. PyTorch on this Zen 3 took 67 s for 104 patches in bf16,
  against 17 s in fp32 (measured).

**The FP16 recipe most likely to pass.** On the Pi's CPU:

- **The graph:** the release's DINOv3 graph (ADR 0022's fp16 forms), with the patch mean pre-scaled.
- **Split sums:** every matrix multiply whose sum runs over more than 128 terms is written as a pairwise sum of 128-wide
  slices, the form ADR 0022 already uses for the encoder's convolutions (`SplitSumConv2d`). That covers the four linear
  layers per block and attention·v.
- **The precision:** FORCE_FP16 on DINOv3 only.
- **Why the split matters:** XNNPACK on the A76 accumulates its fp16 GEMMs in fp16 [S5; S11]. On x86 it accumulates
  them in fp32 [S6], so the x86 result above is an upper bound.
- **The emulation,** in PyTorch, of the A76's fp16 accumulation (measured, 104 patches):
  - without the split, least cosine **0.99930** and largest class-probability change 0.026 (13 photos × 8 patches).
    That passes the fp16 bar's 0.999 cosine, narrowly, and fails the strict bar.
  - with the split, least cosine **0.9999834** and largest change 0.0023. That passes the fp16 bar with margin and falls
    just short of the strict bar's 0.99999.
  - An emulation, not the A76 itself (Unverified until run on the Pi).
- **A speed gain is not yet shown.** The Pi's replay with the release's DINOv3 in fp16 took 39.7 s a photo, about
  6.5 s slower than P2's plain DINOv3 graph in fp32 (§6). The forms' graph has never been timed in fp32 on the Pi.
  Accuracy may be recoverable while speed is not.
- No mode keeps fp32 accumulation with fp16 storage on the A76 in XNNPACK at the pinned commit: its only f32-accumulating
  f16 GEMMs are x86 kernels.
- The A76 also lacks FEAT_FHM (`asimdfhm`, the fp16-to-fp32 widening multiply-add `FMLAL`), so a fast kernel of that kind
  is not available to any runtime on the Pi.

**Under the strict bar,** no 16-bit arithmetic path found passes on the Pi, though the split form comes within 7e-6 of
it in emulation. Only 16-bit weight storage with fp32 arithmetic passes (fp16 weights: least cosine 0.9999989,
measured). That halves the file but not the time: LiteRT upsamples fp16 weights to fp32 on the CPU [S16].

**The next experiment,** with existing tools:

1. Convert DINOv3 at batch 8 twice, in `convert/.venv`:
   - (a) the release's forms with the readout's mean written as `sum(x * (1/N))`;
   - (b) (a) plus 128-wide split sums in the linear layers and attention·v.
   The `/tmp` script built (a) from `build_backbone`, `four_d_attention`, `constant_rope`, `litert_gpu_forms` and
   `fp16_forms`, with the readout rewritten.
2. Check both in fp32 on the CPU against the references, as every form is checked.
3. On the Pi, run `analysis/pi5/pi_replay.py` with `XNNPACK_FLAGS=4 FP16_GRAPHS=dino`, the segmenter in fp32.
4. Run `analysis/pi5/parity_now.py` on its output.

Expected:

- no NaN;
- least cosine near 0.9993 for (a) and 0.99998 for (b);
- (a)'s time about the 39.7 s a photo measured with the defective graph (segmenter fp32, DINOv3 fp16 at batch 8),
  since the defect does not change the arithmetic. (b) adds ops, so its time has to be measured.
- **Time each variant in fp32 too** (`p4_graph.py`, same file, without the flag). The fp16 run so far was slower than
  P2's fp32 DINOv3 (§6). fp16 on the Pi's CPU is worth pursuing only if a forms graph in fp16 beats the plain graph in
  fp32.

Then run ML-7 P7's 782-photo answer check (`p7_answers.py`) on the Pi for the better variant. Report the XNNPACK defect
upstream with the tiny reproducer in §1.2.

This note contradicts no ADR. It proposes one more exact form under ADR 0022, a mean pre-scaled by 1/N, for a runtime
defect rather than for range. Serving DINOv3 in fp16 on a CPU host would need ADR 0019's and 0020's checks, like the
GPU's.

## 1. Why the Pi's CPU failed in fp16

### 1.1 What FORCE_FP16 does in LiteRT 2.3.0

LiteRT v2.3.0 pins XNNPACK at commit `d89ef666` [S1 L83-89]. Its delegate exposes `TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16`
= 4, "Force FP16 inference for FP32 operators" [S2 L33-34].

- **With the flag, every fp32 tensor in the delegated graph becomes fp16** [S7 L1306-1416]:
  - intermediate tensors are defined as fp16;
  - static weights and constants get `XNN_VALUE_FLAG_PACK_TO_FP16`;
  - graph inputs and outputs stay fp32, with convert nodes at the edges.
  - Only the filter and bias of CONV_2D, DEPTHWISE_CONV_2D and TRANSPOSE_CONV are also kept in fp32 [S7 L1217-1251].
- **Nothing keeps LayerNorm, softmax or residual adds in fp32** [S7 L1196-1416]. The one automatic widening is the
  `mul(reduce_sum(x), c)` rewrite below.
- **The delegate's README calls the flag a development tool**: "intended for development workflows, and in particular
  for testing end-to-end accuracy of model when FP16 inference is used" [S8 L574-577].
  - It also says that on x86 with AVX2 the flag runs in emulation: "all elementary floating-point operations are
    computed in FP32, then converted to FP16 and back to FP32" [S8 L579-585].
  - Without the flag, fp16 is used only on hardware with native fp16, and only when the model's metadata declares fp16
    *accumulation* acceptable [S8 L548-571; S7 L1196-1214].
  - There is no "fp16 storage, fp32 accumulation" option among the delegate's flags [S2 L28-61].

### 1.2 The defect: `widen_fp16_accumulators` reads a constant it has just replaced

XNNPACK's subgraph optimiser has a rewrite "to make subgraphs rewritten to be fp16 less likely to overflow" [S3
L3808-3814]. For `mul(reduce_sum(x), c)` or `div(reduce_sum(x), c)` with a static scalar `c`:

- it makes the sum's output fp32;
- it converts `c` to fp32 with `convert_static_value_to_fp32`;
- and it converts the product back to fp16 [S3 L2565-2642].

The rewrite only applies when neither the sum nor `c` has another consumer [S3 L2597-2600].

`convert_static_value_to_fp32` does this [S3 L2550-2560]:

```c
static void convert_static_value_to_fp32(struct xnn_value* value) {
  ...
  value->data = xnn_allocate_memory(sizeof(float));
  float data = get_scalar_value_as_float(value);   /* reads the new, uninitialised buffer */
  memcpy(value->data, &data, sizeof(float));
  ...
```

It replaces `value->data` before it reads the scalar. If the old buffer was XNNPACK's own, it has also freed it. So `c`
becomes whatever the new allocation holds. By then the constant has already been packed to fp16 [S3 L4718], and the
read takes that memory's first two bytes as an fp16. The same lines are on XNNPACK `master` at a5e4de31 [S4 L2635-2645].

Measured with tiny graphs converted by litert-torch 0.9.4 and run on LiteRT 2.3.0 on x86. Input [8, 201, 768], random
values with mean 0.3 and std 2; the error is relative to the fp32 run.

| Graph (TFLite ops) | FORCE_FP16 relative error | Output |
|---|---|---|
| `x * 2` (MUL) | 3.8e-4 | correct |
| `x.sum(-1)` (SUM) | 3.5e-4 | correct |
| `x.mean(-1)` (SUM, MUL by 1/768) | 1.0 | all 0 |
| `x[:, 5:].mean(1)`, the readout's patch mean (SLICE, SUM, MUL by 1/196) | 1.0 | all 0 |
| `x[:, 5:].sum(1) / 196` (litert-torch emits SUM, MUL) | 1.0 | all 0 |
| `cat([x[:, 0], x[:, 5:].mean(1)])` | inf | Inf |
| `x - x.mean(-1)` and `(x*x).mean(-1)` together (1/768 used twice) | 7.1e-4 | correct |
| `(x[:, 5:] * (1/196)).sum(1)` (MUL, then SUM) | 6.2e-4 | correct |
| `cat([x[:, 0], (x[:, 5:] * (1/196)).sum(1)])` | 2.6e-4 | correct |

This explains every detail of the Pi's result:

- **The patch mean breaks; the backbone does not.** DINOv3's 25 LayerNorm forms (`_layer_norm_scaled`) all take the
  mean over 768 channels. The TFLite converter shares their `1/768` constant, so it has many consumers and is not
  rewritten.
- The readout's `1/196` is used once, so it is rewritten and lost.
- **The size of the loss is fixed by the readout's layout.** The readout is [CLS (768), patch mean (768)]. With the mean
  half lost, the cosine is the CLS half's share of the norm, about 0.81–0.83 on every patch. The Pi's per-photo least
  cosines were 0.813–0.827 (`analysis/pi5/results/parity_split.json`).
- **Measured on x86, the release's `dino_b8.tflite` under FORCE_FP16 reproduces the Pi.**

  | Photo | Least cosine, x86 | Least cosine, Pi | CLS half, least cosine | Mean half, least cosine |
  |---|---|---|---|---|
  | 0 | 0.813010 | 0.8130068 | 0.999998 | -0.028 |
  | 1 | 0.827074 | 0.8270661 | 0.999999 | -0.060 |
  | 2 | 0.822270 | 0.8222650 | 0.999999 | -0.101 |

  The x86 emulation and the A76 agree to six digits because the CLS half is near-exact on both. The value is set by
  the lost half, not by arithmetic.

**ML-7 P2's graph** (no fp16 forms) gave NaN on all 26 Pi photos. That fits the overflow ML-7 P7 found on the GPU:

- attention scores up to 1.3×10^5, past fp16's 65,504;
- LayerNorm sums of squares beyond fp16 in 19 of 25 norms, because the register tokens reach 1.5×10^4
  (`docs/plan_inference_runtime.html`, P7).

FORCE_FP16 materialises both in fp16.

Measured on x86: a graph with the pre-scaled mean but without the fp16 forms (only `four_d_attention`,
`constant_rope` and `litert_gpu_forms`) returns NaN under FORCE_FP16. So the forms are still needed on XNNPACK, as on
the GPU.

### 1.3 Precision once the defect is avoided: accumulation differs by ISA

XNNPACK's f16 GEMM (fully connected and batch matmul both use `xnn_init_f16_gemm_config` [S9; S10]) picks its kernels
by ISA.

- **On aarch64 with fp16 arithmetic, including `xnn_uarch_cortex_a76`,** it uses `6x16__asm_aarch64_neonfp16arith_cortex_a75`
  [S5 L167-267]. That kernel loads the bias into fp16 accumulators and adds every product with `FMLA v.8h`, an fp16
  fused multiply-add [S11 L94-104, L136-142]. Its running sums are fp16.
- **On x86 with AVX2,** the same config uses `f16_f32acc` kernels, which accumulate in fp32 [S5 L338-350].
- **f16-f32acc GEMM kernels exist only for AVX2 and AVX512-SKX** at this commit (`src/f16-f32acc-gemm/gen/`). No ARM one
  exists.
- **Reductions and softmax are better off on both ISAs.**
  - The f16 sum used for means accumulates in fp32 on both [S12 L650-701].
  - The f16 softmax subtracts the row maximum and sums its exponentials in fp32 [S13 L34-62; S14 L56-136].
  - On the A76, GELU and rsqrt run as native fp16 kernels (`neonfp16arith_rational_6_4_div`, `neonfp16arith_rsqrt`)
    [S15 L625-630, L986-991].

So the A76 differs from x86's emulation mainly in the GEMMs' accumulation:

- the four linear layers of each block, with inner lengths 768, 768, 768 and 3,072;
- q·kᵀ, over 64;
- attention·v, over 201.

ML-7 P7 found the same mechanism on `yappy`'s GPU for the encoder's long convolutions: running sums of 30–60 for
outputs of about 0.4 (ADR 0022). That GPU's DINOv3 fp16 cosine was 0.99977 (ADR 0020).

## 2. BF16 on the candidate hosts

### 2.1 Hardware

| Host | BF16 arithmetic | Evidence |
|---|---|---|
| Raspberry Pi 5, Cortex-A76 | **No** | LLVM's A76 feature list is Armv8.2 with FullFP16 and DotProd, without BF16 or FP16FML [S17 L778-781]. The Pi's `/proc/cpuinfo` flags (OPS-23) show `fphp asimdhp asimddp` and neither `bf16` nor `asimdfhm`. Linux sets `bf16` from ID_AA64ISAR1_EL1.BF16 and `asimdfhm` from ID_AA64ISAR0_EL1.FHM [S18 L97-101, L142-143, L215-216]. |
| Arm cores that have it | FEAT_BF16 | Optional from Armv8.2; Armv8.6 makes it a default extension [S19 L888-891]. Among cores: Neoverse V1 (Armv8.4), Neoverse N2, Cortex-A510, A710, A715, A520, A720, X2 have it. Cortex-A76, A77, A78, A78C and X1 do not [S17 L745-822, L857-873, L1001-1033]. |
| Intel N100 (Alder Lake-N, Gracemont), `yappy` | **No** | AVX2, FMA, F16C, AVX-VNNI; no AVX-512, no AMX (lscpu on `yappy`, ML-7 §2.4). LLVM's `gracemont` gets the Alder Lake feature list, which has no BF16 [S20 L1272-1296, L1862]. |
| Intel Core Ultra (Meteor Lake) CPU | **No** | LLVM's `meteorlake` also gets the Alder Lake list [S20 L1870]. Arrow Lake adds AVX-NE-CONVERT [S20 L1309-1316]: bf16 loads and conversions, not bf16 arithmetic. |
| Intel Arc iGPU in Meteor Lake (Xe-LPG) | **No** | Mesa 26.2.6 sets `has_bfloat16 = false` and `has_systolic = false` (no XMX matrix engines) for MTL. Arrow Lake-H sets both true [S21 L1109-1147]. ANV exposes VK_KHR_shader_bfloat16 only where `has_bfloat16` holds [S22 L466]. |
| Intel N100's iGPU (Xe-LP, Gfx12) | **No** | Not in Mesa's list for the extension, which is ANV Gfx12.5+ [S23 L584]; ADL-N is Gfx12 (Unverified beyond Mesa's naming). |
| AMD Zen 4 CPU (Ryzen 7040/8040) | **Yes**, AVX512-BF16 | LLVM's `znver4` list includes FeatureBF16 [S20 L1584-1598]. |
| AMD Radeon 680M (RDNA2) / 780M (RDNA3) | **No**, by default | RADV enables VK_KHR_shader_bfloat16 only on GFX12 (RDNA4). On GFX11 it needs an experimental flag, because "GFX11-11.5 has precision issues"; below GFX11 it is never offered [S24 L203-212, L782]. |
| WebGPU / WGSL (Dawn) | **No bf16 type** | WGSL's 16-bit float is `f16`, IEEE binary16, behind the `shader-f16` feature [S25 L1867, L2954, L12142]. Dawn's feature list has `shader f16` and nothing bf16 [S26 L2518]. |

The Vulkan extension itself exists, VK_KHR_shader_bfloat16 [S27]. LiteRT's GPU path reaches the GPU through WebGPU, which
cannot express it.

### 2.2 Runtimes

- **LiteRT / XNNPACK (CPU).**
  - The delegate maps TFLite float32 and float16 tensors to XNNPACK, and no bfloat16 [S7 L234-241]. A graph cannot be
    handed to XNNPACK in bf16 through it.
  - Inside XNNPACK, the bf16→fp32 GEMM config selects kernels only on x86-64 with AVX512-BF16 [S5 L451-478].
  - Other bf16 ops are lowered to fp32 where no native kernel exists [S3 L4781-4784].
  - Not available on any of our hosts except a Zen 4, and not reachable from TFLite there either.
- **LiteRT GPU / WebGPU.** fp16 or fp32 only (`GpuOptions.enforce_f32`, ML-7 §2.4). WGSL has no bf16 (above).
- **ONNX Runtime 1.31 (CPU).**
  - The session option `mlas.enable_gemm_fastmath_arm64_bfloat16` runs fp32 MatMul through a bf16 GEMM [S28 L630-634].
  - It is honoured only on arm64 Linux or macOS, and only when `MlasBf16AccelerationSupported()` holds [S29 L69-72;
    S30 L98-105]. That reads HWCAP2_BF16 [S31 L110-125], so on the A76 the option does nothing.
  - The SME2 overrides are KleidiAI's [S31 L827-833].
  - The CPU provider registers a bf16 LayerNormalization, among few others [S32 L1106]. Coverage for a whole ViT was
    not checked (Unverified).
- **ONNX Runtime 1.31, fp16 on the CPU.** Unlike ML-7 §2.3's finding, the CPU provider now registers fp16 kernels
  [S32 L195-1395]:
  - MatMul, Gemm, Add, Sub, Mul, Div, Erf, Gelu, LayerNormalization and Conv;
  - not Softmax.
  - Its NEON fp16 GEMM accumulates in fp16 too (`float16x8_t accu…` [S33 L606-608]).
  - Not measured here.
- **PyTorch 2.14 (CPU).**
  - `torch.autocast("cpu", dtype=torch.bfloat16)` runs convolutions, `linear`, `matmul`, `bmm`, `addmm` and
    `scaled_dot_product_attention` in bf16 [S34 L338-363].
  - LayerNorm, softmax and the residual adds are not on that list, so they run in the dtype they receive. In a ViT that
    is the fp32 residual stream.
  - `torch.backends.mkldnn.fp32_precision` exists [S35 L103-137]. Whether it gives bf16 fast math on an A76 without
    BF16 is Unverified; without the instructions there is nothing to accelerate.

### 2.3 The model and bf16

- **DINOv3 was trained in bf16.**
  - The paper: training used "bfloat16 and 8-bit floating-point matrix multiplications" (Appendix C) [S36].
  - The repository's training configs set `param_dtype: bf16` and `reduce_dtype: fp32` [S37 `configs/ssl_default_config.yaml`
    L6-8].
  - Its README runs the 7B segmentor and depther under `torch.autocast('cuda', dtype=torch.bfloat16)` [S37 `README.md`
    L377, L518].
- **The released backbones compute RoPE in fp32** (`pos_embed_rope_dtype="fp32"`, ViT-B/16 included [S37
  `dinov3/hub/backbones.py` L277-302]). Training configs use bf16 RoPE.
- No fp16 is used anywhere in the repository's configs (searched).
- timm's port:
  - computes RoPE periods in fp32 and keeps them as a non-persistent buffer in the model's dtype [S45 L1173, L1197-1212,
    L1253-1257];
  - so `.to(torch.bfloat16)` rounds the periods too;
  - at 224 px this measured no worse (below).

## 3. Precision pitfalls, measured

PyTorch on this Zen 3, the first 8 patches of each of the 13 photos (104 patches), batch 8, against `refs.npz`'s PyTorch
fp32 readouts:

- the readout's patch mean taken in fp32 and cast back;
- "CLS" and "mean" are the two 768-d halves of the readout;
- variants marked † used ADR 0022's `fp16_forms`.

| Variant | Least cosine | Median | Least, CLS half | Least, mean half | NaN/Inf | Time, 104 patches |
|---|---|---|---|---|---|---|
| fp32 (sanity) | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 0 | 17 s |
| bf16 weights only, fp32 compute (params and RoPE periods rounded) | 0.9999301 | 0.9999470 | 0.9999259 | 0.9999354 | 0 | 14 s |
| bf16 weights only, RoPE periods kept fp32 | 0.9999301 | 0.9999470 | 0.9999259 | 0.9999354 | 0 | 17 s |
| `torch.autocast('cpu', bfloat16)` | 0.9999415 | 0.9999603 | 0.9999339 | 0.9999550 | 0 | 67 s |
| everything bf16 (`.to(bfloat16)`) | 0.9998739 | 0.9999100 | 0.9998572 | 0.9999121 | 0 | 67 s |
| everything bf16, LayerNorm in fp32 | 0.9998832 | 0.9999101 | 0.9998661 | 0.9999137 | 0 | 63 s |
| everything bf16, RoPE table computed in fp32 | 0.9998696 | 0.9999115 | 0.9998500 | 0.9999079 | 0 | 70 s |
| fp16 weights only, fp32 compute | 0.9999989 | 0.9999993 | 0.9999987 | 0.9999992 | 0 | 14 s |
| everything fp16 (`.half()`) | 0.9999966 | 0.9999986 | 0.9999962 | 0.9999977 | 0 | 53 s |
| everything fp16, LayerNorm in fp32 | 0.9999969 | 0.9999986 | 0.9999965 | 0.9999978 | 0 | 53 s |
| fp16 † | 0.9999965 | 0.9999981 | 0.9999956 | 0.9999977 | 0 | 75 s |
| fp16 †, every GEMM accumulated in fp16 (A76 emulation) | 0.9993021 | 0.9995710 | 0.9991942 | 0.9995429 | 0 | 353 s |
| fp16 †, the same, sums over more than 128 terms split into 128-wide slices added pairwise | 0.9999834 | 0.9999921 | 0.9999795 | 0.9999916 | 0 | 429 s |

What the table shows:

- **bf16's 8 significant bits are the limit, not its range.** Rounding only the weights to bf16 already costs the strict
  bar (0.99993). bf16 activations take it to 0.99987.
  - Keeping LayerNorm in fp32 hardly helps (0.99988).
  - Neither does an fp32 RoPE table (0.99987). At 224 px the 14×14 RoPE angles are not sensitive.
  - Autocast is better (0.99994) because the residual stream and LayerNorm stay fp32.
  - All pass the fp16 bar's 0.999 by a wide margin.
- **fp16 storage costs little where the arithmetic is fp32.** Even an unmodified `.half()` model has no NaN on the CPU
  (0.9999966).
  - PyTorch's CPU kernels compute half-precision ops in fp32 internally, so the 1.3×10^5 attention scores and the large
    LayerNorm sums of squares never exist as fp16 values.
  - The forms matter where a runtime stores those intermediates in fp16: the GPU, and XNNPACK under FORCE_FP16.
- **Accumulation is where fp16 loses most.** Rounding the running sum of every product to fp16, as the A76 kernel does,
  takes the cosine from 0.999996 to 0.99930.
- **The emulation's limits** (Unverified as an exact model of the A76):
  - `addcmul` in fp16 on the CPU computes in fp32 and rounds once per step, like the fused `FMLA`.
  - The patch embedding and the elementwise ops are rounded to fp16, not computed in fp16.
  - GELU and rsqrt use PyTorch's functions, not XNNPACK's fp16 approximations.

The emulation and the GPU measurement agree in size. `yappy`'s GPU, which also accumulates in fp16, measured 0.99977 on
the fixtures (ML-7 P7).

### 3.1 The model's large values (why the forms exist)

- **High-norm tokens.**
  - Darcet et al. found high-norm "artifact" tokens in supervised and self-supervised ViTs, DINOv2 included, and added
    register tokens to absorb them [S39].
  - Sun et al. found "massive activations" in ViTs, in random patch tokens of CLIP and DINOv2 ViT-L. In DINOv2-reg
    ViT-G they appear only in one register token [S40 §5]. Setting them to zero hurts accuracy; setting them to their
    mean does not, so they act as fixed biases [S40 §5, Table 4].
- **DINOv3 B/16 has 4 registers** [S36 Table 2, A.1; S38 L3079-3101].
  - The paper also reports feature-dimension outliers that grow with depth and peak at the output. The final LayerNorm
    scales them down [S36 A.2].
  - On our photos the register tokens reach 1.5×10^4 and the attention scores 1.4×10^5. 19 of 25 LayerNorm sums of
    squares exceed fp16 (`analysis/ml7_gpu/p7_ranges.json`; ML-7 P7).
  - These are why ADR 0022's forms scale each LayerNorm input by its token's max and shift the scores. bf16's range
    removes that need; fp16 with fp32-internal kernels (PyTorch CPU) never meets it.
- **LayerScale.** timm builds DINOv3 B/16 with LayerScale initialised at 1e-5 [S38 L3091].
  - In the checkpoint, 18 of 18,432 LayerScale entries are below fp16's smallest normal number, 6.1e-5 (smallest
    |γ| 3.7e-6, median 0.88).
  - In fp16 they become subnormals, which fp16 represents; XNNPACK's ARM kernels do not flush them (Unverified).
  - Not a cause of the failure here.
- **Training-time practice.** Mixed-precision training keeps an fp32 master copy and scales the loss for fp16
  [S41 abstract]. bf16 needs no such tuning because it has fp32's range [S42 abstract]. Neither paper addresses
  inference with fp16 accumulation.

## 4. Mitigations, each with its evidence

| Mitigation | Effect here | Source |
|---|---|---|
| Pre-scale the patch mean, `sum(x * 1/N)` | Avoids XNNPACK's defect. Release graph under FORCE_FP16 on x86: 0.809 → 0.999998 (measured). | §1.2 |
| fp32 accumulation in GEMMs | x86's FORCE_FP16 has it: 0.999998. A76 has no such XNNPACK kernel. | [S5; S6] |
| Split long sums into 128-wide slices, added pairwise | In the A76 emulation: 0.99930 → 0.99998, and the largest probability change on 13 photos × 8 patches 0.026 → 0.0023 (measured). Costs graph size: each linear layer becomes 6 (or 24 for fc2) matmuls and a tree of adds. | ADR 0022 (`SplitSumConv2d`), §3 |
| LayerNorm on x / max\|x\|, scores shifted by their row max (ADR 0022) | Needed wherever fp16 stores LayerNorm statistics and scores. No cost to accuracy (0.9999965 in PyTorch). | ML-7 P7 |
| LayerNorm, softmax and residual adds in fp32, matmuls in 16-bit | PyTorch autocast bf16: 0.99994. XNNPACK has no per-op precision control in the delegate. | [S34; S7] |
| fp16 weights, fp32 compute | 0.9999989, passes the strict bar. On LiteRT's CPU the weights are upsampled to fp32 before the first inference, so no speed gain [S16]. XNNPACK keeps fp16 weights that feed FC/Conv as fp16 for its GEMMs [S7 L7311-7370], still with fp32 activations. | [S16; S7] |
| ONNX mixed fp16: `convert_float_to_float16(keep_io_types=True, op_block_list=[...])` | `op_block_list` replaces the default list [S43 L140-170, L196-215], so pass the default plus `LayerNormalization`, `Softmax`, `ReduceMean`. `auto_convert_mixed_precision(model, feed_dict, validate_fn, rtol, atol, keep_io_types)` searches for the nodes to keep in fp32 [S44 L36-39]. ML-7 P1 made these copies for ORT's WebGPU. On the Pi's CPU ORT has no fp16 Softmax and accumulates fp16 GEMMs in fp16 (§2.2). Not measured. | [S43; S44] |
| `torch.autocast('cpu', bfloat16)` | 0.99994, but 4× slower here without BF16 hardware. | [S34], §3 |
| SmoothQuant-style per-channel scaling | Built for int8. For fp16 the same idea is what ADR 0022's power-of-two scales do; nothing further found for fp16 inference. | — |

**Selective precision by block was not measured.** The forms already keep every block in range. The emulation's
remaining loss would have to be traced block by block to say where it arises.

## 5. Recipes per runtime

- **LiteRT on the Pi's CPU (fp16):**
  - the release's DINOv3 forms, plus the pre-scaled patch mean;
  - if (a) misses the fp16 bar or its margin is thin: 128-wide split sums in the linear layers and attention·v (§3,
    §4);
  - `CpuOptions(xnnpack_flags=4)` for the DINOv3 graph only, the segmenter in fp32 (`pi_replay.py` already does this
    with `FP16_GRAPHS=dino`);
  - held to ADR 0020's fp16 bar, not the strict bar.
  - The metadata route (`reduced_precision_support` with fp16 accumulation [S8 L554-566; S7 L1196-1214]) does the same
    arithmetic without the flag. It is set through TensorFlow's converter, not litert-torch (Unverified for
    litert-torch).
- **LiteRT, fp16 weights only:**
  - ai-edge-quantizer 0.9.0, in `convert/.venv`, has a `FLOAT_CASTING` algorithm "currently only used for fp16
    quantization" (`algorithm_manager.py` L67, L164; `qtyping.py` L310-320). Its recipe for a litert-torch export was not
    tried (Unverified).
  - The effect would be half the file and the same time on the CPU [S16].
- **ONNX Runtime on the CPU:**
  - fp32 stays the measured, passing path (OPS-23: 24.4 s a photo).
  - bf16 fastmath is inert on the A76 [S29; S31].
  - fp16 has partial kernel coverage and fp16 accumulation (§2.2), unmeasured.
- **PyTorch on the CPU:**
  - bf16 autocast is accurate enough for the fp16 bar.
  - It is useful only on a CPU with BF16 instructions: a Zen 4, or an Arm core with FEAT_BF16.
  - On the Pi and the N100 it would be slower than fp32 (measured 4× slower on Zen 3, which has none).

## 6. Expected speed

- **Measured on the Pi** (OPS-23), photo replay; no fp16 speed gain for DINOv3 is shown yet:
  - **every graph forced to fp16:** 13.3 s a photo, against 44.6 s in fp32, 3.3× faster. These were ML-7 P2's files,
    without the forms, which returned NaN.
  - **only the release's DINOv3 in fp16** (batch 8; the segmenter in fp32; `results/litert_split__replay.json`): 39.7 s
    a photo.
  - **the same replay in fp32 at batch 8, estimated** from P2's per-graph times (2 × 4.17 s encoder, 2 × 0.12 s decoder,
    5 × 4.93 s DINOv3): about 33.2 s.
  - **So the release's DINOv3 graph in fp16 was about 6.5 s a photo slower** than P2's DINOv3 in fp32. The release's
    graph carries the forms' extra ops (ABS, REDUCE_MAX, DIV, SUM … per LayerNorm and attention). Its fp32 time on the
    Pi was never measured, so the cause is unknown: the forms' ops, or XNNPACK's fp16 path on them (Unverified).
  - The defect does not change the arithmetic, so variant (a) should cost about the same as the defective graph. The
    split-sum variant adds ops.
- **No primary source found states the A76's fp16 throughput relative to fp32.**
  - A 128-bit NEON register holds 8 fp16 values or 4 fp32 values, so `FMLA v.8h` does twice the multiply-adds of
    `FMLA v.4s` per instruction [S11].
  - XNNPACK's README claims no figure (Unverified as to speedup).
- **bf16 without hardware support is emulated by conversion.**
  - Measured here: PyTorch bf16 took 67 s against fp32's 17 s for 104 patches.
  - fp16 in PyTorch on this CPU: 53 s.
  - LiteRT's README describes FORCE_FP16 on x86 as a simulation of fp16's precision and range, not a speed path
    [S8 L579-585].
- **16-bit weights halve the weights' memory and file size** [S16]. DINOv3 B/16 is 343 MB in fp32
  (`p7_fp16w/manifest.json`).

## 7. The x86 runs of the release graph, answer level

All 13 photos × 40 patches through LiteRT 2.3.0's XNNPACK on x86, batch 8. Probabilities are the head's softmax averaged
over the 40 patches, compared with the same computed from `refs.npz`'s readouts. Only 13 photos, so these are not ADR
0020's served-photo statistics.

| Graph | Flags | Least cosine | NaN/Inf | Top-1 changed | Largest class-probability change |
|---|---|---|---|---|---|
| Release `dino_b8.tflite` (fp16 forms) | FORCE_FP16 | 0.8088815 | no | 11 of 13 | 0.997 |
| Release forms, patch mean pre-scaled | FORCE_FP16 | 0.9999982 | no | 0 | 3.6e-4 |
| Pre-scaled mean, no fp16 forms | FORCE_FP16 | NaN | yes | 11 of 13 (NaN) | NaN |
| Release forms, patch mean pre-scaled | none (fp32) | 0.99999999998 | no | 0 | 4.9e-7 |

The first row matches the Pi's own result (least cosine 0.809, 11 of 13 answers changed; OPS-23). The rewritten
graph's fp32 row confirms the rewrite is exact at float rounding.

## Sources

Pinned commits: LiteRT `v2.3.0` (4352e261); XNNPACK `d89ef6669a14db203b3b7935b1b3862cb63fb6df` (LiteRT 2.3.0's pin) and
`master` at `a5e4de3149dc68a26b063177e102c16c6cee594e` (read 2026-11-10); LLVM `llvmorg-21.1.0`; Mesa `mesa-26.2.6`;
ONNX Runtime `v1.31.0`; PyTorch `v2.14.0`; timm `v1.0.30`; facebookresearch/dinov3 `6876159a11b4df116f30f667f8c9888617df0751`;
onnxconverter-common `c26a5216ea0a24cd82156e8faf4a080e5400eecd`; Dawn `b73cc737e4e714b5ce17dc5d7316a30e35e23990`; gpuweb
`6bf1c35fffaae37cf9db16c88e7d5d03382b5c52`; Vulkan-Docs `fa30d4e312e843193601be7cb926c7e25d9ec885`; Linux `v6.12`.

1. [S1] LiteRT v2.3.0, `litert_workspace.bzl` L83-89 (XNNPACK pin). https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/litert_workspace.bzl#L83-L89
2. [S2] LiteRT v2.3.0, `tflite/delegates/xnnpack/xnnpack_delegate.h` L28-61 (delegate flags). https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/tflite/delegates/xnnpack/xnnpack_delegate.h#L28-L61
3. [S3] XNNPACK d89ef666, `src/subgraph.c`: `convert_static_value_to_fp32` L2550-2560, `widen_fp16_accumulators` L2565-2642, the call L3808-3814, `xnn_subgraph_optimize` L4709-4787. https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/subgraph.c#L2550-L2642
4. [S4] XNNPACK master a5e4de31, `src/subgraph.c` L2635-2645, L2650-2690 (the same code). https://github.com/google/XNNPACK/blob/a5e4de3149dc68a26b063177e102c16c6cee594e/src/subgraph.c#L2635-L2645
5. [S5] XNNPACK d89ef666, `src/configs/gemm-config.c`: `init_f16_gemm_config` L167-370 (A76 case L243-267, AVX2 L338-350); `init_bf16_f32_gemm_config` L451-478. https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/configs/gemm-config.c#L167-L370
6. [S6] XNNPACK d89ef666, `src/f16-f32acc-gemm/gen/f16-f32acc-gemm-4x16-minmax-avx2-broadcast.c` L73-100 (fp32 accumulators, `_mm256_fmadd_ps`). https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/f16-f32acc-gemm/gen/f16-f32acc-gemm-4x16-minmax-avx2-broadcast.c#L73-L100
7. [S7] LiteRT v2.3.0, `tflite/delegates/xnnpack/xnnpack_delegate.cc`: datatype map L234-241; `promote_to_fp16` L617-626; FP16 enable and metadata L1196-1214; blocklist L1217-1251; tensor definition under force_fp16 L1306-1416; fp16 weight dequantize handling L7311-7370. https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/tflite/delegates/xnnpack/xnnpack_delegate.cc#L1196-L1416
8. [S8] LiteRT v2.3.0, `tflite/delegates/xnnpack/README.md` L542-600, "Floating-Point (IEEE FP16) Operators". https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/tflite/delegates/xnnpack/README.md#L542-L600
9. [S9] XNNPACK d89ef666, `src/operators/fully-connected-nc.c` L1520 (f16 FC uses `xnn_init_f16_gemm_config`). https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/operators/fully-connected-nc.c#L1520
10. [S10] XNNPACK d89ef666, `src/operators/batch-matrix-multiply-nc.c` L612. https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/operators/batch-matrix-multiply-nc.c#L612
11. [S11] XNNPACK d89ef666, `src/f16-gemm/6x16-aarch64-neonfp16arith-cortex-a75.S.in` L94-104 (bias into accumulators), L136-142 (`FMLA v20.8h …`). https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/f16-gemm/6x16-aarch64-neonfp16arith-cortex-a75.S.in#L94-L142
12. [S12] XNNPACK d89ef666, `src/configs/reduce-config.c` L650-701 (`f16_f32acc_rsum`). https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/configs/reduce-config.c#L650-L701
13. [S13] XNNPACK d89ef666, `src/configs/raddstoreexpminusmax-config.c` L34-62. https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/configs/raddstoreexpminusmax-config.c#L34-L62
14. [S14] XNNPACK d89ef666, `src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in` L56-136 (fp32 sum of exponentials). https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in#L56-L136
15. [S15] XNNPACK d89ef666, `src/configs/unary-elementwise-config.c` L625-630 (f16 GELU), L986-991 (f16 rsqrt). https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/configs/unary-elementwise-config.c#L625-L630
16. [S16] Google AI Edge, "Post-training float16 quantization" (LiteRT docs), read 2026-11-10. https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_float16_quant
17. [S17] LLVM llvmorg-21.1.0, `llvm/lib/Target/AArch64/AArch64Processors.td` L741-873, L1001-1040 (per-core features). https://github.com/llvm/llvm-project/blob/llvmorg-21.1.0/llvm/lib/Target/AArch64/AArch64Processors.td#L741-L873
18. [S18] Linux v6.12, `Documentation/arch/arm64/elf_hwcaps.rst` L97-101, L142-143, L215-216. https://github.com/torvalds/linux/blob/v6.12/Documentation/arch/arm64/elf_hwcaps.rst
19. [S19] LLVM llvmorg-21.1.0, `llvm/lib/Target/AArch64/AArch64Features.td` L209-210 (FEAT_FHM), L282-283 (FEAT_BF16), L888-891 (Armv8.6 adds BF16). https://github.com/llvm/llvm-project/blob/llvmorg-21.1.0/llvm/lib/Target/AArch64/AArch64Features.td#L888-L891
20. [S20] LLVM llvmorg-21.1.0, `llvm/lib/Target/X86/X86.td` L1272-1296 (Alder Lake), L1309-1316 (Arrow Lake), L1584-1598 (Zen 4), L1859-1873 (gracemont, meteorlake, arrowlake). https://github.com/llvm/llvm-project/blob/llvmorg-21.1.0/llvm/lib/Target/X86/X86.td#L1272-L1316
21. [S21] Mesa 26.2.6, `src/intel/dev/intel_device_info.c` L1041-1048 (XEHP_FEATURES), L1109-1147 (MTL, ARL). https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.6/src/intel/dev/intel_device_info.c#L1109-1147
22. [S22] Mesa 26.2.6, `src/intel/vulkan/anv_physical_device.c` L466, L1042-1046. https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.6/src/intel/vulkan/anv_physical_device.c#L466
23. [S23] Mesa 26.2.6, `docs/features.txt` L480, L544, L584. https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.6/docs/features.txt#L584
24. [S24] Mesa 26.2.6, `src/amd/vulkan/radv_physical_device.c` L203-212, L782. https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.6/src/amd/vulkan/radv_physical_device.c#L203-212
25. [S25] WGSL spec source, gpuweb 6bf1c35f, `wgsl/index.bs` L1867, L2954, L12142. https://github.com/gpuweb/gpuweb/blob/6bf1c35fffaae37cf9db16c88e7d5d03382b5c52/wgsl/index.bs
26. [S26] Dawn b73cc737, `src/dawn/dawn.json` L2518 (feature names). https://github.com/google/dawn/blob/b73cc737e4e714b5ce17dc5d7316a30e35e23990/src/dawn/dawn.json#L2518
27. [S27] Vulkan-Docs fa30d4e3, `appendices/VK_KHR_shader_bfloat16.adoc`. https://github.com/KhronosGroup/Vulkan-Docs/blob/fa30d4e312e843193601be7cb926c7e25d9ec885/appendices/VK_KHR_shader_bfloat16.adoc
28. [S28] ONNX Runtime v1.31.0, `include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h` L630-634. https://github.com/microsoft/onnxruntime/blob/v1.31.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L630-L634
29. [S29] ONNX Runtime v1.31.0, `onnxruntime/core/providers/cpu/math/matmul.h` L69-72, L106-119. https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/providers/cpu/math/matmul.h#L69-L119
30. [S30] ONNX Runtime v1.31.0, `onnxruntime/core/mlas/inc/mlas.h` L98-105 (`MLAS_SBGEMM_AVAILABLE`). https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/mlas/inc/mlas.h#L98-L105
31. [S31] ONNX Runtime v1.31.0, `onnxruntime/core/mlas/lib/platform.cpp` L110-125 (HWCAP2_BF16), L827-833 (SME2 SBGEMM). https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/mlas/lib/platform.cpp#L110-L125
32. [S32] ONNX Runtime v1.31.0, `onnxruntime/core/providers/cpu/cpu_execution_provider.cc` (MLFloat16 kernels L187-1395; BFloat16 LayerNormalization L1106). https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/providers/cpu/cpu_execution_provider.cc
33. [S33] ONNX Runtime v1.31.0, `onnxruntime/core/mlas/lib/halfgemm_kernel_neon_fp16.cpp` L536-608. https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/mlas/lib/halfgemm_kernel_neon_fp16.cpp#L536-L608
34. [S34] PyTorch v2.14.0, `aten/src/ATen/autocast_mode.cpp` L338-418 (AutocastCPU lists). https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/autocast_mode.cpp#L338-L418
35. [S35] PyTorch v2.14.0, `torch/backends/mkldnn/__init__.py` L83-137. https://github.com/pytorch/pytorch/blob/v2.14.0/torch/backends/mkldnn/__init__.py#L83-L137
36. [S36] Siméoni et al., "DINOv3", arXiv:2508.10104v1 (2025): Table 2, Appendix A.1-A.2, Appendix C. https://arxiv.org/abs/2508.10104
37. [S37] facebookresearch/dinov3 at 6876159a: `dinov3/configs/ssl_default_config.yaml` L6-8; `dinov3/hub/backbones.py` L277-302; `dinov3/models/vision_transformer.py` L27-37, L73; `README.md` L377, L518. https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751
38. [S38] timm v1.0.30, `timm/models/eva.py` L3079-3101 (`vit_base_patch16_dinov3`), L268-269 (fused attention). https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/eva.py#L3079-L3101
39. [S39] Darcet, Oquab, Mairal, Bojanowski, "Vision Transformers Need Registers", arXiv:2309.16588v2 (ICLR 2024). https://arxiv.org/abs/2309.16588
40. [S40] Sun, Chen, Kolter, Liu, "Massive Activations in Large Language Models", arXiv:2402.17762v2 (2024), §5 "Massive Activations in Vision Transformers". https://arxiv.org/abs/2402.17762
41. [S41] Micikevicius et al., "Mixed Precision Training", arXiv:1710.03740v3 (ICLR 2018). https://arxiv.org/abs/1710.03740
42. [S42] Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training", arXiv:1905.12322 (2019). https://arxiv.org/abs/1905.12322
43. [S43] onnxconverter-common c26a5216, `onnxconverter_common/float16.py` L140-170 (`DEFAULT_OP_BLOCK_LIST`), L196-215 (`convert_float_to_float16`). https://github.com/microsoft/onnxconverter-common/blob/c26a5216ea0a24cd82156e8faf4a080e5400eecd/onnxconverter_common/float16.py#L140-L215
44. [S44] onnxconverter-common c26a5216, `onnxconverter_common/auto_mixed_precision.py` L36-39. https://github.com/microsoft/onnxconverter-common/blob/c26a5216ea0a24cd82156e8faf4a080e5400eecd/onnxconverter_common/auto_mixed_precision.py#L36-L39
45. [S45] timm v1.0.30, `timm/layers/pos_embed_sincos.py` L281-297 (`apply_rot_embed_cat`), L1122-1257 (`RotaryEmbeddingDinoV3`). https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/layers/pos_embed_sincos.py#L1122-L1257
