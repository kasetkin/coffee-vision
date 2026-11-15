# Quantization in the inference path: what is usable, for which goal, at what risk

Research note, written 2026-11-15 on the local checkout at `88f9eab`. A reading task: nothing was run, converted or
measured for it. It builds on ML-7 and OPS-23 and contradicts neither.

How each claim is marked:

- **[src]**: read here in source code at a named tag or commit (downloaded raw, or the installed package).
- **[doc]**: read in an official doc page, a paper's own page or a first-party model card. Web pages were read
  through a fetch tool that extracts the answer from the page, so quoted wording is as that tool returned it.
- **[repo]**: a project record; the path is given.
- **[sec]**: reported by a secondary source; not followed to its owner.
- **Unverified**: could not be checked here. All of these are collected in §5.3.

Versions: LiteRT `v2.3.0`; XNNPACK `d89ef666` (LiteRT 2.3.0's pin, per `analysis/pi5/research_dinov3_16bit.md` S1);
`litert-torch` 0.9.4, `ai-edge-quantizer` 0.9.0 and `torchao` 0.18.0 as installed in `convert/.venv`; ONNX Runtime
`v1.31.0`; NNCF `develop` (3.5.0); OpenVINO docs `releases/2025/4`.

## 1. Question and answer

**Question.** How could the project adopt quantization (int8, mixed precision, weight-only, fp16 beyond what it has)
in its inference path, and for which goal on which host?

**Answer.** On the live host quantization has one realistic use, and it is release size, not latency. `yappy`
serves from its GPU, LiteRT's GPU path computes in float [S9], and the project's own int8 encoder on the CPU (2.75 s)
was still slower than the fp32 encoder on the GPU (1.78 s) [R4]. Weight-only int8 or fp16-stored weights would cut
the 816 MB of model files to roughly a quarter or a half without changing the compute precision, and the tool is
already in `convert/.venv` [S5]; both alter the weights, so both are departures under ADR 0022 and need D7. Integer
compute is a CPU and NPU matter: it could help a CPU-only host such as the Pi 5, and it is mandatory for a Hailo NPU.
There the risk is concentrated in two known places. For the segmenter it is LiteMLA's divisor, which the project hit
(IoU 0.0005) and which a paper on EfficientViT reports independently (NaN under base PTQ; fixed only with a custom
quantizer) [S20]; Qualcomm ships EfficientViT only in float and w8a16, not w8a8 [S24]. For DINOv3 it is that the head
and the OOD probe were fitted on fp32 embeddings: published 8-bit results for DINOv2-B range from 19% to 83% ImageNet
accuracy against 83.3% in fp32 depending only on the PTQ method [S19], and no primary source reports int8 DINOv3
features at all. What the project has not tried is almost everything except one configuration: static QDQ on the
encoder with ONNX Runtime's defaults (§2.2). Mixed precision that keeps LiteMLA in float, dynamic quantization of
DINOv3's linear layers, and any quantization on the LiteRT side are all unmeasured.

## 2. What the project already knows

### 2.1 Decisions and bars

- **Serving precision per host** [repo: `docs/adr/0019-serving-runs-litert-accelerator-per-host.md`]: on `yappy`'s GPU
  the segmenter runs fp32 and DINOv3 fp16; 9.2 s a photo, `MemAvailable` drop 3.6 GB. A new host decides each graph's
  precision again under D7.
- **D7's bars** [repo: `docs/adr/0020-converted-models-pass-d7-before-they-serve.md`]: an fp32 graph needs least
  readout cosine ≥ 0.99999 and mask IoU ≥ 0.999. DINOv3 in fp16 needs no NaN or Inf, least patch cosine ≥ 0.999 on
  served photos, no guard verdict change, top-1 changes ≤ 1%, p95 of the largest probability change ≤ 0.05. **There is
  no int8 bar.** The fp16 bar was accepted by the owner for fp16 on `yappy`'s GPU only.
- **Exact forms** [repo: `docs/adr/0022-export-forms-are-exact.md`]: "No clamp, approximation or retraining is used to
  make a runtime fit. A form that is not exact has to be justified as a departure, like fp16 itself."
- **Parity rule** [repo: `CODING_STANDARDS.md` line 7]: a graph served in lower precision than it trains is a recorded
  departure; "an accurate method offline with a cheap one live is rejected".
- **ML-7 D4** [repo: `docs/ml-0007_ticket_inference_runtime.html`, D4]: int8 serving "would feed the head and the probe
  embeddings they were not fitted on ... unless int8 passes D7's gate unchanged. Fitting the head on int8 embeddings
  would move training onto the runtime, against D3." Decided: DINOv3 is not quantised; the int8 encoder does not serve
  because it fails D7.
- **OPS-23** [repo: `docs/ops-0023_ticket_pi5_inference_host.html`, Out of scope]: int8 models and NPUs on the Pi are
  out of that ticket's scope: "both are model changes that would need their own D7 work"; "the Pi's Hailo and Coral
  accelerators need their own compilers and int8 or int4 models".
- **Nothing in `.out-of-scope/` closes quantization** [repo: the five files there cover containers, CPU priority,
  cross-camera evaluation, ResNet18 levers and analytics]. `containerised-serving.md` would bear on a Pi deployment,
  not on quantization itself. `GLOSSARY.md` has no quantization term.

### 2.2 Exactly what was tried for int8

One artefact, `encoder.int8.onnx`, made in ML-7 P1 [repo: `analysis/ml7_gpu/export_onnx.py` L291-305;
`analysis/ml7_gpu/manifest_onnx.json`, key `encoder.int8.onnx`; plan §5 P1]:

| Property | Value | Source |
|---|---|---|
| Model | XL0 image encoder only, 1024 px | `export_onnx.py` L362-364 |
| Tool | `onnxruntime.quantization.quantize_static`, ONNX Runtime 1.31.0 | `export_onnx.py` L292; `convert_onnx/uv.lock` |
| Kind | static, post-training, QDQ format | `export_onnx.py` L302 |
| Weights | int8, per-channel | L302-303 |
| Activations | **uint8**, per-tensor | L303 |
| Calibration method | MinMax (the default; none was passed) | [S11] L522 |
| Calibration set | 32 photos' encoder inputs: 2 per segcrop session, none a fixture, seed 7 | `make_refs.py` L44-46, L207-219; `calibration_photos.txt` |
| Pre-processing | `quant_pre_process(skip_symbolic_shape=True)` | `export_onnx.py` L300 |
| Not set | `nodes_to_exclude`, `op_types_to_quantize`, `reduce_range`, `extra_options` | L302-303 |
| Resulting graph | 265 QuantizeLinear, 458 DequantizeLinear; 7 Div, 30 Gelu, 16 MatMul, 79 Conv | manifest |
| Size | 114 MB against 452 MB fp32 | manifest |

Results [repo: `analysis/ml7_gpu/README.md` §2 Variants; plan Activity 2026-10-13]:

- ONNX Runtime CPU: 2.75 s (fp32 on the same provider: 4.88 s). Mask IoU 0.0005 to 0.85, low-res logits off by up to
  14.5, four answers changed.
- OpenVINO provider, CPU: 2.84 s (fp32: 6.40 s). IoU 0.21.
- The plan's diagnosis: "The size says defect rather than quantisation noise; the likely cause is LiteMLA's denominator,
  near 0 plus 1e-15, quantised to uint8 over a range up to about 5,000 ... one that keeps LiteMLA's division in fp32
  was not tried."

Two details from the tool's source sharpen that diagnosis:

- `Div` is in none of ONNX Runtime's quantization registries [S12 L20-112], and the int8 file still holds its 7 `Div`
  nodes. So the division itself ran in float. Its operands did not: they are outputs of quantized ops (`MatMul`,
  `Slice`, `Pad`), so they passed through a Q/DQ pair. **[src]**
- Inference from the plan's numbers, not measured: a uint8 tensor spanning about 5,000 has a step near 20, so any
  divisor below about 10 becomes 0 and the quotient is taken against `eps = 1e-15`
  (`third_party/efficientvit/efficientvit/models/nn/ops.py` L533, L615).

### 2.3 What was not tried

- **Any quantization of DINOv3.** No int8 DINOv3 graph was ever produced; the cited risk is from the literature
  (ticket §2.3: a DINOv2-based retrieval model fell R@1 89.9 → 62.6; that source was not re-read here).
- **Dynamic quantization of anything.** D4's option (b) named "dynamic int8"; P1 built static QDQ.
- **Mixed precision for the encoder**: LiteMLA (or just its divisor path) excluded from quantization.
- **Other calibration**: Entropy or Percentile, more than 32 photos, symmetric or signed activations, 16-bit
  activations, SmoothQuant, bias correction.
- **The decoder** in any integer form.
- **Any quantization on the LiteRT side**: no int8 `.tflite`, nothing on XNNPACK's integer kernels, nothing on the GPU.
- **Weight-only int8 and fp16-stored weights.** `analysis/pi5/research_dinov3_16bit.md` §"What to try" notes that
  ai-edge-quantizer's fp16 recipe "was not tried (Unverified)".
- **NNCF**, with or without accuracy control.
- **Refitting the head and probe on quantized embeddings** (ruled out by D3 and D4, not measured).

### 2.4 Numbers this note leans on

| Fact | Value | Source |
|---|---|---|
| `yappy`, GPU, fp32 encoder / fp16 DINOv3 | 1.78 s / 5.74 s per photo's graphs | `analysis/ml7_gpu/README.md` §2 |
| `yappy`, CPU fp32, best cell | encoder 4.88 s, DINOv3 24.1 s (ORT) | same |
| `yappy`, CPU, int8 encoder | 2.75 s (ORT), 2.84 s (OpenVINO) | same |
| Pi 5 CPU, fp32 | 24.4 s (ORT), 28.9 s (LiteRT) a photo | OPS-23 ticket §2 |
| Pi 5 CPU, fp16 DINOv3 + fp32 segmenter | about 14 s: DINOv3 9.4-9.8 s, encoder 4.17 s, decoder 0.12 s | OPS-23 ticket §2 |
| Release model files | encoder 452 MB, DINOv3 343 MB, decoder 20.6 MB | `models/dino3b16_segxl0v4_country_aug8_s42.litert/` |
| DINOv3 graph ops | 48 FULLY_CONNECTED, 24 BATCH_MATMUL, 12 GELU, 1 CONV_2D | `analysis/ml7_gpu/manifest_tflite.json` |
| Encoder graph ops | 65 CONV_2D, 14 DEPTHWISE_CONV_2D, 12 BATCH_MATMUL, 7 DIV, 30 GELU | same |
| N100 | AVX2, AVX-VNNI (256-bit); no AVX-512, no AMX | ML-7 ticket §2.4 |

## 3. Findings

### 3.1 LiteRT and litert-torch

**Tooling.**

- **Two routes exist in litert-torch 0.9.4.** (a) PT2E: `litert_torch.quantize.pt2e_quantizer.PT2EQuantizer`, built on
  `torchao.quantization.pt2e` and passed as `QuantConfig(pt2e_quantizer=...)`. (b) Quantize the converted `.tflite`
  with ai-edge-quantizer. **[src]** [S4; S5]
- **The PT2E quantizer annotates few op kinds**: linear, addmm, conv (with relu, bn), add, mul, cat, max_pool2d,
  adaptive_avg_pool2d, GRU io, fixed-qparams; dynamic mode covers linear, addmm and conv only. It has no annotator for
  matmul, softmax, GELU, LayerNorm or division. **[src]** [S4 `pt2e_quantizer.py` L49-71, L263-292;
  `pt2e_quantizer_utils.py` L207-958]. So for a ViT it yields a mixed graph by construction: linears integer, the rest
  float.
- **ai-edge-quantizer 0.9.0 has four recipe families**: dynamic (`dynamic_wi8_afp32`, wi4), weight-only
  (`weight_only_wi8_afp32`, wi4), static (`static_wi8_ai8`, `static_wi8_ai16`), and LiteRT-LM recipes. **[src]**
  [S5 `recipe.py` L16-48, L58-226]
  - Weight-only: "The weights will be explicitly dequantized before being fed into the op to enable float compute thus
    retain model quality." [S5 `recipe.py` L125-149]. Its ops: BATCH_MATMUL, CONV_2D, CONV_2D_TRANSPOSE,
    DEPTHWISE_CONV_2D, EMBEDDING_LOOKUP, FULLY_CONNECTED. [S5 `default_policy.py` L289-296]
  - Dynamic wi8: the same six ops. "The model quality may suffer due to the on-the-fly quantization. If quality is a
    concern, consider using weight-only quantization." [S5 `recipe.py` L66-71; `default_policy.py` L278-285]
  - Static wi8/ai8 and wi8/ai16 cover the ops a ViT needs, **including DIV**, GELU, SOFTMAX, RSQRT, SQRT, MEAN,
    BATCH_MATMUL. [S5 `default_policy.py` `ops_per_config`; `algorithm_manager.py` L91-140]. A blanket static recipe
    would therefore quantize LiteMLA's division too, which ONNX Runtime's tool did not.
  - fp16 weights: a `FLOAT_CASTING` algorithm, "Currently only used for fp16 quantization", for FULLY_CONNECTED,
    CONV_2D, DEPTHWISE_CONV_2D, CONV_2D_TRANSPOSE, EMBEDDING_LOOKUP. [S5 `algorithm_manager.py` L164-185;
    `qtyping.py` L306-320]
  - Selective quantization: a recipe entry is (regex over tensor names, operation, algorithm, config);
    "All the unspecified ops will be kept as FP32". `Quantizer.validate()` compares float and quantized models per
    tensor (MSE by default; cosine available). **[doc]** [S6]
  - Other algorithms registered: OCTAV, MSE, GPTQ, Hadamard rotation, dequantized-weight recovery. **[src]**
    [S5 `algorithm_manager.py` L64-73]
- **The package has been renamed.** "AI Edge Quantizer is now LiteRT Quantizer ... This repository is no longer
  maintained"; the new package is `litert-quantizer`, "Apart from the names, the API is unchanged". **[doc]** [S6]
- **Converter flags are deprecated**: "Post-training quantization through TFLite converter flags
  (`_ai_edge_converter_flags`) is deprecated." **[doc]** [S7]
- The quantizer's own guidance: dynamic for CPU and GPU, static for NPU; dynamic is "Not supported in all hardware
  (e.g., some GPU and NPU)". **[doc]** [S6]

**XNNPACK, LiteRT v2.3.0.**

- **Dynamic-range ops are delegated**, despite the README. The README still says "Post-training dynamic range
  quantization is not supported in XNNPACK" [S2 L618-627]; the delegate source handles a float input with an int8
  filter as "dynamically quantized" in FULLY_CONNECTED (also int4 and int2 filters, and fp16 inputs), CONV_2D,
  TRANSPOSE_CONV and BATCH_MATMUL with a constant int8 right-hand side. **[src]** [S1 L3603-3605, L4056-4059,
  L5018-5028, L6907]. A flag exists to turn it off (`..._DISABLE_DYNAMICALLY_QUANTIZED_OPS`). [S3 L59-61]
  DEPTHWISE_CONV_2D has no such branch (no match in the file), so the encoder's 14 depthwise convolutions would not
  run dynamically quantized on XNNPACK.
- **Statically quantized ops XNNPACK takes**: the README lists 24, among them ADD, CONV_2D, DEPTHWISE_CONV_2D,
  FULLY_CONNECTED, MEAN, MUL, SUB, LOGISTIC, TANH, RESHAPE, TRANSPOSE. **[doc]** [S2]
- **Float only in the delegate**: DIV, GELU, RSQRT, SQRT and SOFTMAX are checked with `CheckTensorFloatType`.
  **[src]** [S1 L4438-4447, L4614-4633, L6050-6055]. A fully integer ViT graph would leave those ops to LiteRT's
  built-in kernels, splitting the graph. The cost of that split was not found in any source.
- **Kernels for both hosts exist** in the pinned XNNPACK: AVX-VNNI kernels for static int8 GEMM and for dynamically
  quantized GEMM, NEON dot-product kernels for both on arm64, and a dynamically-quantized-to-fp16 GEMM (`qd8_f16_qc8w`)
  that needs fp16 arithmetic, which the A76 has. **[src]** [S8 L5914-5916, L4185, L4541-4560, L3422-3759]. Which
  kernel is selected at run time on the N100 and the Pi was not traced. **Unverified.**
- Google's post on dynamic quantization in XNNPACK: Fully Connected and Convolution 2D only; "quadrupled inference
  performance" against fp32 on a Pixel 8 for CNNs and "up to 6.2 times faster" for Stable Diffusion's diffusion model;
  it can be combined with fp16 so that Batch Matrix Multiply and Softmax run in fp16. No ViT and no x86 figure.
  **[doc]** [S10]

**The GPU.**

- **The GPU delegate computes in float.** "Since the GPU backend only supports floating-point execution, we run
  quantized models by giving it a 'floating-point view'": constants are dequantized once at start, 8-bit inputs and
  outputs are converted on the CPU per call, and simulators are inserted between ops to mimic quantized behaviour.
  **[doc]** [S9]. That page describes the TFLite GPU delegate; whether the wheel's WebGPU accelerator that `yappy`
  uses treats an int8 graph the same way is **Unverified** (the accelerator is a prebuilt library).
- **One GPU option does quantize**: `GpuOptions.allow_src_quantized_fc_conv_ops`. "This enables dynamic range
  quantization of the input tensor for large sized fully connected and convolution operations, if the device supports
  it. This will result in accuracy loss, since the input tensor will be quantized to 8-bit. Turning this on will also
  increase the initialization time ... `enable_constant_tensors_sharing` must be true to use this." **[src]**
  [S13 L98-105; S5b `gpu_options.py` L29]. Whether the WebGPU backend on Mesa ANV "supports it", and what it does to
  speed, is **Unverified**.
- So an int8 graph on `yappy`'s GPU would at best be computed as float with simulated rounding: the accuracy cost of
  int8 without its speed. Weight-only and fp16-stored weights are the forms that fit a float GPU.

### 3.2 ONNX Runtime

- **Dynamic for transformers, static for CNNs**: "it is recommended to use dynamic quantization for RNNs and
  transformer-based models", static for CNNs. **[doc]** [S14]. The project's one attempt was static.
- **Transformer path**: "you need to optimize your models using the Transformer Model Optimization Tool before
  quantizing the model"; there is a `QAttention` op for fused attention. **[doc]** [S14]. In ML-7, ORT's Attention
  fusion did not apply to DINOv3 (0 of 12: "the RoPE attention matches none of ORT's patterns"), while GELU (12) and
  LayerNorm (25) fused. **[repo]** plan Activity 2026-10-13. So `QAttention` is not available to this model as
  exported.
- **Ops the tool quantizes**: QDQ registry holds Conv, Gemm, MatMul, Softmax, LayerNormalization, Add, Mul, Resize,
  Pad, Split, Gather, Where and shape ops; dynamic mode (`IntegerOps`) holds Conv, MatMul, Attention, LSTM, Gather,
  Transpose, EmbedLayerNormalization. No Div, Sub, Sqrt, Pow, Erf or Gelu. **[src]** [S12]
- **Excluding nodes and overriding tensors**: `nodes_to_exclude` (names; `get_qdq_config` also takes a predicate),
  `op_types_to_quantize`, and `extra_options` including `TensorQuantOverrides`, `ActivationSymmetric`,
  `CalibMovingAverage`, `SmoothQuant`, `SmoothQuantAlpha`, `MatMulConstBOnly`,
  `OpTypesToExcludeOutputQuantization`. **[src]** [S11 L75-94, L136-216, L247-263, L509-527]. This is the mixed
  precision the plan says was not tried.
- **Calibration**: "three calibration methods: MinMax, Entropy and Percentile". **[doc]** [S14]
- **Debugging**: the docs describe APIs that match weights and activations between the float and quantized models to
  find where they differ most. **[doc]** [S14]
- **x86**: U8S8 on AVX2 and AVX-512 uses VPMADDUBSW, which "might suffer from saturation issues"; `reduce_range`
  (7-bit weights) mitigates it and "is not needed on machines supporting VNNI". "On AVX2 and AVX512 machines, you will
  generally need to enable reduce-range as well if per-channel is enabled." S8S8 with QDQ "should be the first
  choice". **[doc]** [S14]. The N100 has AVX-VNNI, and MLAS installs AVX-VNNI U8S8 GEMM and convolution kernels when
  the CPU reports it. **[src]** [S15 L598-613]. Saturation is therefore not the likely cause of the encoder's failure
  (inference).
- **ARM**: "There is no such issue on other CPU architectures (x64 with VNNI and Arm)"; "processors with dot-product
  instructions can get better performance in general". MLAS installs UDOT/SDOT GEMM and convolution kernels when the
  CPU has dot product. **[doc]** [S14] **[src]** [S15 L786-807]. That the Pi 5's A76 reports dot product was not read
  from the device. **Unverified** (it is an Armv8.2 core; `lscpu` on the Pi would settle it).
- **GPU**: int8 on GPU is a TensorRT matter ("a device that supports Tensor Core int8 computation, like T4 or A100").
  **[doc]** [S14]. Nothing for ORT's WebGPU provider.

### 3.3 OpenVINO and NNCF

- `nncf.quantize(model, calibration_dataset, preset, target_device, subset_size, fast_bias_correction, model_type,
  ignored_scope, advanced_parameters)`. `model_type`: "Model type is needed to specify additional patterns in the
  model. Supported only `transformer` now." With it the default preset is `mixed` (symmetric weights, asymmetric
  activations). `ignored_scope`: "the list of model control flow graph nodes to be ignored during quantization."
  Backends: OpenVINO, ONNX, PyTorch, TorchFX. **[src]** [S16 L137-176, L189-248]
- `nncf.quantize_with_accuracy_control(model, calibration_dataset, validation_dataset, validation_fn, max_drop=0.01,
  drop_type=ABSOLUTE, ...)`: the validation function returns a metric where higher is better. Backends: OpenVINO and
  ONNX only. **[src]** [S16 L284-345, L357-376]. A D7 measure (least mask IoU, least cosine) could be that metric; the
  tool would then choose which layers go back to float. How long it takes on a 1024 px encoder is not stated.
- Advanced parameters include SmoothQuant alphas, an overflow fix and bias-correction controls. **[src]** [S17
  L176-285]
- Transformer-style models are calibrated at batch 1 (batch above 1 is not supported for them). **[doc]** [S18]
- **The N100's iGPU and int8**: OpenVINO's GPU plugin lists u8/i8 "for quantized operations only", and "Hardware
  acceleration for `i8`/`u8` precision may be unavailable on older generation platforms", in which case the model
  runs in the IR's float precision; Iris Xe and Xe MAX are named as accelerating it. **[doc]** [S26]. Whether the
  N100's UHD Graphics does is **Unverified**; the page says `ov::device::capabilities` reports it.
- OpenVINO is not the serving runtime (ADR 0019). NNCF could still be used as a quantizer producing ONNX, but the
  served format is `.tflite`, and no ONNX-QDQ-to-LiteRT route was looked for. **Unverified.**

### 3.4 ViT and DINO evidence

- **What breaks in ViT PTQ** (papers' own abstracts, **[doc]**):
  - FQ-ViT: "serious inter-channel variation in LayerNorm inputs" and an "extreme non-uniform distribution" of
    attention maps. [S27]
  - PTQ4ViT: activations after softmax and GELU are far from Gaussian; with twin uniform quantization and a
    Hessian-guided metric, "less than 0.5% drop at 8-bit quantization" on ImageNet. [S28]
  - RepQ-ViT: "post-LayerNorm activations with severe inter-channel variation" and "post-Softmax activations with
    power-law features". [S29]
  - I-ViT: integer-only ViT with shift-based Softmax and GELU; int8 accuracy "comparable (or even slightly higher)";
    3.72-4.11× speedup, measured with TVM on a GPU's integer units, not on a CPU. [S30]
  - Per-model W8A8 numbers for ViT-B or DeiT-B from these papers' tables were not read. **Unverified.**
- **DINOv2-B/14 at 8 bits depends entirely on the method.** "Activation Quantization of Vision Encoders Needs
  Prefixing Registers" (Kim et al., arXiv 2510.04547v2), ImageNet-1k accuracy, fp32 83.26: naive 19.20, PTQ4ViT 82.97,
  RepQ-ViT 82.27, NoisyQuant 71.46 (W8A8). The paper attributes the damage to outlier tokens arising in the MLP
  output (FC2) of one or two middle blocks. **[doc]** [S19]. Three cautions: the paper calls the metric "zero-shot
  accuracy" and does not say how an image-only model is scored; the naive baseline is not defined beyond
  "round-to-nearest"; and the measure is classification accuracy, not feature cosine, which is what D7 holds.
- **DINOv3 is not in that paper.** Its only mention: "In this regard, DINOv3 is a pleasant exception" (it ships with
  registers). **[doc]** [S19]. No primary source was found that reports int8 DINOv3 features by linear probe, kNN or
  cosine.
- **facebookresearch/dinov3**: issue 287, "Anyone tested quantization with success? my results are bad", open since
  2025-11-26, gives no numbers or method and has no maintainer reply. Issue 288: a user reports that in fp16 "the
  attention dot product (before the softmax) yields values higher than the maximum FP16 value possible", fixed by
  normalising q and k and fine-tuning. **[doc]** [S21; S22]. That agrees with the project's 2.1e6 attention logits
  (ML-7 ticket §2.3). Both are user reports, not statements by the authors.
- **What the 2.1e6 means for int8** (inference, not from a source): a per-tensor 8-bit activation whose range reaches
  2.1e6 has a step near 8,000. ADR 0022's fp16 forms (scores shifted by their row's max, LayerNorm on x / max|x|)
  already shrink those ranges and are exact, so a quantized DINOv3 should start from the fp16-form graph, not the
  plain one. Unmeasured.
- **Hailo's model zoo lists plain ViT-B/16 at 224 px**: float top-1 84.5, on hardware 83.2; 29.1 FPS at batch 1 on
  Hailo-8, 21.8 FPS on Hailo-8L (Dataflow Compiler v2.20.0 per the page). DeiT-B: 80.9 → 80.0. Swin-S: 83.1 → 79.9.
  **[doc]** [S23; S23b]. This is a supervised classifier, not DINOv3 (no RoPE, LayerScale or register tokens), and
  the measure is top-1, not feature parity.
- **Browser builds**: `onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX` holds `model.onnx` (343 MB data),
  `model_quantized.onnx` (86 MB) and `model_q4.onnx` (55.9 MB); no fp16 file. **[doc]** [S31]. No accuracy figure is
  given for the quantized files, and how they were produced was not found. **Unverified.**

### 3.5 EfficientViT and SAM

- **mit-han-lab/efficientvit's README** mentions no quantization, int8, TFLite or QNN; TensorRT appears only in speed
  claims. **[doc]** [S32]
- **Its TFLite script** (`assets/tflite_export.py`) converts classification and segmentation models with tinynn's
  `TFLiteConverter`, swaps bicubic upsampling for bilinear ("bicubic upsampling is not supported in TFLite"), and has
  no quantization option. **[doc]** [S33]. The bilinear swap is the one ML-7 refused (ticket §2.3).
- **Issues**: the repository has no issue about int8 accuracy or LiteMLA's division. Issue 164 (open, no reply)
  reports that the TFLite export of EfficientViT-B1 has no quantized layers. **[doc]** [S34]. Search was by keyword
  through GitHub's API and was cut short by its rate limit; a second pass was not made.
- **Trio-ViT** (arXiv 2405.03882), post-training quantization for EfficientViT-B1/B2 at 224-288 px. **[doc]** [S20]
  - Base PTQ gives **NaN for all four models** (Table IV); the paper attributes the NaN to the divisors in the
    lightweight MSA (§IV-A2): "smaller values within divisors exhibit much greater sensitivity compared to larger
    values."
  - With the MSA's matmuls kept in full precision, W8A8 on B1 still falls to 3.23% top-1, a 76.15-point drop (Table I),
    attributed to inter-channel variation in the depthwise convolutions' inputs and asymmetry in the following
    pointwise convolutions' inputs. So the divisor is not the only problem.
  - Fixes: channel-wise migration, filter-wise shifting, and log2 quantization of the divisors (4 bits). Result:
    78.64 against 79.39 (B1-R224), 80.97 against 82.10 (B2-R224); average drop 0.92.
  - The quantizer is custom (BRECQ-style reconstruction) and the target is an FPGA accelerator. A log2-quantized
    divisor is not an op that LiteRT, ONNX Runtime or OpenVINO run. Code: github.com/shihuihong214/Trio-ViT.
  - It covers EfficientViT-B (classification, and B0 at 1024 px for segmentation), not EfficientViT-SAM-XL0.
- **Qualcomm AI Hub** offers EfficientViT-l2-cls in float and **w8a16** only (ONNX and QNN; QAIRT 2.50, ONNX Runtime
  1.30.0); no w8a8, no accuracy figure. **[doc]** [S24]. A vendor choosing 16-bit activations for this family is
  consistent with the above. No Qualcomm DINOv2 or DINOv3 quantized card was found. **Unverified** (the hub's search
  page could not be read).
- **PTQ4SAM** (arXiv 2405.03144): the bottleneck is "the bimodal distribution in post-Key-Linear activations", plus
  "substantial variations in the post-Softmax distributions" from SAM's two-way cross-attention; with its fixes,
  "lossless accuracy for instance segmentation" at 6 bits on SAM-L. **[doc]** [S35]. This bears on the mask decoder,
  the part this project fine-tunes. The decoder is 0.1 s and 20.6 MB (§2.4), so it has nothing to gain.

### 3.6 What it would be for

**Latency.**

- **`yappy`, GPU**: nothing to gain from int8 (§3.1). The only GPU lever found is `allow_src_quantized_fc_conv_ops`,
  unverified on this backend and lossy by its own description.
- **`yappy`, CPU**: the one measured int8 graph ran 1.8× (ORT) and 2.3× (OpenVINO) faster than fp32 on the same
  provider [repo §2.4], and was still 1.5× slower than the GPU's fp32 encoder. For DINOv3 the GPU's fp16 (5.74 s) is
  4.2× ahead of the best CPU fp32 (24.1 s); int8 would have to exceed that. No primary source gives int8 against fp32
  for ViT-B on an AVX-VNNI CPU. **Not found.**
- **Pi 5, CPU**: today about 14 s. Candidates are the encoder (4.17 s fp32) and DINOv3 (9.4-9.8 s fp16). No primary
  source gives ViT-B int8 timings on a Cortex-A76. **Not found.** A 2026 preprint's abstract reports int8 speedups
  "up to 2.1x" on cores with a dot-product ISA and a 1.7× slowdown without one, same model and runtime; it names no
  ViT. **[doc, single-author preprint, abstract only]** [S36]
- ML-7's ticket cites "Dynamic int8 gave 2.6× on a 4-vCPU Ice Lake in 2022, an older stack" (§2.3); not re-read here.

**Memory and size.**

- ViT-B/16 at 86 M parameters is 343 MB in fp32 and 171 MB in fp16 (`manifest_onnx.json`), and 86 MB as the
  onnx-community 8-bit file [S31]. The XL0 encoder went 452 → 114 MB under int8 [repo §2.2].
- The release ships fp32 files for all three graphs (816 MB), including DINOv3, which the GPU then computes in fp16.
  Weight-only int8 over all three would be about a quarter of that; fp16-stored weights for DINOv3 alone save 171 MB.
- Whether smaller files lower the 3.6 GB `MemAvailable` drop is unknown: with dequantize-at-load, the float weights
  still exist in GPU memory [S9]. **Unverified.**

**New host classes.**

- **Hailo on the Pi 5**: AI HAT+ 13 TOPS is a Hailo-8L, 26 TOPS a Hailo-8, both int8; AI HAT+ 2 is a Hailo-10H
  (40 TOPS, int4). Pi 5 only. **[doc]** [S25]. Hailo compiles and runs ViT-B/16 (§3.4): at 21.8-29.1 FPS, 40 patches
  would take about 1.4-1.8 s (arithmetic on the zoo's figures, batch 1). Whether Hailo's compiler accepts DINOv3's
  RoPE and LayerScale, or EfficientViT's LiteMLA, was not established: its supported-layers page needs a login.
  **Unverified.** It would also be a second runtime with its own compiler, against ADR 0019's single runtime.
- **Coral**: not researched beyond OPS-23's note that it needs its own compiler and int8.
- **Intel NPU**: LiteRT's NPU support includes Intel through OpenVINO (OPS-23 ticket); the N100 has no NPU (task
  brief; not re-checked).
- **Browser**: LiteRT.js runs `.tflite` on WebGPU, WebNN or XNNPACK-in-Wasm; its page says nothing on quantized
  models or size limits (ML-7's ticket records a 2 GB cap). transformers.js selects `fp32`, `fp16`, `q8`, `int8`,
  `uint8`, `q4`, `bnb4`, `q4f16` per model and notes that some encoders are "extremely sensitive to quantization
  settings". **[doc]** [S37; S38]. In a browser, download size is the goal, so the 86 MB and 56 MB DINOv3 files are
  the relevant existence proof; their accuracy is unknown.

### 3.7 Options that keep train/serve parity

- **Weight-only int8** (int8 weights, dequantized to float, float compute). The lowest-risk integer form: only the
  weights change, by at most half a quantization step per channel. It is still an approximation, so ADR 0022 makes it
  a departure to be justified, and D7's fp32 bar (cosine ≥ 0.99999, IoU ≥ 0.999) may or may not hold. Unmeasured.
  Size only; ai-edge-quantizer says so itself. [S5; S6]
- **fp16-stored weights.** For DINOv3 on `yappy`'s GPU the compute is already fp16, so storing fp16 weights may change
  nothing in the output; for the fp32 segmenter it rounds the weights and must pass the fp32 bar. Unmeasured.
- **Mixed precision** (LiteMLA, LayerNorm and attention logits in float; convolutions and linears in int8). Tooling
  exists on all three stacks (§3.1-3.3). Trio-ViT's Table I warns that excluding the attention alone was not enough
  for EfficientViT-B1. The int8 part still fails ADR 0022's "no approximation" unless accepted as a departure with
  its own bar.
- **Dynamic quantization of DINOv3's 48 linear layers.** Needs no calibration set. Activations are quantized per call
  from their own range, so the outlier problem is per input, not per calibration set. It changes the embeddings, so
  D4's objection applies in full.
- **Fitting the head and the OOD probe on quantized embeddings.** This restores parity by construction and is what
  the literature's accuracy figures implicitly assume (a classifier evaluated on the quantized model). The project
  rejected it: it moves training onto the runtime, against ML-7 D3 and D4. It would also tie the training pools to
  one host's integer kernels: the preprint in [S36] reports int8 predictions agreeing 1000/1000 only when two targets
  share an integer kernel, 958-965/1000 otherwise.
- **Quantization-aware training.** Not applicable as the project stands: the backbone is frozen (ADR 0006) and ADR
  0022 excludes retraining to make a runtime fit. For the segmenter only the decoder is fine-tuned, and the encoder is
  where the time is.
- **16-bit activations** (`static_wi8_ai16`, Qualcomm's w8a16). Finer steps for the divisor and the logits. XNNPACK's
  quantized kernels are 8-bit [S2], so on a CPU this would run on LiteRT's built-in kernels. Speed unknown.

## 4. Options by model and host

"—" means not applicable or nothing to gain. Standards touched: A19, A20, A22 are ADRs 0019, 0020, 0022; "parity" is
the `CODING_STANDARDS.md` rule; D3 and D4 are ML-7's.

| Option | Model | `yappy` GPU | `yappy` CPU | Pi 5 CPU | NPU (Hailo) | Browser | Risk | Toolchain | Touches |
|---|---|---|---|---|---|---|---|---|---|
| Weight-only int8 | encoder | size 452 → ~114 MB; speed unchanged | same | same | — | download size | low; unmeasured against IoU ≥ 0.999 | ai-edge-quantizer `weight_only_wi8_afp32` | A22 departure; A20 fp32 bar |
| Weight-only int8 | decoder | 20.6 → ~5 MB; not worth a departure | — | — | — | — | low | same | A22 |
| Weight-only int8 | DINOv3 | size 343 → ~86 MB | same | same | — | download size | low to medium: embeddings move; unmeasured | same | A22; A20 (fp16 bar on GPU, fp32 bar on CPU); parity |
| fp16-stored weights | DINOv3 | size 343 → 171 MB; output possibly unchanged (already fp16) | size only; weights rounded | size only | — | size | lowest | ai-edge-quantizer `FLOAT_CASTING` | A20; A22 if output changes |
| fp16-stored weights | encoder | size 452 → 226 MB | same | same | — | size | low; the fp16 *compute* failed IoU (0.9976), weights alone unmeasured | same | A22; A20 fp32 bar |
| Dynamic int8 (linears, convs) | DINOv3 | — (float GPU); `allow_src_quantized_fc_conv_ops` unverified | speed unknown; must beat 5.74 s on the GPU to matter | speed unknown against 9.4 s fp16; combinable with fp16 in XNNPACK | — | q8 file exists upstream | medium to high: no DINOv3 evidence; head and probe fitted on fp32 | ai-edge-quantizer `dynamic_wi8_afp32`; litert-torch PT2E dynamic; ORT `quantize_dynamic` | D4; parity; A20 needs an int8 bar; A22 |
| Dynamic int8 | encoder | — | convs only; depthwise not taken by XNNPACK | same | — | — | medium: Trio-ViT blames conv inputs as well as the divisor | same | A20 fp32 bar; A22 |
| Static int8, LiteMLA in float | encoder | — | 4.88 → at best ~2.75 s (the failed full-int8 time); slower than the GPU's 1.78 s | 4.17 s → unknown | — | — | high: Trio-ViT Table I (3.23% with MSA matmuls in float) | ORT `nodes_to_exclude`; NNCF `ignored_scope` or accuracy control; ai-edge-quantizer regex recipe | A20; A22 |
| Static int8, blanket | encoder | — | measured: IoU 0.0005-0.85, fails | expected the same | required; LiteMLA support unverified | — | proven failure | ORT `quantize_static` (done) | fails A20 |
| Static int8 or w8a16, calibrated | DINOv3 | — | speed unknown | speed unknown | required; ViT-B runs on Hailo-8/8L, DINOv3 unverified | — | high without a ViT-aware method (19% naive to 83% PTQ4ViT on DINOv2-B) | ai-edge-quantizer `static_wi8_ai8` / `ai16`; ORT QDQ; NNCF `model_type=TRANSFORMER`; Hailo DFC | D4; parity; A19 (second runtime for Hailo); A20; A22 |
| Any int8 | decoder | — | — | — | — | — | PTQ4SAM: bimodal key activations | — | no gain: 0.1 s, 20.6 MB |
| Refit head and probe on quantized embeddings | DINOv3 | — | restores parity for whichever int8 form | same | same | same | ties training to one host's kernels | training pipeline | against D3, D4; ADR 0006 unaffected |
| QAT | either | — | — | — | — | — | — | — | excluded: frozen backbone, A22 |

## 5. What is unknown

### 5.1 To measure here, in order

Each step is cheaper than the next and can stop the ones after it. All comparisons are against PyTorch fp32 under
ADR 0020, on ML-7 P7's sets (the 157 served photos, the negatives, 4 patch seeds) where answers are judged.

1. **Weight-only int8 and fp16-stored weights, all three graphs, offline.** ai-edge-quantizer 0.9.0 is already in
   `convert/.venv`. Record file sizes, least cosine, least mask IoU, answers changed. Run on the CPU first (the
   devcontainer), which answers whether the fp32 bar holds. This is the only option whose goal (release size) applies
   to the live host today.
2. **The same files on `yappy`'s GPU**: does the delegate still take every op (DEQUANTIZE nodes are new; XNNPACK's
   README restricts DEQUANTIZE to per-tensor 8-bit input [S2]), does start-up time change, does the `MemAvailable`
   drop change. D7 on the accelerator, as ADR 0020 requires.
3. **How far int8 moves DINOv3's embeddings, offline, before any speed work.** Dynamic int8 on the 48 linears only,
   starting from the fp16-form graph. Measure against the fp16 bar's quantities (least cosine, top-1 changes, guard
   verdicts, p95 probability change), beside fp16's own figures (0.48% top-1 changes, p95 0.023; ADR 0019) and patch
   sampling's (0.85%, 0.099). If it is far outside, every DINOv3 integer option ends here unless the owner reopens D3.
4. **Where the encoder's error comes from.** ONNX Runtime's QDQ debugging on the existing `encoder.int8.onnx` and the
   32 calibration inputs: per-tensor error, to confirm or refute the divisor diagnosis and to see whether the
   depthwise and pointwise inputs Trio-ViT names matter here too.
5. **A mixed-precision encoder**: LiteMLA's whole block excluded (`nodes_to_exclude`), then Percentile or Entropy
   calibration, then more calibration photos. Judged by mask IoU ≥ 0.999. If NNCF is acceptable as an analysis tool,
   `quantize_with_accuracy_control` with least mask IoU as the metric finds the set of float layers automatically.
6. **Speed, only for what passed 3 or 5, and only on a CPU host**: the Pi 5 with ML-7's replay; XNNPACK and ONNX
   Runtime; with the delegate's report of which ops fell back. On `yappy` the comparison is against the GPU, which
   int8 on the CPU is unlikely to beat (§3.6).
7. **`allow_src_quantized_fc_conv_ops` on `yappy`'s GPU**: one run to learn whether the backend supports it at all.
   Low priority: lossy by description, and both graphs already pass on the GPU.

### 5.2 To decide before measuring (the owner's)

- **An int8 bar.** ADR 0020 has an fp32 bar and an fp16 bar for one graph on one host. Steps 3 and 5 need to know
  what would count as passing; the fp16 bar's quantities are the natural candidate for DINOv3, and the fp32 mask bar
  for the encoder.
- **Whether refitting the head on runtime embeddings stays closed** (D3, D4). If it does, DINOv3 integer compute rests
  on step 3 alone.
- **Whether a second runtime is acceptable for an NPU** (ADR 0019). If not, Hailo is closed regardless of accuracy.
- **Which goal matters**: release size on `yappy` (steps 1-2 suffice), a CPU-only host (steps 3-6), or a new host
  class. *Decided by the owner, 2026-11-15: the goal is speed; size does not matter at all.* §5.4 follows from it.

### 5.3 Claims that could not be verified

1. Whether LiteRT 2.3.0's WebGPU accelerator handles an int8 graph by the "floating-point view" that the GPU delegate
   page describes [S9]. The page is about the TFLite GPU delegate; the accelerator is a prebuilt library.
2. Whether `allow_src_quantized_fc_conv_ops` is supported on the WebGPU backend over Mesa ANV, and its effect.
3. Whether XNNPACK or the GPU accelerator accepts the per-channel DEQUANTIZE nodes a weight-only recipe inserts, or
   folds them at load.
4. Which XNNPACK int8 kernel is selected at run time on the N100 and on the Pi 5; that the Pi 5's A76 reports the
   dot-product feature.
5. Any first-party int8-against-fp32 latency figure for a ViT-B-class model on an AVX-VNNI CPU or a Cortex-A76. None
   found.
6. Any primary report of DINOv3 features under int8 (cosine, linear probe or kNN). None found.
7. How the DINOv2-B accuracy in [S19] is scored ("zero-shot" for an image-only model), and what its naive baseline is.
8. Per-model W8A8 accuracy for ViT-B and DeiT-B in PTQ4ViT, FQ-ViT, RepQ-ViT and I-ViT: only abstracts were read.
9. How the onnx-community DINOv3 `model_quantized.onnx` and `model_q4.onnx` were produced, and their accuracy.
10. Whether Hailo's Dataflow Compiler supports DINOv3's RoPE and LayerScale or EfficientViT's LiteMLA (the
    supported-layers page needs a login); which ViT-B checkpoint the zoo's `vit_base` is; the zoo page's compiler
    version string ("v2.20.0") against Hailo's current numbering.
11. Whether Qualcomm AI Hub has a quantized DINOv2 or DINOv3 entry, or an EfficientViT-SAM entry.
12. Whether the N100's UHD Graphics accelerates i8/u8 in OpenVINO's GPU plugin.
13. Whether mit-han-lab/efficientvit has an issue on int8 accuracy that the keyword search missed (the search was
    rate-limited and not repeated).
14. ML-7's cited figures that were taken from the ticket and not re-read at their source: the DINOv2 retrieval model's
    R@1 89.9 → 62.6, and "dynamic int8 gave 2.6× on a 4-vCPU Ice Lake in 2022".
15. That the N100 has no NPU, and the name of LiteRT's GPU engine: both taken from the task brief.
16. Whether an ONNX QDQ model (from ORT or NNCF) can be carried into a `.tflite` that LiteRT runs. Not looked for.
17. Whether smaller weight files reduce `yappy`'s `MemAvailable` drop.

### 5.4 After the owner's decision: speed only

Owner, 2026-11-15: "weight does not matter at all, the main goal is speed boost". What that leaves of this note:

- **Dropped:** steps 1 and 2 of §5.1, and every weight-only and fp16-stored-weights row of §4. They change file size,
  not compute.
- **On `yappy` as it serves today, quantization has no established speed path.** Of a 12 MP photo's roughly 10 s,
  DINOv3 is 5.74 s and the encoder 1.78 s, both on the GPU (§2.4), and decoding 1.4-1.5 s (`docs/status.md`). The GPU
  computes in float (§3.1). On the CPU, int8 would need more than 4.2× over fp32 for DINOv3 and more than 2.7× for
  the encoder to match the GPU; the one measured int8 graph reached 1.8-2.3× (§3.6), and it failed D7.
- **The one thing to try on `yappy`** (tried, §5.5: it cannot be switched on)**:** step 7, `allow_src_quantized_fc_conv_ops` on DINOv3's graph. It is one run on
  the host. Whether the WebGPU backend supports it, and whether it is faster at all, is unverified (§3.1); it is lossy
  by its own description, so it would need the owner's bar (§5.2).
- **Where integer compute could buy speed is another host.** On a CPU-only host such as the Pi 5 (about 14 s today),
  steps 3 to 6 in that order, with step 3, the offline embedding drift, as the cheap gate. On a Hailo NPU, a plain
  ViT-B's published rate puts 40 patches at about 1.4-1.8 s against `yappy`'s 5.74 s (§3.6), the largest gain in this
  note and the least established: DINOv3 is unverified there, it needs full int8 or int4, and it is a second runtime
  against ADR 0019.

### 5.5 The GPU option test on `yappy`, 2026-11-15: the option cannot be switched on

Step 7 of §5.1, run on the live release (`20261115T0847Z-2807d2a`, LiteRT 2.3.0, the release's own venv and graphs)
with `coffee-cv-web` stopped from 15:26:16 to 15:27:26 UTC. `gpu_quantized_ops.py` runs one graph with one set of
`GpuOptions` per process; `gpu_quantized_ops.sh` stops the service, runs the cells and starts it again. Only the
rehearsal ran (one round, 2 photos); its records are in `results/`. The full run was not started: there was nothing
left to time.

| Graph | Options beside the release's | Result |
|---|---|---|
| DINOv3, fp16 | none | 5.87 s per photo's 40 patches; whole on the GPU; GPU proof ok |
| DINOv3, fp16 | `allow_src_quantized_fc_conv_ops` | 5.87 s; the output is bit-identical to the row above (the same sha256) |
| DINOv3, fp16 | `constant_tensor_sharing` | segmentation fault while the model is created |
| DINOv3, fp16 | both | segmentation fault, the same |
| Encoder, fp32 | none | 1.59 s per photo; whole on the GPU; GPU proof ok |
| Encoder, fp32 | both | segmentation fault, the same |

- **Alone, the option does nothing.** The same time and the same bytes out: it is ignored without constant tensor
  sharing, as its header comment says it would be [S13].
- **Constant tensor sharing crashes this build.** Every process with it died inside
  `libLiteRtWebGpuAccelerator.so`, at the same offset each time (kernel log: `segfault at 100000000 ... in
  libLiteRtWebGpuAccelerator.so`), during `CompiledModel.from_file`, after the WebGPU environment was created and
  before the graph was initialized. The 20 MB decoder alone crashes the same way, in fp16 and in fp32, so it is not
  the graph, the precision or memory.
- **The likely cause, not confirmed** (the accelerator is a prebuilt library): the same header has
  `LrtSetGpuAcceleratorCompilationOptionsSharedTensorMaps`, a pointer to an ML Drift struct that holds the shared
  tensors, marked "not ABI-stable and only for internal usage. It works only when the client is built together with
  the LiteRT and ML Drift delegate" [S13, L196-205]. The Python wrapper sets no such pointer
  (`compiled_model_wrapper.cc`, v2.3.0, L183-209).
- **So the one quantization lever on `yappy`'s GPU is closed**, on LiteRT 2.3.0, the newest release on PyPI on this
  date, and on the nightly too. `ai-edge-litert-nightly` 2.4.0.dev20261114, in a scratch venv on `yappy`
  (`~/ml7/quant/venv_nightly`, the owner's approval, 2026-11-15), with the release's 20 MB decoder beside the running
  service: the default options run; constant tensor sharing, with or without the quantize option, dies with the same
  segmentation fault in `CompiledModel.from_file`.
- **After the run:** the service started again on the same release, and `scripts/deploy_webapp.sh --verify` passed
  through the domain (Colombia on the smoke photo, every example its country, the negative refused).
- The base cells agree with ML-7's figures in §2.4 (5.74 s and 1.78 s then; 2 photos here, one round).

## 6. Sources

Repository records:

- [R1] `docs/adr/0019-serving-runs-litert-accelerator-per-host.md`
- [R2] `docs/adr/0020-converted-models-pass-d7-before-they-serve.md`
- [R3] `docs/adr/0022-export-forms-are-exact.md`
- [R4] `analysis/ml7_gpu/README.md` (§2 tables and Variants, §6 memory and size)
- [R5] `analysis/ml7_gpu/export_onnx.py` L281-305, L362-364; `analysis/ml7_gpu/make_refs.py` L20-21, L44-46, L207-219;
  `analysis/ml7_gpu/calibration_photos.txt`; `analysis/ml7_gpu/manifest_onnx.json`;
  `analysis/ml7_gpu/manifest_tflite.json`; `analysis/ml7_gpu/convert_onnx/uv.lock`
- [R6] `docs/ml-0007_ticket_inference_runtime.html` (§2.3, §2.4, D3, D4, D14);
  `docs/ml-0007_plan_inference_runtime.html` (§5 P1, Activity 2026-10-13 and 2026-10-14)
- [R7] `docs/ops-0023_ticket_pi5_inference_host.html`; `analysis/pi5/research_dinov3_16bit.md`
- [R8] `CODING_STANDARDS.md` line 7; `.out-of-scope/`; `GLOSSARY.md`
- [R9] `third_party/efficientvit/efficientvit/models/nn/ops.py` L533, L615
- [R10] `models/dino3b16_segxl0v4_country_aug8_s42.litert/` (file sizes, `manifest.json`)

External, read 2026-11-15:

1. [S1] LiteRT v2.3.0, `tflite/delegates/xnnpack/xnnpack_delegate.cc`: dynamically quantized BATCH_MATMUL L3603-3605,
   CONV_2D L4056-4059, FULLY_CONNECTED L5018-5028, TRANSPOSE_CONV L6907; float-only binary ops L4438-4447, unary ops
   L4614-4633, SOFTMAX L6050-6055.
   https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/tflite/delegates/xnnpack/xnnpack_delegate.cc
2. [S2] LiteRT v2.3.0, `tflite/delegates/xnnpack/README.md` (quantized operators; L618-627 on dynamic range).
   https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/tflite/delegates/xnnpack/README.md
3. [S3] LiteRT v2.3.0, `tflite/delegates/xnnpack/xnnpack_delegate.h` L28-61 (delegate flags).
   https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/tflite/delegates/xnnpack/xnnpack_delegate.h
4. [S4] litert-torch 0.9.4, `litert_torch/quantize/` (`pt2e_quantizer.py`, `pt2e_quantizer_utils.py`,
   `quant_config.py`), as installed in `convert/.venv/lib/python3.12/site-packages/`.
   https://github.com/google-ai-edge/litert-torch/tree/main/litert_torch/quantize
5. [S5] ai-edge-quantizer 0.9.0, `recipe.py`, `default_policy.py`, `algorithm_manager.py`, `qtyping.py`, `recipes/`,
   as installed in `convert/.venv/lib/python3.12/site-packages/ai_edge_quantizer/`. [S5b] ai-edge-litert 2.2.0,
   `gpu_options.py`, same environment. https://github.com/google-ai-edge/ai-edge-quantizer
6. [S6] ai-edge-quantizer `README.md`, `main` (rename notice, recipe table, selective quantization, validation).
   https://github.com/google-ai-edge/ai-edge-quantizer/blob/main/README.md ; successor:
   https://github.com/google-ai-edge/LiteRT/tree/main/litert_quantizer
7. [S7] litert-torch `docs/pytorch_converter/README.md`, `main`, §Quantization.
   https://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md
8. [S8] XNNPACK d89ef666, `src/configs/gemm-config.c`: `qd8_f16_qc8w` L3422-3759; `qdu8_f32_qc8w` AVX-VNNI L4185;
   `qd8_f32_qc8w` L4401-5003 (arm64 dot product L4541-4560); `qs8_qc8w` AVX-VNNI L5914-5916.
   https://github.com/google/XNNPACK/blob/d89ef6669a14db203b3b7935b1b3862cb63fb6df/src/configs/gemm-config.c
9. [S9] Google AI Edge, "GPU delegates for LiteRT" (quantized models section).
   https://developers.google.com/edge/litert/performance/gpu
10. [S10] TensorFlow blog, "Faster Dynamically Quantized Inference with XNNPack", 2024-04.
    https://blog.tensorflow.org/2024/04/faster-dynamically-quantized-inference-with-xnnpack.html
11. [S11] ONNX Runtime v1.31.0, `onnxruntime/python/tools/quantization/quantize.py`.
    https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/python/tools/quantization/quantize.py
12. [S12] ONNX Runtime v1.31.0, `onnxruntime/python/tools/quantization/registry.py`.
    https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/python/tools/quantization/registry.py
13. [S13] LiteRT v2.3.0, `litert/c/options/litert_gpu_options.h` L98-105;
    `litert/python/litert_wrapper/compiled_model_wrapper/gpu_options.py` L25-30.
    https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/litert/c/options/litert_gpu_options.h
14. [S14] ONNX Runtime docs, "Quantize ONNX models".
    https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
15. [S15] ONNX Runtime v1.31.0, `onnxruntime/core/mlas/lib/platform.cpp` L598-613, L662-686, L786-807.
    https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/mlas/lib/platform.cpp
16. [S16] NNCF `develop` (3.5.0), `src/nncf/quantization/quantize_model.py`; `src/nncf/parameters.py`.
    https://github.com/openvinotoolkit/nncf/blob/develop/src/nncf/quantization/quantize_model.py
17. [S17] NNCF `develop`, `src/nncf/quantization/advanced_parameters.py`.
    https://github.com/openvinotoolkit/nncf/blob/develop/src/nncf/quantization/advanced_parameters.py
18. [S18] NNCF `develop`, `docs/usage/post_training_compression/post_training_quantization/Usage.md`.
    https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/post_training_quantization/Usage.md
19. [S19] Kim et al., "Activation Quantization of Vision Encoders Needs Prefixing Registers", arXiv 2510.04547v2.
    https://arxiv.org/abs/2510.04547 ; https://arxiv.org/html/2510.04547v2
20. [S20] "Trio-ViT: Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer",
    arXiv 2405.03882 (§IV-A, Tables I, IV, V). https://arxiv.org/abs/2405.03882 ; https://arxiv.org/html/2405.03882
21. [S21] facebookresearch/dinov3 issue 287. https://github.com/facebookresearch/dinov3/issues/287
22. [S22] facebookresearch/dinov3 issue 288. https://github.com/facebookresearch/dinov3/issues/288
23. [S23] Hailo Model Zoo, Hailo-8 classification models.
    https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_classification.rst
    [S23b] Hailo-8L:
    https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_classification.rst
24. [S24] Qualcomm, "EfficientViT-l2-cls" model card. https://huggingface.co/qualcomm/EfficientViT-l2-cls
25. [S25] Raspberry Pi documentation, "AI HAT+". https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html
26. [S26] OpenVINO `releases/2025/4`, GPU device documentation.
    https://github.com/openvinotoolkit/openvino/blob/releases/2025/4/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.rst
27. [S27] Lin et al., "FQ-ViT", arXiv 2111.13824 (abstract). https://arxiv.org/abs/2111.13824
28. [S28] Yuan et al., "PTQ4ViT", arXiv 2111.12293 (abstract). https://arxiv.org/abs/2111.12293
29. [S29] Li et al., "RepQ-ViT", arXiv 2212.08254 (abstract). https://arxiv.org/abs/2212.08254
30. [S30] Li and Gu, "I-ViT", arXiv 2207.01405 (abstract). https://arxiv.org/abs/2207.01405
31. [S31] onnx-community, DINOv3 ViT-B/16 ONNX files.
    https://huggingface.co/onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX/tree/main/onnx
32. [S32] mit-han-lab/efficientvit, README. https://github.com/mit-han-lab/efficientvit
33. [S33] mit-han-lab/efficientvit, `assets/tflite_export.py`.
    https://github.com/mit-han-lab/efficientvit/blob/master/assets/tflite_export.py
34. [S34] mit-han-lab/efficientvit issue 164. https://github.com/mit-han-lab/efficientvit/issues/164
35. [S35] Lv et al., "PTQ4SAM", arXiv 2405.03144 (abstract). https://arxiv.org/abs/2405.03144
36. [S36] Shin, "Is INT8 Portable? A Cross-Platform Measurement Study of Quantized Inference on Embedded and
    Automotive Accelerators", arXiv 2609.16085 (abstract only). https://arxiv.org/abs/2609.16085
37. [S37] Google AI Edge, "LiteRT for Web". https://developers.google.com/edge/litert/web
38. [S38] Hugging Face, transformers.js, "Using quantized models (dtypes)".
    https://huggingface.co/docs/transformers.js/guides/dtypes
