# Serving runs both models on LiteRT, with the accelerator and each graph's precision chosen for the host

Both models train on PyTorch and are converted for serving only (ML-7 D3). The live site runs the converted `.tflite` files on LiteRT (`ai-edge-litert`). Pre- and post-processing are ported to numpy, PIL and OpenCV, and training calls the same ports, so the release holds no torch (D5 (b)).

The accelerator, and each graph's precision, are chosen for the host serving them, measured there against PyTorch, and named in the release, not in code. The serving host is not fixed: the site may move back to a CPU-only VM with AVX-512.

## The current host: `yappy`, Intel N100 with an iGPU

LiteRT's WebGPU accelerator, which Dawn runs on Vulkan (Mesa's ANV driver), with a precision per graph (D2, D4):

- **The segmenter**, XL0 encoder and decoder: fp32 (`GpuOptions(enforce_f32=True)`). In fp16 the GPU accumulates its convolutions' products in fp16, and the masks drift past D7's bar even after exact split-sum forms (least IoU 0.9976).
- **DINOv3**, at batch 8: fp16, through exact forms that keep every tensor inside fp16's range ([ADR 0022](0022-export-forms-are-exact.md)).

Chosen over six other cells on that host. The photo replay, models only:

| Cell on `yappy` | Per photo | `MemAvailable` drop |
|---|---|---|
| LiteRT on the GPU, fp32 segmenter and fp16 DINOv3 | 9.2 s | 3.6 GB |
| LiteRT on the GPU, all fp32 | 14.6 s | 4.2 GB |
| PyTorch | 36.0 s | 1.7 GB |
| LiteRT on XNNPACK, the CPU | 47.8 s | 1.4 GB |

ONNX Runtime's CPU, OpenVINO and WebGPU cells were measured too; no CPU cell beat PyTorch.

## A host without a usable GPU

LiteRT stays, with XNNPACK on the CPU or whatever accelerator that host offers, but nothing carries over from `yappy` without measuring:

- **Speed:** LiteRT on XNNPACK was 33% slower than PyTorch on the N100's AVX2. Whether AVX-512 changes that is unknown, so on a new host LiteRT is timed against PyTorch with ML-7 P4's photo replay. If it loses, D2's option (b), PyTorch, serves on that host. Serving holds one runtime, so that is a code change, which the owner accepts (2026-10-15).
- **Precision:** fp16 on the CPU needs hardware fp16 arithmetic (AVX512-FP16); without it XNNPACK runs fp32. The precisions chosen for the GPU do not carry over: each graph's precision is decided again, under D7's bars.
- **The forms:** the GPU's placement forms are harmless on the CPU (exact, ADR 0022). Their speed cost there is measured.

## Consequences

- **What it changes in earlier decisions.** ML-2 Q4 chose "PyTorch runtime for now, at both ends", with a later
  runtime to switch at both ends together, training pools rebuilt, because two runtimes would break parity. Here
  the runtimes differ: training stays on PyTorch and serving runs the converted models. Parity is kept another way:
  the processing around the models is one code path, the ports in `coffeecv/model_io.py` that training calls too,
  and each converted model is held to D7's bars against its PyTorch original on the serving host
  ([ADR 0020](0020-converted-models-pass-d7-before-they-serve.md)). [ADR 0015](0015-xl0-segmenter-owner-judged-labels.md)'s
  "the same code in training and serving" holds in that sense for the segmenter: the same processing, and the same
  weights converted, within D7.
- **A departure from the train/inference parity rule** in `CODING_STANDARDS.md` ("an accurate method offline with a cheap one live is rejected"), wherever a graph serves in lower precision than it trains. On `yappy`: training's metrics are measured with fp32 DINOv3, and serving answers with fp16. Over 157 served photos and 4 patch seeds, fp16 changes 0.48% of top-1 answers and no guard verdict. That is the margin by which reported metrics overstate the live ones. Patch sampling alone changes 0.85% between seeds, and fp16's probability changes are about a quarter of sampling's at p95 (0.023 against 0.099).
- **Kept for each converted model:**
  - **A pinned conversion environment:** `litert-torch`, below the torch that training uses.
  - **The export forms:** [ADR 0022](0022-export-forms-are-exact.md).
  - **D7's gate on the serving host's accelerator:** [ADR 0020](0020-converted-models-pass-d7-before-they-serve.md).
- **The release names its accelerator and each graph's precision.** Moving host means a new measurement, a new manifest and a new release, not a code change.
- **On a GPU host,** the service depends on the host's GPU stack (on `yappy`: Mesa's Vulkan driver, the render node opened by the service, LiteRT's accelerator library from the wheel). The CPU and GPU share the N100's package power limit, so heavy CPU work beside serving slows the GPU: an upload's ssh cost 2–6% of a photo's time.
- **Start-up costs:** every graph sets up at start, so the worker warms them at boot ([ADR 0021](0021-serving-warms-models-at-worker-boot.md)).

Source: [ML-7](../ticket_inference_runtime.html) D2, D3, D4; measurements in `docs/plan_inference_runtime.html` (P4, P7) and `analysis/ml7_gpu/README.md`.
