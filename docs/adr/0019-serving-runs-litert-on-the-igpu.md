# Serving runs both models on LiteRT through the iGPU: the segmenter in fp32, DINOv3 in fp16

Both models train on PyTorch and are converted for serving only (ML-7 D3). The live site runs them on LiteRT (`ai-edge-litert`), on the host's Intel iGPU through LiteRT's WebGPU accelerator, which Dawn runs on Vulkan (Mesa's ANV driver) (D2). Pre- and post-processing stay on torch (D5 (a)). Each graph is compiled at its own precision (D4):

- **The segmenter**, XL0 encoder and decoder: fp32 (`GpuOptions(enforce_f32=True)`). In fp16 its convolutions accumulate their products in fp16, and the masks drift past D7's bar even after exact split-sum forms (least IoU 0.9976).
- **DINOv3**, at batch 8: fp16, through exact forms that keep every tensor inside fp16's range ([ADR 0022](0022-export-forms-are-exact.md)).

Chosen on `yappy` (N100, 4 cores, 7.5 GB) over six other cells: PyTorch, LiteRT on XNNPACK, ONNX Runtime's CPU provider, its OpenVINO provider on the CPU and on the GPU (OpenCL), and its WebGPU plugin. The photo replay, models only, takes:

| Cell | Per photo | `MemAvailable` drop |
|---|---|---|
| LiteRT, fp32 segmenter and fp16 DINOv3 | 9.2 s | 3.6 GB |
| LiteRT, all fp32 | 14.6 s | 4.2 GB |
| PyTorch | 36.0 s | 1.7 GB |

No CPU cell beat PyTorch.

## Consequences

- **A departure from the train/inference parity rule** in `CODING_STANDARDS.md` ("an accurate method offline with a cheap one live is rejected"). Training's metrics are measured with fp32 DINOv3, and serving answers with fp16. Over 157 served photos and 4 patch seeds, fp16 changes 0.48% of top-1 answers and no guard verdict. That is the margin by which reported metrics overstate the live ones. Patch sampling alone changes 0.85% between seeds, and fp16's probability changes are about a quarter of sampling's at p95 (0.023 against 0.099).
- **Kept for each converted model:**
  - **A pinned conversion environment:** `litert-torch`, below the torch that training uses.
  - **The export forms:** [ADR 0022](0022-export-forms-are-exact.md).
  - **D7's gate before release:** [ADR 0020](0020-converted-models-pass-d7-before-they-serve.md).
- **The service depends on the host's GPU stack:** Mesa's Vulkan driver, the render node `/dev/dri/renderD128` opened by the service (a sandbox change), and LiteRT's accelerator library from the wheel.
- **Start-up costs:** the graphs set up at every start, so the worker warms them at boot ([ADR 0021](0021-serving-warms-models-at-worker-boot.md)).
- **The power limit:** the N100's package limit is shared by CPU and GPU. Heavy CPU work beside serving slows the GPU: an upload's ssh cost 2–6% of a photo's time.

Source: [ML-7](../ticket_inference_runtime.html) D2, D3, D4; measurements in `docs/plan_inference_runtime.html` (P4, P7) and `analysis/ml7_gpu/README.md`.
