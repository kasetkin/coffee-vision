# A converted model serves only when proven on the GPU and within D7's bars against PyTorch

A model converted for serving ([ADR 0019](0019-serving-runs-litert-on-the-igpu.md)) is checked against the PyTorch fp32 model it was converted from, on the deploy fixtures and on a larger answer-check set that includes true negatives. It must meet ML-7 D7's bars:

- **The segmenter, fp32:** the least readout cosine is at least 0.99999 for every patch, mask IoU is at least 0.999 for every photo, and every photo whose top-1 or guard verdict changes is listed for the owner.
- **DINOv3, fp16:** no NaN or Inf in any output; the least patch cosine is at least 0.999 on the served photos (those the guard accepts); no guard verdict changes, negatives included; top-1 changes on at most 1% of the served photos; the p95 of the largest class-probability change is at most 0.05.

Each GPU graph counts as running on the GPU only when that is proven, not when it raises no error:

- **The driver's libraries are mapped:** `libvulkan_intel.so` and LiteRT's WebGPU accelerator, and never `libvulkan_lvp.so`, Mesa's CPU renderer.
- **The render node is open.**
- **The i915 counters show the GPU busy.**
- **The delegate's own report places every op on the GPU.** LiteRT's `is_fully_accelerated()` is not trusted: it once said true for a probe graph the delegate had split.

The gates compare against PyTorch within these bars, not bit for bit. The same code on two CPUs already differs: the segmenter's float rounding moved a mask-edge pixel in 48 of 782 photos between the devcontainer and `yappy`, one or two grey levels in a few dozen of 6 million input values.

## Consequences

- `deploy_webapp.sh --compare` and `release_probe.py` move from bit-identical answers to these bars (ML-7 R4).
- The checks need the GPU, so they run on the host, in the smoke-test unit before a release goes live.

Source: [ML-7](../ticket_inference_runtime.html) D7, D14; measurements in `docs/plan_inference_runtime.html` (P2, P4, P7).
