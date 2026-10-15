# Export forms are exact in real arithmetic, and fp16 forms are sized over every photo

When a runtime refuses a graph or loses precision, the model is not approximated to fit. It is rewritten into forms that compute the same function in real arithmetic, in `analysis/ml7_gpu/graphs.py`. Each form is checked against PyTorch in fp32, at float rounding, before conversion.

- **For placement, so LiteRT's delegate takes the whole graph:**
  - the bicubic neck as two matmuls, with the constant on the right;
  - LiteMLA's ones concatenated, not padded;
  - the qkv split in 4-D, with attention written out;
  - DINOv3's RoPE table held constant;
  - its prefix tokens held tiled to the batch;
  - SAM's box prompt without writes into a tensor.
- **For fp16's range:** power-of-two scales, which are exact in floating point, around LiteMLA's linear attention; a denominator guard that is exact because the numerator is 0 wherever the denominator is; LayerNorm on x / max|x| per token; attention scores shifted by their row's max.
- **For fp16's precision:** a convolution over many input channels as a tree of sums over 128-channel slices.

An fp16 form's scales are sized from a range scan of every module's values in fp32, over every photo in `dataset/` (2,027, the 584 true negatives included), not over the fixtures. The negatives reached nearly every site's maximum, and the other photos came within about 15%. The scales keep at least 3× headroom under fp16's maximum.

## Consequences

- No clamp, approximation or retraining is used to make a runtime fit. A form that is not exact has to be justified as a departure, like fp16 itself ([ADR 0019](0019-serving-runs-litert-accelerator-per-host.md)).
- A new model, a new checkpoint of a different architecture, or a new runtime version reruns the range scan and the forms' checks before conversion.
- Split sums cost graph size: the encoder grew from 65 convolutions to 612 at 128 channels, and to 2,420 at 32, where it lost fp16's speed.

Source: [ML-7](../ticket_inference_runtime.html) D4, D14; measurements in `docs/plan_inference_runtime.html` (P1, P2, P7) and `analysis/ml7_gpu/p7_ranges.json`.
