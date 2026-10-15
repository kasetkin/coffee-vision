# Serving loads and warms every model at worker boot, and the boot is measured

The web worker loads every graph it serves and runs one full photo's worth of inference through them before it accepts a request. That means the encoder, the decoder and DINOv3 at its served batch and precision. The boot's duration and memory are logged at every start, and the deploy's smoke test reads them.

Why: a model's first inference pays set-up that later ones do not, and gunicorn kills a request at 30 s.

- **Today (PyTorch):** both models load at worker start but are not warmed, so the first request pays the first inference's set-up. `/classify` alone already sits at the timeout in the sustained clock state (ML-7 P0).
- **On LiteRT's GPU:** the first answer comes 12–20 s after the process starts, and a warm photo takes 8–15 s. The set-up is LiteRT's and Dawn's own: emptying Mesa's shader cache changed nothing (19–21 s either way).
- **On OpenVINO's GPU,** were it ever used: a cold compile adds about 26 s, and its model cache on disk cuts it.

## Consequences

- The worker's boot time grows by the warm-up, about 15–20 s on `yappy`. Readiness checks and the deploy wait for it.
- A warm-up that fails or returns non-finite outputs fails the boot, rather than surfacing on a visitor's request.
- The warm-up input is a fixed photo shipped with the release, so every boot does the same work.

Source: [ML-7](../ticket_inference_runtime.html) D2, R6; measurements in `docs/plan_inference_runtime.html` (P0, P4, P7).
