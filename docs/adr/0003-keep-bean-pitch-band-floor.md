# Keep the bean-pitch FFT band floor at k_lo = 4

`bean_k_lo` looks like it clips the bean-pitch estimate, since many photos land on it, but a sweep showed the floor suppresses 1/f spectral drag: lowering it makes the estimate worse on every capture. Raising it to 5 improves offline pitch accuracy but is not adopted, because offline accuracy is not classification accuracy and `bean_calibration_k` must move with it; changing either takes a paired multi-seed screen ([ADR 0002](0002-paired-multi-seed-evidence.md)).

Origin: 2026-09-03 sweep, `analysis/bean_scale/README.md`; the comment above `bean_k_lo` in `params.yaml`.
