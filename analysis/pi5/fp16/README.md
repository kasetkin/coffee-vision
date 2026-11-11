# OPS-23: fp16 on the Raspberry Pi 5's CPU

How DINOv3 runs on the Pi's CPU in fp16 (LiteRT, XNNPACK flag 4, `FORCE_FP16`) fast and close enough to PyTorch.
Findings and numbers: [ticket OPS-23](../../../docs/ticket_pi5_inference_host.html), section "fp16 on the CPU".

The scripts were run in two places, as in the rest of `analysis/pi5/`:

- **Conversion, here** (repo at `/workspace`), in `convert/.venv`. `convert_seg.py` also imports `imagecodecs`,
  which that venv lacks; put the training venv's copy on `PYTHONPATH`. Outputs go to `$PI16_OUT` (default
  `out/pi16`).
- **Timing, on the Pi**, in the `coffee-bench:ort` image (LiteRT 2.3.0, ONNX Runtime 1.31.0), with
  `/share/coffee-bench` mounted at `/b`: the `.tflite` files under `/b/p16/`, ML-7 P2's inputs at `/b/p2/inputs.npz`,
  records written to `/b/out/p16/`.

| Script | What it does |
|---|---|
| `convert_dino.py VARIANT OUT.tflite` | DINOv3 batch-8 graph with ADR 0022's forms and one of the fp16 variants below; prints its fp32 cosine against ML-7's references |
| `convert_seg.py` | the segmenter's encoder and decoder with `fp16_forms` |
| `dino16.py`, `run16.sh` | 65 DINOv3 calls (13 photos × 40 patches) per variant and flag; `.json` timing record, `.npz` readouts for `../parity_now.py` |
| `micro.py`, `micro_run.py` | one block and its parts, plain and with the forms, timed fp32 against fp16: finds the form that stops the fp16 rewrite |
| `lowering.py` | which TFLite operations a mean, a LayerNorm and a pooling lower to |
| `probe_blocks.py`, `probe_blocks_run.py` | every block's output, fp32 against fp16: where the `allpre` variant goes wrong |
| `enc16.py` | the encoder, fp32 against fp16, embedding cosine against PyTorch |
| `pack_patches.py`, `ans_run.py` | the 288-photo answer check: serving's patches packed as uint8 (rebuilt bitwise on the Pi), DINOv3 run over them, readouts in the layout `../../ml7_gpu/p7_answers.py compare` reads |

`convert_dino.py` variants:

- `prescaled`: the release's forms, readout mean as `sum(x * 1/N)`.
- `split<S>`: `prescaled`, with every Linear and `attn @ v` split into pairwise sums of `S`-wide slices.
- `allpre`: every mean, the LayerNorm's included, as `sum(x * 1/N)`.
- `lnmean`: LayerNorm as `F.layer_norm(x / max|x|, eps=1e-12)` (TFLite `MEAN`), readout as `AVERAGE_POOL_2D`.
- `lnmean_split<S>`: `lnmean`, with Linears over more than `S` inputs split. `S = 768` splits fc2 only. This is the
  variant that passes.

`results/`:

- `<variant>_f<flags>.json`: timing records. `rA_*` and `rB_*` are the alternating re-run.
- `parity_*.json`: D7 on ML-7's 13 photos.
- `mix__replay.json` and `parity_mix.json`: the replay with the encoder in fp16.
- `answers_compare.json`: the 288 photos against PyTorch.
- `answers_*.jsonl`: per-photo times from that check.
