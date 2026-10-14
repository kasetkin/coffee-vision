# ML-7: a GPU or CPU runtime for serving, measured on `yappy`

The report of plan P5 (`docs/plan_inference_runtime.html`, ticket `docs/ticket_inference_runtime.html`). Every
number comes from a committed record in this directory; the raw outputs are in DVC (`artifacts.dvc`).

**In short:** three cells pass D10's test with D15's margin, all of them on the N100's iGPU. They are LiteRT on
Vulkan, ORT's OpenVINO provider on the GPU and ORT's WebGPU plugin. LiteRT on Vulkan is the fastest: a photo's
models take 14.9 s instead of PyTorch's 36.0 s, with parity at float rounding. No CPU cell beats PyTorch. The
GPU's price is memory, and LiteRT's is the highest: 4.1 GB of `MemAvailable` against PyTorch's 1.7 GB. The owner
chooses under D2, D4 and D5. P6 is not needed.

## 1. Where a photo's time goes today (P0)

The current release on `yappy`, PyTorch at 4 threads, the 13 fixtures, the second pass
(`baseline_stages.json`):

| | `/crop` | `/classify` | Photo |
|---|---|---|---|
| Whole, median (p90) | 7.3 s (7.6) | 30.1 s (30.9) | 37.5 s (38.5) |
| XL0 encoder | 6.5 s | 6.5 s | 34.8% |
| Decoder | 0.2 s | 0.2 s | 1.1% |
| DINOv3, 40 patches | | 21.8 s | 57.9% |
| Everything else | 0.6 s | 1.7 s | 6.1% |

The models are 94% of a photo's time. The live record holds only 8 requests: `/crop` 5.2 s and `/classify`
23.4 s (p90 26.1). Those are 1.41× and 1.28× faster than the fixtures, because a lone upload gets the boosted
clock (about 2.15 GHz); from the second photo on, the cores hold about 1.46 GHz. Every number below is in that
sustained state, where `/classify` alone sits at gunicorn's 30 s timeout.

## 2. The seven cells

Each cell is one runtime on one device, fp32, 4 CPU threads. P4 ran three rounds on `yappy` with the site down
(`results.json`). Two kinds of measure:

- **Graphs:** each graph ran in a fresh process: 3 warm-ups, then 20 timed calls (round 1).
- **The photo replay:** decides pass or fail. Each photo runs encoder and decoder twice (`/crop`, then
  `/classify`), then DINOv3 on its 40 patches, two passes over the 13 photos, in one process.

Time per photo in seconds, warm median (cold start: session creation plus the first call):

| | PyTorch | LiteRT CPU | ORT CPU | OpenVINO CPU | **LiteRT Vulkan** | **OpenVINO GPU** | **ORT WebGPU** |
|---|---|---|---|---|---|---|---|
| Encoder | 6.49 (12.1) | 6.91 (7.3) | 4.88 (6.1) | 6.40 (7.8) | 1.78 (4.8) | 2.04 (11.7) | 4.30 (5.5) |
| Decoder | 0.10 (7.5) | 0.10 (0.2) | 0.09 (0.3) | 0.09 (0.9) | 0.09 (0.7) | 0.08 (5.3) | 0.12 (0.5) |
| DINOv3, batch 8 | 21.94 (10.6) | 33.62 (6.8) | 24.12 (5.6) | 23.60 (6.3) | **11.17** (3.8) | **12.94** (12.6) | 13.77 (4.0) |
| DINOv3, batch 40 | **21.86** (28.0) | **33.46** (34.1) | 24.73 (25.7) | 23.63 (25.7) | 13.21 (15.3) | 12.94 (24.6) | **13.50** (14.4) |
| **Photo replay, median** | 36.00 | 47.78 | 38.19 | 36.80 | **14.89** | **17.17** | **22.36** |
| p90 | 36.29 | 47.95 | 38.39 | 36.88 | 14.95 | 17.26 | 22.37 |
| Spread between rounds | 1.2% | 0.4% | 2.1% | 2.6% | 0.2% | 1.5% | 0.0% |
| D7, every round | pass | pass | pass | pass | pass | pass | pass |
| Wholly on its device | yes | yes | yes | yes | encoder only | yes | yes |
| CPU time / wall time | 3.85 | 3.86 | 3.92 | 3.77 | 0.08 | 1.00 | 0.01 |

The replay runs DINOv3 at the batch in bold, the one each cell ran faster in round 1. The 100-call stability
run at batch 8 stays within 0.4% of the 20-call median in every cell.

**Wholly on its device:**
- ORT's GPU cells refuse CPU fallback, so they cannot load at all unless the whole graph is on the GPU.
- LiteRT's delegate reports its own split. The encoder is whole. DINOv3 leaves 5 of 976 ops (BROADCAST_TO) on
  the CPU, and the decoder 38 of 374 (SELECT_V2). This corrects P2's record, which called every Vulkan graph
  whole: P2's fp32 records carry the delegate's report for the encoder only, and the delegate refused the
  same op in P2's fp16 DINOv3. The split costs little, at 0.08 CPU/wall over the replay, and parity is unaffected. A graph form like
  the encoder's could probably move these ops to the GPU too.
- OpenVINO's GPU plugin keeps one CPU core busy while the GPU computes.

**Proof of the GPU:** every GPU cell proved it ran on the iGPU in every round. The driver libraries and the
render node were mapped, and the i915 counters showed the GPU busy for 97–100% of the replay.

### Variants, for information (D4, D12)

| Variant | Time per photo | D7 |
|---|---|---|
| fp16, LiteRT Vulkan | DINOv3 5.74 s, encoder 0.91 s | fail: NaN in DINOv3 and the encoder; decoder IoU 0.9927 |
| fp16, OpenVINO GPU | DINOv3 6.18 s, encoder 0.96 s | fail: encoder NaN; DINOv3 cosine 0.9971; decoder IoU 0.9966 |
| fp16, ORT WebGPU | DINOv3 9.73 s, encoder 2.89 s | fail: encoder NaN; DINOv3 cosine 0.99976; decoder IoU 0.9917 |
| fp16 blanket DINOv3, ORT WebGPU | DINOv3 9.41 s | fail: cosine 0.9993 |
| ORT's fused DINOv3, CPU | 24.30 s (batch 8) | pass; 1% slower than the plain graph |
| int8 encoder, ORT CPU | 2.75 s | fail: IoU 0.0005, 4 answers changed (P1) |
| int8 encoder, OpenVINO CPU | 2.84 s | fail: IoU 0.21 (P3) |

The parity figures come from P1–P3 (`parity_p1_cpu.json`, `parity_p2.json`, `parity_p3.json`) and the times
from P4. fp16 would roughly halve DINOv3's time on the GPU, but no fp16 graph passes D7. The XL0 encoder
overflows in LiteMLA's sum over 4,096 tokens.

## 3. Pass or fail (D10, with D15's margin)

A cell passes if it meets D7 and its replay p90 is at least 25% below PyTorch's (36.29 s), and the gap is wider
than both cells' spread between rounds.

| Cell | p90 | Against PyTorch | D7 | Verdict |
|---|---|---|---|---|
| LiteRT Vulkan | 14.95 s | 58.8% lower | pass | **pass** |
| OpenVINO GPU | 17.26 s | 52.5% lower | pass | **pass** |
| ORT WebGPU | 22.37 s | 38.4% lower | pass | **pass** |
| OpenVINO CPU | 36.88 s | 1.6% higher | pass | fail: not faster |
| ORT CPU | 38.39 s | 5.8% higher | pass | fail: not faster |
| LiteRT CPU | 47.95 s | 32.1% higher | pass | fail: slower |

Each passing cell's gap is more than 30 times the larger of its own spread and PyTorch's (1.2%). D2's preference for a CPU cell does not
apply, because none passes. P6, LiteRT on OpenCL, runs only if no cell passes (D10), so it is not needed.

## 4. Against the live latency

The rows below add P0's pre- and post-processing to each cell's model time: 0.6 s for `/crop` and 1.7 s for
`/classify`. Each endpoint is the cell's warm graph times plus that endpoint's share. "Photo" is the replay
median plus 2.3 s.

| Cell | `/crop` | `/classify` | Photo | OPS-7 would save |
|---|---|---|---|---|
| PyTorch | 7.2 s | 30.1 s | 38.3 s | 6.6 s (17%) |
| LiteRT Vulkan | 2.5 s | 14.7 s | 17.2 s | 1.9 s (11%) |
| OpenVINO GPU | 2.7 s | 16.8 s | 19.5 s | 2.1 s (11%) |
| ORT WebGPU | 5.0 s | 19.6 s | 24.7 s | 4.4 s (18%) |

- P0 measured PyTorch's photo directly at 37.5 s; it ran DINOv3 as batches of 32 and 8, where these rows use
  P4's batch 40. The 0.8 s difference is the error bar on these rows.
- On a passing cell, `/classify` falls from the 30 s timeout to 15–20 s, in the sustained state.
- The GPU cells' clocks were limited too. LiteRT Vulkan and OpenVINO GPU hit the package's power limit (PL1)
  during the replay: the GPU's median clock was 700 and 500 MHz of its 750, and the CPU's 700–850 MHz. Their
  times are under that limit, as serving would be.
- "OPS-7 would save" is one encoder and decoder pass, which OPS-7 would remove.

## 5. Parity per cell

Every cell passed D7 in every round, with identical values in each round. No photo changed its top-1 class, its
guard verdict or its mask's fallback. The continuous measures (D14), the worst photo of the 13:

| Cell | Least cosine | Largest class-probability change | Largest guard-score change | Least mask IoU | Mask pixels differing | Largest low-res logit change |
|---|---|---|---|---|---|---|
| PyTorch (release) | 0.999999999995 | 1.8e-7 | 4e-15 | 0.9999998 | 1 | 3.7e-5 |
| LiteRT CPU | 0.999999999953 | 4.8e-7 | 2e-14 | 0.9999974 | 8 | 2.8e-4 |
| ORT CPU | 0.999999999988 | 3.0e-7 | 3e-14 | 0.9999995 | 3 | 8.1e-5 |
| OpenVINO CPU | 0.999999999956 | 3.0e-7 | 4e-14 | 0.9999995 | 4 | 7.7e-5 |
| LiteRT Vulkan | 0.999999999987 | 2.4e-7 | 9e-16 | 0.9999964 | 8 | 5.1e-4 |
| OpenVINO GPU | 0.999999999992 | 3.0e-7 | 1e-14 | 0.9999979 | 7 | 2.9e-4 |
| ORT WebGPU | 0.999999999921 | 3.0e-7 | 2e-14 | 0.9999990 | 9 | 3.6e-4 |

The guard is the linear probe. Every fixture scores about 0, well inside the distribution and 0.5 from the
threshold, so a change of 1e-14 cannot flip a verdict. The fixtures don't test a photo near the threshold; the
validation photos in the release route's D7 gate would (§8).

The PyTorch row is the release's own code against the references made with the repo's code. Its tiny gap is the
noise floor.

Parity was computed on `yappy`, with the release's code, so the 269 MB of arrays stayed off the slow link. Run
locally with the repo's code on the 15 arrays first fetched, it gives identical reports.

## 6. Memory and size (D1 (b), R5)

`yappy` has 7.5 GB, and the deploy preflight refuses a deploy with less than 5 GB of `MemAvailable`.

| Cell | Replay peak RSS | `MemAvailable` drop | Least `MemAvailable` | Runtime added to the release | Model files |
|---|---|---|---|---|---|
| PyTorch | 2.25 GB | 1.66 GB | 4.77 GB | (torch 769 MB, in the 1.2 GB venv) | 790 MB |
| LiteRT Vulkan | 2.40 GB | 4.18 GB | 2.25 GB | `ai-edge-litert` 58 MB | 815 MB |
| OpenVINO GPU | 2.11 GB | 3.05 GB | 3.37 GB | `onnxruntime-openvino` 252 MB | 815 MB |
| ORT WebGPU | 1.00 GB | 2.67 GB | 3.74 GB | `onnxruntime` 63 MB + `onnxruntime-ep-webgpu` 16 MB | 815 MB |

- **GPU memory is host memory.** The GPU cells' buffers don't show in RSS but do come out of `MemAvailable`.
- **What the drops leave:** LiteRT's drop leaves about 2.2 GB free with nothing else running. The others leave
  3.4–3.7 GB, and PyTorch 4.8 GB.
- **Torch in the worker:** the replay processes did not import torch, but the serving worker will (D5 (a)), so
  serving adds torch's own footprint on top of these figures. P4 did not measure it; R5's smoke test does.
- **Model files:** the converted files are one DINOv3 batch file, the encoder and the decoder. They replace the
  PyTorch weights at about the same size (790 MB).

**Dropping torch (D5 (b)):** about 0.8 GB comes off the 1.2 GB venv (torch 769 MB, timm 16 MB, torchvision
7 MB). The worker's boot loses about 2.8 s of imports, and RSS falls too. That needs the pre- and
post-processing ported and checked against torch under D7. It is the larger saving for D1 (b).

## 7. What each cell costs to keep

- **LiteRT Vulkan:**
  - Comes with the `ai-edge-litert` wheel. Its WebGPU accelerator library is used from the wheel, unbuilt.
  - Needs Mesa's Vulkan driver (22.3.6 on `yappy`).
  - Keeps the LiteRT conversion environment (`convert_litert/`), pinned below the torch that training uses.
  - Keeps the GPU forms (`graphs.litert_gpu_forms`) in the export.
  - The delegate's reports have to be read on every upgrade, because `is_fully_accelerated()` is not
    trustworthy.
- **OpenVINO GPU:**
  - A separate ORT build, `onnxruntime-openvino`, which bundles OpenVINO 2025.4.
  - Needs Intel's NEO OpenCL driver (22.43 on `yappy`).
  - Holds one CPU core busy while the GPU works.
  - Compiles at every cold start: its first answer comes 26 s later than a warm photo would. Its cache cut
    DINOv3's load from 21.7 s to 14.3 s in P3.
  - Needs the zero-bias fix in the export.
- **ORT WebGPU:**
  - Two pip packages, `onnxruntime` and the plugin `onnxruntime-ep-webgpu` (0.4.0, young).
  - Needs Mesa's Vulkan driver.
  - Needs the sequence-to-slice fix in the export.
  - Shares the ONNX export with ORT's CPU provider.

All three GPU cells change the service's sandbox, which has to open `/dev/dri/renderD128`. They also depend on the
host's GPU driver and compile kernels at every start (R6). The deploy gate's `--compare` can no longer demand
bit-identical answers from PyTorch (R4).

## 8. Notes for the serving phases

- **Warm both models at worker start.** Today both models load at worker start but neither is warmed, so the
  first request pays the first inference's set-up inside its 30 s timeout. From process start to the first
  photo's answer:

  | Cell | Load | First answer | Warm photo |
  |---|---|---|---|
  | LiteRT Vulkan | 3.8 s | 20 s | 14.9 s |
  | ORT WebGPU | 1.3–2.4 s | 24–25 s | 22.4 s |
  | OpenVINO GPU, cache emptied | 26–28 s | 43–45 s | 17.2 s |
  | PyTorch | 6–8 s | 42–44 s | 36.0 s |

  Serving must warm both models at worker start and measure the boot time. OpenVINO should keep its model cache
  on disk.
- **One request at a time.** gunicorn runs one sync worker, so one photo holds the GPU at a time. The power
  limit fired with that load alone.
- **Warm the photo-time path too.** P4's replay leaves out pre- and post-processing, which P0 measured at 2.3 s
  a photo.
- **D7 on the validation photos.** P4's parity covers the 13 fixtures only. D7's gate still has to run on the
  validation photos in the release route (R4).

## Files

| File | What |
|---|---|
| `baseline.json`, `baseline_stages.json` | P0: the live log's aggregates and the per-stage baseline (`baseline_stages.py`). |
| `rewrite_check.json`, `manifest_onnx.json`, `manifest_tflite.json`, `parity_p1_cpu.json` | P1: the rewrites against torch, the files, CPU parity (`graphs.py`, `export_onnx.py`, `convert_litert.py`, `make_refs.py`, `parity.py`). |
| `parity_p2.json`, `parity_p3.json` | P2 and P3: the Vulkan and OpenVINO cells (`run_litert.py`, `run_ort.py`, `cellrun.py`, `p2_cells.sh`, `p3_cells.sh`). |
| `results.json` | P4: per graph, stability, the replay and its parity per round (`backends.py`, `p4_graph.py`, `p4_replay.py`, `p4_run.sh`, `p4_results.py`). |
| `artifacts/` (DVC) | The converted files, references, inputs, and every phase's raw outputs and logs (`outputs_p4/` for P4). |
