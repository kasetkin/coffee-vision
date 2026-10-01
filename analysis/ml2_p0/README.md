# ML-2 P0: setup, mask selection, latency, determinism

Ticket: `docs/ticket_segmentation_mask.html` (ML-2). Plan: `docs/plan_segmentation_mask.html` §3.
Run 2026-09-30 on `tmp/segmentation-mask`.

## Done

- **Vendored-code patches** (triton, dc_ae/omegaconf, onnx): `third_party/efficientvit/PATCHES.md`.
  After them the only missing import was Meta's `segment_anything`, now `segment-anything==1.0` in
  `pyproject.toml` / `uv.lock` (devcontainer and the VM's shared venv synced).
- **Weights**: `models_pretrained/efficientvit_sam/efficientvit_sam_l0.pt` from Hugging Face
  `mit-han-lab/efficientvit-sam` at commit `a2f0c592`, 139,410,184 bytes, sha256 `c4f994b0…` (equal to
  HF's LFS ETag). In `manifest.json` with its `source` URL; `verify.py` passes. `dvc add`ed; **not
  pushed** (the DVC remote has been down since 2026-09-24).
- **Loader** `coffeecv/sam_loader.py` and **segmenter** `coffeecv/segment_beans.py` (plan §2.2:
  `SegParams`, `BeanSegmenter.predict_mask`, `mask_and_crop`, `segment_and_crop`, plus `--time` and
  `--candidates`). `tests/test_sam_loader.py` (6 tests) covers: builds with triton/omegaconf/onnx
  blocked; refuses unlisted and sha-mismatched weights; frozen + eval; logits match
  `tests/fixtures/sam_l0_reference.npz` at atol 1e-3. A random-init model misses by >16, so the
  tolerance has a wide margin.

## Mask selection: blocked by the prompt (D5), owner decision needed

`pick_photos.py` drew 10 photos from the segmenter's **train** split (seed 239): one per session,
plus a second from box_pictures. `segment_beans --candidates` wrote every decoder output (the
single-mask token and the three multimask tokens) at full resolution. `contact_sheet.py` then tinted
each one. The sheets are in `outputs/ml2_p0/candidates*/sheets/` (untracked; rebuild with the
commands below).

| prompt | frame-filling photos (pixel_cam, sony_cam, 08-30 ×3, oneplus_flash) | tray photos (box_pictures ×2, iphone, oneplus 08-25) |
|---|---|---|
| `box` (D5 a) | `multi3` ≈ whole frame (0.91–0.99), which is right for these. `single` has holes; on oneplus_flash it is the tray floor *between* beans (0.125) | **every output is the background**: the wood table (iphone, oneplus: 0.32–0.35) or the white paper (box_pictures: 0.64–0.68). iphone `multi3` = the tray walls only |
| `grid3` (D5 b, 3×3 points at 20/50/80% of the photo) | similar to `box` | iphone/oneplus: the **whole tray incl. its walls** (fails accept rule 1); `multi1` = the walls only. box_pictures: the grid lands on the paper, so the paper is segmented |
| `box+grid3` | similar | same as `grid3` |

No fixed whole-photo prompt, with any output selection, gives the bean region on a tray photo. The
tray sessions (box_pictures 180, iphone 92, oneplus 08-25 90) are **362 of 938** pool photos.
So this is not a mask-selection choice: D5's premise fails for pretrained L0. `seg_mask_select` was
therefore **not** written to params.yaml, and `--prompt` / `--mask-select` are required CLI flags
until it is. Options are in the ticket's activity entry of 2026-09-30 (P0).

**Feasibility probe for the fine-tuning route** (`points_probe.py`): the whole-image box plus hand-placed
points, standing in for a judge's corrections (9 include points on the pile; 4-5 exclude points on
table/paper and walls), single-mask output. It gives **the bean pile on all 4 tray photos** (area
0.17 box_pictures, 0.43 iphone, 0.41 oneplus; predicted IoU 0.79-0.81). The outline follows the pile
edge, or the tin's inner edge on box_pictures. That was checked on overviews only, **not yet at full
resolution**. So the D6 correction loop can produce tray labels, and a decoder fine-tuned to answer
the fixed box with those labels (D7) is plausible. Pretrained L0 with any fixed prompt is not.
Outputs: `outputs/ml2_p0/points_probe/`.

## Larger variants: L1, L2, XL0, XL1 (same tests)

Weights come from the same HF commit (`a2f0c592`) and are sha256-checked against HF's ETags, in
`manifest.json`, and `dvc add`ed (not pushed). Loaded with `segment_beans --variant`. The tests were
the same as L0's: the 10 photos, prompts `box`, `grid3` and `box+grid3`, all 4 decoder outputs,
plus the points probe. Sheets: `outputs/ml2_p0/candidates_<variant>_<prompt>/sheets/`,
`outputs/ml2_p0/points_probe_<variant>/`.

| variant | input | params | tray photos, any fixed prompt / output | frame-filling photos | box + hand points | seg ms, devcontainer 4 thr (12 / 19 MP) |
|---|---|---|---|---|---|---|
| L0 | 512 | 34.8M | table / paper / walls / whole tray | `multi3` ≈ whole frame | pile, pred IoU 0.79-0.81 | 629 / 717 |
| L1 | 512 | 47.7M | same; `grid3` `multi2` on iphone = patchy single beans | `multi3` ≈ whole frame | pile, 0.79-0.80 | 788 / 891 |
| L2 | 512 | 61.3M | same (`box`: every output is the table) | `multi3` ≈ whole frame | pile, 0.79-0.81 | 868 / 1043 |
| XL0 | 1024 | 117.0M | same; ~0.4-area outputs are bean snaps + gaps | **worse**: on 08-30 roasted, `single`/`multi2` = gaps between beans, `multi3` 0.77 with holes | pile, 0.87-0.89 | 2163 / 2261 |
| XL1 | 1024 | 203.3M | same | `multi3` ≈ whole frame | pile, 0.86-0.90 | 3475 / 3488 |

Size does not change the finding: no variant gives the bean pile on a tray photo with a fixed
whole-photo prompt. Every variant gives the pile when given include/exclude points, with the same areas
(0.17 / 0.43 / 0.41), so larger models add nothing to the label route. They only raise the
predicted IoU, which does not show up in the masks at overview size. XL costs 3-5x L0's encoder time. VM timing for
L1-XL1 was not run.

## Centre prompts, all five variants (L0-XL1)

Same 10 photos, all 4 outputs. `--prompt center_box`: a centred square of side min(H, W)/10.
`center_point`: one foreground point at the centre. `center_grid4`: 16 foreground points, a centred
4×4 grid spanning a square of side min(H, W)/8. Sheets: `outputs/ml2_p0/candidates_<variant>_<prompt>/sheets/`.
Only the `center_grid4` sheets have the prompt points drawn (the geometry is stored in `candidates.json`
since that run).

- `center_box`: every variant, every output covers 0.1-0.4% of the photo: about one bean.
- `center_point`: `single`/`multi1`/`multi2` = one bean on every photo and variant. `multi3` = the
  bean pile on all tray photos, for every variant (box_pictures 0.17-0.18, iphone 0.43-0.44,
  oneplus 0.41-0.42, oneplus_flash 0.89-0.90). On frame-filling photos it is one bean or a few
  (0.1-3%).
- `center_grid4`: every variant, every output covers 1.5-7%: the beans under and just around the grid.

## Latency (D8: reported, not gated)

`python -m coffeecv.segment_beans --list analysis/ml2_p0/timing_photos.txt --time --threads 4 --prompt box --mask-select multi3`,
median ms. Decode is already paid by today's pipeline; "seg" is everything the segmenter adds.
The training photos are at most 19 MP (sony), not 50 MP as the plan said; the 50 MP row is the sony
photo upscaled to 8192×6144 as a stress case for large web uploads.

| machine | threads | photo | decode | resize | encoder | decoder | upsample | fill+crop | **seg** |
|---|---|---|---|---|---|---|---|---|---|
| VM (EPYC Genoa), idle | 4 | 12 MP pixel | 133 | 66 | 316 | 77 | 16 | 31 | **506** |
| VM, idle | 4 | 19 MP sony | 230 | 66 | 258 | 97 | 24 | 45 | **491** |
| VM, idle | 4 | 50 MP stress | 560 | 293 | 333 | 202 | 70 | 122 | **1021** |
| VM, idle | 8 | 12 / 19 / 50 MP | | | | | | | 494 / 485 / 958 |
| VM, during training | 4 | 12 MP pixel | 147 | 69 | 519 | 120 | 25 | 34 | **766** |
| VM, during training | 4 | 19 MP sony | 250 | 115 | 544 | 148 | 40 | 63 | **911** |
| VM, during training | 4 | 50 MP stress | 597 | 292 | 526 | 295 | 118 | 135 | **1365** |
| devcontainer (Ryzen 5600U) | 4 | 12 / 19 / 50 MP | | | | | | | 629 / 717 / 1223 |
| devcontainer | 1 | 12 / 19 MP | | | | | | | 1588 / 1828 |

Against today's `/classify` at 7.0 s idle, that is about +0.5 s per upload (+1.0 s at 50 MP), minus
the heuristic crop it replaces. 8 threads does not help on the VM.

**During training** (2026-10-01): a `train_baseline` run (resting params, resnet18 full fine-tune, 4
threads, 1.43 s/batch) ran in a scratch copy on the VM while the segmenter was timed at 4 threads. The
segmenter adds 766 / 911 / 1365 ms (12 / 19 / 50 MP), 1.5-1.9x idle; the encoder nearly doubles and
decode barely moves. Masks are bit-identical to the idle VM run. Training slowed from 1.43 to about
1.9 s/batch while the timing ran. Files: `timing_vm_train_t4.json`, `timing_vm_train_50mp_t4.json`.
P6 repeats it for the full `/classify` path. `/preview`
would pay the same cost again if it also segments (plan §6).

The idle VM run used a throwaway copy of these files and its own uv venv in `~/ml2_p0_scratch`,
deleted afterwards. The during-training run (2026-10-01) used the VM repo itself, checked out at
`tmp/segmentation-mask` and with the shared venv synced (`uv sync --locked`, adding `segment-anything==1.0`).

## Determinism

- Same machine, same thread count: **bit-identical masks** on every repeat (all rows above).
- Same machine, 1 vs 4 or 12 threads (devcontainer): 4 of 12.5 M pixels flip (pixel photo) and
  15 of 12.2 M (iphone). The low-res logits differ by ≤3e-4 from float-sum reordering. VM 4 vs 8
  threads: identical on both photos.
- devcontainer vs VM (AVX2 vs AVX-512), 4 threads: **different mask hashes**, the same kind of
  difference `tests/test_dino_backbone.py` documents for DINOv3.

No threshold epsilon removes this (some pixels always sit near any threshold), and
`torch.use_deterministic_algorithms` does not govern CPU reduction order. What follows:
1. Masks are computed once, by one DVC stage on one machine at a fixed thread count, and stored.
   Verdicts (keyed by mask sha256), the review tool and the pools read the stored masks and never
   recompute them. The plan's `seg_predict` stage already works this way.
2. The `segcrop@*` stages pin their thread count and run on the VM. A rebuild elsewhere changes a
   few pixels and the DVC hashes, but not the result.
3. Train/serve parity holds to ~1e-6 of pixels. Serving recomputes on the VM, where the pools are
   built.

## Rebuild

    PYTHONPATH=. python analysis/ml2_p0/pick_photos.py
    for p in box grid3 box+grid3; do
      d=outputs/ml2_p0/candidates; [ $p = box ] || d=${d}_$p
      python -m coffeecv.segment_beans --list analysis/ml2_p0/mask_select_photos.txt --prompt $p --mask-select single --candidates $d
      PYTHONPATH=. python analysis/ml2_p0/contact_sheet.py $d
    done
    python -m coffeecv.segment_beans --list analysis/ml2_p0/timing_photos.txt --time --threads 4 --runs 5 \
      --prompt box --mask-select multi3 --out analysis/ml2_p0/timing_devcontainer_t4.json
