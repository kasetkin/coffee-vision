# ML-34: the four PU21 papers the owner cites, read against our experiment B

Research for [ML-34](../../docs/ml-0034_ticket_hdr_canonical_photo.html) §3, 2026-11-23, on the local checkout at
`a44c635`. A companion to [research_hdr_pu21.md](research_hdr_pu21.md) (the earlier note, "the note" below): it
reads the four papers the owner cites, each from its arXiv PDF, and sets their pipelines beside ours (§3, §6 of the
note; `b_pu21_probe.py`). A finding for the owner, not a decision. Every claim cites its source; the list is at the
end. **Unverified** marks a claim no primary source here confirmed. Numbers marked "computed here" come from PU21's
published parameters ([S5], as in the note's §2) and the X2HDR/Ke et al. fit [S1, S6], run in a scratch script
outside the repo. Nothing in the repo was run or changed.

## Answer

- **All four arXiv IDs resolve to the papers named**, with the named authors [S1–S4]. X2HDR is v2 (28 Oct 2026,
  NeurIPS 2026); LumaGuide v2 (6 Aug 2026); FDIM v1 (27 Apr 2026); Chubarau et al. v1 (1 May 2024).
- **None of the four tests our case**: a frozen, sRGB-pretrained backbone with a trained linear head, for
  classification, on photos that are almost all SDR, against sRGB of a well tone-mapped picture. Three are about
  generation or quality metrics. Where they report a PU21 "win", it is against **linear** HDR (X2HDR, LumaGuide) or
  between two PU21 scalings (Chubarau). FDIM reports no PU21-on/off comparison for its own network at all.
- **Where a paper does put a frozen SDR-trained network on PU-encoded input next to SDR input, SDR wins.**
  - X2HDR: a frozen FLUX VAE reconstructs sRGB LDR frames at 9.86 JOD, PU21-encoded HDR of the same frames at 9.44,
    linear HDR at 8.54 (Table 1, p.4).
  - X2HDR: a frozen SDR-trained quality model (Q-Eval-100K) scores LDR 0.611 and PU21-encoded HDR 0.587, and the
    authors report "a consistent bias in favor of LDR images" (§F.12, p.24–25).
  - LumaGuide: the same quality model "is sensitive to deviations from its learned SDR prior"; more HDR-like PQ
    distributions get lower scores (p.30–31).
  - Our experiment B (note §6) says the same: −0.004 for PU21 in place of sRGB.
- **Chubarau et al.: the owner's sentence is the paper's own conclusion, but it is about HDR, not about SDR.** In
  Table 2 (p.5), Pmax scaling beats "255" on the HDR subset in all four comparisons. That includes the two networks
  never trained on PU input (PieAPP 0.63 against 0.60 SROCC; VTAMIQ 0.76 against 0.71). "255" beats Pmax on the SDR
  subset in all four (by 0.01–0.02), and equals or beats it on the full set in all four. So the note's §3 was half
  right: 255 did win on SDR, but Pmax won on HDR **with and without** fine-tuning on PU data, not only after it.
  §1 below corrects the note.
- **What it means for us.** Our classifier sees bean pixels. In the median gain-map photo 0.3% of them are above
  white (note §6, A), and 1,085 of 2,772 photos have no gain map at all. That is Chubarau's **SDR** regime, where
  "255" won, and where Pmax squeezes SDR into 0–0.43. Our own run pointed the same way: squeezing SDR into 0–0.73
  (pu21_h4) cost twice as much as no squeeze (−0.008 against −0.004).
- **Recommendation:** treat PU21 at the model input as answered for ML-34. None of the four papers predicts a gain
  for our setting, and three of them contain evidence pointing the other way. If the owner wants his Pmax claim
  tested directly, one cheap screen is spelled out in §5. The expected result is below pu21_h4.

## 1. Chubarau, Yoo, Akhavan, Clark 2024 [S2] (HVEI 2024, arXiv:2405.00670v1)

**Task and "boost".** Full-reference image quality assessment (IQA). The score is SROCC/PLCC against human quality
scores on UPIQ (LIVE + TID2013 SDR; Korshunov + Narwaria HDR) (p.3, Table 1 p.4). "Better" is higher correlation.
The comparisons are PU21 "Pmax" against PU21 "255" scaling; fine-tuned against not fine-tuned on PU data; with or
without CORAL domain adaptation. Results quoted from [3] (PU08-based metrics, HDR-VDP, HDR-VQM) sit in the same
table, but **no row feeds sRGB input** to the networks on UPIQ's SDR subset. So PU21-SDR is never compared with
sRGB-SDR in this paper.

**Sequence** (p.2–4):
1. **Source.**
   - SDR: gamma/sRGB code values through a gain-offset-gamma display model, L = (Lmax − Lblk)·F(V) + Lblk (eq. 1,
     p.2), with **Lmax = 100 cd/m², Lblk = 0.5 cd/m²** (200:1) (p.3).
   - HDR: values "either directly encode luminance or can be tone-mapped to a given display characteristic using a
     similar approach" (p.3). How UPIQ's HDR images were put in absolute units is UPIQ's [3], **unverified** here.
2. **PU21 `banding_glare`** ("banding + glare variant", Fig. 1 p.2; p.4). Applied per channel or to luminance: **not
   stated** in the text read (**unverified**).
3. **Scaling**:
   - "Pmax": PU / ~595, so everything is in 0–1 and SDR is in about 0–0.4.
   - "255": PU / 255, so SDR is in 0–1 and HDR reaches about 2.3 (p.3).
   - Any ImageNet mean/std normalisation after that is **not stated**.
4. **Networks.**
   - PieAPP, with a VGG16-like feature extractor, and VTAMIQ, with a ViT. Both feature extractors were "pre-trained
     for classification of sRGB images" (p.4).
   - Both IQA models were then trained end to end on KADID-10k in sRGB (AdamW, 50 epochs; p.4).
   - "PU-" variants are fine-tuned further on PU-encoded KADID-10k. The DA variants add CORAL (Table 3, p.5).
   - Nothing is frozen in any trained variant. The networks see PU21 input at test time without having been trained
     on it only in the "Trained on KADID-10k (sRGB)" rows.

**Table 2 (p.5), SROCC (PLCC in brackets).** The "Trained on KADID-10k (sRGB)" rows are networks never trained on PU
input.

| Model | Input | Full | SDR | HDR |
|---|---|---|---|---|
| PieAPP (sRGB-trained) | PU21 Pmax | 0.85 (0.84) | 0.87 (0.87) | **0.63** (0.65) |
| PieAPP (sRGB-trained) | PU21 255 | **0.86** (0.85) | **0.88** (0.89) | 0.60 (0.63) |
| VTAMIQ (sRGB-trained) | PU21 Pmax | 0.87 (0.87) | 0.90 (0.91) | **0.76** (0.76) |
| VTAMIQ (sRGB-trained) | PU21 255 | **0.88** (0.89) | **0.91** (0.92) | 0.71 (0.72) |
| PU-PieAPP (fine-tuned on PU) | PU21 Pmax | 0.87 (0.87) | 0.89 (0.89) | **0.75** (0.75) |
| PU-PieAPP (fine-tuned on PU) | PU21 255 | 0.87 (0.86) | **0.90** (0.90) | 0.63 (0.66) |
| PU-VTAMIQ (fine-tuned on PU) | PU21 Pmax | 0.89 (0.90) | 0.91 (0.92) | **0.81** (0.81) |
| PU-VTAMIQ (fine-tuned on PU) | PU21 255 | **0.90** (0.91) | **0.93** (0.94) | 0.72 (0.73) |

- The paper's reading (p.5): "performance … increases on SDR and decreases on HDR data when 255 normalization is
  used instead of Pmax … Since our main objective is HDR performance … we emphasize our results under Pmax".
- Conclusion (p.6), the owner's quote verbatim: "the benefit of aligning the full range of PU values to the range of
  values used in pre-training outweighs the benefit of similarity between PU-encoded SDR values and sRGB".
- With domain adaptation, "Pmax offers stronger performance for all setups", except training directly on labelled
  HDR, where the two are "comparable"; the 255 results are omitted (p.6).

**Verdict on the owner's summary and on the note.**
- The owner's sentence is accurate as a quote. It is overstated if read as "Pmax is better". Pmax is better **on
  HDR**. On SDR it is slightly worse in every row, and on the full set it is never better.
- The note's §3 ("Pmax won on HDR after fine-tuning and 255 won on SDR") was **incomplete**. Pmax also won on HDR
  without any fine-tuning on PU data: +0.03 (PieAPP) and +0.05 (VTAMIQ) SROCC. Those networks were still trained
  end to end on sRGB IQA data, so this is not a frozen backbone with a refit linear head.
- The note's inference stands for SDR content: the networks that never saw PU input did better on SDR under "255".

## 2. X2HDR, Wu, Su, Ma, Liao, Mantiuk [S1] (arXiv:2602.04814v2, NeurIPS 2026)

**Task and "boost".** Text-to-HDR generation and single-RAW-to-HDR reconstruction with FLUX.1-dev (and SD-1.5).
- "Boost" in generation means Q-Eval-100K image quality and text alignment, plus effective dynamic range in stops.
- "Boost" in reconstruction means ColorVideoVDP JOD and exposure-optimised PSNR, SSIM, LPIPS and DISTS.
- Baselines: LEDiff, Bracket Diffusion and RawHDR (Tables 2–3, p.7–9). The encoding ablations compare PU21 with
  **linear**, γ 2.2, log (μ = 500) and PQ, all **HDR** encodings (Table 4, p.9).

**Sequence** (§2.1 p.3; §3 p.4–5; §4 p.6; F.1 p.17–18):
1. **Source.**
   - Scene- or capture-referred HDR, mostly relative: merged RAW brackets and public HDR datasets.
   - For the VAE test, frame-matched Blu-ray LDR/HDR pairs (512 pairs, §2.2 p.3).
   - Training crops are kept only if they span at least 5 stops (F.1 p.17).
2. **Absolute scale**: "we globally rescale linear RGB values so that the maximum luminance corresponds to Lpeak =
   4,000 cd/m²" (p.3), per image. This is no display model and no black level, and the scale is per-image, set by
   the image's own peak.
3. **Curve: not the reference PU21 `banding_glare`.**
   - It is the log-quadratic fit V = a(log₂L − Lmin)² + b(log₂L − Lmin), a = 0.001908, b = 0.0078,
     Lmin = log₂0.005 (eq. 1, p.3), taken from Ke et al. [S6].
   - Ke et al. state that this fit is to "the PU21 encoding for a banding model without the influence of glare"
     ([S6] §3, p.3), i.e. the `banding` variant, not the recommended `banding_glare` ([S5]).
   - Computed here: it maps 0.005–10 000 cd/m² onto 0–1, so it is a **full-range (Pmax-like) scaling**. 100 cd/m²
     lands at 0.50; `banding_glare`/595 puts it at 0.43.
   - At 1 cd/m² the fit gives 0.17 against `banding_glare`'s 0.06. The fit tracks the normalised `banding` variant
     within about 0.02 from 0.01 to 10 000 cd/m².
4. **Per channel** ("channel-wise", p.3).
5. **Network.** The FLUX VAE and text encoder are **frozen**. Only LoRA adapters in the denoiser are trained, on the
   encoded HDR (§3, p.5). The VAE is frozen; the generator is not.

**Results that bear on us.**
- **VAE reconstruction** (frozen VAE, nothing trained; Table 1, p.4):

  | Input | JOD | Q⋆PSNR |
  |---|---|---|
  | LDR (sRGB) | 9.86 | 38.8 |
  | PU21-encoded HDR | 9.44 | 35.6 |
  | linear HDR | 8.54 | 27.9 |

  So PU21 beats linear by a wide margin and stays below the SDR version of the same frames.
- **Encoding ablation** (with LoRA training; Table 4, p.9): PU21 ≈ PQ > γ ≈ log > linear.
  - Text-to-HDR quality: 0.668, 0.661, 0.629, 0.627, 0.578.
  - RAW-to-HDR JOD: 7.57, 7.48, 5.99, 6.12, 5.60.
  - The authors note that γ and log histograms are *closer* to LDR statistics than PQ/PU21 are, yet lose after
    fine-tuning. They explain this by the inverse transform after generation amplifying errors (App. E, p.17).
    That argument is about decoding back to linear light; a classifier has no inverse step.
- **Frozen SDR quality model on PU21 input** (F.12, p.24–25), over 512 aligned pairs:
  - image quality: LDR 0.611, PU21-HDR 0.587;
  - text alignment: LDR 0.736, PU21-HDR 0.695;
  - the authors report "a consistent bias in favor of LDR images, even when matched (LDR, HDR) pairs are
    perceptually equivalent".
- **Never PU21-encoded SDR.** LDR always goes in as sRGB in [0, 1]. The encoding of the LDR conditioning input in
  the LDR-to-HDR variant (F.7, p.20) is not stated (**unverified**).

## 3. LumaGuide, Chen, Saini, Adsumilli, Bovik [S3] (arXiv:2607.26237v2)

**Task and "boost".** Training-free HDR text-to-image generation. Sampling is steered so that the luminance
histogram of the predicted image matches a target HDR histogram (Wasserstein-1, 32 bins) (Alg. 1, p.5; p.7).
- The scores are histogram distance, dynamic range in stops, Q-Eval quality and alignment, and a 15-observer JOD
  study (p.7–8, p.27).
- Baselines: unguided FLUX, LEDiff, BracketDiffusion and X2HDR (Table 2, p.8). The domain ablation is **linear
  against PQ** for the guidance loss (Table 3, p.8–9).

**Sequence.**
- No HDR data enters any network as input. The frozen diffusion model's decoded RGB is *interpreted* as PQ code
  values: Y = PQ(Lum(x̂₀)) (Alg. 1, line 4, p.5); "generated images are represented in PQ space" (p.30).
- PQ uses its standard 10 000 cd/m² reference (p.18). There is no display model, black level or PU21 parameter set
  in the main method. Nothing is trained (p.6).
- **PQ vs PU21 appendix** (A.5.4, Table 9, p.23):

  | Resolution | Encoding | Dynamic range (stops) | Q-quality | Q-alignment |
  |---|---|---|---|---|
  | 512 | PQ | 14.99 | 0.568 | 0.814 |
  | 512 | PU21 | 15.95 | 0.556 | 0.813 |
  | 1024 | PQ | 16.02 | 0.584 | 0.833 |
  | 1024 | PU21 | 16.40 | 0.588 | 0.834 |

  The authors keep PQ as the default "due to its robustness and broader compatibility"; PU21 "may provide
  advantages in extreme dynamic range scenarios". How PU21 was implemented there (variant, scaling) is **not
  stated** (**unverified**).
- **The point that bears on us** (p.30–31): Q-Eval quality, a frozen network trained on SDR, falls as outputs
  become more HDR-like in PQ even where they look better. The authors say Q-Eval "is sensitive to deviations from
  its learned SDR prior", and that an HDR luminance redistribution "may be penalized as a distribution shift from
  SDR statistics".
- SDR content in PU21: never.

## 4. FDIM, Wang, Zhang, Zhuang, Zhang, Yu, Zhao [S4] (arXiv:2604.24123v1)

**Task and "boost".** A full-reference video quality metric for traditional and neural codecs. The score is
PLCC/SROCC against mean opinion scores.

**Sequence.**
- **Network** (§III-B, p.4; §III-E, p.6; p.7): a ResNet-18 pretrained on ImageNet, then trained with a feature-
  distance module and fusion layers on DCVQA, which is SDR only (1920×1080 YUV 4:2:0). It is then **fixed**:
  "evaluated using the same FDIM model weights without any dataset-specific fine-tuning".
- **HDR test input** (§III-D, p.5–6):
  - display-encoded RGB → EOTF (sRGB for SDR, PQ/HLG for HDR) → (Lpeak − Lblack)·E(V) + Lblack, clipped at Lpeak,
    plus k·E_amb/π; for PQ the absolute values are used directly, clipped at Lpeak;
  - then PU21 "independently to each channel";
  - then "each channel … normalized to [0,1] for network input".
- **What the paper does not give.**
  - The divisor of that normalisation is not given (**unverified**; "normalized to [0,1]" with peak clipping
    suggests Pmax-like).
  - Lpeak, Lblack and the PU21 parameter set are not in the main text. The paper calls PU21 "a piecewise
    polynomial", which is not the reference closed form [S5] (**unverified** which implementation).
  - The hand-crafted VMAF branch's HDR input is not stated.
- **SDR at test**: the SDR datasets are evaluated as "FDIM", without PU21 (Tables II, IV). The SDR content of
  HDRSDR-VQA, if any, is handled as described in the supplementary material, which was not read (**unverified**).

**Results** (Table V, p.10; SROCC on LIVE-HDR bright / AVT-UHD-1-HDR / HDR-VDC bright+near / HDRSDR-VQA):

| Metric | LIVE-HDR bright | AVT-UHD-1-HDR | HDR-VDC bright+near | HDRSDR-VQA |
|---|---|---|---|---|
| FDIM+PU21 | 0.933 | 0.878 | 0.925 | 0.875 |
| FDIM(Deep)+PU21 | 0.930 | 0.852 | 0.889 | 0.895 |
| ColorVideoVDP | 0.894 | 0.837 | 0.869 | 0.781 |
| VMAF (no PU21) | 0.920 | **0.893** | 0.895 | 0.832 |
| PSNR | 0.814 | 0.618 | 0.706 | 0.755 |
| PSNR+PU21 | 0.816 | 0.646 | 0.668 | 0.770 |

Plain VMAF, which is not given PU21, beats FDIM+PU21 on AVT-UHD-1-HDR and is within 0.02 of it on LIVE-HDR bright.

- **There is no FDIM-without-PU21 row on HDR**, so the paper does not measure what PU21 adds to its own network.
  Its claim is that zero-shot SDR→HDR transfer "works" with PU21 (p.9, p.12).
- The only on/off pairs are PSNR and SSIM, and they are mixed: PU21 helps PSNR a little on LIVE-HDR, AVT and
  HDRSDR-VQA, and hurts it on all four HDR-VDC conditions (e.g. 0.706 → 0.668).
- The paper does not state whether the other SDR metrics (LPIPS, DISTS, TOPIQ) were given PQ code values or
  something else (**unverified**).

## 5. Side by side with ours

| Step | Chubarau [S2] | X2HDR [S1] | LumaGuide [S3] | FDIM [S4] | **Ours (exp. B, note §6)** |
|---|---|---|---|---|---|
| Task | FR IQA (SROCC) | HDR generation / RAW→HDR | training-free HDR generation | FR VQA (SROCC) | 10-class origin classification (macro-F1) |
| Source light | SDR via display model; HDR from UPIQ | relative HDR / RAW, ≥ 5 stops | none (model output read as PQ) | PQ/HLG HDR video; SDR via display model | phone photo, gain map applied; 0.3% of bean pixels above white in the median photo |
| Absolute scale | SDR 100 cd/m², black 0.5 | image max → 4000 cd/m², per image | PQ 10 000 reference | Lpeak/Lblack, values not given | SDR white = 100 cd/m², black 0 (floor 0.005) |
| Curve | PU21 `banding_glare` | quadratic fit to PU21 `banding` (no glare) | PQ (PU21 in an appendix) | "PU21" (implementation unclear) | PU21 `banding_glare` |
| Channels | not stated | per channel | luminance only (for the loss) | per channel | per channel, linear sRGB primaries |
| Scaling | Pmax (/595) or 255 | full range → 0–1 (white 100 → 0.50) | PQ 0–1 | "to [0,1]" (divisor not given) | /PU(top): top 100 (= "255") or 400 cd/m² (white → 0.73) |
| Network | VGG/ViT, all trained (sRGB, then optionally PU) | frozen VAE + LoRA-trained denoiser | all frozen | ResNet-18 trained on SDR, frozen for HDR | **frozen DINOv3 B/16, linear head refit** |
| SDR in PU21 at test | yes (UPIQ SDR) | no | no | no (SDR sets without PU21) | **yes, 99%+ of pixels** |
| PU21-SDR vs sRGB-SDR | not compared | not compared | not compared | not compared | **−0.004 test macro-F1** |
| Baseline the "win" is over | the other PU21 scaling | linear HDR (and γ/log/PQ) | linear-space guidance | other metrics | **sRGB of a tone-mapped photo** |

**Relevance.**
1. **Task.** Reconstruction and generation must *decode back* to linear light. X2HDR's own explanation for PU21
   beating γ/log is error amplification by the inverse transform (App. E). A classifier has no inverse, so that
   advantage does not carry over. IQA compares two images through the same encoding; we read semantic texture
   classes from a single picture.
2. **Data.** Their sources were picked for dynamic range: X2HDR keeps crops of at least 5 stops, LumaGuide targets
   about 15 stops, UPIQ's and the VQA sets' HDR parts are HDR by construction. Our bean regions are SDR in almost
   every photo (note §6, A). The papers' PU21 benefit is about where the HDR values go, and we have very few HDR
   values.
3. **Baseline.** "PU21 beats linear" (X2HDR Table 1/4; LumaGuide Table 3) is **not** "PU21 beats sRGB of a
   tone-mapped picture". Where a paper shows both, sRGB-LDR beats PU21-HDR for a frozen network (X2HDR Table 1 and
   F.12; LumaGuide p.30–31). We already avoid the linear-light failure: our read-back is sRGB.
4. **Frozen.** Every PU21 "win" with a pretrained network came with training on the encoded input: X2HDR's LoRA,
   Chubarau's fine-tune or CORAL. The exception is Chubarau's sRGB-trained rows, where Pmax helped only the HDR
   subset. Our backbone is frozen by ADR 0006; only the head adapts.

**Choices of theirs we did not test, and what each predicts.** Shifts are computed here, as mean shift from 255·V
over all 8-bit codes, the note's §2 measure.

| Untested choice | Where SDR white lands | Mean shift from sRGB | Prediction |
|---|---|---|---|
| Pmax full range, `banding_glare`/PU(10 000) (Chubarau) | 0.43 | −62 levels with black 0.5 (pu21_h4 was −18, pu21_sdr +23) | Worse than pu21_h4. Compressing SDR toward 0.73 already doubled the cost. Chubarau's sRGB-trained networks lost on SDR under Pmax in every row; the gain was on HDR content, which we barely have. |
| Display black 0.5 cd/m² (Chubarau) | sRGB 0 → 0.087 ("255") | +27 against our +23 | Moves further from sRGB; no reason to expect a gain. |
| PQ (LumaGuide, X2HDR ablation) | — | +47 (note §2) | At best PQ ≈ PU21 in both generation papers; further from sRGB than PU21. Below pu21_sdr is the likelier outcome. |
| X2HDR's quadratic `banding` fit, full range | 0.50 at 100 cd/m² | −39 | As Pmax: compresses SDR. |
| X2HDR's per-image peak → 4000 cd/m² | 0.89, varying per photo | +44 | A per-photo brightness change set by the brightest highlight (the note §3 already advised against it). |
| Fine-tuning or domain adaptation on PU input (Chubarau, X2HDR) | — | — | This is where every paper's PU21 gain came from. It needs an unfrozen backbone, which ADR 0006 rules out. |

None of these predicts a different sign from experiment B for our data. The one prediction a paper supports with a
number, Pmax helping on HDR content, needs HDR content, and in the median photo 0.3% of bean pixels have it.

## 6. Recommendation for ML-34

- **Do not adopt PU21 at the model input, and do not schedule more PU21 arms for ML-34.** The cited papers show
  PU21 beating linear HDR, which we never use. They show PU21 needing training to beat its alternatives. Where they
  put a frozen SDR network on SDR against PU-encoded input, the SDR input won.
- **Correct the note's §3**: in Chubarau's Table 2, Pmax beat "255" on HDR also without fine-tuning on PU data
  (+0.03 / +0.05 SROCC); "255" beat Pmax on SDR in every row (§1).
- **If the owner wants his Pmax claim tested on our data,** one screen, run exactly as experiment B (unaugmented,
  C = 0.1, seeds 42/123/7, the same patches), a diagnostic and not a recorded run:
  - arm `pu21_pmax_hdr`: pu21_h4_hdr's construction with top = 100 × white (10 000 cd/m², so /PU(10 000)) and
    `BLACK = 0.5`;
  - compare it, paired by seed, with pu21_h4_hdr and with base;
  - fix the bar before the run: adopt only if it beats base by ADR 0002's bar;
  - expected: below pu21_h4_hdr (−0.004 against base), because SDR white falls to 0.43;
  - cost: about one build plus three embeddings (about an hour a seed, note §6).
  It is not needed to decide D1–D4. Its value is settling the claim, not finding accuracy.

## 7. Measured here: full-range (Pmax) scaling

Measured on our data after this note was written (2026-11-23, `b_pu21_probe.py`, arm `pu21_pmax_hdr`: as
pu21_h4_hdr, but PU21(10,000 cd/m²) = 255, so SDR white at 110 of 255; display black 0, as every other arm, not
the 0.5 suggested above; 8-bit crops, so SDR gets ~110 levels where Chubarau fed floats). Unaugmented head, three
seeds, test / val macro-F1: 0.9422 / 0.9421 (s42), 0.9441 / 0.9325 (s123), 0.9238 / 0.9420 (s7).
- Against today's crops: test −0.0043 (−0.0064 to −0.0017), lower at every seed; val −0.0026 (−0.0081 to +0.0032).
- Against pu21_h4_hdr (the same HDR light, 4 × white = 255): test −0.0003 (−0.0026, +0.0039, −0.0021), val −0.0031
  (−0.0017, −0.0074, −0.0002). Full-range scaling does not beat the narrower one on our data, and neither beats sRGB:
  Chubarau's HDR-subset result does not carry over to crops that are almost all standard range.

## Unverified

- Chubarau et al.: whether PU21 was applied per channel or to luminance, and whether ImageNet mean/std followed the
  Pmax/255 scaling, are not stated in the text. How UPIQ's HDR images were put in absolute units is UPIQ's [3],
  not read.
- X2HDR: how the Blu-ray HDR frames (Table 1) were linearised and whether the 4000 cd/m² rescale was applied there
  is not stated beyond §2.1's "for all HDR images". The LDR conditioning encoding in F.7 is not stated.
- LumaGuide: the PU21 implementation (variant, scaling) behind Table 9 is not described.
- FDIM: the PU21 implementation ("piecewise polynomial"), Lpeak, Lblack, the [0,1] divisor, the VMAF branch's HDR
  input, the SDR metrics' HDR input, and the handling of HDRSDR-VQA's SDR content are in a supplementary not read
  (not in the arXiv PDF).
- No source found tests PU21 (or PQ) input on a **frozen** backbone with a **refit linear head** for
  classification. The predictions in §5 are reasoning from these papers and experiment B, not evidence.

## Sources

PDF page numbers are the file's own pages. All four papers were read from the arXiv PDFs downloaded on 2026-11-23
and converted to text; tables were read from the extracted text.

- S1 Wu, Su, Ma, Liao, Mantiuk, "X2HDR: HDR Image Generation in a Perceptually Uniform Space", arXiv:2602.04814v2
  (NeurIPS 2026): https://arxiv.org/pdf/2602.04814v2 (abstract p.1–2; §2.1 eq. 1 and the 4000 cd/m² rescale p.3;
  §2.2 and Table 1 p.3–4; §3 p.4–5; Table 4 and §5.4 p.9–10; App. D p.16; App. E p.17; F.1 p.17–18; F.4 p.18–19;
  F.7 p.20–21; F.12 p.24–25)
- S2 Chubarau, Yoo, Akhavan, Clark, "Adapting Pretrained Networks for Image Quality Assessment on High Dynamic Range
  Displays", HVEI 2024, arXiv:2405.00670v1: https://arxiv.org/pdf/2405.00670v1 (eq. 1 and Fig. 1 p.2; "Training
  with PU-encoded Data" p.3; Experimental Setup p.4; Table 2 and "Normalization scheme" p.5; Table 3 p.5;
  domain-adaptation text and Conclusion p.6)
- S3 Chen, Saini, Adsumilli, Bovik, "LumaGuide: Distribution Shaping for Training-Free HDR Generation in Diffusion
  Models", arXiv:2607.26237v2: https://arxiv.org/pdf/2607.26237v2 (Alg. 1 p.5; §4.1 and Table 1 p.7; Table 2 p.8;
  Table 3 p.8–9; A.3 p.18; A.5.4 Table 9 p.23; subjective study p.27; Q-Eval discussion p.30–31)
- S4 Wang, Zhang, Zhuang, Zhang, Yu, Zhao, "FDIM: A Feature-distance-based Generic Video Quality Metric for
  Versatile Codecs", arXiv:2604.24123v1: https://arxiv.org/pdf/2604.24123v1 (zero-shot claim p.2; §II-D p.3;
  §III-B ResNet-18 p.4; §III-D eq. 13–15 p.5–6; §III-E p.6; §IV-B p.7; §V-C p.9; Table V p.10; Conclusion p.12)
- S5 `gfxdisp/pu21` at `78340c0`, `matlab/pu21_encoder.m` L49–55 (parameter sets), README L10 (use
  `banding_glare`), as cited in the note [S2 there]:
  https://github.com/gfxdisp/pu21/tree/78340c0c4c20908c6bdcf0931dfde763c53a24ad
- S6 Ke, Luo, Xiang, Fan, Ranjan, Chapiro, Mantiuk, "Training Neural Networks on RAW and HDR Images for Restoration
  Tasks", arXiv:2312.03640v3: https://arxiv.org/pdf/2312.03640v3 (§3, eq. 2–3, p.3: the quadratic PU21 fit, "a
  banding model without the influence of glare")
- S7 This repo: [research_hdr_pu21.md](research_hdr_pu21.md) §2, §3, §6; `analysis/ml34/b_pu21_probe.py`
  (`PU21_BANDING_GLARE`, `WHITE = 100`, `BLACK = 0`, `encode`, `_one_hdr`); [ADR
  0006](../../docs/adr/0006-frozen-dinov3-b16-head.md) (frozen backbone); [ADR
  0002](../../docs/adr/0002-paired-multi-seed-evidence.md) (the bar)
