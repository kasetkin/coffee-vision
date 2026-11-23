# ML-34: does HDR light help the models, and where would PU21 sit?

Research for [ML-34](../../docs/ml-0034_ticket_hdr_canonical_photo.html) (D1, D2, D3; R1), 2026-11-22, on the local
checkout at `a44c635`. A finding for the owner, not a decision. Every claim cites its source; the list is at the end.
**Unverified** marks a claim no primary source here confirmed. The numbers marked "computed here" come from PU21's
published equation and parameters [S2], run in a scratch script outside the repo. Nothing in the repo was run or
changed. §6 (added after, the same day and the next) records the experiments run on this checkout for the
note's E1–E3, by `analysis/ml34/a_light_above_white.py` and `b_pu21_probe.py`.

## Answer

- **The literature gives no reason to expect a gain on this dataset.** HDR input helped in two kinds of study. In
  one, the test scenes had extreme contrast: a scene of about 18 stops, or simulated exposures 4–10× off [S5, S6,
  S11]. In the other, the network or a learnable front end was trained on the new encoding [S8, S10, S12]. Under
  ordinary light, HDR or RAW input matched SDR, within about 1 mAP [S7, S8, S10]. With a detector pretrained on SDR,
  raw HDR did worse than SDR (26.3 against 31.0 mAP) [S7]. No primary source tests a **frozen**, sRGB-pretrained
  backbone with a linear probe fed PU, PQ or log input. The nearest result is SDR-pretrained IQA networks used
  without fine-tuning: good on PU21-encoded SDR, weak on PU21-encoded HDR [S12].
- **The dataset already answered half the question.** ADR 0041 put the gain map's extra light into the photos. The
  retrained classifier then moved by −0.0009 on test, inside the seed range. In the bean region the boost is about
  0.43 stops ([S18]: ML-29 P8). *Corrected by measurement (§6, experiment A):* the boosted light does **not** stay
  below the tone map's knee everywhere. In the median gain-map photo 1.5% of bean pixels are above the knee and 0.3%
  above white; 72 of 206 photos have over 5% of the bean region above the knee, 33 over 5% above white, the iPhones
  most (9–29% above the knee in the median photo of each model). The tone map is the identity on most bean pixels,
  not on all, and only bean pixels reach the classifier (the crop fills the rest with grey).
- **PU21** is one closed-form curve, V = p7·(((p1 + p2·Y^p4)/(1 + p3·Y^p4))^p5 − p6), clamped at 0. Y is absolute
  luminance from 0.005 to 10 000 cd/m². There are four parameter sets, and the authors recommend only
  `banding_glare` [S1, S2]. 100 cd/m² maps to about 256 and 10 000 cd/m² to about 595 [S2]. PU21 is **not** fitted to
  sRGB: only its scale is set, so that 0.1–100 cd/m² spans 256 steps [S1 p.4]. The older PU08 was fitted to sRGB
  [S3]. Computed here, an sRGB image on a 100 cd/m² display, PU21-encoded and divided by PU(white), sits **+23 levels
  above its own sRGB codes on average, up to +36 in the midtones**. HLG differs by +10 (max 20), PQ by +47.
- **Where PU21 could sit.** Only at the model input. That changes every photo's input, SDR photos included, by far
  more than D27's tolerance of 1 level, so it is **D2 (b): both models retrained**. It is not compatible with D2
  (a). PU21 has no part in D1 (how the store keeps light above white). That needs PQ, HLG, a float/linear JPEG XL or
  a `jhgm` gain map, and the pinned libjxl can write all of them except a ready-made gain-map bundle (§4).
- **Recommendation:** decide D1 and D2 (a) without PU21. If HDR input for the models is ever wanted, run PU21 as a
  separate, recorded experiment: frozen head only, crops only, PU21 at 100 cd/m² as SDR white, "255" scaling. Pass it
  only if it beats today's sRGB read-back by the ADR 0002 bar. §5 lists the runs.

## 1. Do CV models gain from HDR information?

What each study fed the network, against what, and what came out.

| Study | Task, data | HDR / RAW input | Baseline | Result | Conditions |
|---|---|---|---|---|---|
| Mukherjee, Bessa, Melo-Pinto, Chalmers 2021 [S5] | detection; SDR dataset converted to HDR; real out-of-distribution (OOD) HDR test set | detectors trained on HDR (**encoding not verified**: full text not reachable) | the same detectors trained on SDR | "10–12% more accuracy" (abstract) | high-contrast scenes "under extreme lighting conditions" only |
| Mukherjee, Melo, Filipe, Chalmers, Bessa 2020 [S6] | detection, Faster R-CNN / SSD512 / R-FCN pretrained on VOC/COCO, **not retrained**; 1,289 HDR images, ≈18 stops average (p.6) | HDR tone-mapped to 8-bit sRGB by 7 operators | the middle exposure | all but one operator "outperform the traditional middle (0th) exposure" (p.9); the mAP table is an image, not extracted | extreme scenes; HDR used only through a tone map into the frozen detector's sRGB |
| Koçdemir (MSc thesis, METU) 2021 [S7] | Faster R-CNN **from scratch**, Cityscapes 16-bit | HDR linear; HDR with γ 2.2; log; 7 tone maps | best-exposed LDR | AP@0.5 55.1 (LDR) to 56.1 (HDR + γ); mAP 33.1 against 33.3 (Table 3.1, p.42). No luminance, dynamic-range or entropy band where HDR is better (p.43) | ordinary driving scenes. "We do not observe a strong benefit of using HDR content" |
| Koçdemir 2021, ch. 4 [S7] | RetinaNet, **COCO-pretrained**, fine-tuned on the 2020 OOD set | HDR linear; HDR + γ | standard LDR; tone maps | HDR 26.3, HDR + γ 29.8, standard LDR 31.0, Mantiuk TMO 31.3 mAP (Table 4.2, p.61). "When the detection network is pre-trained with LDR images … the gap … widens" (p.23) | pretrained on SDR: linear HDR loses about 5 mAP; a γ curve recovers most of it |
| Ljungbergh et al. 2023 [S8] | Faster R-CNN R50-FPN, PASCALRAW (4,259 daylight images) | RAW RGGB; RAW + learnable γ; RAW + learnable Yeo-Johnson | rawpy RGB | RGB 50.5 ± 0.5 AP; RAW alone 31.3; learnable γ 51.4; Yeo-Johnson 52.6 ± 0.4 (Table 1, p.8) | linear input loses 19 AP. A learned curve gains about 2 AP, credited to dark regions (p.8). Pretraining not stated (**unverified**) |
| Buckler, Jayasuriya, Sampson 2017 [S9] | 8 tasks incl. CIFAR-10 ResNets, fine-tuned from pretrained weights on each pipeline | ISP stages removed one at a time | full ISP | Gamma compression and demosaicing are "the only important stages"; without gamma the data revert to a log-normal distribution (p.4) | the curve matters; colour transforms barely do (ResNet44 6.3% → 6.6% without gamut mapping) |
| Cui, Harada 2024 (RAW-Adapter) [S10] | RetinaNet R18/R50, PASCALRAW normal / dark / over-exposed (synthetic, ×2.5–3.5) | RAW + learnable input/model adapters, whole network trained | demosaiced RAW; ISPs | R50: normal 89.7 against 89.2; over-exposed 89.5 against 88.8; dark 86.6 against 82.6 (Table 2, p.12). Training from scratch on RAW "significantly impair[s]" results (Fig. 2, p.3) | +0.5 mAP in normal light, +4 in the dark; sRGB pretraining is what makes RAW workable |
| Onzon, Mannan, Heide 2021 [S11] | automotive detection, 1,600 HDR pairs | learned auto-exposure trained end to end | classical auto-exposure | +2.3 to +6.6 mAP as the exposure shift grows ×1.5 to ×10 (Table 1, p.6). An LDR-pretrained detector not fine-tuned: 13–20 mAP against 21–32 fine-tuned (Table 2) | the gain grows with exposure error |
| Chubarau et al. 2024 [S12] | IQA, UPIQ; PieAPP (VGG) and VTAMIQ (ViT), **pretrained on sRGB** | PU21 `banding_glare`, display 100 cd/m², black 0.5 | the same networks without fine-tuning | without fine-tuning: SDR SROCC 0.87–0.91, HDR 0.60–0.76. Fine-tuned on PU-encoded SDR: HDR up to 0.81 (Table 2, p.5) | closest to our case. A network pretrained on sRGB "is not expected to produce reliable predictions on PU-encoded HDR data" (p.3) |
| Ke, …, Mantiuk 2023/25 [S13] | restoration: denoise, deblur, super-resolution | linear; PQ, PU21, μ-law per RGB channel, 0.005–4,000 cd/m², scaled to 0–1 | linear + L1 | +1.5 to +9 dB for an encoded input; "the differences in performance between the three transforms are mostly small" (§7) | trained per encoding; pretraining not stated |

What the table says, for this project:

1. **Linear light into an sRGB-trained network is worse than SDR.** The loss was 19 AP on RAW [S8] and 5 mAP with a
   pretrained detector [S7]. Studies call a compressive curve (γ, log, PQ, PU) the one stage that must stay [S9]. The
   models here never see linear light, so this argues only against a linear read-back.
2. **Gains need extreme light.** Mukherjee et al. [S5, S6] chose high-contrast scenes on purpose (≈18 stops); Onzon's
   gain grows with the exposure error [S11]; RAW-Adapter's is +4 mAP in the dark and +0.5 in normal light [S10]. Under
   ordinary light, HDR or RAW input was within about 1 mAP of SDR [S7, S8, S10]. The coffee photos are close-ups of
   beans under kitchen light. ML-29 measured a 0.43-stop map boost on beans [S18]; experiment A (§6) found a
   minority of photos, mostly iPhones, with bean highlights above the tone map's knee.
3. **Where HDR won, the network adapted.** Every gain above came with training: from scratch, fine-tuned, a learned
   ISP, or adapters. The two frozen cases are Mukherjee 2020 [S6] and Chubarau's no-fine-tuning rows [S12]. Mukherjee
   2020 wins by tone-mapping HDR **into** sRGB for the frozen network, which is what canonicalization does today. The
   Chubarau rows show a sRGB-pretrained network losing accuracy once PU values go above the SDR range.
4. **No study tests our exact case**: a frozen ViT with a linear probe on PU, PQ or log input, for fine-grained
   classification. **Unverified** by any source found.

DINOv3's own pretraining applied ColorJitter (brightness 0.4, contrast 0.4, saturation 0.2, hue 0.1, p = 0.8) and
grayscale (p = 0.2) to sRGB images [S14]. The backbone has therefore seen global tone changes within ±40% of
brightness, but only on values in the sRGB range. Whether that makes its features tolerant of PU21's midtone lift
(§2) is an inference, **unverified**.

## 2. PU21 and PU08

**PU08** (Aydın, Mantiuk, Seidel, SPIE HVEI 2008) [S3]
- Maps luminance from 10⁻⁵ to 10⁸ cd/m² to "approximately perceptually uniform code values" (p.1). It is built by
  the recursion f_i = f_{i−1}(1 + t(f_{i−1})) on CSF thresholds t (eq. 7, p.6).
- **Fitted to sRGB**: s·PU(L) + m against sRGB luma, least squares over 0.1–80 cd/m² (eq. 8, p.6). It is stored as a
  look-up table, not a formula (p.6).
- Validated as interchangeable with sRGB on a CRT-like display: less than 1 dB PSNR and less than 0.01 SSIM apart
  (p.8). Values below about 0.5 cd/m² go negative [S1 p.4].

**PU21** (Mantiuk, Azimi, PCS 2021) [S1], official code `gfxdisp/pu21` at `78340c0` [S2]
- Derived from a CSF measured from 0.0002 to 10 000 cd/m². That CSF loses sensitivity above 100 cd/m², which the
  CSFs behind PQ and PU08 do not (p.2–3).
- It models banding artefacts, not CSF peaks, and optionally glare at L_g = 0.5 cd/m²:
  S_g(L) = S(L + L_g)·L/(L + L_g) (eq. 3, p.4). P(L) is integrated numerically on log L (eq. 2, p.3), from 0.005 to
  10 000 cd/m², then fitted with one invertible formula (p.4).
- **Anchoring** (p.4): "the PU values are rescaled by a constant, so that the range of luminance between 0.1 cd/m² and
  100 cd/m² is mapped to the 256 steps". Only the scale is matched to sRGB, not the curve.
- Results on UPIQ (380 HDR pairs): `banding_glare` gives the best correlation for PSNR, VSI, FSIM and MS-SSIM, and is
  comparable to PU08 for SSIM. It beats PQ and PU08 (Fig. 6, p.4–5).
- Code [S2]: `matlab/pu21_encoder.m`.
  - L26–27: `L_min = 0.005`, `L_max = 10000`.
  - L49–55: the four parameter sets.
  - L66–71: "100 [nit] is mapped to 256 to mimic the input to SDR quality metrics"; Y "MUST be scaled in the absolute
    units (nits, cd/m^2)".
  - L78–80: clamp, then `V = max(p7*(((p1 + p2*Y.^p4)./(1 + p3*Y.^p4)).^p5 - p6), 0)`.
  - L92–94: the inverse.
  - `README.md` L10: "Please use only the `banding_glare` variant."
  - BSD-3-Clause.

| Variant (`pu21_encoder.m` L49–55) | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| banding | 1.070275272 | 0.4088273932 | 0.153224308 | 0.2520326168 | 1.063512885 | 1.14115047 | 521.4527484 |
| **banding_glare** (default, L43) | 0.353487901 | 0.3734658629 | 8.277049286e-05 | 0.9062562627 | 0.09150303166 | 0.9099517204 | 596.3148142 |
| peaks | 1.043882782 | 0.6459495343 | 0.3194584211 | 0.374025247 | 1.114783422 | 1.095360363 | 384.9217577 |
| peaks_glare | 816.885024 | 1479.463946 | 0.001253215609 | 0.9329636822 | 0.06746643971 | 1.573435413 | 419.6006374 |

Values computed here from that equation (PU units; PQ shown as 256 × its signal for comparison):

| cd/m² | 0.005 | 0.1 | 1 | 10 | 26 (18% grey at 203) | 100 | 203 | 406 | 1000 | 4000 | 10 000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banding_glare | 0.0 | 5.7 | 36.5 | 123.6 | 174.4 | 256.4 | 303.8 | 352.9 | 420.1 | 527.5 | 595.4 |
| PQ × 256 | 3.9 | 16.0 | 38.4 | 76.7 | 97.3 | 130.1 | 148.7 | 167.5 | 192.5 | 231.1 | 256.0 |

So 0.1–100 cd/m² spans 250.7 steps, not exactly 256. One stop above SDR white adds about 46–48 PU (about 18% of
the SDR range). A photo whose light reaches 4× white uses 1.37× PU(white).

**How PU21 is meant to be used with SDR-trained metrics** [S1 §II-A, S2, S4]
- **Display model first.** L = (L_peak − L_black)·f(V) + L_black + L_amb (eq. 1, p.2). For HDR content: tone-map to
  a particular HDR display, then the display model, then PU (Fig. 1, p.2).
- The code's gain-gamma-offset model (`pu21_display_model_gog.m`):
  - defaults (L41–59): contrast 1000:1, γ 2.2, ambient 0 lux, k_refl 0.005;
  - L77: L = (Y_peak − Y_black)·V^γ + Y_black;
  - L82–83: Y_black = E_amb/π·k_refl + Y_peak/contrast.
- The SDR example (`examples/ex_sdr_images.m` L14–19) uses 100 cd/m², 1000:1 and 10 lux. The HDR example
  (`ex_hdr_images.m` L9–14) scales the image's own peak to a 4000 cd/m² display.
- Mantiuk 2016 [S4 §2.3, p.3]: relative HDR should be mapped to a target display, "in the simplest case … so that the
  peak image value is mapped to the display peak luminance".
- Chubarau et al. used 100 cd/m² and a black of 0.5 cd/m² (200:1) for SDR [S12 p.3].

**Luminance or per channel.**
- The paper encodes "absolute linear RGB or luminance" [S1 p.1].
- The code applies PU21 per RGB channel for colour metrics (PSNR, FSIM, VSI) and to luminance (BT.709 weights,
  `pu21_metric.m` L108–115, L138) for SSIM and MS-SSIM.
- Mantiuk 2016 [S4 §2.4, p.3]: PU-PSNR on RGB "may perform substantially worse than for luminance alone", because RGB
  metrics do not separate luminance distortions from chroma distortions.
- For a network input that must keep colour, per-channel is the only form used in the CNN papers [S12, S13]. Per
  channel, a curve on R, G and B changes saturation as an sRGB curve does; it is not hue-preserving.

**PU21 against the other curves, for an SDR picture.** Computed here: sRGB codes V → linear → each curve, scaled so
that SDR white is 1, then compared with 255·V. Lower is closer to what DINOv3 was pretrained on.

| Curve, white placement | Mean shift (levels) | Worst | V = 0.2 / 0.46 / 0.8 → | 2× and 4× white → |
|---|---|---|---|---|
| sRGB (today) | 0 | 0 | 51 / 117 / 204 | clipped at 1 |
| HLG OETF, 75% = white (BT.2408) | +10.3 | 20.0 | 55 / 128 / 220 | 1.18, 1.35 |
| PU21 banding_glare, white 100 cd/m², black 0.1 | +23.3 | 35.9 | 75 / 153 / 223 | 1.18, 1.37 |
| PU21 banding_glare, white 203 cd/m², black 0.2 | +31.0 | 46.5 | 89 / 163 / 226 | 1.16, 1.33 |
| PQ, white 203 cd/m², ÷ PQ(203) | +47.2 | 69.3 | 118 / 180 / 232 | 1.13, 1.26 |
| log₁₀, 10⁻³–1 | +53.0 | 82.0 | 129 / 192 / 236 | 1.10, 1.20 |

- Even with the best affine scale and offset, PU21 differs from sRGB by 10.6 levels RMS, worst 22.7 (100 cd/m²).
- All four HDR curves give about 16–37% more range per doubling above white, so none buys much headroom.
- PQ and log were designed as storage or coding curves; log has no roll-off at low luminance [S1 p.2].
- HLG is relative and scene-referred, and is the nearest to sRGB here. Its backward compatibility is BT.2100's design
  aim. That aim is **unverified** from the Recommendation itself; it is consistent with the numbers.
- PU21 is the only curve here validated as an input to SDR-designed metrics on HDR [S1, S12]. That is an IQA result,
  not a recognition result.

## 3. Where PU21 would sit in this pipeline

**Today** (`coffeecv/canonical.py`, `gain_map.py`, `tone_map.py` docstrings):
- original → LittleCMS → linear Rec.2020, with 1.0 = SDR white (203 cd/m², BT.2408);
- the gain map applied there, or a PQ/HLG source linearised there;
- the BT.2408 EETF on max(R, G, B) back into [0, 1];
- Rec.2020, BT.709 transfer, 16-bit JPEG XL;
- `load_canonical` → 8-bit sRGB, clipped;
- crops → `model_io.eval_patch`: /255, ImageNet mean and std.

Light above white exists only between gain-map application and the tone map.

- **(a) The store keeps light above white; the read-back applies PU21 instead of tone map + sRGB.** This is D1 (a) or
  (b) plus D2 (b). Every photo's model input changes, SDR photos included, by +23 levels on average (§2 table). That
  is far outside D27's "mean ≤ 1.0". So it means retraining the classifier head, the OOD probe (fit on the same
  features, ADR 0005/0016) and, if the segmenter reads it too, the segmenter.
- **(b) The store stays as it is (standard range, ADR 0041); PU21 at the model input only.** This keeps nothing above
  white: PU21 of an already tone-mapped [0, 1] picture is only a different tone curve. It is the same retrain as (a)
  with no HDR information gained. Pointless, except as the control run for (a).
- **(c) PU21 as an extra channel or a training-only alternative.** A frozen DINOv3 takes exactly 3 channels; a fourth
  means a new patch embedding, so the backbone is no longer frozen (against ADR 0006). As training-only augmentation
  it changes nothing at serving, and the store would need the HDR light anyway.

**The choices inside (a)**, if it were tried:
- **Absolute luminance of 1.0.** PU21 needs cd/m² [S2 L66–71].
  - **100 cd/m²** puts SDR white at PU 256, as PU21 and Chubarau's "255" scaling intend [S1 p.4, S12 p.3]. It
    minimises the change to SDR photos (+23 against +31 levels) and keeps headroom in values above 1.
  - **203 cd/m²** is what 1.0 *means* in the store (BT.2408, `gain_map.py`, `tone_map.py`). Mapping 1.0 to 203 cd/m²
    is right for storage but shifts SDR content further from sRGB under PU21.
  - **The photo's own headroom as peak** (PU21's HDR example, 4000 cd/m² = the image's maximum) makes each photo's
    SDR white land at a different PU value, set by its brightest highlight. For a classifier this is a per-photo
    brightness change driven by the background. Not recommended.
  - So, for the model input: 1.0 → 100 cd/m², a display black of about 0.1 cd/m² (1000:1, [S2]) and no reflections.
    That choice is a recommendation, **unverified** by any recognition study.
- **Scaling into DINOv3's range.**
  - "255" scaling (PU/255): SDR stays in [0, 1] and HDR goes to about 2.3. Pretrained networks are "not expected to
    produce reliable predictions" on the part above 1 [S12 p.3].
  - "Pmax" scaling (PU/595): everything is in [0, 1], but SDR is compressed to about 0.43 of the range.
  - In Chubarau's study, Pmax won on HDR and 255 won on SDR (Table 2, p.5). *Corrected 2026-11-23
    ([research_pu21_papers.md](research_pu21_papers.md)):* Pmax won on the HDR subset also for the networks never
    trained on PU input (PieAPP 0.63 against 0.60, VTAMIQ 0.76 against 0.71), not only after fine-tuning; 255 won on
    the SDR subset in all four comparisons and tied or won on the full set. With a frozen
    backbone only the head adapts, which favours "255", where SDR photos stay near the pretrained distribution.
  - Then ImageNet normalisation as today. A value of 1.37 (4× white) becomes (1.37 − 0.485)/0.229 ≈ 3.9σ, outside
    anything DINOv3 saw [S14].
- **Colour.** Apply per channel on linear Rec.2020 converted to linear sRGB primaries first, so that the three
  channels mean what DINOv3's channels meant. Out-of-sRGB values are negative there and must be clipped, as today.
  PU21 on luminance with chroma carried separately would give the backbone a picture whose channels mean something
  it never saw.
- **SDR photos (no map, 1,085 of 2,772).** Under PU21 they would no longer look like sRGB: midtones lift by 20–36
  levels (V 0.2 → 75; V 0.46 → 153). Beans are dark, so their crops move most. ADR 0041's gain-map brightening of
  +15 to +25 levels on beans left accuracy unchanged after a retrain [S18]. A shift of similar size is plausible to
  tolerate after a retrain, but not without one.

**D2.** PU21 is incompatible with D2 (a) by construction. Under D2 (a) the read-back must reproduce D33's EETF and the
sRGB transform, which is the ticket's R1 and does not involve PU21. PU21 belongs to D2 (b) only.

## 4. Storage facts for D1

- **imagecodecs 2026.10.10** (pinned, `uv.lock`) bundles libjxl 0.12.0 (`libjxl-…so.0.12.0` in the installed wheel;
  [S19] README L96).
  - `jpegxl_encode` takes `transfer=` LINEAR, SRGB, PQ, HLG, BT709, GAMMA, DCI and float32/float16 input; "Float must
    be in nominal range [0, 1] unless perceptualheuristics is False" ([S15] `_jpegxl.pyx` L79–89, L153–176, L427–445).
  - It sets no intensity target, so libjxl's default applies: 10 000 for PQ, 1000 for HLG, 255 otherwise ([S16]
    `lib/jxl/luminance.cc` L13–24).
  - It has no box API ([S15] L678: "TODO: decode from ISOBMFF container").
  - Checked here: a float32 linear BT.2100 picture with values up to 4 round-trips with `perceptualheuristics=False`,
    and uint16 PQ and HLG encode and decode.
- **The repo's own `coffeecv/jxl_encode.py`** calls the bundled libjxl through ctypes. The wheel's libjxl exports
  `JxlEncoderAddBox`, `JxlEncoderUseBoxes`, `JxlDecoderSetDecompressBoxes` and `JxlDecoderSetDesiredIntensityTarget`.
  It does **not** export `JxlGainMapWriteBundle`, `ReadBundle` or `GetBundleSize`; those are declared in
  `lib/include/jxl/gain_map.h` at v0.12.0 [S16] (symbol check, this machine [S17]).
  - So a `jhgm` box (ISO 21496-1 metadata + an optional colour encoding + a naked JPEG XL codestream for the map, per
    `gain_map.h` L25–63) would be serialised by our own code and added with `JxlEncoderAddBox`.
  - libjxl 0.12.0's decoder does not apply a gain map. `decode.h` has no gain-map rendering, so the read-back would
    apply it with `gain_map.py`'s maths.
- **intensity_target.** For PQ, libjxl's linear 1.0 is the intensity target, not 203 cd/m² (prior note §2).

## 5. Implications for ML-34

**Recommendation (not a decision).**
1. **Keep PU21 out of D1 and D2 (a).** If the owner wants the bright preview with D2 (a), the HDR light goes into the
   store as PQ, HLG, float or `jhgm`. The models' read-back reproduces today's EETF + sRGB exactly, and R1 is
   measured as the ticket says. PU21 adds nothing to that path.
2. **Do not expect accuracy from HDR input on this dataset.**
   - The literature's gains come from extreme contrast or exposure error, and from training the input path [S5–S13].
   - The coffee crops are dark beans under ordinary light; most of their gain-map light is below the knee, but a
     third of gain-map photos have over 5% of the bean region above it (§6, experiment A).
   - ADR 0041's own A/B on the same light was a null result.
3. **If HDR input is still wanted, test it under D2 (b) as a recorded experiment** (`docs/agents/experiments.md`,
   ADR 0002's paired multi-seed bar):
   - **E0, R1 check:** the read-back of an HDR-keeping store against today's canonical photos, D27's metric, all
     1,687 gain-map photos plus the SDR ones (needed for any D1).
   - **E1, bean-region headroom:** the share of bean-mask pixels with max(R, G, B) > 1 after the map, per camera,
     and their 99.9th percentile. If it is near zero, stop: there is nothing for PU21 to keep where the classifier
     looks.
   - **E2, the curve alone:** today's canonical photos (standard range) read back through PU21 (1.0 → 100 cd/m²,
     black 0.1, per channel in linear sRGB primaries, "255" scaling, clipped at 1). The frozen head is refit with
     three seeds, paired against today's three. This isolates the cost or benefit of the curve with no HDR content.
   - **E3, PU21 with headroom:** as E2, but from scratch canonical photos made without the tone map (linear float,
     values above 1 kept). Run it unclipped, and also with a "Pmax" scaling (PU/PU(max headroom)). Compare with E2 on
     the gain-map photos only and on all photos.
   - **E4, control curves:** the same as E3 with HLG (75% white), the nearest HDR curve to sRGB (§2), and with
     today's EETF. This separates "HDR helps" from "PU21 helps".
   - Also report: per-class F1, the OOD probe's AUROC on `analysis/ml29` p11's set, and the segmenter left on sRGB
     (its fine-tune is ML-33's).
   - The pass bar is fixed before the runs. A null result in E3 against E2 and E4 settles it on this dataset.

## 6. Measured here (2026-11-22 to 2026-11-23, checkout `a44c635`)

Two experiments for §5's E1–E3, run on this machine (12 CPUs, no GPU). Both are diagnostics, not recorded runs
(`docs/agents/experiments.md`): nothing is archived to `experiments/` and nothing is adopted from them.

### Experiment A (E1): light the tone map changes, inside the bean mask

`python -I analysis/ml34/a_light_above_white.py /workspace` → [a_light_above_white.txt](a_light_above_white.txt).
Every segmenter mask of `data/seg_masks_ml19/xl0_v7r_s123` whose original carries a gain map (206 measured; 40
empty masks, 1 map not applied). The original decoded as the store decodes it (`canonical.decode`, map applied), the
light read **before** the tone map; the peak and knee are D33's (`tone_map.picture_peak`, `eetf_knee`).

| | median photo | worst photo |
|---|---|---|
| picture peak (× white) / EETF knee (× white) | 1.70 / 0.76 | |
| bean pixels above the knee (changed by the tone map) | 1.5% | 61% |
| bean pixels above white (lost to an SDR canonical photo) | 0.3% | 33% |
| bean max(R, G, B), 50th / 99th percentile (× white) | 0.24 / 0.84 | |
| bean luminance, 50th / 99th percentile | 41 / 142 cd/m² (white = 203) | |

- 72 of 206 photos have over 5% of the bean region above the knee; 33 over 5% above white.
- By camera (median photo, above the knee): Pixel 9 Pro 0.5% (97 photos), OnePlus 7 Pro 1.3% (70), Galaxy Z Fold8
  Ultra 5.6% (11), iPhone 15 9.4% (9), iPhone 15 Pro Max 20% (10), iPhone 16 Pro Max 29% (4).
- The knee sits well below white: 0.81 at a peak of 1.5×, 0.60 at 2.7× (Apple's median headroom), 0.49 at 4×. The
  EETF compresses bean highlights that are below white too.
- Only bean pixels reach the classifier: `segment_beans.mask_and_crop` fills everything outside the mask with the
  ImageNet-mean grey (`segment_beans.py` L126). Light above white elsewhere in the photo (windows, lamps, the tray)
  can matter to the segmenter, never to the classifier.

### Experiment B (E2, E3): the frozen head on PU21-encoded crops

`python -I analysis/ml34/b_pu21_probe.py /workspace build|probe|report`. Each arm is a scratch pool
(`outputs/ml34_b/<arm>/pool`, gitignored) with the tracked pool's names, so the same split and the same patch boxes;
the arms differ in their crops' pixels alone. The head is fit by `fit_frozen_head`'s own pieces (`frozen_config`,
`build_fold_datasets`, `embed`, `fit_arm` at C = 0.1), **unaugmented only** (no views: ~9,200 patches an arm and
seed, about an hour each at 6 threads), seeds 42, 123, 7. The pool includes ML-36's photos, so the numbers are not
comparable with exp286–291's.

PU21 as §3 recommends: `banding_glare`, per channel on linear sRGB primaries, 1.0 (SDR white) = 100 cd/m², display
black 0 (PU21's own floor of 0.005 cd/m², not §3's 0.1), scaled so that `top` maps to 255 and stored as 8-bit crops
(so clipped at `top`). The fill grey is put back outside the mask after encoding.

- **base**: today's crops.
- **pu21_sdr** (E2, the curve alone): today's crops through PU21, top = white ("255" scaling).
- **pu21_h4**: today's crops through PU21 with 2 stops of headroom reserved, top = 4 × white (SDR white lands at
  PU(100)/PU(400) ≈ 0.73 of the range). Nothing above white is in these crops: the cost of reserving headroom alone.
- **pu21_h4_hdr** (E3): as pu21_h4, but the 1,288 crops of photos with an applied gain map are cut, at the crop's
  recorded box (`segcrop_report.json`), from the original's light decoded with its map and **no tone map**; the
  other 523 crops are pu21_h4's bytes. Check: that light rendered as standard range matches the stored crop on calm
  bean pixels to 0.9–2.0 of 255 levels (the store's lossy coding).
- **pu21_h4_tm** (the control for pu21_h4_hdr): the same originals and boxes, tone-mapped by D33's EETF at the
  picture's peak, as canonicalization does, then pu21_h4's curve.

Patch macro-F1, unaugmented head (val / test):

| arm | s42 | s123 | s7 |
|---|---|---|---|
| base | 0.9451 / 0.9486 | 0.9406 / 0.9458 | 0.9387 / 0.9285 |
| pu21_sdr | 0.9370 / 0.9441 | 0.9425 / 0.9419 | 0.9403 / 0.9253 |
| pu21_h4 | 0.9395 / 0.9396 | 0.9392 / 0.9377 | 0.9396 / 0.9228 |
| pu21_h4_hdr | 0.9439 / 0.9448 | 0.9399 / 0.9401 | 0.9422 / 0.9259 |
| pu21_h4_tm | 0.9377 / 0.9409 | 0.9399 / 0.9395 | 0.9371 / 0.9241 |
| pu21_pmax_hdr | 0.9421 / 0.9422 | 0.9325 / 0.9441 | 0.9420 / 0.9238 |

Paired by seed:

- **pu21_sdr − base**: val −0.0016 (−0.0081 to +0.0019), test −0.0038 (−0.0045 to −0.0031). The PU21 curve in
  place of sRGB costs the frozen backbone a little on test, at every seed.
- **pu21_h4 − base**: val −0.0020 (−0.0056 to +0.0009), test −0.0076 (−0.0090 to −0.0057). Reserving headroom
  costs about twice as much:
  every SDR value is compressed into the lower 73% of the input range.
- **pu21_h4_hdr − pu21_h4**, the HDR information with the curve held fixed: val +0.0044, +0.0006, +0.0026; test
  +0.0052, +0.0025, +0.0031. **Positive at every seed, mean +0.0025 val, +0.0036 test**, but smaller than the seed
  range of either arm (test 0.9228–0.9396).
- **pu21_h4_hdr − pu21_h4_tm**, the HDR light alone (the same originals, the same boxes, the same curve; only the
  tone map differs): val +0.0061, −0.0001, +0.0051 (mean +0.0037); test +0.0039, +0.0007, +0.0018 (mean +0.0021).
  Not negative at any seed, inside the seed range, and uneven: clear at s42 and s7, nothing at s123.
- **pu21_h4_tm − pu21_h4**, the lossless original against the store's lossy JPEG XL: val −0.0012, test +0.0015.
  So of pu21_h4_hdr's +0.0036 on test over pu21_h4, about +0.002 is the HDR light and +0.0015 the lossy coding.
- **pu21_pmax_hdr** (added 2026-11-23 after [research_pu21_papers.md](research_pu21_papers.md); as pu21_h4_hdr with
  PU21(10,000 cd/m²) = 255, Chubarau et al.'s full-range scaling): against base test −0.0043, val −0.0026; against
  pu21_h4_hdr test −0.0003, val −0.0031. Full-range scaling is no better than 4 × white here.
- **pu21_h4_hdr − base**: val +0.0005 (−0.0012 to +0.0035), test −0.0040 (−0.0057 to −0.0026). With the curve's
  cost included, HDR input is still below today's sRGB crops on test at every seed.

What it says, and what it does not:

1. **The curve costs more than the HDR light gives.** PU21 alone −0.004 on test; headroom −0.008; HDR light in that
   headroom gives back +0.004. Nothing here beats today's crops.
2. **The HDR light's own effect is about +0.002 on test, +0.004 on val**, once the control (pu21_h4_tm) takes out the
   lossless-original difference. Small, inside the seed range, and not at every seed (s123: none). It sits on top of
   a curve that costs −0.008: the net is still below today's crops.
3. **Unaugmented heads only.** The shipped head is augmented (8 views); colour jitter there could absorb or mask a
   curve's effect. One recorded augmented run is 3.3–4.8 h a seed (ML-29 P10).
4. **ADR 0002's bar is not met by any arm**, and was not fixed before these runs: they are a screen, not evidence for
   adoption.

## Unverified

- Mukherjee et al. 2021 [S5]: only the abstract was read (IEEE Xplore refused automated download). Its input encoding
  (PU, log or linear) and its exact metric are not confirmed. The task's attribution to Debattista did not hold; the
  authors are Mukherjee, Bessa, Melo-Pinto and Chalmers. Debattista co-wrote the exposure-compression method that
  [S7] uses for its LDR baseline.
- Mukherjee 2020's per-method mAP values are in images in the PDF and were not extracted.
- Ke et al. [S13]: read through the arXiv HTML (v3) by a fetch tool, not from the PDF. Its numbers above are as that
  reading gave them.
- Whether Ljungbergh et al.'s backbone was ImageNet-pretrained (Detectron2's default) is not stated in the text read.
- No study found that feeds PU, PQ or log input to a **frozen** ViT with a linear probe for classification. §3's
  choices (100 cd/m², "255", per channel in sRGB primaries) are reasoning, not evidence.
- BT.2100's backward-compatibility aim for HLG: not re-read for this note.

## Sources

PDF page numbers are the file's own pages.

- S1 Mantiuk, Azimi, "PU21: A novel perceptually uniform encoding for adapting existing quality metrics for HDR",
  PCS 2021: https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2021_PU21.pdf (eq. 1 p.2; §III p.2–4; eq. 2 p.3; eq. 3,
  rescaling, Fig. 5–6 p.4; conclusions p.5)
- S2 `gfxdisp/pu21`, commit `78340c0c4c20908c6bdcf0931dfde763c53a24ad` (2022-08-02):
  https://github.com/gfxdisp/pu21/tree/78340c0c4c20908c6bdcf0931dfde763c53a24ad
  - `matlab/pu21_encoder.m` L26–27, L43, L49–55, L62–95
  - `matlab/pu21_metric.m` L63–72, L83–99, L108–115, L138
  - `matlab/pu21_display_model_gog.m` L41–59, L77, L82–83
  - `matlab/examples/ex_sdr_images.m` L14–19
  - `matlab/examples/ex_hdr_images.m` L9–14
  - `README.md` L8–10
- S3 Aydın, Mantiuk, Seidel, "Extending quality metrics to full luminance range images", SPIE HVEI 2008, 68060B:
  https://www.cl.cam.ac.uk/~rkm38/pdfs/aydin08eqmflri.pdf (p.1, p.6 eq. 7–8, p.8)
- S4 Mantiuk, "Practicalities of predicting quality of high dynamic range images and video", ICIP 2016:
  https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2016prac_hdr_metrics.pdf (§2.3, §2.4, p.3)
- S5 Mukherjee, Bessa, Melo-Pinto, Chalmers, "Object Detection Under Challenging Lighting Conditions Using High
  Dynamic Range Imagery", IEEE Access 9:77771–77783, 2021, doi:10.1109/ACCESS.2021.3082293 (abstract via the
  OpenAlex and Semantic Scholar APIs; full text not read)
- S6 Mukherjee, Melo, Filipe, Chalmers, Bessa, "Backward Compatible Object Detection Using HDR Image Content", IEEE
  Access 8:142736–142746, 2020:
  https://wrap.warwick.ac.uk/id/eprint/140398/1/WRAP-Backward-compatible-object-detection-HDR-image-Chalmers-2020.pdf
  (p.1, p.6, p.8–9)
- S7 Koçdemir, "An analysis on the effect of dynamic range on object detection with deep neural networks", MSc
  thesis, METU, 2021: https://open.metu.edu.tr/bitstream/handle/11511/93209/10425385.pdf (p.23, Table 3.1 p.42,
  p.43, Table 4.2 p.61)
- S8 Ljungbergh, Johnander, Petersson, Felsberg, "Raw or Cooked? Object Detection on RAW Images", arXiv:2301.08965v2:
  https://arxiv.org/pdf/2301.08965v2 (Table 1 and §4.2–4.4, p.8)
- S9 Buckler, Jayasuriya, Sampson, "Reconfiguring the Imaging Pipeline for Computer Vision", ICCV 2017,
  arXiv:1705.04352v3: https://arxiv.org/pdf/1705.04352v3 (abstract p.1; §3–4, p.4)
- S10 Cui, Harada, "RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images", ECCV 2024,
  arXiv:2408.14802v1: https://arxiv.org/pdf/2408.14802v1 (Fig. 2 p.3; dataset p.9–10; Tables 2–3 p.12)
- S11 Onzon, Mannan, Heide, "Neural Auto-Exposure for High-Dynamic Range Object Detection", CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/papers/Onzon_Neural_Auto-Exposure_for_High-Dynamic_Range_Object_Detection_CVPR_2021_paper.pdf
  (Tables 1–2, p.6)
- S12 Chubarau, Yoo, Akhavan, Clark, "Adapting Pretrained Networks for Image Quality Assessment on High Dynamic Range
  Displays", arXiv:2405.00670v1: https://arxiv.org/pdf/2405.00670v1 (p.3 "Training with PU-encoded Data"; Table 2
  and normalisation, p.5)
- S13 Ke, Luo, Xiang, Fan, Ranjan, Chapiro, Mantiuk, "Training Neural Networks on RAW and HDR Images for Restoration
  Tasks", arXiv:2312.03640v3: https://arxiv.org/html/2312.03640v3 (§3, §5.1–5.2, §6, Table 2, §7; read through a
  fetch tool)
- S14 `facebookresearch/dinov3`, commit `6876159a11b4df116f30f667f8c9888617df0751`, `dinov3/data/augmentations.py`
  L121–137: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/data/augmentations.py
- S15 imagecodecs `imagecodecs/_jpegxl.pyx`, v2026.10.10 (L79–89, L153–176, L427–445, L678):
  https://github.com/cgohlke/imagecodecs/blob/v2026.10.10/imagecodecs/_jpegxl.pyx
- S16 libjxl v0.12.0:
  - `lib/include/jxl/gain_map.h` L25–63, L76–118
  - `lib/jxl/luminance.cc` L13–24
  - `lib/include/jxl/codestream_header.h` (`intensity_target`, `min_nits`)
  - `lib/include/jxl/encode.h` (`JxlEncoderAddBox`)
  - https://github.com/libjxl/libjxl/tree/v0.12.0
- S17 The pinned wheel's `imagecodecs.libs/libjxl-75330aa4.so.0.12.0`, exported symbols checked with ctypes on this
  machine, 2026-11-22.
- S18 This repo:
  - [ADR 0041](../../docs/adr/0041-a-photos-gain-map-is-applied-at-canonicalization.md) (−0.0009 on test)
  - [analysis/ml29/README.md](../ml29/README.md), "How much a gain map would brighten the bean region" and P8's
    report
  - [research_tone_mapping.md](../ml29/research_tone_mapping.md) §1–2 (BT.2100, BT.2408 white at 203 cd/m², libjxl's
    intensity target) and §5 (iii) (bean light below the EETF knee)
  - `coffeecv/canonical.py`, `gain_map.py`, `tone_map.py`, `model_io.py` docstrings
- S19 imagecodecs `README.rst` v2026.10.10 L96 (libjxl 0.12.0), as cited in
  [research_16bit_colour.md](../ml19/research_16bit_colour.md) S5.
