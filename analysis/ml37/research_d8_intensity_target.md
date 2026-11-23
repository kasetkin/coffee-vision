# ML-37 D8: the JPEG XL intensity target of the linear canonical photo

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D8 (and R2), 2026-11-23, on the local
checkout at `01e6ac1`. This is a recommendation for the owner, not a decision: D8 stays open until the owner picks an
option. Every number comes from the three scripts beside this note (`d8_intensity_target.py`, `d8_by_level.py`,
`d8_decoder_view.py`) and their printed results (`d8_*.txt`). The source reading is libjxl, jxl-rs, imagecodecs and
Chromium at the versions in §8, read in `tmp/ml37/d8/downloads/` (not committed).

## Summary

- **Recommended: (a) 203.** In libjxl and jxl-rs the intensity target of a linear (non-PQ) file is the luminance
  at which (1, 1, 1) is shown, not a peak (§1). D1 makes 1.0 reference white, and BT.2408 and coffeecv's
  `tone_map.REFERENCE_WHITE` put that at 203 cd/m². 255 is libjxl's default, the level its XYB was tuned at. No
  standard names it as a white.
- **The default is 255, and imagecodecs cannot change it.** `jpegxl_encode` has no parameter for it. The repo's
  ctypes encoder sets it with one field, and ML-37 needs that encoder anyway for half-float (§2).
- **It does change the lossy encode, through one path.** The XYB transform scales linear light by
  `intensity_target / 255` before quantization, and the decoder undoes it (§3). At 203, libjxl codes a picture 0.796×
  as bright. The quantizer, the AQ field and the AC strategy all see the dimmer picture. At these efforts nothing
  else reads the target.
- **Measured on 20 dataset photos (§4):** compared with 255, the file is 4.5% smaller in the store (q95/e7) and 6.1%
  smaller on the front (q90/e3). The read-back's 32×32 block-mean error rises 9–10%: from 0.055 to 0.060 sRGB levels
  in the store and from 0.110 to 0.121 on the front. The largest block goes from 3.15 to 3.52 in the store and stays
  at 6.06 on the front. That is the trade a lower quality setting makes: 203 at q95 sits on 255's curve near q94.6,
  and 203 at q90 near q88.8. At 255, linear half-float matches today's uint16 BT.709 encode (block mean 1.00×), so
  the 9% in R2 comes from the target, not from linear input. Everything stays at least 8× inside D27's tolerance
  (mean ≤ 1.0, no block over 6). The front's one block of 6.06 is there with either target, and also in today's
  encode (6.09).
- **By light level (§5):** 203 raises the relative error 5–8% from mid-grey up to 4× white and 10–19% in the deep
  shadows (below 1/64 of white). Light above white behaves like the mid-tones. The target has no other interaction
  with it in the coder.
- **Decoders (§6):** the float samples that come back do not depend on the target (measured). The target matters only
  to a decoder that renders to PQ (libjxl, jxl-rs), which puts 1.0 at the target: 203 vs 255 cd/m², and a photo's
  8× highlight at 1624 vs 2040. libjxl's own tone mapping ignores a linear file's target, and Chromium reads it only
  for PQ and HLG files.
- **If the owner wants today's error back at 203:** front q91 (0.995× the bytes, block mean 0.112) and store q96
  (1.085×, 0.050). Optional; not needed for D27.

## 1. What the intensity target is

**In the codestream.** `intensity_target` is a half-float field of the `ToneMapping` bundle in the image header,
in cd/m². Its default is 255. It is written only when the header's `extra_fields` is set, and a non-default tone
mapping sets that flag. Beside it are `min_nits`, `relative_to_max_display` and `linear_below` (libjxl
`lib/jxl/image_metadata.cc` L384–405, L292–345; jxl-rs `jxl/src/headers/image_metadata.rs` L150–162). Cost: 203
in place of the default adds `extra_fields` and the bundle, a few bytes.

**What the documentation says.** libjxl's public header calls it the "Upper bound on the intensity level present in
the image in nits. For unsigned integer pixel encodings, this is the brightness of the largest representable value
[…] An encoder is allowed to set 255 for SDR images without computing a histogram. Leaving this set to its default of
0 lets libjxl choose a sensible default value based on the color encoding." (`lib/include/jxl/codestream_header.h`
L130–138; the same text in `lib/jxl/image_metadata.h` L157–162). cjxl's `--intensity_target` repeats it
(`tools/cjxl_main.cc` L198–202). ISO/IEC 18181-1 is paywalled and was not read. libjxl is its reference software, and
jxl-rs, an independent decoder from the same project, implements the field the same way, so their code stands in for
the spec here.

**What the code does: the luminance of (1, 1, 1).** For every transfer that is not PQ, libjxl treats the intensity
target as the luminance at which the value 1.0 is shown. It is not used as a peak:
- The CMS interface: "for colorspaces where luminance is relative (essentially: not PQ), indicates the luminance at
  which (1, 1, 1) will be displayed […] intensity_target cd/m² in PQ should map to and from (1, 1, 1) in the
  relative one" (`lib/include/jxl/cms_interface.h` L86–98).
- The decoder's PQ output from any other transfer uses `TF_PQ(orig_intensity_target)` (`render_pipeline/stage_from_linear.cc`
  L56–66; `cms/transfer_functions-inl.h` L137–141; `cms/jxl_cms.cc` L175–190). The linear value 1.0 becomes
  `intensity_target` cd/m².
- The XYB encoder and decoder scale linear light by `intensity_target / 255` and back (§3).
- cjxl's EXR reader takes OpenEXR's `whiteLuminance`, the luminance of RGB (1, 1, 1), as the intensity target, and
  djxl's EXR writer writes it back the same way (`lib/extras/dec/exr.cc` L228–229, L519;
  `lib/extras/enc/exr.cc` L160).

For an integer file, whose values stop at 1.0, the two readings agree: 1.0 is the largest representable value. For
D1's float file, which keeps light above 1.0, they do not. The code's reading, "1.0 is shown at `intensity_target`
cd/m²", is the one any libjxl or jxl-rs decoder acts on (§6). Under D1, 1.0 is reference (diffuse) white, so the
intensity target is the luminance of reference white. BT.2408-8 §2.1 and Table 1 put HDR reference white at
**203 cd/m²** ([research_tone_mapping.md](../ml29/research_tone_mapping.md) §1). A "true" upper bound, 203 × the
photo's peak, would declare that 1.0 is shown at that luminance, so every reader would put the picture 2–8 times too
bright. It is not an option.

**255 is not a white.** libjxl's default for any non-PQ, non-HLG transfer is `kDefaultIntensityTarget = 255`
(`lib/jxl/base/common.h` L104; `lib/jxl/luminance.cc` L13–24: PQ 10,000, HLG 1000, "SDR" 255). It is the luminance at
which libjxl's XYB was tuned: at 255 the scaling in §3 is 1. No standard names 255 cd/m² as a reference white.
BT.2408 gives 203, and coffeecv uses 203 everywhere it gives light a luminance: `tone_map.REFERENCE_WHITE`, used by
`pq_to_relative` and by D33's EETF.

## 2. Who writes it: libjxl's default, imagecodecs and the repo's encoder

- **libjxl.** `JxlEncoderInitBasicInfo` sets `intensity_target = 0` (`lib/jxl/encode.cc` L1254).
  `JxlEncoderSetBasicInfo` keeps a non-zero value and marks it set (L1382–1390). Otherwise
  `JxlEncoderSetColorEncoding` calls `SetIntensityTarget`, which gives 255 for a linear transfer (L1191–1193;
  `luminance.cc`). Measured: `intensity_target = 0` writes a file that declares 255, the same size to the byte as one
  written with 255, with the same errors (§4, the "f16 IT=0" and "IT=255" rows, read back with
  `JxlDecoderGetBasicInfo`).
- **imagecodecs 2026.10.10** cannot set it. `jpegxl_encode` has no such parameter (`imagecodecs/_jpegxl.pyx`
  L153–176). It calls `JxlEncoderInitBasicInfo` and never touches the field (L308–445), so a linear float file
  from it always declares 255. Its decoder never calls `JxlDecoderSetDesiredIntensityTarget`. The function is
  declared in `libjxl.pxd` L548 and unused in `_jpegxl.pyx`.
- **The repo's ctypes encoder** (`coffeecv/jxl_encode.py`) already lays out `BasicInfo.intensity_target` (libjxl
  0.12.0's struct). Setting it is one assignment before `JxlEncoderSetBasicInfo`. `d8_intensity_target.encode` does
  exactly that and needs nothing else. So D8 (a) needs the repo's encoder, which ML-37 needs anyway for the float16
  form (§2 of the ticket), and D8 (b) is what imagecodecs writes.
- **The repo's header check.** `coffeecv/jxl_header.read_header` stops after the colour encoding and skips the
  `ToneMapping` bundle. A host that refuses a canonical photo with the wrong intensity target (R6) needs about ten
  lines more: read `tone_mapping` when `extra_fields` is set (`image_metadata.cc` L344–345), with a default of 255.

## 3. What it does to the lossy encode

**The XYB transform reads it.** With `uses_original_profile = false` (both settings), libjxl converts the input to
linear sRGB and then to XYB. `ComputePremulAbsorb` multiplies the opsin absorbance matrix by
`mul = intensity_target / 255.0f` (`lib/jxl/enc_xyb.cc` L214–231, called from `ToXYB` L233–288 with
`metadata->m.IntensityTarget()`, `enc_frame.cc` L1609). XYB is then built from `cbrt(M · (IT/255) · rgb + bias) −
cbrt(bias)`, with a bias of 0.00379 per channel (`cms/opsin_params.h` L36). The decoder divides it back out:
`InitSIMDInverseMatrix(…, orig_intensity_target)` scales the inverse matrix by `255 / intensity_target`
(`opsin_params.cc` L35–45; `dec_xyb.cc` L233–241; jxl-rs `render/stages/xyb.rs` L156, L429). So the samples that come
back do not depend on the target, but **the encoder codes the picture as if it were `IT/255` as bright**. At 203 the
light entering XYB is ×0.796. That is ×0.927 in XYB's cube-root range and up to ×0.796 near black, where the bias
makes the curve linear. Every quantizer, the adaptive-quantization field (`enc_adaptive_quantization.cc`, computed
from the XYB image) and the AC strategy then work on the dimmer picture at the same distance. A dimmer picture is
coded with coarser relative steps: fewer bytes and more error, more so in the shadows. §4 and §5 measure it. The
same applies to today's uint16 BT.709 encode: the scaling sits after the colour conversion, whatever the input.

**What does not read it, at these settings:**
- The butteraugli quantization loop takes `params.intensity_target` from the file only for PQ or HLG, and 80 for
  anything else (`enc_adaptive_quantization.cc` L949–953). It runs only at effort 8 and above
  (`speed_tier <= kKitten`, L1282; `common.h` L50–51; effort sets the tier as `10 - effort`, `encode.cc` L1682). Neither 7 nor 3 runs it.
- `JxlEncoderDistanceFromQuality` is a fixed map, 95 → 0.55 and 90 → 1.0 (`encode.cc` L1626–1631).
- The colour conversion into linear sRGB passes the target to the CMS, which uses it only for PQ and HLG
  (`cms/jxl_cms.cc` `BeforeTransform` L104–125, `ApplyHlgOotf` L888–892).
- `DownsampleImage2_Iterative` uses the default 255 (`enc_heuristics.cc` L754–755), but only for resampling, which
  neither setting uses.

## 4. Measured: size and error at the store's and the front's settings

`d8_intensity_target.py` (printed results in `d8_intensity_target.txt`) took a sample of 20 dataset originals:
seed 0, stratified by camera and by gain map, 8 with a gain map (peaks 1.48–3.18 × white) and 12 without. Each one was
decoded as the store decodes it, made into D1's light (linear Rec.2020, 1.0 = white, light above white kept) and
area-averaged 2× on each side to bound the encode time. Each was then encoded with libjxl 0.12.0 (the library in
imagecodecs' wheel) and read back as D2 reads it: matrix to linear sRGB, D6 (a)'s EETF at the photo's peak, clip, sRGB
curve, ×255 in float. The errors are against the source through the same read-back. "Block" means ML-29 D27's 32×32
block means. "Dark" means blocks whose mean luma is under 64 levels. Sizes are totals over the 20 photos at half
scale. The ratio is the mean per-photo ratio to IT=255, with its range in brackets.

| setting | variant | declared | MB | size vs 255 | block mean | block max | dark blocks | pixel RMSE |
|---|---|---|---|---|---|---|---|---|
| store q95/e7 | f16, IT=0 (unset) | 255 | 18.34 | 1.000 | 0.055 | 3.15 | 0.057 | 1.726 |
| | f16, IT=255 | 255 | 18.34 | 1.000 | 0.055 | 3.15 | 0.057 | 1.726 |
| | **f16, IT=203** | 203 | 17.56 | 0.955 (0.912–0.973) | 0.060 | 3.52 | 0.063 | 1.800 |
| | f16, IT=255 at q94 | 255 | 16.50 | 0.896 | 0.063 | 3.59 | 0.063 | 1.894 |
| | f16, IT=203 at q96 | 203 | 19.77 | 1.085 | 0.050 | 3.08 | 0.050 | 1.661 |
| | f16, IT=80 | 80 | 14.08 | 0.758 | 0.086 | 4.15 | 0.104 | 2.237 |
| | f16, IT=1000 | 1000 | 22.54 | 1.244 | 0.037 | 2.90 | 0.033 | 1.430 |
| | today: u16 BT.709 | 255 | 18.05 | 0.980 | 0.055 | 2.97 | 0.056 | 1.729 |
| | u16 BT.709, IT=203 | 203 | 17.28 | 0.936 | 0.060 | 3.59 | 0.062 | 1.806 |
| front q90/e3 | f16, IT=0 (unset) | 255 | 11.66 | 1.000 | 0.110 | 6.06 | 0.112 | 2.583 |
| | f16, IT=255 | 255 | 11.66 | 1.000 | 0.110 | 6.06 | 0.112 | 2.583 |
| | **f16, IT=203** | 203 | 10.99 | 0.939 (0.902–0.956) | 0.121 | 6.06 | 0.126 | 2.721 |
| | f16, IT=255 at q89 | 255 | 11.09 | 0.949 | 0.119 | 5.88 | 0.121 | 2.689 |
| | f16, IT=255 at q88 | 255 | 10.59 | 0.904 | 0.127 | 5.79 | 0.126 | 2.792 |
| | f16, IT=203 at q91 | 203 | 11.61 | 0.995 | 0.112 | 5.84 | 0.115 | 2.604 |
| | f16, IT=1000 | 1000 | 16.10 | 1.407 | 0.068 | 4.55 | 0.063 | 1.923 |
| | today: u16 BT.709 | 255 | 11.44 | 0.975 | 0.109 | 6.09 | 0.112 | 2.599 |
| | u16 BT.709, IT=203 | 203 | 10.78 | 0.916 | 0.121 | 5.92 | 0.126 | 2.738 |

Uncoded half-float alone (rounding only): block mean 0.001, largest block 0.05.

What the table shows:
- **Unset is 255.** The IT=0 and IT=255 files have the same size to the byte in all 40 photo × setting pairs and the
  same errors, and libjxl's decoder reads 255 from both.
- **203 trades error for size, as quality does.** Per photo, 203 costs 1.087× the block-mean error in the store
  (range 1.054–1.149) and 1.101× on the front (1.037–1.147). On dark blocks it costs 1.128× and 1.135×. Placed on
  255's own rate-distortion curve by linear interpolation between q94 and q95 (store) or q88 and q89 (front), 203 at
  q95 equals 255 at about q94.6 (0.0585 against the measured 0.060). 203 at q90 equals 255 at about q88.8 (0.1208
  against 0.121). Within this sample's resolution, the target is a quality offset and changes nothing else. The
  larger targets confirm it: 1000 and 10000 buy error with bytes along the same direction.
- **Linear half-float costs nothing against today's encode at the same target.** Per photo, f16 at 255 has 1.001×
  the block-mean error of today's u16 BT.709 file in the store (0.910–1.026) and 1.008× on the front (0.965–1.165).
  The whole of the 9–11% R2 would see at 203 is the target. Today's file moves the same way when given 203 (0.936×
  the size, 0.060), because the scaling happens after the colour conversion.
- **The gain-map photos alone** show the same ratios: 0.956× / 0.942× the size, block mean 0.053 → 0.057 (store)
  and 0.105 → 0.116 (front) (`d8_intensity_target.txt`, second table).
- **D27 is never at risk from the target.** The worst photo's block mean is 0.116 (store, 203) and 0.177 (front,
  203), against 0.109 and 0.163 at 255. One front photo has a block over 6 (6.06) with either target and in today's
  encode (6.09). That block belongs to the front's q90, not to D8.

## 5. Measured: where the error moves, by light level

`d8_by_level.py` (`d8_by_level.txt`) encoded the six gain-map photos with the highest peaks (2.0–3.0 × white) and two
without a map at 203 and 255. It compared the decoded linear luminance (BT.2100 weights, 4×4 block means) with the
source's, binned by the source's luminance in octaves. The table gives the median relative error.

| source Y (× white) | blocks | store 203 | store 255 | ratio | front 203 | front 255 | ratio |
|---|---|---|---|---|---|---|---|
| 1/1024–1/512 | 6,992 | 0.0447 | 0.0376 | 1.189 | 0.0612 | 0.0518 | 1.181 |
| 1/256–1/128 | 25,039 | 0.0141 | 0.0125 | 1.123 | 0.0226 | 0.0198 | 1.140 |
| 1/64–1/32 | 79,596 | 0.0061 | 0.0056 | 1.090 | 0.0111 | 0.0100 | 1.104 |
| 1/16–1/8 | 189,025 | 0.0035 | 0.0033 | 1.058 | 0.0065 | 0.0061 | 1.074 |
| 1/4–1/2 | 387,148 | 0.0022 | 0.0021 | 1.045 | 0.0040 | 0.0037 | 1.075 |
| 1/2–1 | 248,554 | 0.0018 | 0.0017 | 1.046 | 0.0031 | 0.0029 | 1.068 |
| 1–2 | 109,285 | 0.0013 | 0.0012 | 1.058 | 0.0024 | 0.0022 | 1.068 |
| 2–4 | 11,554 | 0.0010 | 0.0010 | 1.056 | 0.0018 | 0.0017 | 1.058 |

(All bins are in the file; four are left out here.) The pattern is §3's mechanism. Near black, the bias makes XYB
almost linear, so the ×0.796 dimming reaches the coded values in full and the error rises most (×1.12–1.19). Higher
up, the cube root compresses the dimming to ×0.927, and the error rises 5–8%. **Light above white is no exception:**
1–4 × white behaves like the mid-tones (×1.06), and libjxl neither clips nor treats it specially. XYB is float, and
the cube root carries values above 1 the same way. The target's only interaction with light above white is the
decoder's (§6).

## 6. Decoders and displays

`d8_decoder_view.py` (`d8_decoder_view.txt`) encoded flat grey patches at 0.18, 1, 2, 4 and 8 × white (q95/e7,
linear half-float) with each target. It decoded them with libjxl 0.12.0 four ways:

| decode | IT=203 file | IT=255 file |
|---|---|---|
| as stored, linear float (imagecodecs, the read-back) | 0.180, 1.001, 2.000, 3.997, 7.998 | 0.180, 1.000, 1.998, 3.994, 7.998 |
| as stored, desired intensity target 100 | same | same |
| to PQ, in cd/m² | 36.5, 203.2, 405.9, 811.3, 1624 | 45.9, 255.1, 509.4, 1019, 2040 |
| to PQ, desired intensity target 1000, in cd/m² | same | same |

- **The read-back does not see the target.** imagecodecs asks for the file's own linear encoding and never sets a
  desired target (§2). libjxl undoes the XYB scaling exactly (§3). The two files differ only by the coding noise.
  So D2's read-back, the models and the preview get the same numbers with either value, apart from the 4–6% in §4.
- **A decoder that renders to PQ puts 1.0 at the target.** This follows `stage_from_linear.cc` L56–66 in libjxl and
  `linear_to_pq` in jxl-rs (`jxl/src/color/tf.rs` L376–396). At 203, reference white lands where BT.2100/BT.2408
  displays and D33's PQ reading expect it. At 255 it lands 1/3 of a stop too bright, and so does the light above
  white: a photo's 8× highlight lands at 2040 cd/m² instead of 1624.
- **libjxl's own tone mapping ignores a linear file's target.** `JxlDecoderSetDesiredIntensityTarget` is "provided
  for convenience and […] not meant to be considered authoritative" (`decode.h` L830–841). Its stage tone-maps only a
  PQ source brighter than the request, or an HLG source (`render_pipeline/stage_tone_mapping.cc` L36–56). For a
  linear file it changes only the number `JxlDecoderGetBasicInfo` reports (`decode.cc` L2247–2249). Measured: no
  effect, in linear or in PQ.
- **Chromium reads the target only for PQ and HLG files.** It passes the target on as CLLI MaxCLL, the content's
  peak, and skips 255 and PQ's 10000 as "no statement" (`jxl_image_decoder.cc` L179–195). For a linear file it uses
  neither value, so D8 does not change what a browser shows. For a PQ or HLG file, Chromium reads the field as the
  documented upper bound, which the libjxl code does not do for a linear file (§1). The canonical photo is linear,
  so that reading never applies to it.

## 7. Recommendation

**(a) 203, keeping quality 95 / effort 7 in the store and 90 / 3 on the front.**

- **It is what the field means for this file.** libjxl and jxl-rs use a linear file's intensity target as the
  luminance of 1.0 (§1). D1 makes 1.0 reference white, which BT.2408, D33's EETF and `tone_map.REFERENCE_WHITE`
  all put at 203. With 255, every PQ rendering of the canonical photo comes out a third of a stop too bright (§6),
  and the file says something about its own white that the repo says nowhere else.
- **It costs nothing extra to write.** ML-37 already needs the repo's ctypes encoder for half-float, and the
  target is one field there (§2). The header check (R6) can read it in about ten lines.
- **Its effect on the models' input is a small quality offset.** Files are 4.5% / 6.1% smaller, and the block-mean
  error is 9–10% higher (+0.005 levels in the store, +0.011 on the front). That equals about 0.4 quality steps in
  the store and 1.2 on the front. It is at least 8× inside D27, and the error concentrates in the deepest shadows
  (§4, §5). R2's question is answered separately: linear half-float input does not lose detail against today's
  uint16 BT.709 at the same target (1.00×). The difference R2 would measure at 203 is the target alone.
- **If the owner prefers today's error exactly,** raise the front to q91 (the same bytes as 255 at q90, block mean
  0.112 against 0.110) and the store to q96 (+8.5% bytes, 0.050 against 0.055). That decision is about quality and
  belongs to the owner. It is independent of D8, and nothing in D27 asks for it.

**Against (b) 255.** It is what imagecodecs writes, and it gives slightly lower error at the same quality. But that
lower error is only what a higher quality setting would also buy, and it costs the file a false statement about its
white. (b) would be the choice only if the canonical photo were still written through imagecodecs' `jpegxl_encode`.
ML-37 §2 already moves the write to the repo's strip encoder, which needs a float16 form.

**Confidence.** High on the meaning and the code paths: they are read in libjxl 0.12.0 and jxl-rs and confirmed by
measurement (the default, the decoder views, the XYB scaling's effect). Medium-high on the size of the effect: 20
photos, encoded at half scale. The ratios were steady per photo (store 1.05–1.15), but full-scale files may move the
absolute numbers.

**Unverified.**
- ISO/IEC 18181-1's own wording (paywalled). libjxl, its reference software, and jxl-rs stand in for it.
- Full-resolution encodes, and efforts 8 and above. At effort 8 and above the butteraugli loop runs, but it uses 80
  for any non-PQ, non-HLG file, so it should not add a dependence of its own (§3).
- libjxl versions other than 0.12.0.
- Browsers other than Chromium, and HDR viewers that render to PQ through something other than libjxl or jxl-rs.
- Any effect on the models: no training was run. The evidence here is error in sRGB levels only.

## 8. Sources

Read in `tmp/ml37/d8/downloads/` (not committed). The paths are relative to each project's root.

- **libjxl 0.12.0**, tag `v0.12.0` (commit `a7a9c78`). This is the version linked in imagecodecs' wheel
  (`imagecodecs.jpegxl_version()` = "libjxl 0.12.0"). Files read: `lib/include/jxl/codestream_header.h`,
  `lib/include/jxl/cms_interface.h`, `lib/include/jxl/decode.h`, `lib/jxl/image_metadata.{h,cc}`,
  `lib/jxl/base/common.h`, `lib/jxl/luminance.cc`, `lib/jxl/encode.cc`, `lib/jxl/enc_xyb.cc`, `lib/jxl/enc_frame.cc`,
  `lib/jxl/enc_adaptive_quantization.cc`, `lib/jxl/enc_heuristics.cc`, `lib/jxl/cms/opsin_params.h`,
  `lib/jxl/opsin_params.cc`, `lib/jxl/dec_xyb.cc`, `lib/jxl/cms/jxl_cms.cc`, `lib/jxl/cms/transfer_functions-inl.h`,
  `lib/jxl/decode.cc`, `lib/jxl/render_pipeline/stage_from_linear.cc`,
  `lib/jxl/render_pipeline/stage_tone_mapping.{h,cc}`, `lib/extras/dec/exr.cc`, `lib/extras/enc/exr.cc` and
  `tools/cjxl_main.cc`.
- **jxl-rs** `jxl` crate 0.8.0, commit `9a08ca6` (2026-10-20): `jxl/src/headers/image_metadata.rs`,
  `jxl/src/render/stages/xyb.rs`, `jxl/src/color/tf.rs`.
- **imagecodecs 2026.10.10**, the source distribution: `imagecodecs/_jpegxl.pyx`, `imagecodecs/libjxl.pxd`.
- **Chromium**, `third_party/blink/renderer/platform/image-decoders/jxl/jxl_image_decoder.cc`, the main branch as
  fetched on 2026-11-23 (decodes through jxl-rs). The revision was not recorded.
- **ITU-R BT.2408-8** §2.1 and Table 1 (HDR reference white 203 cd/m²), as read and cited in
  [analysis/ml29/research_tone_mapping.md](../ml29/research_tone_mapping.md) §1.
- **ISO/IEC 18181-1** (the JPEG XL codestream): not read (paywalled).
- **The repo at `01e6ac1`:** `coffeecv/jxl_encode.py`, `coffeecv/jxl_header.py`, `coffeecv/tone_map.py`,
  `coffeecv/canonical.py`, and the ML-37 ticket (§2, §5 R2, §6 D8).
- **Measurements**, beside this note: `d8_intensity_target.py` → `d8_intensity_target.txt` (per-row data in
  `tmp/ml37/d8/d8_rows.csv`, 500 rows: 20 photos × 25 variants), `d8_by_level.py` → `d8_by_level.txt`,
  `d8_decoder_view.py` → `d8_decoder_view.txt`. Run with
  `cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d8_<name>.py`, 2 threads each, peak RSS
  717 MB.
