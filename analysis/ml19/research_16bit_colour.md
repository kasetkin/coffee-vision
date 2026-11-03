# ML-19: 16-bit and float colour management, and JPEG XL from Python

Research note for [ML-19](../../docs/ticket_canonicalization.html), written 2026-11-03. Sources are primary: source code at a
named tag, official docs, PyPI metadata and wheels unpacked under `/tmp`. "Measured" means a run of imagecodecs
2026.10.10 from its PyPI wheel, on an AVX2-only x86_64 machine (AMD Zen 3, glibc 2.41, NumPy 2.5.4). The scripts lived
in `/tmp` and are not committed. "Unverified" marks claims that could not be checked here.

## Answer

**Use `imagecodecs`, pinned to one exact version (today 2026.10.10), for both colour conversions and for the JPEG XL
encode and decode.** It is the only PyPI package found that does all of the following:

- It runs LittleCMS on uint16, float16, float32 and float64 NumPy arrays [S1].
- It encodes JPEG XL from uint16 or float32 input, with an enumerated Rec.2020 / D65 / BT.709 colour encoding [S2].
- It bundles its C libraries in the manylinux wheel: LittleCMS 2.19 and libjxl 0.12.0. Pinning the Python package
  therefore pins both libraries (measured; [S3]).

The recipe:

1. **Original to Rec.2020: `cms_transform` on float32**, with `intent='RELATIVE_COLORIMETRIC'` passed explicitly (the
   default is perceptual [S1 L355-357]).
   - LittleCMS never applies its lossy optimizations to float formats [S10].
   - Float output is unbounded with parametric curves, so it keeps the slightly negative values that Display P3 red
     produces in Rec.2020 (measured).
   - uint16 with `flags=CMS.FLAGS.NOOPTIMIZE` gives the same precision. Default-flag uint16 is lossy (measured below).
2. **Encode: `jpegxl_encode(img, level=95, effort=7, primaries='BT2100', transfer='BT709')`.**
   - `level=95` is distance 0.55, the same mapping `cjxl -q 95` uses [S2 L228-235, S20].
   - Pass `effort` explicitly. imagecodecs defaults to 5, libjxl to 7 [S2 L214, S21].
   - Use uint16 or float32 input. If the input is float, it must lie in [0, 1] unless `perceptualheuristics=False`
     [S2 L175], so clip the small negatives first.
3. **Decode: `jpegxl_decode`.** It returns uint16 for a file encoded from 16-bit input, and float32 for a file encoded
   from float [S2 L827-853].
4. **Rec.2020 to sRGB: `cms_transform` to `outdtype=uint8`**, from float32 input, or from uint16 input with `NOOPTIMIZE`.
   The two gave identical 8-bit output (measured).
5. **Pin the Rec.2020 profile bytes, not just the library.** Commit the 536-byte ICC profile that libjxl generates for
   `RGB_D65_202_Rel_709`, and check its SHA-256 in a test.
   - That profile uses the same parametric curve libjxl uses (measured; [S22]).
   - imagecodecs cannot build a parametric curve itself. `cms_profile` offers only a pure gamma or a sampled table
     [S1 L621-676].
   - Two Rec.2020 profiles that differ only in their s15.16 colorant rounding changed 3% of 8-bit output pixels by
     1 code (measured).

**Risks.**

- **imagecodecs is a one-maintainer library with an unstable API.** It ships breaking changes in most releases, eight
  releases so far in 2026 [S4, S5]. Mitigations:
  - lock the exact version and wheel hash in `uv.lock`;
  - never build it from sdist, which would link the system LittleCMS;
  - keep a golden-output test.
- **glibc's libm is not pinned by the wheel.** LittleCMS evaluates parametric curves with `pow()`, and glibc chooses a
  `pow` variant at run time: FMA if FMA and AVX2 are present, otherwise SSE2 [S30]. Both machines have AVX2 and FMA,
  so they take the same branch. A different glibc version is not pinned, though. Run the same container image on both
  machines.
- **libjxl's output may not be bit-identical across CPU vendors** (unverified).
  - The imagecodecs wheel compiles only the SSE2, SSE4 and AVX2 Highway targets (measured), so the AVX-512 machine also
    runs AVX2 code.
  - But libjxl calls `ApproximateReciprocal` in both the encoder and the decoder dequantizer [S24, S25]. On AVX2 that is
    the `vrcpps` instruction [S26], whose exact results differ between Intel and AMD [S27, S28].
  - Mitigations: encode each canonical photo once and store the file, never re-encode it. Check the decoded output's
    hash on both machines. Compare colours with a tolerance, not bit-for-bit, when the two machines' encodes are
    compared.
- **Library upgrades change output.**
  - LittleCMS 2.16 changed float-to-integer transforms, and 2.18 changed a 16-bit Lab precision path [S11].
  - imagecodecs upgrades also bump the bundled LittleCMS and libjxl.
  - Treat any version bump as a re-canonicalization, as ADR 0033 already requires for colour maths changes.

**Not recommended.**

- **pyvips-binary:** its libvips is built without libjxl. Its `icc_transform` offers only 8- or 16-bit device data,
  with no way to set NOOPTIMIZE. Its `jxlsave` cannot write an enumerated Rec.2020 encoding. Details under question 2.
- **pillow-jxl-plugin:** it is GPL-3.0, cannot set the colour encoding, and decodes 16-bit RGB to 8-bit.
- **python-lcms2:** it is GPL-3.0 and bundles LittleCMS 2.17, has one maintainer and no JPEG XL.

These recommendations agree with ADR 0033 and do not contradict any ADR.

## 1. How established projects colour-manage 16-bit and float images

| Project | Uses LittleCMS? | Precision and flags | Notes |
|---|---|---|---|
| GIMP 3 (babl/GEGL) | For non-matrix profiles only | lcms format matched to the buffer: 8, 16, half, `TYPE_RGB_FLT`, `TYPE_RGB_DBL` [S12 L1800-1809]. The image's transforms to sRGB use `NOOPTIMIZE` and black-point compensation [S13 L1059-1060]. | Matrix/TRC profiles go through babl's own float code (`babl_fish`) unless `GIMP_COLOR_TRANSFORM_DISABLE_BABL` is set [S14 L232-243]. babl falls back to lcms for profiles with lookup tables [S15 L1077-1080]. |
| darktable | As a fallback | `TYPE_RGBA_FLT` / `TYPE_LabA_FLT`, flags 0 [S16 L1676-1691] | Matrix/TRC profiles use its own float matrix plus curve code (same lines). It builds Rec.2020 and Rec.709 profiles in code with `cmsCreateRGBProfile` and `cmsBuildParametricToneCurve` [S17 L65, L457-464]. |
| RawTherapee | Yes | `TYPE_RGB_FLT`, `TYPE_Lab_FLT`, `TYPE_Lab_DBL`, with `cmsFLAGS_NOOPTIMIZE \| cmsFLAGS_NOCACHE` ("NOCACHE is important for thread safety") [S18 L276-280, L368-376] | |
| Krita | Yes (its colour engine) | Format per colour space: `TYPE_BGRA_16` for 16-bit integer, `TYPE_RGBA_FLT` for 32-bit float [S19a, S19b] | Registers the LittleCMS fast-float plugin when built with it [S19 L333-335], so its float output depends on that GPL plugin. |
| ImageMagick | Yes | `TYPE_RGB_DBL` by default ("highres"); `TYPE_RGB_16` only if `profile:highres-transform` is false on a non-HDRI build; `cmsFLAGS_HIGHRESPRECALC` [S6 L1218-1227, L1291-1301, L1447] | |
| libvips 8.18 / pyvips | Yes | Device RGB is `TYPE_RGB_8` or `TYPE_RGB_16` only (`depth` 8 or 16); the PCS is Lab or XYZ, float or 16-bit; `cmsFLAGS_NOCACHE` plus optional black-point compensation [S7 L229-233, L444-453, L1132] | No float device-RGB path, and no `NOOPTIMIZE`, so 16-bit RGB to RGB goes through LittleCMS's 33³ lookup table (see question 4). |
| libjxl 0.12 (its CMS) | Optional | lcms path: `TYPE_RGB_FLT`, `cmsFLAGS_NOCACHE \| BLACKPOINTCOMPENSATION \| HIGHRESPRECALC` [S22 L586-590, L1326-1333]. skcms is the CMake default on little-endian machines [S23 L120-126, L160-161]. | The imagecodecs wheel builds it with lcms; pillow-jxl-plugin builds it with skcms (measured). |
| OpenColorIO | Not for pixels | Own ICC reader: "The matrix/TRC model … is parsed … Other types of ICC profiles are not currently supported" [S8 L23-31] | lcms is linked only by the `ociobakelut` tool, to write ICC profiles [S8a]. |
| OpenImageIO | No | Colour conversion goes through OCIO; no LittleCMS found in `color_ocio.cpp` (grep only) [S9] | Partly verified. |
| colour-science 0.4.7 | No | Pure NumPy (`RGB_to_RGB`) [S31]; no ICC parser in the package (grep of the wheel) | Its BT.2020 OETF uses α 1.099 / β 0.018, or 1.0993 / 0.0181 for 12-bit systems [S31a L52-60]. libjxl's 709 curve uses 1.099 / 0.018 [S22b]. |
| tifffile 2026.10.15 | No | "color space transformations … ICC … metadata are not implemented" [S32]; ICC bytes are only read or written as tag 34675 | |
| scikit-image 0.26 | No | No ICC handling in the package (grep of the wheel) | |
| Pillow 12.3 | Yes (ImageCms) | Modes map to `TYPE_RGBA_8`, `TYPE_CMYK_8`, `TYPE_GRAY_16` (I;16) and 8-bit gray, Lab, YCbCr [S33 L215-241]; "Pillow doesn't yet support multichannel images with a depth of more than 8 bits per channel" [S34 L69-70] | Matches the ticket's finding that 16-bit RGB modes fail. |

**Why they make these choices** (from the sources above):

- Float formats avoid LittleCMS's lossy pipeline optimizations, which never apply to float [S10 L662-663, L1061-1062,
  L1409-1410].
- They keep out-of-range values.
- `NOCACHE` makes one transform safe to call from many threads; libvips, libjxl and RawTherapee say so in their
  comments.
- The projects that care most about accuracy (GIMP, darktable, babl) skip LittleCMS for matrix/TRC profiles and do the
  maths themselves in float.

## 2. Python bindings to LittleCMS for 16-bit and float arrays

| Package | cp312 manylinux x86_64 wheel | C library bundled? | License | Maintenance | Building a Rec.2020 / BT.709 profile in code |
|---|---|---|---|---|---|
| **imagecodecs 2026.10.10** | Yes (`cp312-abi3-manylinux_2_28`) | Yes: `liblcms2-da5b85cc.so.2.0.19` (`cms_version()` gives `lcms2 2.19.0`; the README lists lcms2 2.19.1 [S5 L81]), `libjxl 0.12.0`, `libjxl_cms` linked to that lcms. No fast-float plugin is shipped or registered: no plugin library and no `cmsPlugin` call in `_cms` (measured). | BSD-3-Clause; bundled lcms is MIT | One maintainer (Christoph Gohlke); 8 releases in 2026; "The API is not stable yet" [S5 L315-317] | `cms_profile('rgb', whitepoint=…, primaries=…, gamma=… or transferfunction=uint16/float32 table)` [S1 L515-676]. No parametric curve option (see "Profile construction" below). |
| **pyvips 3.2.0 + pyvips-binary 8.18.7** | Yes (`cp37-abi3-manylinux_2_28`) | Yes, statically linked into one `libvips` .so. lcms2 2.19.1 per the packaging repo's `versions.properties` (main branch at time of reading) [S35]. The build string in the binary says `libjxl: false`, and there is no `jxlsave` or `jxlload` operation (measured). | pyvips MIT, pyvips-binary LGPL-3.0-or-later | Active (libvips project) | `icc_transform` takes a profile file name or ICC bytes and `depth` 8 or 16 [S7 L1132, L1250]; no profile construction from primaries. |
| python-lcms2 0.1.1 | Yes (`cp312-manylinux_2_28`) | Yes, statically linked; `get_version()` gives 2170 (lcms 2.17) (measured) | GPL-3.0 (classifier and LICENSE file) | One author; 3 releases (0.0.20 to 0.1.1, May 2026) [S36] | Formats `RGB_16`, `RGB_FLT`, `RGB_DBL`, `RGB_HALF_FLT`. Flags include `NOTPRECALC` (= 0x100 = `cmsFLAGS_NOOPTIMIZE`) but not `NOCACHE`. Parametric profiles only from a built-in list (Adobe RGB, ProPhoto and others, no Rec.2020) through the private `_lcms2.create_rgb_profile` (measured). |
| pylcms2 0.0.21 | No (only `cp313-win_amd64`) | | none stated | | Unusable here [S37]. |
| littlecms 0.1b3 (2016), lcms 0.3.0 (2020) | No wheels | | BSD / MIT | Stale | Unusable here [S37]. |
| colour-science 0.4.7 | Pure Python | Not applicable | BSD-3-Clause | Active | Exact matrix and transfer maths in NumPy, no ICC. NumPy 2.5 dispatches `np.power` to SVML on AVX-512 (`X86_V4`) CPUs [S38 L1079-1081, S38a L8, L64], so the same NumPy code can give last-bit different results on the AVX2 and AVX-512 machines. |

**Profile construction with imagecodecs (measured).** There are three ways to get a Rec.2020 / BT.709 profile:

- **A sampled curve.** `cms_profile(... transferfunction=<4096 float32 entries>)` writes a `curv` tag of 4,096 16-bit
  entries (8,204 bytes per channel). In float mode LittleCMS clips sampled curves to [0, 1], so negatives are lost.
- **A gamma profile with the curve tags swapped.** Build the profile with `gamma=2.2` (it gets a `para` type-0 curve),
  then rewrite its three curve tags as `para` type 3 with the BT.709 parameters (1/0.45, 1/1.099, 0.099/1.099, 1/4.5,
  0.081). That is about 25 lines of pure Python. The rewritten profile passed `cms_profile_validate`, and LittleCMS
  then keeps negative values in float mode.
- **The profile libjxl generates.** libjxl writes exactly that `para` type-3 curve [S22b L1099-1103]. Its 536-byte
  profile can be read once through pillow-jxl-plugin's `info['icc_profile']` or from djxl. The measured profile has
  tags `desc` "RGB_D65_202_Rel_709", `cicp`, `chad`, XYZ colorants and `para` (2.222229, 0.909912, 0.090088, 0.222229,
  0.080994). This is the profile the recommendation commits.

`cms_info` reports a v4 profile's D50 media white point and D50-adapted primaries: for the Rec.2020 profile it gave
white (0.3457, 0.3585), not D65. Do not use it to check a profile's D65 white point.

## 3. JPEG XL encoders and decoders reachable from Python

| Route | Rec.2020 / BT.709 enumerated encoding | Input | Lossy, distance, effort | Decode at ≥16 bits and read the encoding |
|---|---|---|---|---|
| **imagecodecs `jpegxl_encode`** (since 2026.5.10 [S4]) | `primaries='BT2100'` (`JXL_PRIMARIES_2100`), `transfer='BT709'`. White is fixed to D65 and the intent to relative; it calls `JxlEncoderSetColorEncoding` [S2 L427-458]. | uint8, uint16 (16-bit), float16, float32 [S2 L356-375] | `level` gives the distance through `JxlEncoderDistanceFromQuality`; `distance`, `effort` (default 5), `numthreads` [S2 L153-171, L214, L228-235] | `jpegxl_decode` gives uint16 or float32 from the header's bit depth [S2 L827-853]. It does not expose the colour encoding or ICC profile; there is no `JxlDecoderGetColorAs*` call in the source. |
| pyvips `jxlsave` | No. With an ICC profile attached it calls `JxlEncoderSetICCProfile`; otherwise it sets sRGB or linear sRGB [S39 L585-622]. | 8/16-bit, float | `Q` / `distance`, `effort`, `bitdepth` [S39 L961-975, L1016-1020] | Not in pyvips-binary at all (measured); it needs a system libvips with libjxl, which is not pinned by PyPI. |
| pillow-jxl-plugin 1.3.8 | No: `Encoder(parallel, has_alpha, lossless, quality, num_threads)` only | 8-bit modes | quality only | Decoding a 16-bit RGB file returned mode `RGB` uint8 (measured). It returns the ICC libjxl generates (useful once, to extract the Rec.2020 profile). Static libjxl 0.12.0 with skcms; GPL-3.0-or-later (wheel metadata and SBOM). |
| cjxl / djxl CLI | `-x color_space=RGB_D65_202_Rel_709` [S21 L204-216]. It is ignored if the input file already carries colour information [S40 L22-33], so feed PPM (16-bit) or PFM (float). | PPM/PFM/PNG | `-q 95` (distance 0.55), `-e 7` [S20, S21] | djxl output at 16 bits or float. Static Linux builds are released as assets of v0.12.0 [S41]. It is not on PyPI, and it adds a subprocess and temporary files. |

**Colour handling inside libjxl, for this pipeline:**

- **Encoder.** A lossy (XYB) encode of input that is neither sRGB nor linear sRGB converts the input to linear sRGB
  with the CMS first [S42 L233-287]. With the imagecodecs wheel that CMS is LittleCMS in float.
  The libjxl wrapper skips LittleCMS when the source and destination encodings are equal [S22 L1191-1197].
- **Decoder.** An XYB file whose original encoding is enumerated with the 709 curve is decoded straight back to that
  encoding, without any CMS [S43 L127-165]. So `jpegxl_decode` returns Rec.2020 / BT.709 pixel values.
- **Enumerations and quality mapping.** `JXL_PRIMARIES_2100` = 9 ("as specified in Rec. ITU-R BT.2100-1"; these are the
  BT.2020 primaries). `JXL_TRANSFER_FUNCTION_709` = 1 [S44 L63-96]. `JxlEncoderDistanceFromQuality(q)` is
  `0.1 + (100 - q) * 0.09` for q ≥ 30 [S20 L1626-1631].
- **libjxl's BT.709 curve** uses the constants 1.099 / 0.099 / 0.018 / 4.5 / 0.45 [S22b]. The SIMD path uses
  `FastPowf` ("maximum error 1e-6").

**Measured (imagecodecs 2026.10.10, 1024×768 16-bit gradient image with noise):**

- `level=95` and `distance=0.55` gave identical bytes.
- `numthreads` 1, 4 and 16 gave byte-identical files.
- Decoding was thread-independent.
- uint16 and float32 input of the same image gave files of the same size but different bytes.
- Error after the q95 round trip: mean 249, p99 833 and maximum 2,444 sixteen-bit codes, which is about 1, 3 and 9.5
  eight-bit codes.
- A 3 MP noise image at effort 7 took 1.5 s to encode.

**Highway SIMD dispatch:**

- libjxl 0.12's CMake enables the AVX2 and AVX3_DL targets by default, and disables AVX3, AVX3_SPR and AVX3_ZEN4
  [S23 L128-129, L194-201]. A default build would therefore run AVX-512 (AVX3_DL) code on an Ice Lake or newer CPU.
- The imagecodecs and pillow-jxl wheels contain only the `N_SSE2`, `N_SSE4` and `N_AVX2` namespaces, and no
  `N_AVX3*` (measured by searching the binaries). Both machines therefore run the AVX2 code.
- The AVX2 and AVX-512 targets compute `ApproximateReciprocal` with different instructions: `_mm256_rcp_ps` against
  `_mm512_rcp14_ps` [S26].
- The JPEG XL decoder conformance suite checks output against RMS and peak-error tolerances (for example rms 0.000977),
  not bit-exactness [S29]. libjxl makes no bit-exactness promise; none was found in its sources.

## 4. Determinism

**LittleCMS (2.19):**

- **No SIMD in the core.** A grep of `src/` and `include/` for SSE, AVX and intrinsics found nothing, so the transforms
  are scalar C.
- **SIMD only in the plugin, and the plugin is unused here.** Only the separate fast-float plugin has SSE code with a
  CPUID check (`fast_8_matsh_sse.c`, `fast_float_sup.c`). That plugin is GPL-3 [S45 L5] and must be registered with
  `cmsPlugin`. The imagecodecs wheel neither ships nor registers it (measured). Krita registers it [S19].
- **The same wheel gives the same output on both machines, with two exceptions.** A manylinux wheel is compiled for the
  baseline x86-64 instruction set, so FMA contraction cannot change with the CPU (inference from the wheel tag; the
  compile flags are unverified). The two exceptions:
  - **libm `pow`.** glibc picks its `pow` at run time: FMA if FMA and AVX2 are present, otherwise SSE2 [S30]. Both
    machines get the FMA variant, but glibc itself is not pinned by the wheel.
  - **Build-time optimizations.** These are chosen when the transform is created and depend on the pixel format and
    flags, not on the CPU:
    - 8-bit RGB input gets the matrix-shaper optimization [S10 L1693-1694].
    - 16-bit RGB input without `NOOPTIMIZE` gets a 33³ lookup table (`OptimizeByResampling`) [S10 L1822-1828,
      L1960-1962; S46 L696-703]. Pre-linearization curves are added only when explicitly requested [S10 L1071-1074].
    - Float formats are never optimized, and the 1-pixel cache is off for them [S10 L662-663; S47 L1163-1165].
- **Repeated runs were identical** (measured).

**What the 16-bit lookup table costs (measured):**

- **Display P3 to Rec.2020/709**, 262,144 random 16-bit pixels, against a float64 NumPy reference:

  | Mode | Max error (16-bit codes) | Mean error (16-bit codes) |
  |---|---|---|
  | uint16, default flags | 174 | 10.4 |
  | uint16, `NOOPTIMIZE` | 21.7 (5.5 away from the 709 knee) | 1.35 |
  | float32 | 21.3 (5.1 away from the knee) | 1.30 |

  The residual error near the knee (16 codes at encoded value 0.081) comes from the s15.16 rounding of the `para`
  parameters and the colorant matrices.
- **Rec.2020 to 8-bit sRGB**, 200,000 random Rec.2020 colours:
  - uint16 with default flags differed from float32 in 23.5% of pixels, by up to 22 codes. 28% of the colours that are
    inside the sRGB gamut differed too: lookup-table nodes clipped at the gamut edge pull in their neighbours.
  - uint16 with `NOOPTIMIZE` and float32 gave identical 8-bit output.
- **Display P3 red lies just outside Rec.2020.** The P3 pixel (0.989, 0.065, 0.002) has linear Rec.2020 blue −0.00096.
  Float transforms with `para` curves keep the negative value (encoded −0.0043). uint16 output and sampled curves clip
  it.
- **Speed**, 12 MP, single thread: uint16 with default flags 0.31 s, uint16 `NOOPTIMIZE` 2.07 s, float32 2.12 s.

**Changes between LittleCMS versions that can change output** [S11]:

- 2.13: "Added automatic linear space detection".
- 2.16: "floating point transforms float -> integer are now honored as float".
- 2.18: "Fixed a loss of precision when Lab16 is used as input color space on integer transforms".
- 2.20 (released after the wheel's 2.19): fixes to premultiplied alpha on planar 16-bit and optimized matrix-shaper
  transforms, and to planar half-float and Lab double formatters.

Pin the exact version. A golden test should catch an upgrade.

**libjxl:**

- **Within one machine, deterministic** (measured):
  - output is independent of thread count;
  - at effort 7 the encoder runs no butteraugli round trip, which is used only at effort 8 and above [S48 L1282];
  - at distance 0.55 the edge-preserving filter is off, since it needs distance ≥ 0.7 [S49 L333-343].
- **Across machines, CPU-dependent in principle:**
  - The dequantizer bias in the decoder, and its counterpart in the encoder, use `ApproximateReciprocal` [S24 L62-64,
    S25 L142-146; `enc_group.cc` L374].
  - On the AVX2 target that is `vrcpps`. Intel only bounds its error (≤ 1.5·2⁻¹²), and Intel and AMD return different
    values [S27, S28].
  - Not tested here (only one machine was available). If the two machines use different vendors' CPUs, expect possible
    last-bit differences in decoded uint16 values. Those rarely reach an 8-bit code after the sRGB conversion.

## Sources

Paths without a host are inside the named archive or repository at the named tag. Line numbers are for that version.

- S1 imagecodecs `imagecodecs/_cms.pyx`, v2026.10.10: https://github.com/cgohlke/imagecodecs/blob/v2026.10.10/imagecodecs/_cms.pyx (L335-511 `cms_transform`, L515-728 `cms_profile`, L903 dtypes `BHdfe`)
- S2 imagecodecs `imagecodecs/_jpegxl.pyx`, v2026.10.10: https://github.com/cgohlke/imagecodecs/blob/v2026.10.10/imagecodecs/_jpegxl.pyx
- S3 imagecodecs wheel `imagecodecs-2026.10.10-cp312-abi3-manylinux_2_28_x86_64.whl`, `imagecodecs.libs/`: https://pypi.org/project/imagecodecs/2026.10.10/#files
- S4 imagecodecs `CHANGES.rst`, v2026.10.10: https://github.com/cgohlke/imagecodecs/blob/v2026.10.10/CHANGES.rst ; PyPI release history: https://pypi.org/project/imagecodecs/#history
- S5 imagecodecs `README.rst`, v2026.10.10 (L81 lcms2 2.19.1, L96 libjxl 0.12.0, L315-317 API not stable): https://github.com/cgohlke/imagecodecs/blob/v2026.10.10/README.rst
- S6 ImageMagick `MagickCore/profile.c` (main): https://github.com/ImageMagick/ImageMagick/blob/main/MagickCore/profile.c
- S7 libvips `libvips/colour/icc_transform.c`, v8.18.7: https://github.com/libvips/libvips/blob/v8.18.7/libvips/colour/icc_transform.c
- S8 OpenColorIO `src/OpenColorIO/fileformats/FileFormatICC.cpp` (main): https://github.com/AcademySoftwareFoundation/OpenColorIO/blob/main/src/OpenColorIO/fileformats/FileFormatICC.cpp ; S8a `src/apps/ociobakelut/CMakeLists.txt` L32: https://github.com/AcademySoftwareFoundation/OpenColorIO/blob/main/src/apps/ociobakelut/CMakeLists.txt
- S9 OpenImageIO `src/libOpenImageIO/color_ocio.cpp` (main): https://github.com/AcademySoftwareFoundation/OpenImageIO/blob/main/src/libOpenImageIO/color_ocio.cpp
- S10 Little-CMS `src/cmsopt.c`, tag lcms2.19: https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmsopt.c
- S11 Little-CMS `ChangeLog`, tags lcms2.19 and lcms2.20: https://github.com/mm2/Little-CMS/blob/lcms2.20/ChangeLog
- S12 GIMP `libgimpcolor/gimpcolorprofile.c` (master): https://gitlab.gnome.org/GNOME/gimp/-/blob/master/libgimpcolor/gimpcolorprofile.c
- S13 GIMP `app/core/gimpimage-color-profile.c` (master): https://gitlab.gnome.org/GNOME/gimp/-/blob/master/app/core/gimpimage-color-profile.c
- S14 GIMP `libgimpcolor/gimpcolortransform.c` (master): https://gitlab.gnome.org/GNOME/gimp/-/blob/master/libgimpcolor/gimpcolortransform.c
- S15 babl `babl/babl-icc.c` (master): https://gitlab.gnome.org/GNOME/babl/-/blob/master/babl/babl-icc.c
- S16 darktable `src/iop/colorin.c` (master): https://github.com/darktable-org/darktable/blob/master/src/iop/colorin.c
- S17 darktable `src/common/colorspaces.c` (master): https://github.com/darktable-org/darktable/blob/master/src/common/colorspaces.c
- S18 RawTherapee `rtengine/iplab2rgb.cc` (dev); also `rtengine/improcfun.cc` L434-440: https://github.com/RawTherapee/RawTherapee/blob/dev/rtengine/iplab2rgb.cc
- S19 Krita `plugins/color/lcms2engine/LcmsEnginePlugin.cpp` (master): https://invent.kde.org/graphics/krita/-/blob/master/plugins/color/lcms2engine/LcmsEnginePlugin.cpp ; S19a `colorspaces/rgb_f32/RgbF32ColorSpace.cpp` L25, S19b `colorspaces/rgb_u16/RgbU16ColorSpace.cpp` L24 (same directory)
- S20 libjxl `lib/jxl/encode.cc` L1626-1631, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/encode.cc
- S21 libjxl `tools/cjxl_main.cc` (L114-115 effort default 7, L204-216 `-x color_space`, L688 quality to distance), v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/tools/cjxl_main.cc
- S22 libjxl `lib/jxl/cms/jxl_cms.cc`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/cms/jxl_cms.cc ; S22b `lib/jxl/cms/jxl_cms_internal.h` L1099-1103 and `lib/jxl/cms/transfer_functions-inl.h` L96-133 (same tag)
- S23 libjxl `CMakeLists.txt`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/CMakeLists.txt
- S24 libjxl `lib/jxl/quantizer-inl.h`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/quantizer-inl.h
- S25 libjxl `lib/jxl/dec_group.cc` and `lib/jxl/enc_group.cc`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_group.cc
- S26 Highway (libjxl 0.12.0 submodule, commit 457c891 = tag 1.2.0) `hwy/ops/x86_256-inl.h` L2673-2675 and `hwy/ops/x86_512-inl.h` L1898-1900: https://github.com/google/highway/blob/1.2.0/hwy/ops/x86_256-inl.h
- S27 RCPPS/RSQRTPS error bound (Intel SDM extract): https://felixcloutier.com/x86/rsqrtps ; R. O'Callahan, "Emulating AMD approximate arithmetic instructions on Intel" (2021): https://robert.ocallahan.org/2021/09/emulating-amd-rsqrtss-etc-on-intel.html
- S28 LHCb merge request "Remove use of operations with different results on AMD and Intel": https://gitlab.cern.ch/lhcb/Rec/-/merge_requests/2488
- S29 JPEG XL conformance `scripts/conformance.py` L28-73 and `testcases/bicycles/test.json`: https://github.com/libjxl/conformance
- S30 glibc `sysdeps/x86_64/fpu/multiarch/e_pow.c` and `ifunc-fma4.h` (GitHub mirror, master): https://github.com/bminor/glibc/blob/master/sysdeps/x86_64/fpu/multiarch/ifunc-fma4.h
- S31 colour-science 0.4.7 `colour/models/rgb/rgb_colourspace.py` (`RGB_to_RGB`, L1305): https://github.com/colour-science/colour/blob/develop/colour/models/rgb/rgb_colourspace.py ; S31a `colour/models/rgb/transfer_functions/itur_bt_2020.py`: https://github.com/colour-science/colour/blob/develop/colour/models/rgb/transfer_functions/itur_bt_2020.py
- S32 tifffile 2026.10.15 module docstring (`tifffile/tifffile.py`), wheel: https://pypi.org/project/tifffile/2026.10.15/
- S33 Pillow `src/_imagingcms.c` L215-241, tag 12.3.0: https://github.com/python-pillow/Pillow/blob/12.3.0/src/_imagingcms.c
- S34 Pillow `docs/handbook/concepts.rst` L69-70, tag 12.3.0: https://github.com/python-pillow/Pillow/blob/12.3.0/docs/handbook/concepts.rst
- S35 libvips-packaging `versions.properties` (main): https://github.com/kleisauke/libvips-packaging/blob/main/versions.properties ; pyvips-binary on PyPI: https://pypi.org/project/pyvips-binary/8.18.7/
- S36 python-lcms2 on PyPI: https://pypi.org/project/python-lcms2/0.1.1/ ; source: https://github.com/wzjoriv/python-lcms2
- S37 PyPI JSON for `pylcms2`, `littlecms`, `lcms`: https://pypi.org/project/pylcms2/ , https://pypi.org/project/littlecms/ , https://pypi.org/project/lcms/
- S38 NumPy `numpy/_core/meson.build` L1079-1081, tag v2.5.4: https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/meson.build ; S38a `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src` (main): https://github.com/numpy/numpy/blob/main/numpy/_core/src/umath/loops_umath_fp.dispatch.c.src
- S39 libvips `libvips/foreign/jxlsave.c`, v8.18.7: https://github.com/libvips/libvips/blob/v8.18.7/libvips/foreign/jxlsave.c
- S40 libjxl `lib/extras/dec/color_hints.cc`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/extras/dec/color_hints.cc
- S41 libjxl v0.12.0 release assets (includes `jxl-linux-x86_64-static.tar.lz`): https://github.com/libjxl/libjxl/releases/tag/v0.12.0
- S42 libjxl `lib/jxl/enc_xyb.cc`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/enc_xyb.cc
- S43 libjxl `lib/jxl/dec_xyb.cc`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_xyb.cc
- S44 libjxl `lib/include/jxl/color_encoding.h`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/include/jxl/color_encoding.h ; `lib/include/jxl/decode.h` L707-719 (same tag)
- S45 Little-CMS `plugins/README.1ST`, tag lcms2.19: https://github.com/mm2/Little-CMS/blob/lcms2.19/plugins/README.1ST
- S46 Little-CMS `src/cmspcs.c`, tag lcms2.19: https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmspcs.c
- S47 Little-CMS `src/cmsxform.c` (L918-939 float transforms), tag lcms2.19: https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmsxform.c
- S48 libjxl `lib/jxl/enc_adaptive_quantization.cc`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/enc_adaptive_quantization.cc
- S49 libjxl `lib/jxl/enc_frame.cc`, v0.12.0: https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/enc_frame.cc

**Unverified:**

- whether libjxl's output actually differs between the two machines (needs a run on the second machine);
- the compiler flags used for the imagecodecs and pyvips-binary wheels;
- which Highway version the imagecodecs wheel uses;
- the exact lcms2 version inside pyvips-binary 8.18.7 (taken from the packaging repo's main branch, not from a tag);
- OpenImageIO's non-use of LittleCMS (one file grepped).
