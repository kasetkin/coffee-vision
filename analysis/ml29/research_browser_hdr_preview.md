# ML-29: a brighter, wider-gamut photo preview in phone browsers

Research for the owner, 2026-11-18, on the local checkout at `aecebaf`, following ML-29 P11
([ADR 0041](../../docs/adr/0041-a-photos-gain-map-is-applied-at-canonicalization.md)). The question: can the page show
a photo preview that uses Display P3's gamut and the panel's HDR brightness on a Pixel 9 Pro (Chrome on Android) and on
an iPhone 15 Pro, 16 Pro or 17 Pro (Safari on iOS)? A finding for the owner, not a decision: anything below that
changes what canonicalization keeps is a change to ADR 0033 and ADR 0041, and the owner's call. Every claim cites its
source (list at the end). **(unconfirmed)** marks a claim no primary source confirmed. "Source reading" means
Chromium's or Android's code was read at `main` on 2026-11-18 and nothing was run. No phone was tested.

## Summary

- **Pixel 9 Pro, Chrome on Android.** *Gamut: probably, but unconfirmed for this phone.* Chrome colour-manages
  tagged images and switches its window to Android's wide-colour mode. Whether it reports a wide-gamut screen in the
  Pixel's default "Adaptive" colour setting needs a test on the phone. *HDR: yes for a gain-map JPEG or AVIF in
  `<img>`, by source reading*: Blink decodes the gain map, and Chrome follows the display's current HDR/SDR ratio. The
  panel is rated "up to 2,000 nits (HDR)", 8 bits per channel ("Full 24-bit depth"); Google states no gamut figure.
- **iPhone 15 Pro, 16 Pro, 17 Pro, Safari.** *Gamut: yes.* WebKit colour-matches every tagged image to the display,
  and all three panels are "Wide color (P3)". *HDR: yes from Safari 26 (iOS 26)* for HDR images in `<img>`, with
  gain-map JPEGs named in the 26.3, 26.4 and 27 release notes. All three are rated 1600 nits peak (HDR). Outdoor peak
  is 2000 nits on the 15 Pro and 16 Pro, 3000 on the 17 Pro.
- **Today the page gets neither.** The preview is a server-made JPEG at quality 85, with no ICC profile, made from the
  8-bit sRGB read-back. The browser treats it as sRGB. Colours outside sRGB were clipped at the read-back. Light above
  white was tone-mapped away at canonicalization (ADR 0041).
- **Serving the canonical JPEG XL directly would gain nothing.** Safari has decoded JPEG XL since 17, and Chrome since
  155, by the beta blog and the schedule. But the canonical photo is standard range by construction. Its 16 bits
  give finer steps, not light above white. Values above white need a PQ or HLG transfer or a gain map.
- **What each gain needs.** *Gamut* needs only a serving change: a P3-tagged preview made from the canonical photo's
  Rec.2020 samples instead of the sRGB read-back. *Brightness* needs an HDR rendition that the canonical photo no
  longer holds. There are two ways. (i) The page shows the visitor's own file, which the browser already has. That
  reverses OPS-7 D8's "never the file itself". (ii) Canonicalization keeps the pre-tone-map light as a gain map. That
  changes ADR 0033 and ADR 0041. Neither is decided here.

## 1. What the page shows today

- The image element is `<img id="preview">` (`webapp/static/index.html:337`), with two canvases over it for the bean
  region (`index.html:339–340`, drawn with `getContext('2d')`, default sRGB, at `index.html:624`, `:637`).
- The page sets `preview.src` to `data:image/jpeg;base64,${part.thumbnail}`, the server's thumbnail
  (`index.html:1017–1042`). The comment there reads "the server's JPEG thumbnail, never the file itself -- browsers
  other than Safari cannot show a HEIC, and none a RAW". The examples do the same: "The browser need not be able to
  show a JPEG XL: the page shows the server's JPEG thumbnail (D7)" (`index.html:789`).
- The inference host makes the thumbnail. `/analyze` decodes the canonical photo with `canonical.load_canonical` to
  8-bit sRGB (`webapp/app.py:484`). `_preview_part` then makes `Image.fromarray(seg.photo)`, `thumbnail((1024, 1024))`
  and `save(buf, format="JPEG", quality=85)` (`webapp/app.py:67`, `:525–536`). No `icc_profile` is passed, so the JPEG
  is untagged. Its docstring promises "the pixels the model sees" (`app.py:527`).
- An untagged image "must be treated as being in the sRGB color space" (CSS Color 4 §3.5). WebKit says the same: "If an
  image doesn't have a tagged profile, WebKit assumes that it is sRGB" (WebKit, "Improving Color on the Web").
- Before that, the canonical photo is "Rec.2020 (D65, BT.709 transfer) … clipped after the conversion". It is read
  back "to 8-bit sRGB, clipped" (`coffeecv/canonical.py` docstring). A gain map is applied and "tone-mapped back into
  the canonical photo's standard range" (ADR 0041), with the D33 operator, whose peak goes to white
  ([research_tone_mapping.md](research_tone_mapping.md) §6). So the page receives sRGB, 8-bit, at or below white.

## 2. Wide gamut in the two browsers

**Specifications.**
- CSS Color 4 §3.4: a tagged image has an ICC profile "or other, equivalent methods", such as PNG's `cICP` chunk,
  "one of various SDR or HDR color spaces, such as Display P3 or BT.2100 HLG". Untagged images are sRGB (§3.5).
- Media Queries 5 §6.4: `color-gamut: srgb | p3 | rec2020` gives "the approximate range of colors that are supported by
  the UA and output device".
- HTML canvas ("Color spaces and color space conversion", under §4.12.5): `PredefinedColorSpace` is `"srgb"`, `"srgb-linear"`, `"display-p3"` and
  `"display-p3-linear"`. `CanvasRenderingContext2DSettings` has `colorSpace` (default `"srgb"`), `toneMapping` and
  `colorType` (`"unorm8"`, `"float16"`). The spec works an example: an sRGB-clamped value lies inside the gamut on a
  `display-p3` output device, where "no projection will be needed".
- WebGPU's `GPUCanvasConfiguration` has `colorSpace` (a `PredefinedColorSpace`, default `"srgb"`) and `toneMapping`.

**Chrome.**
- chromestatus lists "Canvas color management" (2D canvas and `ImageData` in Display P3) as on by default from 94,
  desktop and Android. It lists "WebGL canvas color management" from 104, and the `color-gamut` media query from 58.
- AVIF: Blink builds the image's colour space from its CICP (nclx) primaries and transfer, treating unspecified as
  BT.709 and sRGB (`avif_image_decoder.cc`, source reading). ICC profiles go through `SetEmbeddedColorProfile`
  (`image_decoder.cc`).
- Android side: `DisplayAndroid.getIsWideColorGamut()` is true only when both the display and the display server are
  wide-gamut. The second comes from `Configuration.isScreenWideColorGamut()`. Chrome's `WindowAndroid` sets the window
  to `ActivityInfo.COLOR_MODE_WIDE_COLOR_GAMUT` when asked to (`setWideColorEnabled`) (source reading).
- Android's documentation says `isScreenWideColorGamut()` "returns true only if the display is wide color gamut
  capable and the device supports wide color gamut color rendering". A display "might be wide color gamut capable
  but not color-managed, in which case, the system will not grant an app the wide color gamut mode" (Android, "Enhance
  graphics with wide color content"). The reference page adds: "colorspace aware but not necessarily color-managed.
  The final colors may still be changed by the screen depending on user settings."

**The Pixel's colour setting.**
- Pixel Help lists Natural ("the most accurate colors") and Boosted ("accurate colors with more vividness").
  "Adaptive (Pixel 3 & later): See the most vivid colors and keep color details. This is on by default."
- In AOSP's `DisplayTransformManager.setColorMode`, Natural and Boosted are `DISPLAY_COLOR_MANAGED` (saturation 1.0
  and 1.1). Saturated is `UNMANAGED`, and Automatic is `DISPLAY_COLOR_ENHANCED`.
- AOSP requires a colour-managed device to report Display-P3 and sRGB colour modes from its composer
  (source.android.com, "Color management").
- Two things are **unconfirmed** because no Google source states them: that the Pixel's "Adaptive" is AOSP's
  Automatic, and whether Chrome then sees `isScreenWideColorGamut()` true on a Pixel 9 Pro. The test is
  `matchMedia('(color-gamut: p3)')` on the phone in each colour setting.

**Safari.**
- "WebKit color-matches all images on both iOS and macOS … whether it is normal or wide gamut". WebKit supports
  `color-gamut` (WebKit, "Improving Color on the Web").
- Safari 15.2 "Added wide gamut support for canvas". 16.4 added `display-p3` in WebGL canvas. 26.2 added
  `display-p3-linear` colours in CSS. 27 added `srgb-linear` and `display-p3-linear` to `PredefinedColorSpace`
  (Safari release notes).
- Every iPhone in the question has "Wide color (P3)" (Apple tech specs).
- WebKit's statements about ICC profiles predate CICP. **Unconfirmed**: whether Safari honours a PNG `cICP` chunk.
  AVIF nclx is part of the format, so it is likely, but no Apple source says so.

## 3. HDR images in the two browsers

**What makes an image HDR.**
- Headroom is peak luminance over HDR reference white. It "depends on the level of HDR Reference White, user
  preference, and viewing conditions" (CSS Color HDR §2.2).
- An image brighter than white carries either a PQ or HLG transfer (BT.2100) or a gain map over a standard-range base.
  For a gain map, the display weight comes from `max_display_boost`, "the maximum available boost supported by a
  display, at a given point in time", which "can change over time based on device settings and other factors, such as
  ambient light conditions, or how many bright pixels are on the screen" (Ultra HDR v1.1, "Display").
- So a 16-bit sample of 0–1 under the BT.709 transfer, which is what the canonical photo holds, has no headroom. It
  needs a transfer that reaches above white, or a map.

**Chrome.**
- Gain-map JPEG (Ultra HDR, ISO 21496-1): Blink's `JPEGImageDecoder::GetGainmapInfoAndData` finds the gain-map image.
  `DeferredImageDecoder::ActivateLazyGainmapDecoding` decodes it once all data has arrived. No runtime flag gates it
  in `runtime_enabled_features.json5` or `features.cc` (source reading).
- Gain-map AVIF: the AVIF decoder reads libavif's (crabbyavif's) `gainMap` into `SkGainmapInfo`, base SDR or HDR
  (source reading). JPEG XL gain maps: the jxl-rs decoder has a `JxlRsGainMap` path (source reading). PQ and HLG:
  the JPEG XL decoder passes content light level only "when tone mapping PQ or HLG", which shows PQ and HLG images are
  tone-mapped to the display (source reading).
- **Unconfirmed**: the milestone at which gain-map JPEGs became default-on. "Chrome v116+" comes from a Mozilla bug
  (bugzilla 1832622), not a Chrome source. On Android, Chrome reads `Display.getHdrSdrRatio()`, "the current hdr/sdr
  ratio expressed as the ratio of targetHdrPeakBrightnessInNits / targetSdrWhitePointInNits". It listens for changes
  (`PhysicalDisplayAndroid.java`, source reading; Android `Display` reference).
- **Unconfirmed**: that a gain-map `<img>` reaches the Pixel 9 Pro panel above SDR white. No Google page states it for
  Chrome on Android, and no test was run here.
- `dynamic-range` media query: on by default from 98, desktop and Android (chromestatus). Media Queries 5 §6.5 defines
  `high` as high peak brightness, high contrast and "color depth … greater than 24 bit". It also says the feature
  "does not test whether such a mode is active". The Pixel 9 Pro is rated "Full 24-bit depth", so whether
  `dynamic-range: high` matches on it is **unconfirmed**.
- `dynamic-range-limit`: on by default from 136, desktop, Android and WebView (chromestatus; Chrome 136 release
  notes). The CSS Color HDR draft gives `standard | no-limit | constrained | dynamic-range-limit-mix()`, initial
  `no-limit`. `standard` caps at HDR reference white; `constrained` is "somewhat greater … such that a mix of SDR and
  HDR content can be comfortably viewed together".
- Canvas: WebGPU "extended range (HDR)" is on from 129 (chromestatus). "Canvas2D tone mapping" (`toneMapping:
  'extended'`) is Proposed for 160, so a 2D canvas cannot show HDR in Chrome today. The "HDR headroom query" is only
  Proposed.

**Safari.**
- Safari 26.0: "Added support for HDR Images on iOS 26, iPadOS 26, macOS 26 and visionOS 26". It also added "HDR support
  in WebGPU Canvas", and partial `dynamic-range-limit` support: "`standard` and `no-limit`, non-animatable". The WebKit
  blog: "Using standard converts all of the HDR content to SDR" (WebKit Features in Safari 26.0).
- Later releases fixed gain maps in particular cases:
  - 26.2: HDR images in CSS backgrounds, borders and SVG.
  - 26.3: "positioned or transformed `<img>` elements containing HDR JPEGs with gain maps would incorrectly render as
    SDR".
  - 26.4: the same for "opacity-altered" `<img>`.
  - 27: GPU-backed gain-map decoding, and RGB gain maps that had been decoded to 8 bits.
  - Source: the Safari release notes.
- The iPhones in the question all run iOS 26, which supports iPhone 11 and later (Apple Support).
- **Unconfirmed**: which formats Safari renders as HDR besides gain-map JPEGs. None of these pages names Apple's HEIC
  gain map, AVIF gain maps, PQ or HLG AVIF, or HDR JPEG XL. Safari's `color-gamut` and `dynamic-range` are shipped,
  with `video-dynamic-range` not supported (chromestatus's Safari note).

**Limited headroom and low power.**
- Apple: `UIScreen.currentEDRHeadroom` "limits all rendered content, and can change depending on its configuration".
  `potentialEDRHeadroom` can change too. Android: `getHdrSdrRatio()` is the current ratio. Ultra HDR: the weight falls
  smoothly to the SDR rendition as `max_display_boost` falls.
- So the same file shows less above white at high screen brightness or in sunlight. The base image is what remains.
- Safari 17 "Added support for enforcing low-power mode … by tone mapping HDR video to SDR". **Unconfirmed**: whether
  iOS Low Power Mode or Android's Battery Saver limits HDR images. No primary source says so.

## 4. JPEG XL: can the canonical photo be served directly?

- **Safari**: "Added support for JPEG XL" in Safari 17 (release notes; WebKit Features in Safari 17.0).
- **Chrome**: the earlier libjxl-based entry never shipped (chromestatus 5188299478007808, still "Proposed"). The Intent to Ship for jxl-rs decoding was posted on 2026-08-24, with
  three LGTMs and "Yes" to all six Blink platforms, Android included (blink-dev).
  - The Chrome 155 beta post, "in beta as of September 16, 2026", lists "JPEG XL decoding support (image/jxl) in blink".
  - chromiumdash gives Chrome 155 stable on 2026-10-06. Android stable is 158 today.
  - In source, `enable_jxl_decoder = true` (`third_party/blink/renderer/config.gni`) and
    `kJXLImageFormat` is `FEATURE_ENABLED_BY_DEFAULT`.
  - chromestatus still shows the entry as "Proposed", milestone 155. **Unconfirmed**: that 155 stable shipped it on.
    The developer.chrome.com release-notes page for 155 is a 404, and the index points to chromestatus.
- **Colour**: Chrome's JPEG XL decoder takes the colour profile from jxl-rs as `output_color_profile().try_as_icc()`.
  A bare codestream's enumerated Rec.2020 / BT.709 encoding, like the canonical photo's, would get a synthesised ICC
  and be colour-managed (source reading; **unconfirmed** by a test). For Safari it is **unconfirmed**.
- **Verdict**: serving the canonical photo directly would bring the gamut along where the browser decodes it. It cannot
  bring brightness, because the file holds none above white (§1). It also costs a full canonical photo (up to 18 MP, ADR 0036) where the
  thumbnail's long side is 1024 px, and Chrome below 155 and other browsers cannot show it. The thumbnail exists for
  reasons the format does not change (`index.html:1017`).

## 5. The panels (manufacturers' spec pages)

| Phone | Panel | SDR / typical | HDR peak | Outdoor / peak | Gamut, depth (as stated) |
|---|---|---|---|---|---|
| Pixel 9 Pro | 6.3" Super Actua, LTPO OLED, 1280 × 2856, 1–120 Hz | — | up to 2,000 nits (HDR) | up to 3,000 nits (peak) | "HDR support", "Full 24-bit depth for 16 million colors"; no gamut stated |
| iPhone 15 Pro | 6.1" Super Retina XDR OLED, ProMotion to 120 Hz | 1000 nits max (typical) | 1600 nits (HDR) | 2000 nits (outdoor) | "HDR display", "Wide color (P3)" |
| iPhone 16 Pro | 6.3" Super Retina XDR OLED | 1000 nits | 1600 nits (HDR) | 2000 nits (outdoor); 1 nit min | same |
| iPhone 17 Pro | 6.3" Super Retina XDR OLED, anti-reflective | 1000 nits | 1600 nits (HDR) | 3000 nits (outdoor); 1 nit min | same |

Google's footnote: "HDR brightness measured at 100% on-pixel ratio. Peak brightness measured at 5% on-pixel ratio"
(Pixel tech specs; the footnote is read from that page, which shares it across phones). The Pixel 9 Pro specs are on
Google's support page, because the Store page now shows the Pixel 11 range. Google publishes neither the Pixel 9 Pro's
gamut coverage nor any panel bit depth beyond "24-bit". Apple publishes no bit depth.

## 6. What a preview could gain here

**(a) Gamut: a serving change only.**
- The canonical photo keeps Rec.2020 colours (ADR 0033: "so the stored photos keep their wide-gamut colour even though
  the models read sRGB today"). The preview throws them away at `load_canonical`'s 8-bit sRGB read-back, before the
  JPEG is made (§1).
- A preview converted from the canonical photo's Rec.2020 samples to Display P3 would keep the colours that are inside
  P3 but outside sRGB. On a wide-gamut screen (§2) the browser would show them.
- Possible forms: a JPEG with a Display P3 ICC profile (the format Ultra HDR's own base image uses: "Tip: Use a
  Display-P3 color profile"), a PNG with `cICP`, or an AVIF with nclx 12/13. A 10-bit AVIF avoids P3's coarser 8-bit
  steps.
- What it costs:
  - A second decode or conversion on the inference host (or the front).
  - `_preview_part`'s promise, "the pixels the model sees", becomes "the photo the model sees, in a wider gamut".
  - The overlays stay as they are: the canvases draw only the tint and outline.
- How much it would show on roasted beans is not measured here. A count of canonical pixels outside sRGB would answer
  that before any code is written.

**(b) Brightness: needs an HDR rendition the canonical photo no longer has.**
- ADR 0041 applies the map and tone-maps back into standard range, so the light above white is gone. More container
  bits would not bring it back (§3).
- The options, none decided:
  1. **Show the visitor's own file in the page**, through an object URL. No bytes cross the slow link, and the browser
     applies the file's own gain map: Chrome for Ultra HDR JPEG and AVIF (source reading); Safari 26+ for gain-map
     JPEG.
     - It reverses OPS-7 D8 and the comments at `index.html:1017` and `app.py:526–528`.
     - It works only for formats the browser decodes: no RAW, no HEIC outside Safari, and HEIC's Apple gain map in
       Safari **unconfirmed**.
     - The preview is no longer the canonical photo. Its standard-range base is darker than the map-applied canonical
       photo (ADR 0041: median +20.1 of 255).
     - The crop box and mask are fractions of the upright canonical photo. They line up only if the browser turns the
       file by its EXIF orientation as canonicalization does. CSS's default `image-orientation: from-image` suggests
       it does, but that is **unconfirmed** here.
  2. **Make an HDR preview from the pre-tone-map light**, as an Ultra HDR JPEG or a gain-map AVIF. The bundled
     `imagecodecs` 2026.10.10 has `ultrahdr_encode` (float16 linear, "1.0 == 203 nits", optional SDR companion,
     `gamut` and `transfer` arguments; read from its docstring). libavif writes ISO 21496-1 maps. Where it is made
     matters:
     - The light above white exists only on the front, during canonicalization. The inference host receives only the
       canonical photo (ADR 0033).
     - Making it on the front means a second artefact beside the canonical photo, in the one streamed request (ADR
       0023), and more memory against the front's 600 MB (ADR 0041, R6).
     - Making it on the host means the canonical photo must carry the map: a JPEG XL gain map, which Chrome's jxl-rs
       path reads (source reading). That changes the canonical format, which is "bare codestream … nothing else"
       (`canonical.py`).
     - Either way it amends ADR 0033 and ADR 0041, and ADR 0036 for the front's settings.
     - A map in the canonical photo would also change what the store and training read, so it is a
       re-canonicalization question (ADR 0033, Consequences).
  3. Leave brightness out and take (a) only.
- `dynamic-range-limit` decides how bright an HDR preview gets beside the page's SDR text. Safari supports only
  `standard` and `no-limit`. `constrained` exists in Chrome's draft only (§3).
- Any server-made preview must carry no EXIF or other metadata (ADR 0013), as today's does not.

**Closed ideas.** None of the five in `.out-of-scope/` (containerised serving, CPU priority for sweeps, cross-camera
evaluation, ResNet18 recipe levers, a visitor-analytics service) touches previews, colour or HDR. Decisions touched:
ADR 0033, 0036 and 0041 (option b2), ADR 0023 (one streamed request), OPS-7 D8 (b1), ADR 0013 (metadata).

## Not confirmed

- That Chrome on a Pixel 9 Pro reports a wide-gamut screen, in "Adaptive" or in "Natural"; and that Pixel's
  "Adaptive" is AOSP's Automatic (enhanced) mode.
- That a gain-map image in Chrome on the Pixel 9 Pro reaches above SDR white, and from which Chrome version gain maps
  were on by default (the "116" figure is from a Mozilla bug, not from Chrome).
- Whether `dynamic-range: high` matches on the Pixel 9 Pro (its panel is stated as 24-bit).
- That Chrome 155 stable shipped JPEG XL enabled (chromestatus still says Proposed; the beta post and the source say
  on), and how Safari colour-manages an enumerated-colour JPEG XL.
- Which formats Safari 26+ shows as HDR besides gain-map JPEGs: Apple HEIC gain maps, AVIF gain maps, PQ/HLG AVIF,
  HDR JPEG XL.
- Whether Safari honours PNG `cICP`.
- Whether Low Power Mode (iOS) or Battery Saver (Android) limits HDR images; only Safari's video tone mapping is
  documented.
- That the page's `<img>` turns a visitor's file by EXIF orientation as canonicalization does (option b1).
- The Pixel 9 Pro's gamut coverage and panel bit depth; Apple's bit depth.

## Sources

Specifications
- W3C CSS Color 4, §3.4–3.5: <https://www.w3.org/TR/css-color-4/>
- CSS Media Queries 5, §6.4 `color-gamut`, §6.5 `dynamic-range` (editor's draft): <https://drafts.csswg.org/mediaqueries-5/>
- CSS Color HDR 1, §2.2 headroom, §3.1 `dynamic-range-limit` (editor's draft): <https://drafts.csswg.org/css-color-hdr-1/>
- WHATWG HTML, canvas, "Color spaces and color space conversion": <https://html.spec.whatwg.org/multipage/canvas.html>
- WebGPU, `GPUCanvasConfiguration`: <https://gpuweb.github.io/gpuweb/>
- Ultra HDR Image Format v1.1 (Android): <https://developer.android.com/media/platform/hdr-image-format>

Chrome and Chromium
- chromestatus API (fetched 2026-11-18): JPEG XL <https://chromestatus.com/feature/5114042131808256>; `dynamic-range-limit`
  <https://chromestatus.com/feature/5146250411769856>; canvas colour management
  <https://chromestatus.com/feature/5807007661555712>; `dynamic-range` <https://chromestatus.com/feature/5680926106320896>;
  WebGPU extended range <https://chromestatus.com/feature/6196313866895360>; Canvas2D tone mapping
  <https://chromestatus.com/feature/5077164932333568>; HDR headroom query <https://chromestatus.com/feature/5172386273689600>;
  WebGL canvas colour management <https://chromestatus.com/feature/4814886323355648>; `color-gamut`
  <https://chromestatus.com/feature/5354410980933632>
- Intent to Ship, JPEG XL (blink-dev, 2026-08-24): <https://groups.google.com/a/chromium.org/g/blink-dev/c/-gDojQbDPRI>
- Chrome 155 beta: <https://developer.chrome.com/blog/chrome-155-beta>; Chrome 136 release notes:
  <https://developer.chrome.com/release-notes/136>; release-notes index: <https://developer.chrome.com/release-notes>
- chromiumdash schedule and releases: <https://chromiumdash.appspot.com/fetch_milestone_schedule?mstone=155>,
  <https://chromiumdash.appspot.com/fetch_releases?channel=Stable&platform=Android&num=3>
- Chromium source at `main`, <https://chromium.googlesource.com/chromium/src/+/main/>:
  `third_party/blink/renderer/platform/image-decoders/{jpeg/jpeg_image_decoder.cc, avif/avif_image_decoder.cc,
  jxl/jxl_image_decoder.cc, image_decoder.cc, BUILD.gn}`, `third_party/blink/renderer/platform/graphics/deferred_image_decoder.cc`,
  `third_party/blink/renderer/config.gni`, `third_party/blink/common/features.cc`,
  `third_party/blink/renderer/platform/runtime_enabled_features.json5`, `third_party/rust/jxl/v0_8/wrapper/lib.rs`,
  `ui/android/java/src/org/chromium/ui/base/WindowAndroid.java`,
  `ui/android/java/src/org/chromium/ui/display/{DisplayAndroid,PhysicalDisplayAndroid}.java`
- Secondary, cited only as unconfirmed: Mozilla bug 1832622 <https://bugzilla.mozilla.org/show_bug.cgi?id=1832622>

Android and Pixel
- Displaying Ultra HDR: <https://developer.android.com/media/grow/ultra-hdr/display>
- Wide colour content: <https://developer.android.com/training/wide-color-gamut>
- `Configuration.isScreenWideColorGamut()`: <https://developer.android.com/reference/android/content/res/Configuration#isScreenWideColorGamut()>
- `Display.getHdrSdrRatio()`: <https://developer.android.com/reference/android/view/Display#getHdrSdrRatio()>
- AOSP colour management: <https://source.android.com/docs/core/display/color-mgmt>
- AOSP `DisplayTransformManager.java`, `ColorDisplayManager.java`:
  <https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/services/core/java/com/android/server/display/color/DisplayTransformManager.java>,
  <https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/core/java/android/hardware/display/ColorDisplayManager.java>
- Pixel display colours: <https://support.google.com/pixelphone/answer/6111557?hl=en>
- Pixel hardware tech specs (Pixel 9 Pro): <https://support.google.com/pixelphone/answer/7158570?hl=en>

Apple and WebKit
- Safari release notes (17, 15.2, 16.4, 26, 26.2, 26.3, 26.4, 27):
  <https://developer.apple.com/documentation/safari-release-notes/safari-17-release-notes>,
  <https://developer.apple.com/documentation/safari-release-notes/safari-15_2-release-notes>,
  <https://developer.apple.com/documentation/safari-release-notes/safari-16_4-release-notes>,
  <https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes>,
  <https://developer.apple.com/documentation/safari-release-notes/safari-26_2-release-notes>,
  <https://developer.apple.com/documentation/safari-release-notes/safari-26_3-release-notes>,
  <https://developer.apple.com/documentation/safari-release-notes/safari-26_4-release-notes>,
  <https://developer.apple.com/documentation/safari-release-notes/safari-27-release-notes>
- WebKit Features in Safari 26.0: <https://webkit.org/blog/17333/webkit-features-in-safari-26-0/>
- WebKit Features in Safari 17.0: <https://webkit.org/blog/14445/webkit-features-in-safari-17-0/>
- Improving Color on the Web (WebKit): <https://webkit.org/blog/6682/improving-color-on-the-web/>
- `UIScreen.currentEDRHeadroom`, `potentialEDRHeadroom`:
  <https://developer.apple.com/documentation/uikit/uiscreen/currentedrheadroom>,
  <https://developer.apple.com/documentation/uikit/uiscreen/potentialedrheadroom>
- Tech specs: iPhone 15 Pro <https://support.apple.com/en-us/111829>, iPhone 16 Pro
  <https://support.apple.com/en-us/121031>, iPhone 17 Pro <https://support.apple.com/en-us/125090>
- iOS 26 compatible iPhones: <https://support.apple.com/guide/iphone/iphone-models-compatible-with-ios-26-iphe3fa5df43/26/ios/26>

This repo
- `webapp/static/index.html:337–340, 624, 637, 789, 1017–1042`; `webapp/app.py:67, 484, 525–536`;
  `coffeecv/canonical.py` and `coffeecv/gain_map.py` docstrings; ADR 0013, 0023, 0033, 0036, 0041;
  [research_tone_mapping.md](research_tone_mapping.md); `.out-of-scope/`.
