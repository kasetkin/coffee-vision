---
status: accepted
---

# The canonical photo is linear light, read back once

The canonical photo (GLOSSARY.md, "Canonicalization") stores linear light: lossy JPEG XL, Rec.2020 primaries, D65, the linear transfer, 16-bit integers, intensity target 203 (BT.2408's reference white), values in [0, 1]. Light above white is still rolled off by ML-29 D33's EETF at canonicalization, so nothing above white is kept. Every reader opens it the same one way, `coffeecv/readback.py`: the uint16 samples, numpy's 3 × 3 matrix to linear sRGB, clipped, the sRGB curve, rounded to 8 bits once. That picture is the model picture: both models read it in training and in serving, and the preview is made from it. The store and the front make the photo the same way, LittleCMS on float32, and differ only in effort, quality and cap.

The owner chose linear light because the BT.709 storage curve it replaces had two readings: our read-back undid it exactly through a committed profile, but a viewer shown the file applied a display gamma, and Chromium drew canonical photos about 11 levels darker (OPS-25 D8). Half-float was rejected: research found no measurable gain for either model above 8 bits, and linear uint16 compresses exactly as well as half-float on the same picture (`analysis/ml37/research_canonical_int_vs_half.md`). Keeping 8-bit model input, rounded once, was chosen over a float path to the models for the same reason. numpy does the read-back rather than LittleCMS: it is within 0.0002 of a level of a float64 reference where LittleCMS's s15.16 profiles are 0.15 off, and about twice as fast. The front's integer colour path (LittleCMS's optimized transform) was dropped for the float one: into linear uint16 it quantizes light to 1/16384, up to 0.8 of a level before the encode.

## Consequences

- **Amends [ADR 0033](0033-photos-are-canonicalized-once-at-the-front.md)** (ML-19 D7, D31, D46): the canonical photo is no longer Rec.2020 with the BT.709 curve, and one library no longer does all of the colour: LittleCMS converts originals into linear Rec.2020, and numpy reads the photo back. numpy is pinned exactly, as imagecodecs is (`readback.NUMPY`); a new version is a re-test of every model input. `coffeecv/data/rec2020_bt709.icc` stays, its hash tested, as the source profile of a JPEG XL that declares Rec.2020 with the BT.709 curve, such as a canonical photo made before this change and sent as an original.
- **Amends [ADR 0036](0036-the-front-makes-its-own-canonical-photo.md)** (ML-37 D12): the front's colour is the store's float path, for every upload. Its encoder settings and its cap still differ from the store's.
- The inference host accepts only this form. A canonical photo in the form before it is refused, naming its transfer and intensity target, so the store, the front and the host change in one release, and the example photos are re-made in it.
- A tool that reads the samples without the read-back sees linear light as if it were a curve: a much darker picture. Every reader in this repo goes through `canonical.load_canonical` or `readback.to_srgb`; the research scripts under `analysis/` that read the form before are records, left as they were.

Source: [ML-37](../ml-0037_ticket_linear_canonical_photo.html) D1, D2, D3, D8, D9, D12 and D19; P3.
