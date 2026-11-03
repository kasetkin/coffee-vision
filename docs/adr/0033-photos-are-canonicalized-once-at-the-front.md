---
status: accepted
---

# Photos are canonicalized once, at the front, and both models read only canonical photos

Every photo both models read is first turned into its canonical photo (GLOSSARY.md, "Canonicalization"): the EXIF rotation applied, a photo over 60 MP downscaled to 50 MP, colour-managed from its own profile, its main image only, no metadata, stored as lossy JPEG XL at quality 95 in Rec.2020 and decoded to 8-bit sRGB for the models. One library, `imagecodecs` at one exact version, does all of it that touches colour or JPEG XL: both colour conversions, by the LittleCMS it bundles, at float precision, and the JPEG XL encode and decode, by the libjxl it bundles. The same version runs in training, on the front and on the inference host. A visitor's photo is canonicalized on the front, before the slow link to the inference host, which accepts only canonical photos. The dataset is canonicalized into a DVC store that the segmenter's labels, its fine-tune, the crop stage and the OOD tools read; the originals in `dataset/` stay untouched.

We chose this over decoding each original as it comes (today), because that path ignores colour profiles and orientation, so 47% of the dataset reached the models as Display P3 numbers read as sRGB. We chose one lossy generation at the front over canonicalizing on the inference host, because only the front can make the photo smaller before the 2.1 Mbit/s link, and over canonicalizing on both, which would encode a photo twice. We chose Rec.2020 storage over storing sRGB, the owner's call, so the stored photos keep their wide-gamut colour even though the models read sRGB today. We chose `imagecodecs` over Pillow's ImageCms, which converts 8-bit RGB only and would band Rec.2020; over OpenCV, which has no ICC colour management; over pyvips, whose wheel has no JPEG XL and whose 16-bit conversion goes through a lossy lookup table; and over `pillow-jxl-plugin`, which cannot write the Rec.2020 encoding and is GPL-3.0. One wheel that bundles both C libraries means one pin holds both, where separate packages could drift apart (`analysis/ml19/research_16bit_colour.md`).

## Consequences

- **Amends [ADR 0031](0031-the-front-and-the-inference-host.md):** the front gets the decoders, colour management and a JPEG XL encoder, and about 500 MB of memory. It is no longer standard-library only.
- Training and serving read the same canonical photo, so a change to canonicalization (its quality, effort, cap or colour maths) changes both models' input and needs a re-canonicalization and a retrain. Tests on every commit hold the colour conversion fixed.
- A change of `imagecodecs`' version is a change to canonicalization: it moves LittleCMS or libjxl, so it needs a re-canonicalization and a retrain, never a routine upgrade. A lock check on every commit refuses a package held at different versions in training's, the web service's and the front's locks. The Rec.2020 profile is a committed file, its hash tested, since profiles that differ only in rounding move pixels.
- `pillow-jxl-plugin` leaves every environment: `imagecodecs` decodes a visitor's JPEG XL too.
- Label masks line up with canonical photos, which are rotated. Earlier sessions' masks line up with the unrotated originals, and are carried into the canonical geometry, not reused as they are.

Source: [ML-19](../ticket_canonicalization.html) D1, D3, D5, D7, D12, D27, D31, D41, D44, D45, D46.
