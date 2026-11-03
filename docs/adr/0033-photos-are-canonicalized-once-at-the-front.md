---
status: accepted
---

# Photos are canonicalized once, at the front, and both models read only canonical photos

Every photo both models read is first turned into its canonical photo (GLOSSARY.md, "Canonicalization"): the EXIF rotation applied, at most 50 MP, colour-managed from its own profile, its main image only, no metadata, stored as lossy JPEG XL at quality 95 in Rec.2020 and decoded to 8-bit sRGB for the models. A visitor's photo is canonicalized on the front, before the slow link to the inference host, which accepts only canonical photos. The dataset is canonicalized into a DVC store that the segmenter's labels, its fine-tune, the crop stage and the OOD tools read; the originals in `dataset/` stay untouched.

We chose this over decoding each original as it comes (today), because that path ignores colour profiles and orientation, so 47% of the dataset reached the models as Display P3 numbers read as sRGB. We chose one lossy generation at the front over canonicalizing on the inference host, because only the front can make the photo smaller before the 2.1 Mbit/s link, and over canonicalizing on both, which would encode a photo twice. We chose Rec.2020 storage over storing sRGB, the owner's call, so the stored photos keep their wide-gamut colour even though the models read sRGB today.

## Consequences

- **Amends [ADR 0031](0031-the-front-and-the-inference-host.md):** the front gets the decoders, colour management and a JPEG XL encoder, and about 500 MB of memory. It is no longer standard-library only.
- Training and serving read the same canonical photo, so a change to canonicalization (its quality, effort, cap or colour maths) changes both models' input and needs a re-canonicalization and a retrain. Tests on every commit hold the colour conversion fixed.
- Label masks line up with canonical photos, which are rotated. Earlier sessions' masks line up with the unrotated originals, and are carried into the canonical geometry, not reused as they are.

Source: [ML-19](../ticket_canonicalization.html) D1, D3, D5, D7, D12.
