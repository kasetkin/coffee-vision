---
status: accepted
---

# A colour declaration is honoured or refused, never guessed

A photo file can state what its colour numbers mean in several ways (GLOSSARY.md, "Colour declaration"): an ICC profile, a coded tag (HEIF's and AVIF's `nclx`, PNG's `cICP`), a JPEG XL header enumeration, or the EXIF interoperability tag R03 for Adobe RGB. Canonicalization honours whichever the file carries, in that order of precedence, and reads a photo as sRGB only when it declares nothing or declares "unspecified". A declaration it cannot convert from is a refusal with a message that names it, never a silent read as sRGB. Until this decision every format but JPEG XL was read through its ICC profile or as sRGB, so a file whose only declaration was a coded tag was accepted and read wrong: of the owner's darktable exports of one photo, a linear Rec.2020 AVIF and HEIC reached the models 65 of 255 levels off, and a Display P3 HEIC 2 levels off (`analysis/ml29/README.md`).

We chose this over reading every unrecognised declaration as sRGB, which accepts more files but gives a wrong answer with nothing to show it was wrong, where a refusal tells the visitor what to export instead. We chose the ICC profile first, over the coded tag first, because the HEIF and AVIF specifications give the profile precedence for colour, and because it keeps every photo that has a profile, which is every HEIC and 1,715 of the 2,556 JPEGs in the dataset, reading exactly as it does today. We chose to build each source profile in code from the declared primaries, white point and curve, over a committed profile per declaration, because custom primaries (darktable's Adobe RGB JPEG XL) make a fixed set impossible.

## Consequences

- Two legacy declarations are deliberately not honoured, and their files are read as sRGB: PNG's `gAMA` and `cHRM` chunks, and a BMP header's calibrated endpoints. They are often wrong or zero in files found in the wild, browsers disagree on them, and honouring a pure 2.2 gamma would shift 7 dataset PNGs that carry sRGB's own values. PNG's `iCCP`, `cICP` and `sRGB` chunks and a BMP's embedded ICC profile are honoured.
- A JPEG XL with an embedded ICC profile is refused, not guessed: the profile sits compressed inside the codestream and `imagecodecs` at its pinned version does not return it.
- [ML-19](../ml-0019_ticket_canonicalization.html) D16 stands for a photo with no declaration, or with an ICC profile LittleCMS cannot use as RGB: it is read as sRGB.
- A declaration with the PQ or HLG curve is honoured by tone-mapping into the canonical photo's standard range, which [ADR 0033](0033-photos-are-canonicalized-once-at-the-front.md) keeps.
- The canonical photo's own committed Rec.2020 profile (ML-19 D46) is not built in code and does not change.

Source: [ML-29](../ml-0029_ticket_what_canonicalization_reads.html) D4, D5, D7, D8, D20 and D22, decided by the owner on 2026-11-14.
