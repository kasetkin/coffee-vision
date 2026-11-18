---
status: accepted
---

# A photo's gain map is applied at canonicalization, by this repo's own code

When a photo carries a gain map (GLOSSARY.md, "Gain map"), canonicalization applies it to the main image at full weight and tone-maps the result back into the canonical photo's standard range, with the operator an HDR main image gets (ML-29 D33). The store's settings and the front's do the same, so training and serving keep reading the same kind of photo. A file with two maps uses the ISO 21496-1 one first, then the vendor's (Ultra HDR, Apple). In the dataset, 1,687 of the 2,772 photos carry a map that is applied. Their canonical photos came out brighter at the median by 20.1 of 255 levels (ML-29 P8).

The owner adopted this on 2026-11-18, although the bar fixed before the results (D16) was not met. That bar was a paired three-seed improvement beyond the seed range with no class's F1 dropping beyond it. With the classifier retrained on gain-map crops (exp289–291 against exp286–288), the change on test averaged −0.0009 (from −0.0032 to +0.0031). The photos the two arms shared moved as much as the swapped ones. The owner's reason: "they do not break anything on our dataset, but it theory can have positive effect with new photos." The rejected alternative was what D16's numbers favoured: the main image only, as before. It reads every photo as the phone shows it in standard range, not as the HDR version the phone also stored.

The maps are read and applied by `coffeecv/gain_map.py`, not by a library (D37). The bundled libultrahdr agrees with it within 0.006–0.014 stops on 60 dataset JPEGs, and stays the check on it. It was not chosen because it refuses the per-channel Ultra HDR form, reads no HEIF or AVIF, and stops at 8192 × 8192. `gain_map.py`'s docstring opens with where to look first when a gain-map photo comes out wrong. A canonical photo made with a map names the map in its decoder (`jpeg+Ultra HDR`).

A map is applied to the light in linear Rec.2020 (D38). That is exact only for a map that scales red, green and blue alike: one channel, and one value in each per-channel field. Every dataset photo's map is one. ISO 21496-1 applies other maps in the base image's or the alternate image's primaries. Those maps are left unapplied, and the photo is its main image, until a ticket implements the primaries.

## Consequences

- **Amends [ADR 0033](0033-photos-are-canonicalized-once-at-the-front.md):** the canonical photo is made from the main image with its gain map applied, not from the main image only.
- A change to `coffeecv/gain_map.py` is a change to canonicalization, like one to `canonical.py`. It needs the store re-canonicalized and the models retrained. `dvc.yaml` names it as a dependency of the canonicalize, segcrop and test-mask stages.
- The front's release carries `gain_map.py`. A map is applied strip by strip into one float copy of the photo, and a map of more than 16 million samples (pixels times channels, `canonical.MAP_MAX_SAMPLES`) is left unapplied before it is made float, so the front stays under its 600 MB (R6). On this machine, held to 2 CPUs: 394–484 MB and 14–18 s for the dataset's largest gain-map photos of each form, against 264–304 MB and 2.5–4 s without (`analysis/ml29/p11_front_cost.py`). The dataset's largest map has 12.48 million samples.
- A map no reader here understands is no map: three iPhone 13 HEICs in the dataset.
- The segmenter in service (`xl0_v7r_s123`) was fine-tuned on canonical photos made before this. Its next fine-tune ([ML-33](../ml-0033_ticket_segmenter_on_pass17.html)) reads the new ones.

Source: [ML-29](../ml-0029_ticket_what_canonicalization_reads.html) D10, D12, D14, D15, D16, D33, D37 and D38; P8 to P11 (`analysis/ml29/README.md`).
