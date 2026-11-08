# Example photos

The originals of the example photos the page offers (GLOSSARY.md, "Example photo"; ticket OPS-16), moved
here from the dataset on 2026-11-03 (ticket ML-16 D3). Their lossy JPEG XL re-encodes are public, in
`webapp/static/examples/` (ADR 0032); these originals stay private, like the rest of `dataset/`.

The page offers four since ML-16's release (ML-16 D12, 2026-11-08): the Vietnam one, `PXL_20261007_043818022.jpg`,
left the list when `xl0_v7r_s123` found no beans in it and the whole-photo fallback answered Brazil. Its original
stays here, still in no set, so a later release can bring it back on a segmenter that finds its pile.

**Read by no pipeline, and in no training or evaluation set**: this folder has no `class_*` folders, is not in
`dvc.yaml`'s segcrop list, is listed in `coffeecv/coverage_report.py`'s `EXCLUDED`, and none of its photos is in
`labels/ml5/seg_dataset.yaml` or an OOD manifest. A model the page's answer comes from has not seen them once it is
trained after this move. Only `scripts/make_example_photos.py` reads them, finding each by its sha256.

`example_photos.manifest.csv` gives each photo's example name (`examples.json`), its country (empty for the
negative), its sha256, the path it was moved from, and any byte copy deleted with it (the two copies in
`dataset/segmenter_positives/`). The bytes are unchanged: all five were stripped where they were before
(`labels/ml3/strip_manifest.csv`), and a re-check here found no private tags; the HEIC keeps its HEIF thumbnail
item, which exiftool cannot remove.

A new example photo comes from outside every set, or leaves them before the next retrain (ADR 0032).
