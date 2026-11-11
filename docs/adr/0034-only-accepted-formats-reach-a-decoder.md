---
status: accepted
---

# Only accepted formats reach a decoder, and they are listed once, in a data file

A photo is read only if it is in an accepted format (GLOSSARY.md, "Accepted format"): one the page and `llms.txt` name. The accepted formats are written once, in `coffeecv/data/accepted_formats.csv`: one row per format, giving its chip label, extensions, MIME types and decoder. One recogniser in `coffeecv/canonical.py` tells a file's row from its first bytes and its name. The front's check refuses a file with no row, and canonicalization opens a file with its row's decoder only, never Pillow's other plugins. Training's store reads the same file, so an accepted format means the same in training and in serving. Every other list of formats in the code, the refusal message included, is read from the file, and a test holds the public texts (`llms.txt`, `openapi.json`, the page's chips and `accept` lists) to it.

We chose to refuse formats that a decoder could read (a plain TIFF, and anything else Pillow knows) over letting canonicalization decide, because the front is a public endpoint and Pillow 12.3.0, when a file fails the parser its signature picked, tries its other 42 plugins in turn. We chose one data file and one recogniser over a list in each module held equal by a test, because the lists that existed had already drifted: OPS-15 dropped GIF and BMP without deciding to, and the front and canonicalization told HEIF apart by different brand rules.

Amended 2026-11-09 by [OPS-21](../ticket_android_image_and_file.html) D5: on Android the page's three inputs depart from "every accepted format and nothing else", because what Chrome and Firefox open follows from the `accept` list. Choose Photo's list is the file's MIME types alone: an extension in it keeps Chrome from its photo picker. Select File's names no format, since an image type in it brings the camera back: it is one made-up extension, `.coffeecv`, that no MIME table knows, so both browsers open Files unfiltered (D11; under `application/octet-stream` Firefox's Files greyed out every photo). The camera button's is `image/*`, since a camera writes JPEG or HEIC and that list is the one that opened the camera directly. The front's check still refuses a file with no row, whatever list it was picked through, and `tests/test_accepted_formats_named.py` holds all five of the page's lists, the Android three included.

Amended 2026-11-11 by [ML-25](../ticket_intake_2026_11_10.html) D6: dataset intake takes GIF. The owner kept a GIF in the dataset for diversity; canonicalization already read it, but intake's list did not name it, so its metadata went unchecked and the segmenter dataset left it out. `strip_metadata` now strips a GIF's XMP, keeps its comment and compares every frame; intake's list is still narrower than the accepted formats.

## Consequences

- A new format, TIFF included, is a new row and its recogniser rule, the contract's texts updated in the same change, and a test that canonicalizes it. Nothing reads a format that has no row.
- The file ships in both releases' manifests, since both hosts import `canonical.py`, which reads it.
- Dataset intake keeps its own, narrower list (`strip_metadata.PHOTO_EXTENSIONS`): what metadata stripping is checked on. GIF joined it on 2026-11-11 (ML-25 D6). `canonicalize_folder` reports a file in `dataset/` with no row and goes on.

Source: [OPS-18](../ticket_gif_bmp_uploads.html) D3, D4, D10, D11, D12, D17; [ML-25](../ticket_intake_2026_11_10.html) D6.
