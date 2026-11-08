---
status: accepted
---

# Only accepted formats reach a decoder, and they are listed once, in a data file

A photo is read only if it is in an accepted format (GLOSSARY.md, "Accepted format"): one the page and `llms.txt` name. The accepted formats are written once, in `coffeecv/data/accepted_formats.csv`: one row per format, giving its chip label, extensions, MIME types and decoder. One recogniser in `coffeecv/canonical.py` tells a file's row from its first bytes and its name. The front's check refuses a file with no row, and canonicalization opens a file with its row's decoder only, never Pillow's other plugins. Training's store reads the same file, so an accepted format means the same in training and in serving. Every other list of formats in the code, the refusal message included, is read from the file, and a test holds the public texts (`llms.txt`, `openapi.json`, the page's chips and `accept` lists) to it.

We chose to refuse formats that a decoder could read (a plain TIFF, and anything else Pillow knows) over letting canonicalization decide, because the front is a public endpoint and Pillow 12.3.0, when a file fails the parser its signature picked, tries its other 42 plugins in turn. We chose one data file and one recogniser over a list in each module held equal by a test, because the lists that existed had already drifted: OPS-15 dropped GIF and BMP without deciding to, and the front and canonicalization told HEIF apart by different brand rules.

## Consequences

- A new format, TIFF included, is a new row and its recogniser rule, the contract's texts updated in the same change, and a test that canonicalizes it. Nothing reads a format that has no row.
- The file ships in both releases' manifests, since both hosts import `canonical.py`, which reads it.
- Dataset intake keeps its own, narrower list (`strip_metadata.PHOTO_EXTENSIONS`): what metadata stripping is checked on. `canonicalize_folder` reports a file in `dataset/` with no row and goes on.

Source: [OPS-18](../ticket_gif_bmp_uploads.html) D3, D4, D10, D11, D12, D17.
