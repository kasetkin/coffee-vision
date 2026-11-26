---
status: accepted
---

# The front draws the preview

The preview (GLOSSARY.md) is made at the front, not on the inference host. Once the front's check service has made a photo's canonical photo, it opens the visitor's answer itself, a 200 of NDJSON lines, and its first line is `{"part": "preview", "jpeg": ...}`: the canonical photo read back as the host reads it (`canonical.load_canonical`, the model picture), shrunk to 1,024 px and sent as a JPEG (`coffeecv/preview.py`), made on a thread while the photo crosses to the host. The host sends no picture: its first line is `{"part": "segmentation"}`, the mask and the box, then the classification or an error, and the front relays them after its own line, never before. A refusal the front gives before the stream (too large, not an image, unreadable, busy at its gate) stays an HTTP status; once the preview has gone out, a host that is offline, busy or fails becomes an error line, `{"part": "error", "outcome": "offline" | "busy" | "failed", "message": ...}`, which the page shows as the same refusal. A preview the front cannot make is left out and logged, and the host's lines still go.

The owner chose this because the picture reached the browser only after the photo had crossed the slow link and been decoded and segmented on the host, about 5 s after the front already held the same picture. Made at the front it is on the page about 0.7 s after the canonical photo, its making overlaps the crossing, and the page still shows it when the host is offline or busy. It is the models' picture to the level, from the same read-back on the same file. Shrinking in linear light before the read-back was rejected: 0.47 s saved off the visitor's wait for a picture that is not the models' (up to 94 levels off at edges). Keeping the picture in the host's line too, so a caller reading the old second-line rule still finds one, was rejected: 150 KB more across the slow link, for no visitor.

## Consequences

- **Amends [ADR 0023](0023-webapp-answers-a-photo-in-one-streamed-request.md):** the answer's first line is the front's preview, not the host's thumbnail with the mask and box; those come in the host's segmentation line after it. The public API's line order changes (`openapi.json`, `llms.txt`), and the error line carries an `outcome` when the front writes it.
- **Amends [ADR 0031](0031-the-front-and-the-inference-host.md):** the front no longer only relays the host's answer; it writes the answer's status, headers and first line, and turns a host refusal after them into a line. A host busy past the preview is not retried by the page.
- The front's memory holds the canonicalization and, after it, the preview's decode and strip read-back (about 100 MB more at 17 MP, ML-37 R10), never both at once; its unit's MemoryMax is reviewed from the peak its log line records with the preview in it.
- The logs follow the lines (ML-37 D23, D25 to D27): the front's line has the preview's build and when it went out, on its own clock, or why it was not made, and the host's first line's arrival; the host's has the segmentation's; the page's timeline is at version 2. The reading script reads both versions.
- Whether libjxl decodes bit-identically on the front's CPU and the host's is checked at a release (ML-37 D22), not assumed.

Source: [ML-37](../ml-0037_ticket_linear_canonical_photo.html) D4, D15, D16, D19, D22 to D27; P4.
