# OPS-18 R1: the example photos as GIF and BMP

`python analysis/ops18/r1_gif_examples.py http://127.0.0.1:8100`, 2026-11-08, on the branch `ops18-gif-bmp-uploads`:
a local front (`webapp/front/check_service.py`) forwarding to a local inference host (`webapp/app.py` under Flask's
server, `dino3b16_segxl0v7r_country_aug8_s42`'s LiteRT model on this machine's CPU). A record, not a gate (D6). Each
cell is the top country and its score, or the answer's refusal.

| Example | Expected | As sent (canonical JPEG XL) | GIF | BMP (24-bit) |
|---|---|---|---|---|
| brazil-grinder | Brazil | Brazil 1.00 | Brazil 1.00 | error: 413 this photo is over the 40 MB limit |
| ethiopia-green | Ethiopia | Ethiopia 0.99 | Ethiopia 0.63 | Ethiopia 0.99 |
| indonesia-roasted | Indonesia | Indonesia 1.00 | Indonesia 0.99 | error: 413 this photo is over the 40 MB limit |
| cappuccino | refused (no beans) | refused_ood: Unrecognized | refused_ood: Unrecognized | refused_ood: Unrecognized |

- GIF: every example keeps its verdict. Ethiopia's score drops from 0.99 to 0.63 (256 colours, Pillow's adaptive
  palette); Brazil and Indonesia barely move; the cappuccino is still refused.
- BMP: Brazil (50 MP) and Indonesia (19 MP) are over the 40 MB upload cap as BMPs (about 3 bytes a pixel), as the
  ticket's section 2 expects; Ethiopia and the cappuccino (12 MP) answer as their originals do.
- Memory, from the front's log lines: the 50 MP GIF peaked at 617 MB while canonicalized (21.8 s), the 12 MP ones at
  410-441 MB. ML-19 P8 measured 461 MB for an 8-bit 60 MP photo on imagecodecs' JPEG path; Pillow's path (GIF, BMP,
  and WebP and AVIF before this ticket) holds the palette image, its RGB conversion and the array at once. The front's
  unit has MemoryMax 600M with swap allowed.
