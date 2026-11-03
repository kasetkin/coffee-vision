"""ML-19 P1: jxl_header.py's reading of a file's colour encoding against libjxl's own.

    UV_CACHE_DIR=... uv run --script pipeline.py info     # once, for the environment below
    <that environment's python> analysis/ml19/p1/check_header.py make DIR
    /home/vscode/.venv/bin/python -I analysis/ml19/p1/check_header.py check DIR [MORE.jxl ...]

`make` writes small JPEG XL files of every primaries and transfer imagecodecs can write, plus 8-bit, float, grey,
lossless, alpha and container files. `check` needs training's environment, which still has pillow-jxl-plugin: libjxl
gives the decoder an ICC profile built from the file's colour encoding, and that profile's description names the
encoding (RGB_D65_202_Rel_709, or sRGB, DisplayP3, Rec2100PQ, Rec2100HLG for the common ones). The reader must
agree with it on every file.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jxl_header import read_header  # noqa: E402

TF = {"709": "BT709", "SRG": "sRGB", "Lin": "Linear", "PeQ": "PQ", "HLG": "HLG", "DCI": "DCI"}
PR = {"SRG": "sRGB", "202": "BT2100", "DCI": "P3"}   # libjxl names the P3 primaries DCI
NAMED = {"sRGB": ("RGB", "sRGB", "sRGB"), "DisplayP3": ("RGB", "P3", "sRGB"),
         "Rec2100PQ": ("RGB", "BT2100", "PQ"), "Rec2100HLG": ("RGB", "BT2100", "HLG")}
INTENT = {"Rel": "Relative", "Per": "Perceptual"}


def make(out: Path) -> None:
    import imagecodecs as ic
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    cases = [(f"{p}_{t}_u16", {"primaries": p, "transfer": t}, np.uint16, 3)
             for p in ("SRGB", "BT2100", "P3") for t in ("BT709", "SRGB", "LINEAR", "PQ", "HLG", "DCI")]
    cases += [("default_u8", {}, np.uint8, 3), ("default_u16", {}, np.uint16, 3), ("default_f32", {}, np.float32, 3),
              ("rgba_u16_2020", {"primaries": "BT2100", "transfer": "BT709"}, np.uint16, 4),
              ("grey_u8", {}, np.uint8, 1), ("lossless_u8", {"lossless": True}, np.uint8, 3),
              ("container_2020", {"primaries": "BT2100", "transfer": "BT709", "usecontainer": True}, np.uint16, 3)]
    for name, kw, dtype, channels in cases:
        a = rng.random((37, 53, channels) if channels > 1 else (37, 53))
        a = a.astype(np.float32) if dtype == np.float32 else (a * np.iinfo(dtype).max).astype(dtype)
        (out / f"{name}.jxl").write_bytes(ic.jpegxl_encode(a, level=None if kw.get("lossless") else 95, **kw))
    print(f"{len(cases)} files in {out}")


def check(files: list[Path]) -> int:
    import pillow_jxl  # noqa: F401 -- registers JPEG XL with Pillow
    from PIL import Image, ImageCms
    bad = 0
    for f in files:
        h = read_header(f.read_bytes())
        with Image.open(f) as im:
            size, icc = im.size, im.info["icc_profile"]
        desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
        if desc in NAMED:
            want = (*NAMED[desc], "Relative")
        elif desc.startswith("Gra_"):
            _, _, intent, tf = desc.split("_")
            want = ("Grey", None, TF[tf], INTENT[intent])
        else:
            _, _, prim, intent, tf = desc.split("_")
            want = ("RGB", PR[prim], TF[tf], INTENT[intent])
        c = h.colour
        ok = (c.colour_space, c.primaries, c.transfer, c.rendering_intent) == want and c.white_point == "D65" \
            and size == (h.width, h.height)
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {f.name:24} {desc:22} extra={h.extra_channel_types} boxes={h.boxes}")
    print(f"{bad} of {len(files)} disagree with libjxl")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(Path(sys.argv[2]))
    else:
        paths = [Path(a) for a in sys.argv[2:]]
        sys.exit(1 if check([q for p in paths for q in (sorted(p.glob("*.jxl")) if p.is_dir() else [p])]) else 0)
