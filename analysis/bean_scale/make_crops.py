"""Pick 10 random photos per rig and render a centred crop for manual bean counting.

Ground truth is defined as equivalent centre-to-centre spacing:
    spacing_px = crop_side_px / sqrt(N_beans_in_crop)
which is the quantity that decides how many beans land in a patch, and is directly
comparable to the FFT "period" already recorded. Crop side is a fixed fraction of
the photo's short side so every rig yields a similar, countable bean count
regardless of its magnification.
"""
import json, random
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("/workspace/data/cropped")
OUT = Path("/tmp/claude-1000/-workspace/45b9528f-eaa3-4dc2-a210-c85635b16fea/scratchpad/gt")
RIGS = {"old_box": "2026-08-07__box_pictures_all_classes",
        "pixel_cam": "2026-08-09__pixel_cam",
        "sony_cam": "2026-08-09__sony_cam"}
FRAC = 0.40      # of the short side -> roughly 20-35 beans, countable in one look
VIEW = 900       # display size

rng = random.Random(20260811)
manifest = []
for rig, session in RIGS.items():
    photos = sorted((ROOT / session).glob("*/*__cropped.jpg"))
    for i, p in enumerate(rng.sample(photos, 10)):
        im = Image.open(p).convert("RGB")
        side = int(min(im.size) * FRAC)
        cx, cy = im.width // 2, im.height // 2
        crop = im.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))
        view = crop.resize((VIEW, VIEW), Image.LANCZOS)
        d = ImageDraw.Draw(view)
        # 3x3 guide grid: counting per cell is far more reliable than counting a whole square
        for k in (1, 2):
            d.line([(k * VIEW // 3, 0), (k * VIEW // 3, VIEW)], fill=(255, 40, 40), width=2)
            d.line([(0, k * VIEW // 3), (VIEW, k * VIEW // 3)], fill=(255, 40, 40), width=2)
        name = f"{rig}__{i:02d}.png"
        view.save(OUT / name)
        manifest.append({"name": name, "rig": rig, "photo": str(p.relative_to(ROOT)),
                         "crop_side_px": side, "view_px": VIEW})
json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
print(f"{len(manifest)} crops written")
for rig in RIGS:
    s = [m["crop_side_px"] for m in manifest if m["rig"] == rig]
    print(f"  {rig:<10} crop side {min(s)}-{max(s)} px of source")
