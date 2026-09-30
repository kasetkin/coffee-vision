"""Ticket ML-2 P1: known-good test masks for the judge, from placed points (owner, 2026-09-30).

The owner's decision replaces D12's "pretrained masks as they come": for each photo, include points on the
bean pile and exclude points on tray, rim, table or paper are placed by eye (by Claude, from a gridded
overview), and pretrained L0 draws the mask from the whole-image box plus those points -- the prompt the
P0 points probe showed gives the pile on tray photos. On a frame-filling photo points make a speckled
mask with holes in the shadows between beans, so those items carry no points and `output: multi3`: the box
alone, widest output, which P0 found covers the frame. The owner then accepts or declines each mask.
The points are the record of how each mask was made and are committed with it.

    python -m coffeecv.seg_base_masks pick    # draw N photos from base_candidates into the points file
    python -m coffeecv.seg_base_masks grid    # gridded overviews to place points on
    python -m coffeecv.seg_base_masks run     # masks + review sheets (overview + full-resolution tiles)

Points file: labels/ml2/base_points.yaml (git). Masks: data/seg_masks/base_points/ (1-bit PNG + index.csv,
computed once on one machine and stored: mask bits differ by a few pixels across CPUs, P0).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from coffeecv import seg_lists
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.sam_loader import weights_for
from coffeecv.segment_beans import BeanSegmenter, SegParams, mask_and_crop, mask_sha256

POINTS_FILE = REPO_ROOT / "labels" / "ml2" / "base_points.yaml"
MASK_DIR = REPO_ROOT / "data" / "seg_masks" / "base_points"
SHEET_DIR = REPO_ROOT / "outputs" / "ml2_p1" / "base_points"
N_PICK = 20
MIN_ROASTED = 2
_STREAM = 5                                  # under seg_split_seed; seg_lists uses 1-4 and 6-7
MAGENTA = (255, 0, 255)


def item_id(entry: dict) -> str:
    """File-name-safe id; stems alone repeat across sessions (IMG_xxxx). OOD positives have a batch, no
    session or class."""
    p = Path(entry["path"])
    if "session" not in entry:
        return f"{entry['batch']}__{p.stem}"
    return f"{entry['session']}__{entry['class']}__{p.stem}"


def pick(n: int, rest: bool = False) -> None:
    """Draw `n` of base_candidates into a new points file, or with `rest` append every candidate the file
    does not hold yet (points already placed are never touched)."""
    _, lists = seg_lists.load_lists()
    cands = lists["base_candidates"]
    if rest:
        doc = yaml.safe_load(POINTS_FILE.read_text())
        have = {it["path"] for it in doc["items"]}
        new = [{"id": item_id(e), "path": e["path"], "sha256": e["sha256"], "roast": e.get("roast", "unknown"),
                "include": [], "exclude": []} for e in cands if e["path"] not in have]
        header = "".join(l for l in POINTS_FILE.read_text().splitlines(True) if l.startswith("#"))
        POINTS_FILE.write_text(header + yaml.safe_dump({"items": doc["items"] + new}, sort_keys=False, width=200,
                                                       default_flow_style=None))
        print(f"appended {len(new)} items to {POINTS_FILE.relative_to(REPO_ROOT)}")
        return
    if POINTS_FILE.exists():
        raise FileExistsError(f"{POINTS_FILE.relative_to(REPO_ROOT)} exists; points already placed are not redrawn")
    stratum = lambda e: (e["session"], e["roast"])                    # noqa: E731
    sizes = defaultdict(int)
    for e in cands:
        sizes[stratum(e)] += 1
    quota = seg_lists.allocate(dict(sizes), n, {k: MIN_ROASTED for k in sizes if k[1] == "roasted"})
    seed = RunConfig.from_params_yaml().seg_split_seed
    picked = seg_lists.stratified_draw(cands, stratum, quota, np.random.default_rng([seed, _STREAM]))
    items = [{"id": item_id(e), "path": e["path"], "sha256": e["sha256"], "roast": e["roast"],
              "include": [], "exclude": []} for e in picked]
    POINTS_FILE.write_text(
        "# Ticket ML-2 P1: points for the judge's test masks (coffeecv/seg_base_masks.py). (x, y) are fractions\n"
        "# of the photo's width and height. include = on the bean pile, exclude = tray, rim, table, paper.\n"
        + yaml.safe_dump({"items": items}, sort_keys=False, width=200, default_flow_style=None))
    print(f"wrote {len(items)} items to {POINTS_FILE.relative_to(REPO_ROOT)}")


def load_items() -> list[dict]:
    return yaml.safe_load(POINTS_FILE.read_text())["items"]


def small(rgb: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s


def grid(long_side: int = 1100) -> None:
    """Overview with a labelled 0.1 grid, so points can be read off as fractions."""
    out = SHEET_DIR / "grid"
    out.mkdir(parents=True, exist_ok=True)
    for it in load_items():
        if (out / f"{it['id']}.jpg").exists():
            continue
        im_arr, _ = small(load_rgb_image(REPO_ROOT / it["path"]), long_side)
        h, w = im_arr.shape[:2]
        im = Image.fromarray(im_arr)
        d = ImageDraw.Draw(im)
        for k in range(1, 10):
            x, y = round(k / 10 * (w - 1)), round(k / 10 * (h - 1))
            d.line([(x, 0), (x, h)], fill=(0, 255, 255), width=1)
            d.line([(0, y), (w, y)], fill=(0, 255, 255), width=1)
            d.text((x + 2, 2), f".{k}", fill=(255, 255, 0))
            d.text((2, y + 2), f".{k}", fill=(255, 255, 0))
        im.save(out / f"{it['id']}.jpg", quality=88)
    print(f"wrote grids to {out.relative_to(REPO_ROOT)}")


def outline(rgb: np.ndarray, mask: np.ndarray, px: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0
    out = rgb.copy()
    out[edge] = MAGENTA
    return out


def boundary_tiles(mask: np.ndarray, n: int, tile: int) -> list[tuple[int, int]]:
    """Top-left corners of `n` full-resolution tiles centred on points evenly spaced along the mask's
    outer contour (all contours, longest first, by arc length)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    pts = np.concatenate([c[:, 0, :] for c in sorted(contours, key=len, reverse=True)])
    h, w = mask.shape
    corners = []
    for i in np.linspace(0, len(pts), n, endpoint=False).astype(int):
        x, y = pts[i]
        corners.append((int(np.clip(x - tile // 2, 0, max(w - tile, 0))),
                        int(np.clip(y - tile // 2, 0, max(h - tile, 0)))))
    return corners


def review_sheet(rgb: np.ndarray, mask: np.ndarray, it: dict, iou: float, overview: int = 1400,
                 n_tiles: int = 6, tile: int = 700) -> Image.Image:
    """Left: overview with the outline, the points (green include, red exclude), the D4 crop box dashed and
    the tile positions T1..Tn. Right: the tiles at full resolution with the outline, 2 columns."""
    h, w = rgb.shape[:2]
    ov, s = small(rgb, overview)
    ms = cv2.resize(mask.astype(np.uint8), ov.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    im = Image.fromarray(outline(ov, ms, 1))
    d = ImageDraw.Draw(im)
    crop = mask_and_crop(rgb, mask, SegParams(mask_select="single", prompt="box"))
    if not crop.info.get("fallback"):
        x0, y0, bw, bh = crop.info["box"]
        x1, y1 = x0 + bw, y0 + bh
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) * s / 12))
            at = lambda t: (s * (a[0] + (b[0] - a[0]) * t / n), s * (a[1] + (b[1] - a[1]) * t / n))  # noqa: E731
            for k in range(0, n, 2):
                d.line([at(k), at(k + 1)], fill=(255, 255, 0), width=2)
    for pts, colour in ((it["include"], (0, 255, 0)), (it["exclude"], (255, 0, 0))):
        for fx, fy in pts:
            x, y = fx * (w - 1) * s, fy * (h - 1) * s
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=colour, outline="black")
    corners = boundary_tiles(mask, n_tiles, tile)
    for k, (tx, ty) in enumerate(corners, 1):
        d.rectangle([tx * s, ty * s, (tx + tile) * s, (ty + tile) * s], outline=(0, 200, 255), width=2)
        d.text((tx * s + 4, ty * s + 4), f"T{k}", fill=(0, 200, 255))
    d.text((8, 8), f"{it['id']}  area={mask.mean():.3f}  pred_iou={iou:.3f}", fill=MAGENTA)

    tiles = []
    for k, (tx, ty) in enumerate(corners, 1):
        t = Image.fromarray(outline(rgb[ty:ty + tile, tx:tx + tile], mask[ty:ty + tile, tx:tx + tile], 2))
        ImageDraw.Draw(t).text((6, 6), f"T{k}", fill=(0, 200, 255))
        tiles.append(t)
    cols, rows = 2, (len(tiles) + 1) // 2
    W = im.width + cols * tile
    H = max(im.height, rows * tile)
    sheet = Image.new("RGB", (W, H), "white")
    sheet.paste(im, (0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, (im.width + (i % 2) * tile, (i // 2) * tile))
    return sheet


def run(only: list[str] | None) -> None:
    items = [it for it in load_items() if (it["include"] or it.get("output")) and (not only or it["id"] in only)]
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    (SHEET_DIR / "review").mkdir(parents=True, exist_ok=True)
    seg = BeanSegmenter(SegParams(mask_select="single", prompt="box", weights=weights_for("l0")))
    index_file = MASK_DIR / "index.csv"
    index = {}
    if index_file.exists():
        index = {r["id"]: r for r in csv.DictReader(index_file.read_text().splitlines())}
    for it in items:
        rgb = load_rgb_image(REPO_ROOT / it["path"])
        mask, iou = seg.predict_with_points(rgb, it["include"], it["exclude"], it.get("output", "single"))
        Image.fromarray(mask).convert("1").save(MASK_DIR / f"{it['id']}.png", optimize=True)
        index[it["id"]] = {"id": it["id"], "path": it["path"], "photo_sha256": it["sha256"],
                           "mask_sha256": mask_sha256(mask), "area_frac": f"{mask.mean():.4f}",
                           "pred_iou": f"{iou:.4f}", "n_include": len(it["include"]),
                           "n_exclude": len(it["exclude"]), "output": it.get("output", "single"),
                           "weights_sha256": seg.weights_sha256}
        review_sheet(rgb, mask, it, iou).save(SHEET_DIR / "review" / f"{it['id']}.jpg", quality=88)
        print(f"{it['id']}  area={mask.mean():.3f}  pred_iou={iou:.3f}")
    rows = [index[k] for k in sorted(index)]
    with open(index_file, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pick", "grid", "run"])
    ap.add_argument("--n", type=int, default=N_PICK)
    ap.add_argument("--only", nargs="*", help="item ids to (re)run")
    ap.add_argument("--rest", action="store_true", help="pick: append the candidates not yet in the points file")
    args = ap.parse_args(argv)
    if args.command == "pick":
        pick(args.n, args.rest)
    elif args.command == "grid":
        grid()
    else:
        run(args.only)


if __name__ == "__main__":
    main()
