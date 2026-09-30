"""Planted defects for measuring the judge (ticket ML-2 P1, D14, plan §4.3).

A base mask (owner-accepted, D25) is corrupted in a known way and sent to the judge blind, next to the
clean base. Five types:

    shift        the whole mask moved; tray on one side, beans lost on the other
    erode        the outline pulled in all round (from the frame edge too); beans left outside
    dilate       the outline grown outward onto tray, rim or table
    drop_corner  a triangle cut off one corner of the mask's bounding box
    add_band     a strip of tray/table outside the pile joined to the mask

D14: a defect is *consequential* if the D4 crop of the defect mask has IoU <= 0.90 with the clean crop, or
it adds non-bean area >= 5% of the bean region (the base mask is the bean region). Catch rate on
consequential defects is gated; on small ones it is only reported. Magnitudes straddle that boundary:
each type is evaluated on a ladder of severities, and variants are drawn from the smallest consequential
steps and the largest small steps, so the judge is tested near the line and not on gross errors.
A type that cannot be planted on a base (nothing outside a frame-filling pile to dilate onto or join) is
skipped for that base and counted.

"Snaps to single beans" and "outlines a non-coffee pile" cannot be planted realistically; the owner's
blind audit in P2 covers them (plan §4.3).

    python -m coffeecv.seg_defects make --set pilot   # 20 items from base_dev (D15 pilot)
    python -m coffeecv.seg_defects make --set dev     # every base_dev base (prompt development)
    python -m coffeecv.seg_defects make --set final   # base_heldout, once, fresh seed (D14 final check)

Manifest: labels/ml2/defects/<set>.csv (git): opaque item id -> base, type, magnitude, the metrics and
the mask sha. Defect masks: outputs/ml2_p1/defects/<set>/ (regenerable; the manifest's sha pins them).
Clean items point at the base mask itself. An existing manifest is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from coffeecv import seg_lists
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.seg_base_masks import MASK_DIR as BASE_MASK_DIR
from coffeecv.segment_beans import d4_box, mask_sha256

TYPES = ("shift", "erode", "dilate", "drop_corner", "add_band")
CROP_IOU_MAX = 0.90          # D14: consequential if the crop IoU is <= this ...
ADDED_FRAC_MIN = 0.05        # ... or added non-bean area >= this share of the bean region
LADDER = np.linspace(0.02, 1.0, 50)
WINDOW = 4                   # draw from this many ladder steps on each side of the boundary
SEARCH_LONG_SIDE = 1024      # the ladder runs on a copy this size; the drawn variants at full resolution
MANIFEST_DIR = REPO_ROOT / "labels" / "ml2" / "defects"
OUT_DIR = REPO_ROOT / "outputs" / "ml2_p1" / "defects"
MANIFEST_FIELDS = ("item", "base", "path", "photo_sha256", "mask", "mask_sha256", "type", "t", "crop_iou",
                   "added_frac", "consequential", "trial")


@dataclass(frozen=True)
class SetSpec:
    bases: str               # list in photo_lists.yaml
    stream: int              # rng stream under seg_split_seed; seg_lists uses 1-8, seg_base_masks 5
    clean_trials: int        # each clean base judged this many times (false-fail trials)
    n_cons: int              # consequential variants per type per base
    n_small: int             # small variants per type per base


SETS = {
    # 20 items: 5 clean, and per type 2 consequential + 1 small, each on a different base
    "pilot": SetSpec("base_dev", 11, 1, 2, 1),
    "dev": SetSpec("base_dev", 12, 2, 1, 1),
    "final": SetSpec("base_heldout", 13, 2, 2, 1),
}
PILOT_CLEAN = 5


def box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def consequence(clean: np.ndarray, defect: np.ndarray) -> tuple[float, float, bool]:
    """(crop IoU, added non-bean area / bean region, consequential) of `defect` against the base mask."""
    n = int(np.count_nonzero(clean))
    iou = box_iou(d4_box(clean), d4_box(defect)) if defect.any() else 0.0
    added = int(np.count_nonzero(defect & ~clean)) / n
    return iou, added, iou <= CROP_IOU_MAX or added >= ADDED_FRAC_MIN


class Planter:
    """Every defect of one base mask as a function of severity t in (0, 1], with the random choices
    (direction, corner, side) fixed at construction so the ladder varies the magnitude only."""

    def __init__(self, mask: np.ndarray, rng: np.random.Generator | None = None, choices: dict | None = None):
        self.m = mask.astype(bool)
        h, w = self.m.shape
        self.scale = float(np.sqrt(np.count_nonzero(self.m)))       # side of a square of the same area
        ys, xs = np.nonzero(self.m)
        self.bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        x0, y0, x1, y1 = self.bbox
        room = {"left": x0, "right": w - 1 - x1, "top": y0, "bottom": h - 1 - y1}
        if choices is None:
            sides = [s for s, r in room.items() if r >= 0.02 * self.scale]
            choices = {"angle": rng.uniform(0, 2 * np.pi), "corner": int(rng.integers(4)),
                       "side": str(rng.choice(sides)) if sides else None,
                       "band_frac": rng.uniform(0.3, 0.6),          # band length / bbox side
                       "band_pos": rng.uniform(0, 1)}               # where along the side
        self.choices = choices
        self.angle, self.corner, self.side = choices["angle"], choices["corner"], choices["side"]
        self.band_frac, self.band_pos = choices["band_frac"], choices["band_pos"]
        self.room = room.get(self.side, 0)
        self._din = self._dout = None

    def resized(self, long_side: int) -> "Planter":
        """The same defects on a downscaled copy of the mask, for a cheap magnitude search."""
        h, w = self.m.shape
        f = min(1.0, long_side / max(h, w))
        small = cv2.resize(self.m.astype(np.uint8), (max(1, round(w * f)), max(1, round(h * f))),
                           interpolation=cv2.INTER_NEAREST) > 0
        return Planter(small, choices=self.choices)

    def plantable(self, kind: str) -> bool:
        if kind == "add_band":
            return self.side is not None
        if kind == "dilate":
            return self.m.mean() < 0.95
        return True

    def _dist_in(self) -> np.ndarray:
        if self._din is None:                                       # the frame edge counts as boundary
            padded = np.pad(self.m, 1).astype(np.uint8)
            self._din = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        return self._din

    def _dist_out(self) -> np.ndarray:
        if self._dout is None:
            self._dout = cv2.distanceTransform((~self.m).astype(np.uint8), cv2.DIST_L2, 5)
        return self._dout

    def __call__(self, kind: str, t: float) -> np.ndarray:
        m, s = self.m, self.scale
        h, w = m.shape
        if kind == "shift":
            dx, dy = (round(t * 0.5 * s * f(self.angle)) for f in (np.cos, np.sin))
            out = np.zeros_like(m)
            out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
                m[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
            return out
        if kind == "erode":
            return m & (self._dist_in() > t * 0.25 * s)
        if kind == "dilate":
            return m | (self._dist_out() <= t * 0.25 * s)
        x0, y0, x1, y1 = self.bbox
        if kind == "drop_corner":
            yy, xx = np.ogrid[:h, :w]
            u = (xx - x0) / max(x1 - x0, 1)
            v = (yy - y0) / max(y1 - y0, 1)
            u = 1 - u if self.corner in (1, 2) else u
            v = 1 - v if self.corner in (2, 3) else v
            return m & ~(u + v < t)
        if kind == "add_band":
            out = m.copy()
            depth = max(1, round(t * self.room))
            horiz = self.side in ("top", "bottom")
            lo, hi = (x0, x1) if horiz else (y0, y1)
            length = max(1, round((hi - lo) * self.band_frac))
            a = lo + round((hi - lo - length) * self.band_pos)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if self.side == "top":
                out[y0 - depth:cy, a:a + length] = True
            elif self.side == "bottom":
                out[cy:y1 + 1 + depth, a:a + length] = True
            elif self.side == "left":
                out[a:a + length, x0 - depth:cx] = True
            else:
                out[a:a + length, cx:x1 + 1 + depth] = True
            return out
        raise ValueError(kind)


def variants(p: Planter, kind: str, n_cons: int, n_small: int, rng: np.random.Generator) -> list[dict]:
    """Up to `n_cons` consequential and `n_small` small variants near the D14 boundary: evaluate the ladder
    on a downscaled copy, then draw steps from the WINDOW smallest consequential ones and the WINDOW largest
    small ones below the first consequential step. Each variant is planted at full resolution and records
    its mask and its full-resolution metrics, which decide whether it is consequential."""
    if not p.plantable(kind):
        return []
    q = p.resized(SEARCH_LONG_SIDE)
    ladder = []
    for t in LADDER:
        d = q(kind, float(t))
        iou, added, cons = consequence(q.m, d)
        ladder.append((float(t), iou, added, cons))
    first = next((i for i, r in enumerate(ladder) if r[3]), len(ladder))
    cons_idx = [i for i in range(first, len(ladder)) if ladder[i][3]]
    small_idx = [i for i in range(first) if ladder[i][1] < 1.0 or ladder[i][2] > 0][::-1]
    out = []
    for want, idx, n in ((True, cons_idx, n_cons), (False, small_idx, n_small)):
        # the window in random order, then the steps further from the boundary, until n variants land on
        # the wanted side at full resolution (a step right at the boundary can flip when upscaled)
        order = [*rng.permutation(idx[:WINDOW]), *idx[WINDOW:]]
        got = 0
        for i in order:
            if got == n:
                break
            t = ladder[i][0]
            d = p(kind, t)
            iou, added, cons = consequence(p.m, d)
            if cons == want:
                out.append({"type": kind, "t": t, "crop_iou": iou, "added_frac": added, "consequential": cons,
                            "mask": d})
                got += 1
    return out


def base_masks(list_name: str) -> list[dict]:
    """The accepted base masks of a list, with the stored mask checked against the list's sha."""
    _, lists = seg_lists.load_lists()
    index = {r["path"]: r for r in csv.DictReader((BASE_MASK_DIR / "index.csv").read_text().splitlines())}
    out = []
    for e in lists[list_name]:
        r = index[e["path"]]
        f = BASE_MASK_DIR / f"{r['id']}.png"
        m = np.array(Image.open(f)) > 0
        if mask_sha256(m) != e["mask_sha256"]:
            raise ValueError(f"{f}: mask differs from the accepted one in {list_name}")
        out.append({"id": r["id"], "path": e["path"], "photo_sha256": e["sha256"], "file": f, "mask": m})
    return out


def make(set_name: str) -> Path:
    spec = SETS[set_name]
    manifest = MANIFEST_DIR / f"{set_name}.csv"
    if manifest.exists():
        raise FileExistsError(f"{manifest.relative_to(REPO_ROOT)} exists; defect sets are generated once")
    rng = np.random.default_rng([RunConfig.from_params_yaml().seg_split_seed, spec.stream])
    bases = base_masks(spec.bases)
    rows, skipped = [], Counter()
    if set_name == "pilot":
        # one item per base where possible: clean on 5 bases, each defect variant on another base
        order = list(rng.permutation(len(bases)))
        plan = [(b, "clean", 0, 0) for b in order[:PILOT_CLEAN]]
        rest = order[PILOT_CLEAN:] + order[:PILOT_CLEAN]
        for kind in TYPES:
            for want_cons in [True] * spec.n_cons + [False] * spec.n_small:
                plan.append((None, kind, int(want_cons), int(not want_cons)))
        for b, kind, nc, ns in plan:
            if kind == "clean":
                rows.append(_clean_row(bases[b], 0))
                continue
            for j, b in enumerate(rest):                  # the first base this type can be planted on
                got = variants(Planter(bases[b]["mask"], rng), kind, nc, ns, rng)
                if got:
                    rows.append(_defect_row(bases[rest.pop(j)], got[0]))
                    break
                skipped[kind] += 1
    else:
        for b in bases:
            for trial in range(spec.clean_trials):
                rows.append(_clean_row(b, trial))
            p = Planter(b["mask"], rng)
            for kind in TYPES:
                got = variants(p, kind, spec.n_cons, spec.n_small, rng)
                if not got:
                    skipped[kind] += 1
                rows.extend(_defect_row(b, v) for v in got)
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    ids = set()
    out_dir = OUT_DIR / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        while (item := f"{set_name}_{rng.integers(16 ** 8):08x}") in ids:
            pass
        ids.add(item)
        r["item"] = item
        if "defect_mask" in r:
            f = out_dir / f"{item}.png"
            Image.fromarray(r.pop("defect_mask")).convert("1").save(f, optimize=True)
            r["mask"] = str(f.relative_to(REPO_ROOT))
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} items to {manifest.relative_to(REPO_ROOT)}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(Counter(
        (r["type"], r["consequential"]) for r in rows).items())))
    if skipped:
        print("  not plantable (base, type): " + ", ".join(f"{k} x{v}" for k, v in sorted(skipped.items())))
    return manifest


def _clean_row(b: dict, trial: int) -> dict:
    return {"base": b["id"], "path": b["path"], "photo_sha256": b["photo_sha256"],
            "mask": str(b["file"].relative_to(REPO_ROOT)), "mask_sha256": mask_sha256(b["mask"]), "type": "clean",
            "t": 0.0, "crop_iou": 1.0, "added_frac": 0.0, "consequential": "", "trial": trial}


def _defect_row(b: dict, v: dict) -> dict:
    return {"base": b["id"], "path": b["path"], "photo_sha256": b["photo_sha256"], "defect_mask": v["mask"],
            "mask_sha256": mask_sha256(v["mask"]), "type": v["type"], "t": round(v["t"], 4),
            "crop_iou": round(v["crop_iou"], 4), "added_frac": round(v["added_frac"], 4),
            "consequential": int(v["consequential"]), "trial": 0}


def load_manifest(set_name: str) -> list[dict]:
    return list(csv.DictReader((MANIFEST_DIR / f"{set_name}.csv").read_text().splitlines()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["make"])
    ap.add_argument("--set", choices=list(SETS), required=True)
    args = ap.parse_args(argv)
    make(args.set)


if __name__ == "__main__":
    main()
