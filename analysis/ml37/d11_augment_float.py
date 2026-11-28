"""ML-37 R5 / D11: every albumentations transform in params.yaml's pipelines on float32 [0, 1] against uint8.

Research only (analysis/ml37/research_d11_float_resize_and_augment.md). Writes analysis/ml37/d11_augment_float.txt.

One canonical photo read back as in d11_resize_parity.py: u8 (load_canonical's) and f32 (the float read-back). Each
transform runs alone (p = 1; OneOf's children one by one and OneOf itself), built by coffeecv.augment's strict
builder, at the size its pipeline sees it: seg_ft.augment on the photo at the SAM side (1024 long, with a label for
the spatial ones), head_augment.patch on a 448 patch, head_augment.input on the 224 input. For each of SEEDS seeds a
fresh pipeline with that seed runs on
  u8       the uint8 picture
  grid     u8 / 255 (float32 on the 8-bit grid): the transform's float path on today's values
  f32      the float read-back (off the 8-bit grid)
and reports
  grid vs u8     |grid x 255 - u8| (max, mean) and the mean signed difference, in 8-bit levels
  on-grid share  the share of f32's output values within 1e-3 level of the 8-bit grid, minus the input's own share:
                 near 100% means the transform rounded the float picture to 8 bits inside
  f32 range      the float output's min and max (outside [0, 1] = not clipped; > 1 = a fill or value in 0-255 units)
  label          for spatial transforms with a label: whether the label came out the same for u8 and f32
Then each whole pipeline (groups expanded) on f32: errors, warnings, dtype and range over SEEDS seeds.
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml

from coffeecv import augment, model_io

cv2.setNumThreads(2)
_spec = importlib.util.spec_from_file_location("parity", Path(__file__).with_name("d11_resize_parity.py"))
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity)

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".txt")
PHOTO = REPO / "data/canonical/2026-09-11__oneplus"
SEEDS = 24


def pick_photo() -> Path:
    return sorted(PHOTO.rglob("*.jxl"))[0]


def on_grid(x: np.ndarray) -> float:
    v = x.astype(np.float64) * 255
    return float((np.abs(v - np.rint(v)) < 1e-3).mean())


def entries(spec, groups, where):
    """(name, entry) for each transform at p = 1; OneOf's children one by one, then OneOf itself."""
    out = []
    for t in augment.expand(spec, groups, where=where):
        if t["transform"] == "OneOf":
            for c in t["args"]["transforms"]:
                out.append((f"{c['transform']} (in OneOf)", {**c, "p": 1.0}))
            out.append(("OneOf", {**t, "p": 1.0}))
        else:
            out.append((t["transform"], {**t, "p": 1.0}))
    return out


def run(entry, image, label, seed, check):
    pipe = augment.build([entry], where="d11", seed=seed, check=check)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = pipe(image=image, mask=label) if label is not None else pipe(image=image)
    return out["image"], (out.get("mask") if label is not None else None), [str(x.message)[:120] for x in w]


def main() -> None:
    from coffeecv.seg_finetune import IGNORE, _fill_mask_is_ignore
    params = yaml.safe_load((REPO / "params.yaml").read_text())
    groups = params["augment"]
    path = pick_photo()
    u8, f32 = parity.read_back(path)
    h, w = u8.shape[:2]
    nh, nw = model_io.sam_preprocess_shape(h, w, 1024)
    photo = {"u8": parity.pil_u8(u8, (nw, nh)), "f32": np.clip(parity.pil_f(f32, (nw, nh)), 0, 1)}
    y0, x0 = (h - 896) // 2, (w - 896) // 2
    patch = {"u8": parity.pil_u8(u8[y0:y0 + 896, x0:x0 + 896], (448, 448)),
             "f32": np.clip(parity.pil_f(f32[y0:y0 + 896, x0:x0 + 896], (448, 448)), 0, 1)}
    inp = {"u8": parity.pil_u8(patch["u8"], (224, 224)), "f32": np.clip(parity.pil_f(patch["f32"], (224, 224)), 0, 1)}
    del u8, f32
    label = np.zeros((nh, nw), np.uint8)
    label[nh // 4: 3 * nh // 4, nw // 5: 4 * nw // 5] = 1

    lines = [f"ML-37 R5: albumentations {augment.version()} on float32 vs uint8; photo {path.relative_to(REPO)} "
             f"({w}x{h}); {SEEDS} seeds per transform, each alone at p = 1",
             f"input on-grid share (f32): photo {on_grid(photo['f32']):.4f}, patch {on_grid(patch['f32']):.4f}, "
             f"input {on_grid(inp['f32']):.4f}", "",
             f"{'pipeline / transform':<46} {'grid vs u8: max':>15} {'mean|d|':>8} {'mean d':>8} "
             f"{'on-grid f32':>11} {'f32 min':>8} {'f32 max':>8}  label  notes"]
    pipelines = [("seg_ft.augment", params["seg_ft"]["augment"], photo, label, _fill_mask_is_ignore),
                 ("head_augment.patch", params["head_augment"]["patch"], patch, None, None),
                 ("head_augment.input", params["head_augment"]["input"], inp, None, None)]
    base_grid = {"seg_ft.augment": on_grid(photo["f32"]), "head_augment.patch": on_grid(patch["f32"]),
                 "head_augment.input": on_grid(inp["f32"])}
    for where, spec, img, lab, check in pipelines:
        for name, entry in entries(spec, groups, where):
            mx, sa, s, n, grid_share, lo, hi, notes, label_same = 0.0, 0.0, 0.0, 0, [], np.inf, -np.inf, set(), True
            for seed in range(SEEDS):
                try:
                    ou, mu, wu = run(entry, img["u8"], lab, seed, check)
                    og, _, _ = run(entry, img["u8"].astype(np.float32) / 255, lab, seed, check)
                    of, mf, wf = run(entry, img["f32"], lab, seed, check)
                except Exception as e:  # noqa: BLE001
                    notes.add(f"ERROR {type(e).__name__}: {str(e)[:100]}")
                    break
                notes.update(f"warn: {m}" for m in set(wf) - set(wu))
                if og.dtype != np.float32 or of.dtype != np.float32:
                    notes.add(f"float in -> {of.dtype} out")
                d = og.astype(np.float64) * 255 - ou.astype(np.float64)
                mx, sa, s, n = max(mx, float(np.abs(d).max())), sa + float(np.abs(d).sum()), s + float(d.sum()), n + d.size
                grid_share.append(on_grid(of))
                lo, hi = min(lo, float(of.min())), max(hi, float(of.max()))
                if lab is not None:
                    label_same &= bool(np.array_equal(mu, mf)) and mf.dtype == np.uint8 and set(np.unique(mf)) <= {0, 1, IGNORE}
            if n:
                g = float(np.mean(grid_share)) - base_grid[where]
                lines.append(f"{(where + ': ' + name)[:46]:<46} {mx:15.3f} {sa / n:8.4f} {s / n:+8.4f} {g:+11.4f} "
                             f"{lo:8.4f} {hi:8.4f}  {'same' if lab is not None and label_same else ('DIFF' if lab is not None else '  - ')}"
                             f"  {'; '.join(sorted(notes))}")
            else:
                lines.append(f"{(where + ': ' + name)[:46]:<46} {'; '.join(sorted(notes))}")

    lines += ["", "Whole pipelines on f32 (groups expanded), one build per seed:"]
    for where, spec, img, lab, check in pipelines:
        full = augment.expand(spec, groups, where=where)
        errs, warns, lo, hi, dt = set(), set(), np.inf, -np.inf, set()
        for seed in range(SEEDS):
            try:
                pipe = augment.build(full, where=where, seed=seed, check=check)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    o = pipe(image=img["f32"], mask=lab)["image"] if lab is not None else pipe(image=img["f32"])["image"]
                warns.update(str(x.message)[:120] for x in w)
                dt.add(str(o.dtype))
                lo, hi = min(lo, float(o.min())), max(hi, float(o.max()))
            except Exception as e:  # noqa: BLE001
                errs.add(f"{type(e).__name__}: {str(e)[:120]}")
        lines.append(f"  {where:<20} dtypes {sorted(dt)} range [{lo:.4f}, {hi:.4f}] errors {sorted(errs) or 'none'} "
                     f"warnings {sorted(warns) or 'none'}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
