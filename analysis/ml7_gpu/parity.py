"""ML-7 P1 (ticket D7): a runtime's outputs against PyTorch's, on the fixtures, with the repo's own code.

    python analysis/ml7_gpu/parity.py --refs <refs.npz> --got <outputs.npz> [--out report.json]

D7's gate, with the owner's aggregation (2026-10-13): the readout's cosine with PyTorch fp32 >= 0.99999 for every
patch of every photo (the least one), and mask IoU >= 0.999 for every photo, after the shared post-processing.
A photo whose top-1 or guard verdict changes is listed for the owner, not blocking. Beside them, not gating
(ticket D14): the largest change in a class probability, the probe score's change next to the photo's distance
from the 0.5 threshold, and the largest change in the low-resolution mask logits before the threshold.

`got` holds what a runtime returned for refs' inputs: `readouts` [P, 40, 1536] and/or `low_masks` [P, 3, 256, 256]
with `ious` [P, 3]. Either half may be missing (a graph checked alone); the gate reads what is there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

COSINE_MIN = 0.99999
IOU_MIN = 0.999


def least_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """The least cosine between matching vectors (the last axis) of a and b: every patch, not their mean."""
    a, b = a.reshape(-1, a.shape[-1]).astype(np.float64), b.reshape(-1, b.shape[-1]).astype(np.float64)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float(cos.min())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def gate(rows: list[dict]) -> dict:
    """D7 over the per-photo rows: pass if every photo's least cosine and mask IoU (where measured) reach the
    bars; the photos whose top-1 or guard verdict changed, listed."""
    cos = [r["least_cosine"] for r in rows if r.get("least_cosine") is not None]
    iou = [r["mask_iou"] for r in rows if r.get("mask_iou") is not None]
    least_cos, least_iou = (min(cos) if cos else None), (min(iou) if iou else None)
    ok = (least_cos is None or least_cos >= COSINE_MIN) and (least_iou is None or least_iou >= IOU_MIN)
    changed = [r["photo"] for r in rows
               if False in (r.get("top1_same"), r.get("verdict_same"), r.get("fallback_same"))]
    return {"pass": ok, "least_cosine": least_cos, "least_mask_iou": least_iou, "changed_answer": changed,
            "bars": {"cosine": COSINE_MIN, "mask_iou": IOU_MIN}}


# ---------------------------------------------------------------------------------------- the repo's answers
def _answers(readouts: np.ndarray, head: dict, ref: dict | None, probe: dict | None) -> dict:
    """What classify_one answers from a photo's readouts: its head, mean probabilities and guard (coffeecv.infer)."""
    from coffeecv.infer import PROBE_THRESHOLD, ood_scores, probe_score

    logits = readouts @ head["weight"].T + head["bias"]
    z = np.exp(logits - logits.max(1, keepdims=True))
    probs = (z / z.sum(1, keepdims=True)).mean(0)
    out = {"probs": probs, "top1": int(probs.argmax())}
    if probe is not None:
        out["probe"] = float(probe_score(readouts, probe))
        out["refused"] = out["probe"] > PROBE_THRESHOLD
    elif ref is not None:
        out["probe"] = float(np.median(ood_scores(readouts, ref)[0]))
    return out


def _final_mask(low: np.ndarray, input_size, original_size, seg) -> np.ndarray:
    """multi3's low-res logits -> the photo's bool mask, by the segmenter's own post-processing (_upsample)."""
    import torch
    pr = seg.predictor
    pr.input_size, pr.original_size = tuple(int(v) for v in input_size), tuple(int(v) for v in original_size)
    return seg._upsample(torch.from_numpy(np.ascontiguousarray(low)))


def unpack_mask(bits: np.ndarray, original_size) -> np.ndarray:
    h, w = (int(v) for v in original_size)
    return np.unpackbits(bits)[: h * w].reshape(h, w).astype(bool)


def _fallback(mask: np.ndarray, seg) -> bool:
    """The D18 verdict, by the segmenter's own rule (mask_and_crop), which reads only the mask."""
    from coffeecv.segment_beans import mask_and_crop
    return bool(mask_and_crop(np.zeros((*mask.shape, 3), np.uint8), mask, seg.p).info["fallback"])


def compare(refs: dict, got: dict, context) -> dict:
    """Per-photo measures and D7's gate. `context` gives the repo's head, guard and segmenter (make_refs.Context)."""
    rows = []
    names = [str(n) for n in refs["photos"]]
    for i, name in enumerate(names):
        row: dict = {"photo": name, "least_cosine": None, "mask_iou": None}
        if "readouts" in got:
            r_ref, r_got = refs["readouts"][i], got["readouts"][i]
            a_ref = _answers(r_ref, context.head, context.ref, context.probe)
            a_got = _answers(r_got, context.head, context.ref, context.probe)
            row.update(least_cosine=least_cosine(r_got, r_ref),
                       readout_max_abs=float(np.abs(r_got - r_ref).max()),
                       nan_or_inf=bool(not np.isfinite(r_got).all()),
                       top1_same=a_ref["top1"] == a_got["top1"],
                       prob_max_change=float(np.abs(a_got["probs"] - a_ref["probs"]).max()))
            if "probe" in a_ref:
                row.update(probe_change=a_got["probe"] - a_ref["probe"],
                           probe_distance_from_threshold=a_ref["probe"] - 0.5 if context.probe else None,
                           verdict_same=a_ref.get("refused") == a_got.get("refused"))
        if "low_masks" in got:
            low_ref, low_got = refs["low_masks"][i], got["low_masks"][i]
            m_got = _final_mask(low_got[2], refs["input_size"][i], refs["original_size"][i], context.seg)  # multi3
            m_ref = unpack_mask(refs["mask_bits"][i], refs["original_size"][i])
            fb = _fallback(m_got, context.seg)
            row.update(mask_iou=mask_iou(m_got, m_ref), fallback_same=fb == bool(refs["fallback"][i]),
                       mask_pixels_differ=int(np.count_nonzero(m_got ^ m_ref)),
                       low_logit_max_change=float(np.abs(low_got - low_ref).max()),
                       iou_pred_max_change=float(np.abs(got["ious"][i] - refs["ious"][i]).max()),
                       low_nan_or_inf=bool(not np.isfinite(low_got).all()))
        rows.append(row)
    return {"photos": rows, "gate": gate(rows)}


def summary_line(label: str, report: dict) -> str:
    g, rows = report["gate"], report["photos"]
    parts = [f"{label}: {'PASS' if g['pass'] else 'FAIL'}"]
    if g["least_cosine"] is not None:
        parts.append(f"least cosine {g['least_cosine']:.7f}")
        parts.append(f"max prob change {max(r['prob_max_change'] for r in rows):.2e}")
        if any(r.get("probe_change") is not None for r in rows):
            parts.append(f"max |probe change| {max(abs(r['probe_change']) for r in rows):.2e}")
    if g["least_mask_iou"] is not None:
        parts.append(f"least mask IoU {g['least_mask_iou']:.6f}")
        parts.append(f"max low-res logit change {max(r['low_logit_max_change'] for r in rows):.2e}")
    if any(r.get("nan_or_inf") or r.get("low_nan_or_inf") for r in rows):
        parts.append("NaN/Inf PRESENT")
    if g["changed_answer"]:
        parts.append(f"answer changed: {', '.join(g['changed_answer'])}")
    return "; ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--got", type=Path, nargs="+", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="the shipped checkpoint refs were made from")
    ap.add_argument("--out", type=Path, default=None, help="write every report as one JSON")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_refs import Context

    refs = dict(np.load(args.refs, allow_pickle=False))
    context = Context.load(args.checkpoint)
    reports = {}
    for path in args.got:
        reports[path.name] = compare(refs, dict(np.load(path, allow_pickle=False)), context)
        print(summary_line(path.name, reports[path.name]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(reports, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
