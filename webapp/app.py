"""Public web service for the shipped coffee-bean classifier -- one endpoint, one
model, no state. Wraps `coffeecv.infer.classify_one` rather than reimplementing
any of its scale/OOD logic, so this service and the CLI can never silently drift
on what counts as a refusal (see `coffeecv/infer.py`'s own docstring on parity).

Model, config, class labels and the OOD reference are all loaded once here, at
import time -- gunicorn is run with a single sync worker (see
`webapp/deploy/coffee-cv-web.service`), so "once at import" means once for the
life of the process, not once per request.
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, Response, jsonify, request
from PIL import Image

from coffeecv.config import REPO_ROOT
from coffeecv.dataset import RAW_EXTENSIONS, load_class_labels, load_rgb_image
from coffeecv.infer import classify_one, config_for_checkpoint, crop_to_bean_region, load_model, reference_path_for

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT = REPO_ROOT / "models" / "allrigs_oneplusmerged_s17.pt"
N_PATCHES = 40
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # matches nginx's client_max_body_size
PREVIEW_MAX_DIM = 1024  # a thumbnail, not the classification input -- keep it light

# static_folder=None: this app serves exactly one route. nginx serves the static
# page and only proxies /classify here -- disabling Flask's own default
# static-file route removes that surface even in-process, redundantly with nginx
# never routing to it.
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

cfg, cfg_source = config_for_checkpoint(CHECKPOINT, None)
logger.info("loaded checkpoint config from %s", cfg_source)
class_labels = load_class_labels(REPO_ROOT / cfg.classes_file)
class_ids = sorted(class_labels)
model, head = load_model(CHECKPOINT, cfg.model_name, len(class_ids), cfg.dropout)

ref_path = reference_path_for(CHECKPOINT)
ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
if ref is None:
    logger.warning("OOD guard unavailable: no reference at %s -- predictions below will be unguarded", ref_path)


def _entry_to_response(entry: dict) -> tuple[dict, int]:
    """`classify_one`'s CLI-shaped dict -> (JSON body, HTTP status) for the frontend.

    Scale/OOD refusals are 200s -- a well-formed answer, just not a prediction.
    "Unmeasurable" is different: that's not a photo the pipeline could even read,
    which is a bad request, not a legitimate refusal -- 400, same as the "no file
    uploaded" case above it.

    The OOD guard's `entry["ood"]["note"]` is deliberately NOT reused verbatim
    here, unlike everywhere else this file borrows CLI-shaped text. That note is
    written for the CLI's audience -- someone debugging thresholds, who wants the
    percentile and the reasoning. A phone user just needs the decision, not the
    diagnostic; "warned"/"refused_ood" below carry that decision, and one or two
    words is enough to say it.
    """
    verdict = entry["verdict"]
    if verdict == "REFUSED (unmeasurable)":
        return {"verdict": "refused_unreadable",
                "message": "Couldn't read this as a photo of coffee beans."}, 400
    if verdict == "REFUSED (scale)":
        return {"verdict": "refused_scale", "message": entry["scale_note"]}, 200
    if verdict == "REFUSED (out of distribution)":
        return {"verdict": "refused_ood", "message": "Unrecognized"}, 200

    response = {
        "verdict": "predicted",
        "ranked": [{"id": cid, "label": label, "score": score}
                   for cid, label, score in entry["ranked"]],
    }
    warnings = []
    ood = entry.get("ood")
    if ood and ood.get("warned"):
        warnings.append("Uncertain")
    crop = entry.get("crop")
    if crop and crop.get("needs_review"):
        warnings.append("Uncertain framing")
    if warnings:
        response["verdict"] = "warned"
        response["warning"] = " / ".join(warnings)
    return response, 200


@contextmanager
def _saved_upload(upload):
    """Write an upload to a tempfile and clean it up unconditionally afterward.

    Suffix is normally a fixed generic one, since Pillow sniffs format from
    content and doesn't need it. RAW is the one exception, and matters for
    correctness, not just convenience: DNG and friends are valid TIFF
    containers that embed a small compatibility preview for programs that
    don't understand RAW, so Pillow's TIFF plugin will happily "succeed" at
    opening one and silently return that degraded embedded preview instead of
    failing -- there's no way to detect "this needs rawpy instead" from
    content alone (confirmed: the same DNG opens as 1536x2040 via rawpy's real
    decode vs 1280x964 via Pillow's embedded-preview fallback). The extension
    is the only signal `load_rgb_image`'s RAW branch can dispatch on, so
    unlike every other format here, it's used -- but only ever checked
    against `RAW_EXTENSIONS`, coffeecv.dataset's own fixed allowlist, never
    trusted as an arbitrary string: `Path(...).suffix` only ever looks at the
    last path segment regardless of what a crafted filename contains before
    it, and the result must exactly match one of those literal entries or it
    falls back to the generic suffix same as before.
    """
    orig_suffix = Path(upload.filename or "").suffix.lower()
    suffix = orig_suffix if orig_suffix in RAW_EXTENSIONS else ".upload"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        upload.save(tmp_path)
        yield Path(tmp_path)
    finally:
        os.unlink(tmp_path)


@app.post("/classify")
def classify():
    upload = request.files.get("photo")
    if upload is None or upload.filename == "":
        return jsonify(error="no photo uploaded"), 400

    # User override for a live crop they don't trust (see /crop) -- the
    # server never takes the *box* from the client, only this one bit, and
    # classify_one() still does its own detection unless told to skip it.
    skip_crop = request.form.get("skip_crop") == "1"

    with _saved_upload(upload) as path:
        entry = classify_one(path, cfg, class_ids, class_labels, model, head, ref,
                              n_patches=N_PATCHES, skip_crop=skip_crop)

    body, status = _entry_to_response(entry)
    return jsonify(body), status


@app.post("/crop")
def crop():
    """Detect (not apply) the live-inference crop, so the frontend can preview
    it before /classify. Advisory only: classify_one() always recomputes its
    own crop server-side from the uploaded bytes via patches_for_photo, so
    nothing returned here is ever trusted back -- a client could send any box
    it wants and it would change nothing about what actually gets classified.
    """
    upload = request.files.get("photo")
    if upload is None or upload.filename == "":
        return jsonify(error="no photo uploaded"), 400

    with _saved_upload(upload) as path:
        try:
            rgb = load_rgb_image(path)
        except (ValueError, OSError) as exc:
            return jsonify(error=f"could not read this as an image: {exc}"), 400

    h, w = rgb.shape[:2]
    _, crop_info = crop_to_bean_region(rgb)
    if crop_info is None:
        return jsonify(cropped=False, box=None, needs_review=False)

    x, y, bw, bh = crop_info["box"]
    return jsonify(
        cropped=True,
        # Fractions of the full decoded image (x0, y0, x1, y1), not thumbnail
        # pixels -- keeps this independent of /preview's PREVIEW_MAX_DIM, so
        # the frontend can scale it against whatever it's actually displaying.
        box=[x / w, y / h, (x + bw) / w, (y + bh) / h],
        needs_review=crop_info["needs_review"],
    )


@app.post("/preview")
def preview():
    """Decode any format this app accepts and re-encode as a JPEG thumbnail.

    Exists so the browser never needs native decode support for HEIC/AVIF/JXL
    to show a preview -- Pillow already has to decode these to classify them
    (`load_rgb_image` is the same function `classify_one` uses under the hood),
    so reusing that here for a universally-displayable preview is free, and
    guaranteed to agree with what the model actually sees.
    """
    upload = request.files.get("photo")
    if upload is None or upload.filename == "":
        return jsonify(error="no photo uploaded"), 400

    with _saved_upload(upload) as path:
        try:
            rgb = load_rgb_image(path)
        except (ValueError, OSError) as exc:
            return jsonify(error=f"could not read this as an image: {exc}"), 400

    thumb = Image.fromarray(rgb)
    thumb.thumbnail((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return Response(buf.getvalue(), mimetype="image/jpeg")


if __name__ == "__main__":
    # Local/dev only -- production runs under gunicorn (see webapp/deploy/). Bound
    # to loopback only, debug never on: Werkzeug's debugger is remote-code-execution
    # as a feature if it's ever reachable.
    app.run(host="127.0.0.1", port=5000, debug=False)
