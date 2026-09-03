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
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import WatchedFileHandler
from pathlib import Path

from flask import Flask, Response, g, jsonify, request
from PIL import Image
from werkzeug.exceptions import HTTPException

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

# --- structured request logging -- see docs/logging_plan.html for the field
# list, the GDPR reasoning, and what's deliberately excluded (client IP,
# filenames, image bytes, anything beyond a coarse User-Agent category). ---

LOG_DIR = Path(os.environ.get("COFFEE_CV_LOG_DIR", "/var/log/coffee-cv"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


class _JsonFormatter(logging.Formatter):
    """One JSON object per line. Only fields actually set on the record are
    included, so a route that has nothing to say about e.g. `verdict` just
    omits the key rather than writing it as null on every line.
    """

    _FIELDS = (
        "request_id", "endpoint", "status", "latency_ms", "verdict",
        "top1_class", "top1_score", "ood_median", "ood_warned",
        "crop_needs_review", "skip_crop", "upload_format", "upload_bytes",
        "decoded_w", "decoded_h", "decode_ms", "crop_detect_ms", "inference_ms",
        "beans_across", "bean_pitch_px",
        "ua_category", "exc_type",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for field in self._FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# Separate logger + dedicated file, not the root `logger` above: this is
# structured, machine-parsed, and kept a full year (see docs/logging_plan.html
# section 06) -- mixing it into journald via basicConfig would compete with
# the system journal's own retention budget for everything else on the box.
# WatchedFileHandler (not a plain FileHandler) is what lets logrotate rotate
# this file by renaming it -- the handler notices the inode changed on its
# next write and reopens the new file itself, no signal or restart needed.
request_logger = logging.getLogger("coffee_cv.request")
request_logger.setLevel(logging.INFO)
request_logger.propagate = False
_request_log_handler = WatchedFileHandler(LOG_DIR / "app.jsonl")
_request_log_handler.setFormatter(_JsonFormatter())
request_logger.addHandler(_request_log_handler)


def _coarse_user_agent(ua: str) -> str:
    """OS + browser family only -- never the raw string. The one place this is
    used (decode-failure / error logging) needs "which browsers cluster around
    this failure", not a fingerprintable exact version. See docs/logging_plan.html
    section 03-C/D.
    """
    if not ua:
        return "unknown"
    if "iPhone" in ua or "iPad" in ua:
        os_name = "iOS"
    elif "Android" in ua:
        os_name = "Android"
    elif "Macintosh" in ua:
        os_name = "macOS"
    elif "Windows" in ua:
        os_name = "Windows"
    elif "Linux" in ua:
        os_name = "Linux"
    else:
        os_name = "unknown"
    if "Edg" in ua:
        browser = "Edge"
    elif "CriOS" in ua or "Chrome" in ua:
        browser = "Chrome"
    elif "FxiOS" in ua or "Firefox" in ua:
        browser = "Firefox"
    elif "Safari" in ua:
        browser = "Safari"
    else:
        browser = "unknown"
    return f"{os_name} {browser}"


def _log_decode_failure(exc: Exception) -> None:
    request_logger.warning("decode failure", extra={
        "request_id": getattr(g, "request_id", None),
        "endpoint": request.path,
        "exc_type": type(exc).__name__,
        "ua_category": _coarse_user_agent(request.headers.get("User-Agent", "")),
    })


@app.before_request
def _start_request():
    g.request_id = uuid.uuid4().hex[:8]
    g.t0 = time.monotonic()


@app.after_request
def _log_request(response):
    # Not shown in the JSON error body (see _entry_to_response) -- this header
    # is what lets nginx's access log (docs/logging_plan.html phase 2) capture
    # the same id via $upstream_http_x_request_id, joining its line to this one.
    response.headers["X-Request-Id"] = g.request_id
    extra = {
        "request_id": g.request_id,
        "endpoint": request.path,
        "status": response.status_code,
        "latency_ms": round((time.monotonic() - g.t0) * 1000),
    }
    for field in ("verdict", "top1_class", "top1_score", "ood_median", "ood_warned",
                  "crop_needs_review", "skip_crop", "upload_format", "upload_bytes",
                  "decoded_w", "decoded_h", "decode_ms", "crop_detect_ms", "inference_ms",
                  "beans_across", "bean_pitch_px"):
        value = getattr(g, field, None)
        if value is not None:
            extra[field] = value
    request_logger.info("request", extra=extra)
    return response


@app.errorhandler(Exception)
def _log_unhandled(exc):
    # Flask/Werkzeug's own HTTP errors (404, 405, ...) are not bugs -- let
    # them return normally instead of flattening every one into a 500.
    if isinstance(exc, HTTPException):
        return exc
    request_logger.error("unhandled exception", exc_info=exc, extra={
        "request_id": getattr(g, "request_id", None),
        "endpoint": request.path,
        "exc_type": type(exc).__name__,
        "ua_category": _coarse_user_agent(request.headers.get("User-Agent", "")),
    })
    return jsonify(error="internal error"), 500


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


def _log_classify_fields(entry: dict, body: dict) -> None:
    """Populate flask.g with the fields _log_request will pick up on its way out.

    Kept separate from _entry_to_response, which shapes the client-facing JSON
    and stays a pure mapping with no Flask-context dependency. Every field here
    is read with .get()/an `if` guard rather than assumed present, because how
    much of `entry` exists depends on which verdict was reached -- a scale
    refusal has decode/crop timing but no inference timing, an unmeasurable
    refusal has neither.
    """
    g.verdict = body["verdict"]
    crop = entry.get("crop")
    if crop:
        g.crop_needs_review = crop.get("needs_review")
    ood = entry.get("ood")
    if ood:
        g.ood_median = ood.get("median")
        g.ood_warned = bool(ood.get("warned"))
    ranked = entry.get("ranked")
    if ranked:
        g.top1_class = ranked[0][0]
        g.top1_score = round(ranked[0][2], 4)
    if "decoded_wh" in entry:
        g.decoded_w, g.decoded_h = entry["decoded_wh"]
    # Framing geometry. Nothing gates on these -- the scale guard that used to
    # was deleted because it could not fire (see coffeecv/infer.py's docstring).
    # They are logged because any future guard has to get its thresholds from
    # the real distribution of what people photograph, and the original's
    # thresholds were unreachable precisely because they came from theory
    # instead. bean_pitch_px is included alongside because it is what reveals
    # the estimator's band-floor pinning on real traffic rather than on the
    # 192-photo sample it was found in. See docs/scale_guard_plan.md.
    if "beans_across" in entry:
        g.beans_across = entry["beans_across"]
    if "bean_pitch_px" in entry:
        g.bean_pitch_px = entry["bean_pitch_px"]
    timing = entry.get("timing_ms")
    if timing:
        g.decode_ms = timing.get("decode")
        g.crop_detect_ms = timing.get("crop_detect")
        g.inference_ms = timing.get("inference")


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
        g.upload_format = orig_suffix.lstrip(".") or "unknown"
        g.upload_bytes = os.path.getsize(tmp_path)
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
    g.skip_crop = skip_crop

    with _saved_upload(upload) as path:
        entry = classify_one(path, cfg, class_ids, class_labels, model, head, ref,
                              n_patches=N_PATCHES, skip_crop=skip_crop)

    body, status = _entry_to_response(entry)
    _log_classify_fields(entry, body)
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
            _log_decode_failure(exc)
            return jsonify(error=f"could not read this as an image: {exc}"), 400

    h, w = rgb.shape[:2]
    g.decoded_w, g.decoded_h = w, h
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
            _log_decode_failure(exc)
            return jsonify(error=f"could not read this as an image: {exc}"), 400

    g.decoded_h, g.decoded_w = rgb.shape[:2]
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
