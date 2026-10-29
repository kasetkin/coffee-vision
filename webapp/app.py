"""Public web service for the shipped coffee-bean classifier -- one endpoint, one
model, no state. /analyze runs `coffeecv.infer.classify_one`'s two steps,
`segment_decoded` and `classify_segmented`, rather than reimplementing any of
its scale/OOD logic, so this service and the CLI can never silently drift on
what counts as a refusal (see `coffeecv/infer.py`'s own docstring on parity).
Its answer streams back in two parts, the preview and then the classification
(ticket OPS-7, ADR 0023).

Model, config, class labels and the OOD reference are all loaded once here, at
import time -- gunicorn is run with a single sync worker (see
`webapp/deploy/coffee-cv-web.service`), so "once at import" means once for the
life of the process, not once per request.

The models run on LiteRT (ticket ML-7, ADR 0019), from the converted model beside the checkpoint
(coffeecv.litert), on the accelerator and at the precisions its manifest names; this process imports no
torch. Before the first request the worker runs a fixed photo through /analyze's whole path (ADR 0021), and logs how long its boot took and the memory it holds. On the GPU, the warm-up is also ADR 0020's proof
that the GPU ran it: a worker that cannot show it does not serve.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import resource
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import WatchedFileHandler
from pathlib import Path

import numpy as np
from flask import Flask, Response, g, jsonify, request, stream_with_context
from PIL import Image
from werkzeug.exceptions import HTTPException

BOOT_T0 = time.monotonic()

from coffeecv.config import REPO_ROOT  # noqa: E402 -- after BOOT_T0, so the boot's time includes the imports
from coffeecv.class_list import load_classes  # noqa: E402
from coffeecv.infer import (_sha, classify_segmented, config_for_checkpoint, PROBE_THRESHOLD,  # noqa: E402
                            inference_tta_for, load_ood_probe, load_ood_reference, probe_path_for,
                            reference_path_for, segment_decoded, sig12, warm_up)
from coffeecv.litert import gpu_idle_ms, gpu_mapped, gpu_proof, litert_dir_for, load_models  # noqa: E402
from coffeecv.photo_io import RAW_EXTENSIONS, load_rgb_image  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, seg_params  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The shipped model. COFFEE_CV_CHECKPOINT (a path relative to the repo) overrides it without a code
# edit -- for smoke-testing a candidate inside the real systemd sandbox before repointing this line.
CHECKPOINT = REPO_ROOT / os.environ.get("COFFEE_CV_CHECKPOINT", "models/dino3b16_segxl0v6_country_aug8_s42.pt")
# Its converted model (coffeecv.litert): beside it unless COFFEE_CV_LITERT_DIR (relative to the repo) names another,
# for the same smoke tests. COFFEE_CV_LITERT_RUNTIME is LiteRT's accelerator directory (release.env names the
# release venv's ai_edge_litert/, which holds the WebGPU library) and COFFEE_CV_WARMUP_PHOTO the boot's warm-up
# photo (ADR 0021); both relative to the release, or absolute.
LITERT_DIR = REPO_ROOT / os.environ["COFFEE_CV_LITERT_DIR"] if "COFFEE_CV_LITERT_DIR" in os.environ \
    else litert_dir_for(CHECKPOINT)
LITERT_RUNTIME = REPO_ROOT / os.environ["COFFEE_CV_LITERT_RUNTIME"] if os.environ.get("COFFEE_CV_LITERT_RUNTIME") \
    else None
N_PATCHES = 40
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # matches nginx's client_max_body_size
PREVIEW_MAX_DIM = 1024  # a thumbnail, not the classification input -- keep it light

# static_folder=None: this app serves exactly one route. nginx serves the static
# page and only proxies /analyze here -- disabling Flask's own default
# static-file route removes that surface even in-process, redundantly with nginx
# never routing to it.
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

cfg, cfg_source = config_for_checkpoint(CHECKPOINT, None)
logger.info("loaded checkpoint config from %s", cfg_source)
# The model's own frozen list (infer.classes_path_for): folder ids for a model fitted before ticket ML-3,
# countries after it. class_ids is in output order.
classes = load_classes(REPO_ROOT / cfg.classes_file)
class_ids, class_labels = list(classes.keys), dict(classes.labels)
# COFFEE_CV_LITERT_ON_CPU=1: every graph on this machine's CPU at fp32 whatever the manifest names, for runs off the
# serving host (the deploy's local expected answers, ML-7 P10). Never set by the service.
models = load_models(LITERT_DIR, LITERT_RUNTIME, on_cpu=os.environ.get("COFFEE_CV_LITERT_ON_CPU") == "1")
if models.manifest["checkpoint_sha256"][:16] != _sha(CHECKPOINT):
    raise SystemExit(f"{LITERT_DIR} was converted from checkpoint {models.manifest['checkpoint_sha256'][:16]}, "
                     f"not {CHECKPOINT.name}")
classifier = models.classifier
logger.info("LiteRT model %s: %s", LITERT_DIR.name,
            json.dumps({g: {k: spec[k] for k in ("file", "accelerator", "precision", "batch")}
                        for g, spec in models.manifest["graphs"].items()}))
# Dihedral TTA is a property of the model, read from its card (off for a frozen ViT,
# where 8 views would cost ~46 s per photo -- docs/dinov3_integration_plan.md §6.2). LiteRT serves none.
TTA = inference_tta_for(CHECKPOINT, cfg.model_name)
if TTA:
    raise SystemExit(f"{CHECKPOINT.name}'s card asks for TTA, which the LiteRT service does not serve")
logger.info("model %s (%s), %d classes, TTA off", CHECKPOINT.name, cfg.model_name, len(class_ids))

ref_path = reference_path_for(CHECKPOINT)
ref = load_ood_reference(CHECKPOINT, classifier.embedding_dim)
if ref is None:
    logger.warning("OOD guard unavailable: no reference at %s -- predictions below will be unguarded", ref_path)

# The probe is optional and its presence *is* the deployment decision, the same
# way the reference's presence decides whether the guard runs at all. Deploying
# it means copying one file next to the weights; rolling back means removing it,
# with no code change and no restart-time flag to get wrong. A mismatched probe
# raises here, at import, rather than serving confident nonsense per request.
probe = load_ood_probe(CHECKPOINT, embedding_dim=classifier.embedding_dim)
if probe is None:
    logger.info("OOD guard: centroid metric (no probe beside %s)", CHECKPOINT.name)
else:
    logger.info("OOD guard: linear_probe from %s, refusing above a fixed %.1f (ADR 0016)",
                probe["_path"], PROBE_THRESHOLD)

# Ticket ML-2: a crop_method "segment" model finds the bean region with the segmenter it was trained with, here
# its converted graphs; BeanSegmenter refuses graphs whose weights or decoder are not the config's, at import, so
# a mismatch fails the boot, not the first request.
segmenter = BeanSegmenter(seg_params(cfg), engine=models.sam) if cfg.crop_method == "segment" else None
if segmenter is not None:
    logger.info("bean region: segmenter %s + decoder %s", segmenter.weights_sha256[:16],
                (segmenter.decoder_sha256 or "pretrained")[:16])


def _file_sha(path: Path) -> str | None:
    return _sha(path) if path.exists() else None


def _dvc_md5(path: Path) -> str | None:
    """The md5 DVC recorded for a tracked file (`<file>.dvc`), i.e. which DVC object is deployed."""
    dvc_file = Path(f"{path}.dvc")
    if not dvc_file.exists():
        return None
    for line in dvc_file.read_text().splitlines():
        if line.strip().startswith("- md5:") or line.strip().startswith("md5:"):
            return line.split("md5:", 1)[1].strip()
    return None


def _code_identity() -> dict:
    """Which code is serving. A deploy stamp from write_build_info.py is authoritative; without one, fall
    back to git HEAD at import and say so -- under an rsync deploy HEAD can describe other code."""
    stamp = REPO_ROOT / "webapp" / "BUILD_INFO.json"
    if stamp.exists():
        info = json.loads(stamp.read_text())
        return {"commit": info.get("commit"), "dirty": info.get("dirty"), "source": "BUILD_INFO.json"}
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD") or None,
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
                "source": "git HEAD at import (no BUILD_INFO.json; may not match rsynced code)"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "source": "unknown"}


# What is serving, fixed at import and put on EVERY request line (not just the startup line, which log
# rotation eventually drops): a year of logs mixes model versions, and a score is meaningless without
# the embedding space and threshold that produced it. Hashes are the same 16-hex sha256 prefix the OOD
# artifacts use for `checkpoint_sha`.
BUILD = {
    "model_sha": _file_sha(CHECKPOINT),
    "model_dvc_md5": _dvc_md5(CHECKPOINT),
    "backbone_sha": models.manifest["backbone"]["weights_sha256"][:16],
    "litert": {"manifest_sha": _file_sha(LITERT_DIR / "manifest.json"),
               **{g: f"{spec['accelerator']} {spec['precision']}" for g, spec in models.manifest["graphs"].items()}},
    "ood_reference_sha": _file_sha(ref_path),
    "ood_probe_sha": _file_sha(probe_path_for(CHECKPOINT)) if probe is not None else None,
    "ood_threshold": PROBE_THRESHOLD if probe is not None else None,
    "tta": TTA,
    "crop_method": cfg.crop_method,
    **({"segmenter_sha": segmenter.weights_sha256[:16],
        "seg_decoder_sha": (segmenter.decoder_sha256 or "")[:16] or None} if segmenter is not None else {}),
    "code": _code_identity(),
}
logger.info("serving %s %s", CHECKPOINT.name, json.dumps(BUILD))

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
        "request_id", "endpoint", "status", "latency_ms", "preview_ms", "outcome", "parts_sent", "model", "build",
        "verdict",
        "top1_class", "top1_score", "ood_median", "ood_probe", "ood_method", "ood_warned",
        "crop_needs_review", "skip_crop", "upload_format", "upload_bytes",
        "decoded_w", "decoded_h", "decode_ms", "crop_detect_ms", "inference_ms",
        "beans_across", "bean_pitch_px",
        "mask_area_frac", "bean_frac_in_crop", "retained_frac", "seg_fallback", "patches_below_share",
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
    # A streamed answer is still being computed here: its own generator writes the line when it ends (OPS-7 R2).
    if not getattr(g, "stream_logs_itself", False):
        _write_request_line(response.status_code)
    return response


def _write_request_line(status: int) -> None:
    """The request's one line in the request log, from the fields its route set on flask.g."""
    extra = {
        "request_id": g.request_id,
        "endpoint": request.path,
        "status": status,
        "latency_ms": round((time.monotonic() - g.t0) * 1000),
        "model": CHECKPOINT.name,
        "build": BUILD,
    }
    for field in ("preview_ms", "outcome", "parts_sent",
                  "verdict", "top1_class", "top1_score", "ood_median", "ood_probe",
                  "ood_method", "ood_warned",
                  "crop_needs_review", "skip_crop", "upload_format", "upload_bytes",
                  "decoded_w", "decoded_h", "decode_ms", "crop_detect_ms", "inference_ms",
                  "beans_across", "bean_pitch_px",
                  "mask_area_frac", "bean_frac_in_crop", "retained_frac", "seg_fallback", "patches_below_share"):
        value = getattr(g, field, None)
        if value is not None:
            extra[field] = value
    request_logger.info("request", extra=extra)


@app.errorhandler(Exception)
def _log_unhandled(exc):
    # Flask/Werkzeug's own HTTP errors (404, 405, ...) are not bugs -- let
    # them return normally instead of flattening every one into a 500.
    if isinstance(exc, HTTPException):
        return exc
    _log_exception(exc)
    return jsonify(error="internal error"), 500


def _log_exception(exc: Exception) -> None:
    request_logger.error("unhandled exception", exc_info=exc, extra={
        "request_id": getattr(g, "request_id", None),
        "endpoint": request.path,
        "exc_type": type(exc).__name__,
        "ua_category": _coarse_user_agent(request.headers.get("User-Agent", "")),
    })


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
        # Both scores on every line, not just the deciding one: the centroid
        # median keeps the log series continuous across the metric switch, and
        # the probe score is what the next recalibration will be fitted against.
        g.ood_probe = ood.get("probe")
        g.ood_method = ood.get("method")
        g.ood_warned = bool(ood.get("warned"))
    ranked = entry.get("ranked")
    if ranked:
        g.top1_class = ranked[0][0]
        g.top1_score = sig12(ranked[0][2])
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
    # Segmenter models only (ML-2): how much of the photo the mask covers, what the crop keeps, whether the
    # D18 whole-photo fallback fired, and how many patches the D17 rule had to top up. Nothing gates on
    # them yet (D9); a gate needs their real-traffic distribution first.
    for field in ("mask_area_frac", "bean_frac_in_crop", "retained_frac", "seg_fallback", "patches_below_share"):
        if field in entry:
            setattr(g, field, entry[field])
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


def _mask_png_b64(mask: np.ndarray) -> str:
    """The whole photo's bool mask as the preview part sends it (ticket OPS-6 D7): scaled to long side <= PREVIEW_MAX_DIM,
    aspect kept, never up (area average, then >= 0.5), as a 1-bit palette PNG, index 0 (not bean) transparent
    and index 1 (bean) white, base64. The page stretches it to the photo's displayed rect, as it does the
    box's fractions, so its size need not equal the thumbnail's to the pixel."""
    h, w = mask.shape
    scale = min(1.0, PREVIEW_MAX_DIM / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    frac = Image.fromarray(mask.astype(np.float32))   # mode "F": BOX averages the 0/1 values exactly
    if frac.size != size:
        frac = frac.resize(size, Image.Resampling.BOX)
    img = Image.fromarray((np.asarray(frac) >= 0.5).astype(np.uint8))
    img.putpalette([0, 0, 0, 255, 255, 255])
    buf = io.BytesIO()
    img.save(buf, "PNG", bits=1, transparency=0, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/analyze")
def analyze():
    """One photo, one upload, one answer in two parts (ticket OPS-7, ADR 0023): `application/x-ndjson`, one JSON
    object per line. The photo is decoded and segmented once, before the answer starts, so an unreadable photo is
    still a plain 400 and a segmenter error the error handler's 500. Then the stream: the preview part
    (`_preview_part`), then the classification part, `_entry_to_response`'s body, or an error part if the
    classification fails, since the 200 has gone out with the preview (R4).

    skip_crop=1 is the user's override (D1, "Use full photo instead"): the segmenter does not run and the whole
    photo is classified -- the server never takes a box from the client, only this one bit. Its preview has no
    mask and no box (D9).
    """
    upload = request.files.get("photo")
    if upload is None or upload.filename == "":
        return jsonify(error="no photo uploaded"), 400
    skip_crop = request.form.get("skip_crop") == "1"
    g.skip_crop = skip_crop

    with _saved_upload(upload) as path:
        t0 = time.monotonic()
        try:
            photo = load_rgb_image(path)
        except (ValueError, OSError) as exc:
            _log_decode_failure(exc)
            return jsonify(error=f"could not read this as an image: {exc}"), 400
        decode_ms = round((time.monotonic() - t0) * 1000)
    g.decoded_h, g.decoded_w = photo.shape[:2]
    seg = segment_decoded(photo, cfg, skip_crop=skip_crop, segmenter=segmenter, decode_ms=decode_ms)
    preview = _preview_part(seg)
    g.stream_logs_itself = True
    return Response(stream_with_context(_analyze_stream(seg, preview)), mimetype="application/x-ndjson",
                    # Both nginx servers stream it only if asked to; ours passes the header on (OPS-7 D11).
                    headers={"X-Accel-Buffering": "no"})


def _analyze_stream(seg, preview: dict):
    """/analyze's two lines, and its one request-log line when the stream ends, however it ends (OPS-7 R2, D19):
    `outcome` is complete, error, or client_gone (the generator closed at a yield: the client left, or the server
    gave up writing to it), and `parts_sent` counts the lines handed to the server."""
    g.parts_sent, g.outcome = 0, "client_gone"
    try:
        g.parts_sent, g.preview_ms = 1, round((time.monotonic() - g.t0) * 1000)   # D13: when the preview went out
        yield json.dumps(preview) + "\n"
        try:
            entry = classify_segmented(seg, cfg, class_ids, class_labels, classifier, ref, n_patches=N_PATCHES,
                                       tta=TTA, probe=probe)
            body, _ = _entry_to_response(entry)   # the status went out with the preview
            _log_classify_fields(entry, body)
            line, ending = {"part": "classification", **body}, "complete"
        except Exception as exc:   # any failure after the 200 is an error part, logged as the handler logs one
            _log_exception(exc)
            line, ending = {"part": "error", "message": "Something went wrong -- try a different photo."}, "error"
        g.parts_sent = 2
        yield json.dumps(line) + "\n"
        g.outcome = ending
    finally:
        _write_request_line(200)


def _preview_part(seg) -> dict:
    """The answer's first part (ticket OPS-7 D8): the photo as a JPEG thumbnail, so the browser never needs to
    decode HEIC/AVIF/JXL/RAW itself, and guaranteed to be the pixels the model sees (`load_rgb_image`'s); the
    whole photo's bean mask (`_mask_png_b64`; null on a fallback, on skip_crop and for a tray-heuristic model);
    `seg_fallback` (ML-2 D18: the whole photo is classified); and the crop box. Display only: nothing here is
    ever trusted back from the client.
    """
    thumb = Image.fromarray(seg.photo)
    thumb.thumbnail((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    part = {"part": "preview", "thumbnail": base64.b64encode(buf.getvalue()).decode("ascii"),
            "mask": None if seg.photo_mask is None else _mask_png_b64(seg.photo_mask),
            "seg_fallback": seg.seg_fallback}
    if seg.crop_info is None:
        return {**part, "cropped": False, "box": None, "needs_review": False}
    h, w = seg.photo.shape[:2]
    x, y, bw, bh = seg.crop_info["box"]
    # Fractions of the full decoded photo (x0, y0, x1, y1), not thumbnail pixels -- keeps this independent of
    # PREVIEW_MAX_DIM, so the page can scale it against whatever it is actually displaying.
    return {**part, "cropped": True, "box": [x / w, y / h, (x + bw) / w, (y + bh) / h],
            "needs_review": seg.crop_info["needs_review"]}


def _mem_available_mb() -> float | None:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 1024, 1)
    return None


def warm_up_photo() -> Path:
    """COFFEE_CV_WARMUP_PHOTO (relative to the repo, or absolute), else the deploy's smoke fixture where the repo
    has it (a checkout, not a release). The same photo every boot, so every boot does the same work (ADR 0021)."""
    if os.environ.get("COFFEE_CV_WARMUP_PHOTO"):
        return REPO_ROOT / os.environ["COFFEE_CV_WARMUP_PHOTO"]
    fixtures = REPO_ROOT / "webapp" / "deploy" / "fixtures.txt"
    smoke = [ln.split()[2] for ln in fixtures.read_text().splitlines() if ln.startswith("smoke ")] \
        if fixtures.exists() else []
    if not smoke:
        raise SystemExit("no warm-up photo: set COFFEE_CV_WARMUP_PHOTO (ADR 0021)")
    return REPO_ROOT / smoke[0]


_idle0, _t0 = gpu_idle_ms(), time.monotonic()
WARM_UP = warm_up(warm_up_photo(), cfg, class_ids, class_labels, classifier, ref, probe, segmenter, N_PATCHES)
if models.on_gpu:
    _idle1 = gpu_idle_ms()
    WARM_UP["gpu"] = gpu_proof(gpu_mapped(), None if None in (_idle0, _idle1) else _idle1 - _idle0,
                               (time.monotonic() - _t0) * 1000)
WARM_UP.update(boot_s=round(time.monotonic() - BOOT_T0, 2),
               peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               mem_available_mb=_mem_available_mb())
logger.info("boot %s", json.dumps(WARM_UP))
if "gpu" in WARM_UP and not WARM_UP["gpu"]["ok"]:
    raise SystemExit(f"the GPU is not proven to have run the warm-up (ADR 0020): {WARM_UP['gpu']['why_not']}")


if __name__ == "__main__":
    # Local/dev only -- production runs under gunicorn (see webapp/deploy/). Bound
    # to loopback only, debug never on: Werkzeug's debugger is remote-code-execution
    # as a feature if it's ever reachable.
    app.run(host="127.0.0.1", port=5000, debug=False)
