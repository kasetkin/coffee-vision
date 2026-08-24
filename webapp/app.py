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

import json
import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

from coffeecv.config import REPO_ROOT
from coffeecv.dataset import load_class_labels
from coffeecv.infer import classify_one, config_for_checkpoint, load_model, reference_path_for

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT = REPO_ROOT / "models" / "allrigs_mixstyle05_e100p20_s17.pt"
N_PATCHES = 40
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # matches nginx's client_max_body_size

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
    """
    verdict = entry["verdict"]
    if verdict == "REFUSED (unmeasurable)":
        return {"verdict": "refused_unreadable",
                "message": "Couldn't read this as a photo of coffee beans."}, 400
    if verdict == "REFUSED (scale)":
        return {"verdict": "refused_scale", "message": entry["scale_note"]}, 200
    if verdict == "REFUSED (out of distribution)":
        return {"verdict": "refused_ood", "message": entry["ood"]["note"]}, 200

    response = {
        "verdict": "predicted",
        "ranked": [{"id": cid, "label": label, "score": score}
                   for cid, label, score in entry["ranked"]],
    }
    ood = entry.get("ood")
    if ood and ood.get("warned"):
        response["verdict"] = "warned"
        response["warning"] = ood["note"]
    return response, 200


@app.post("/classify")
def classify():
    upload = request.files.get("photo")
    if upload is None or upload.filename == "":
        return jsonify(error="no photo uploaded"), 400

    # Fixed generic suffix, never derived from the client-supplied filename --
    # Pillow sniffs the format from file content, it doesn't need the extension,
    # so there's no reason to trust (or even look at) what the client called it.
    fd, tmp_path = tempfile.mkstemp(suffix=".upload")
    os.close(fd)
    try:
        upload.save(tmp_path)
        entry = classify_one(Path(tmp_path), cfg, class_ids, class_labels, model, head, ref,
                              n_patches=N_PATCHES)
    finally:
        os.unlink(tmp_path)

    body, status = _entry_to_response(entry)
    return jsonify(body), status


if __name__ == "__main__":
    # Local/dev only -- production runs under gunicorn (see webapp/deploy/). Bound
    # to loopback only, debug never on: Werkzeug's debugger is remote-code-execution
    # as a feature if it's ever reachable.
    app.run(host="127.0.0.1", port=5000, debug=False)
