"""ML-37 P4 (plan finding 7, R3, R10): the front's peak memory and time with the preview made after the encode (D15),
against its unit's MemoryMax=600M, on P3's photos (ML-29 R6's seven and the front-forms study's 40,
analysis/ml37/p3_front_cost.py).

    python -I analysis/ml37/p4_front_cost.py --code CODE_ROOT --data DATA_ROOT [--photos r6|forms|all]
        [--out FILE.jsonl] [--work DIR]

Each photo goes through the check service itself (webapp/front/check_service.py from `--code`), run in a fresh process
under `taskset -c 0,1` (the front's 2 vCPU), as the front runs it: this process posts the photo to it and plays the
inference host, which reads the forwarded photo and answers with a segmentation line and a classification line. The
row is the check service's own log line: its peak resident memory (VmHWM, reset as canonicalization starts, read
again once the preview is made, so it holds both, ML-37 D27), the canonicalization's and the preview's times, when the
preview went out (to_preview_ms, from the answer's opening), the preview's size, and whether the preview came first.
This machine cannot hold a process to MemoryMax: the peak is what the cgroup would have to hold. Not on the front.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p3_front_cost import R6, forms, make_ceiling  # noqa: E402


class Host(BaseHTTPRequestHandler):
    """The inference host: reads the forwarded photo, then answers as the app does, in two lines."""

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write(b'{"part": "segmentation", "mask": null, "box": null, "cropped": false, '
                         b'"seg_fallback": false, "needs_review": false}\n')
        self.wfile.write(b'{"part": "classification", "verdict": "refused_ood", "message": "Unrecognized"}\n')

    def log_message(self, *args):
        pass


def serve(code: Path, inference: str, log_dir: Path, port_file: Path) -> None:
    """The check service from `code`, on a free loopback port written to `port_file`."""
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / "webapp" / "front"))
    import check_service
    server = check_service.make_server(("127.0.0.1", 0), inference, log_dir)
    port_file.write_text(str(server.server_address[1]))
    server.serve_forever()


def post(port: int, photo: Path) -> tuple[str, list[str], int]:
    """The photo as the page sends it: its request ID, its answer's lines' names and the preview's JPEG bytes."""
    import http.client
    boundary, rid = uuid.uuid4().hex, uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="{photo.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n").encode() + photo.read_bytes() + \
        f"\r\n--{boundary}--\r\n".encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request("POST", "/analyze", body=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                                         "X-Request-Id": rid})
    resp = conn.getresponse()
    text = resp.read()
    conn.close()
    if resp.status != 200:
        return rid, [f"HTTP {resp.status}: {text[:200]!r}"], 0
    lines = [json.loads(ln) for ln in text.splitlines()]
    jpeg = next((len(base64.b64decode(ln["jpeg"])) for ln in lines if ln.get("part") == "preview"), 0)
    return rid, [ln["part"] for ln in lines], jpeg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--photos", choices=("r6", "forms", "all"), default="all")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--work", type=Path, default=Path("/tmp/ml37_p3"))
    ap.add_argument("--serve", nargs=3, metavar=("INFERENCE", "LOG_DIR", "PORT_FILE"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.serve:
        serve(args.code.resolve(), args.serve[0], Path(args.serve[1]), Path(args.serve[2]))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    host = ThreadingHTTPServer(("127.0.0.1", 0), Host)
    threading.Thread(target=host.serve_forever, daemon=True).start()
    inference = f"http://127.0.0.1:{host.server_address[1]}"
    photos = {**(R6 if args.photos in ("r6", "all") else {}), **(forms() if args.photos in ("forms", "all") else {})}
    for name, rel in photos.items():
        path = make_ceiling(args.work) if rel is None else args.data / rel
        with tempfile.TemporaryDirectory() as tmp:
            port_file = Path(tmp) / "port"
            proc = subprocess.Popen(["taskset", "-c", "0,1", sys.executable, "-I", __file__, "--code", str(args.code),
                                     "--data", str(args.data), "--serve", inference, tmp, str(port_file)],
                                    stderr=subprocess.DEVNULL)
            try:
                for _ in range(600):
                    if port_file.exists() and port_file.read_text():
                        break
                    time.sleep(0.1)
                t = time.perf_counter()
                rid, parts, jpeg = post(int(port_file.read_text()), path)
                seconds = time.perf_counter() - t
                for _ in range(100):   # the line is written once the answer has gone out
                    lines = [json.loads(ln) for ln in (Path(tmp) / "requests.jsonl").read_text().splitlines()]
                    line = next((ln for ln in lines if ln.get("request_id") == rid), None)
                    if line:
                        break
                    time.sleep(0.05)
            finally:
                proc.kill()
                proc.wait()
        row = {"photo": name, "path": rel, "parts": parts, "preview_jpeg_bytes": jpeg, "seconds": round(seconds, 2),
               **{k: line.get(k) for k in ("outcome", "status", "width", "height", "canonicalize_ms",
                                           "preview_build_ms", "to_preview_ms", "crossing_ms", "total_ms", "peak_mb",
                                           "swap_mb", "preview_error", "error")
                  if line and line.get(k) is not None}}
        out = json.dumps(row)
        print(out, flush=True)
        if args.out:
            with args.out.open("a") as f:
                f.write(out + "\n")
    host.shutdown()


if __name__ == "__main__":
    main()
