"""ML-7 P0 (ticket R1, D14): where each photo's time goes today, before any conversion.

Two halves, both read-only:

  log      the live record: the service's request log (app.jsonl and its rotations, gzipped or not), reduced
           to per-endpoint aggregates of latency_ms, decode_ms, crop_detect_ms and inference_ms over the
           successful /crop and /classify requests of one model (--model: the log keeps every model the site
           has served). Only the aggregates leave the host. Standard library only.

  stages   the release's own code, per stage: run from inside a release directory, like release_probe.py,
           with that release's venv. It imports webapp.app (so the worker's boot is timed as gunicorn pays it,
           torch's import included), then posts every fixture photo to /crop and then /classify through
           Flask's test client, the order an upload takes on the site, for --passes passes. Nothing in the
           release is changed: the stages are timed by wrapping the release's own functions in this process
           (StageClock), the segmenter's stages come from its own timing_ms, and DINOv3 is timed per batch by
           forward hooks on the served model, so its head (one Linear) is inside dino_b32 and dino_b8. The
           request log goes to a scratch directory, never to the service's, and is removed at the end.

On yappy (the site is stopped, ticket D11), as the deploy user, with this file copied to ~/ml7/. The live
record first, so `stages` can set the fixtures beside it; then `stages`, detached from the SSH session
(docs/agents/remote-sweeps.md), from the current release with its own venv; then the two outputs home:

    ssh webappremote 'python3 ~/ml7/baseline_stages.py log /var/log/coffee-cv/app.jsonl* \\
        --model allrigs_dino3b16_segxl0v3_country_s42.pt --out ~/ml7/baseline.json'
    scripts/remote_launch.sh 'python -B ~/ml7/baseline_stages.py stages --list /opt/coffee-cv/fixtures/fixtures.txt \\
        --dir /opt/coffee-cv/fixtures/photos --live ~/ml7/baseline.json --out ~/ml7/baseline_stages.json' \\
        webappremote /opt/coffee-cv/current /opt/coffee-cv/current/.venv \\
        /home/coffee-deploy/ml7/stages.log /home/coffee-deploy/ml7/stages_status.log
    scp webappremote:ml7/baseline.json webappremote:ml7/baseline_stages.json analysis/ml7_gpu/

`stages` prints, per request, the whole time, the stages and the share no stage explains; then the P0 checks
(plan §4): the stages add up to each whole path within 10%; the fixtures' whole paths against the live record;
and the share of each photo's time spent in the models (the encoder and decoder of both segmenter passes, and
DINOv3), together and each. Under half together is the plan's stop for the owner. The report records this
file's sha256, so the run can be matched to a commit.

read_fixtures and the release-import check repeat release_probe.py's: that file is deploy tooling, not in the
release, and this one runs from ~/ml7/ beside neither.

Locally, at toy scale (the repo is laid out like a release; the model comes from COFFEE_CV_CHECKPOINT):

    COFFEE_CV_CHECKPOINT=models/allrigs_dino3b16_segxl0v3_country_s42.pt python -B \\
        analysis/ml7_gpu/baseline_stages.py stages --list webapp/deploy/fixtures.txt --dir <flat copy> \\
        --limit 1 --passes 1 --out /tmp/stages.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple

sys.dont_write_bytecode = True   # nothing written into the release directory

ENDPOINTS = ("/crop", "/classify")    # in the order an upload calls them
LOG_FIELDS = ("latency_ms", "decode_ms", "crop_detect_ms", "inference_ms")
STAGE_TOL = 0.10                      # plan §4: the stages add up to the whole path within 10%
MODEL_SHARE_STOP = 0.5                # plan §4: stop for the owner if the models are under half of a photo's time
DINO_STAGE = "dino_b"                 # + the batch size: one stage per DINOv3 batch shape
MODEL_STAGES = {"encoder": ("encoder",), "decoder": ("decoder",), "dinov3": (DINO_STAGE,)}   # share -> prefixes


def _name(endpoint: str) -> str:
    return endpoint.strip("/")


# ------------------------------------------------------------------------------------------------ pure helpers
def quantiles(values: Iterable[float]) -> dict:
    """n, min, median, p90 (linear interpolation between order statistics) and max; None when empty."""
    v = sorted(values)
    if not v:
        return {"n": 0, "min": None, "median": None, "p90": None, "max": None}
    pos = 0.9 * (len(v) - 1)
    lo = int(pos)
    p90 = v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (pos - lo)
    return {"n": len(v), "min": v[0], "median": statistics.median(v), "p90": round(p90, 3), "max": v[-1]}


def read_log(paths: Iterable[Path], bad: list[str]) -> Iterator[dict]:
    """The JSON records of request-log files, gzipped or not; a line that is not a JSON object goes to `bad`."""
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad.append(f"{path.name}: {line[:80]!r}")
                    continue
                if isinstance(rec, dict):
                    yield rec
                else:
                    bad.append(f"{path.name}: {line[:80]!r}")


def _is_model_request(r: dict) -> bool:
    return r.get("msg") == "request" and r.get("endpoint") in ENDPOINTS and r.get("status") == 200


def models_seen(records: Iterable[dict]) -> dict[str, int]:
    """How many successful model requests each served model answered."""
    return dict(Counter(r.get("model") for r in records if _is_model_request(r)))


def summarise_requests(records: Iterable[dict], model: str | None = None) -> dict:
    """Per model endpoint, the successful requests' count and the quantiles of each timing field it logs; only
    `model`'s requests when it is given."""
    rows: dict[str, list[dict]] = {e: [] for e in ENDPOINTS}
    for r in records:
        if _is_model_request(r) and (model is None or r.get("model") == model):
            rows[r["endpoint"]].append(r)
    return {e: {"n": len(rs), **{f: quantiles(r[f] for r in rs if r.get(f) is not None) for f in LOG_FIELDS}}
            for e, rs in rows.items() if rs}


class StageClock:
    """Milliseconds per named stage for the request in flight. `wrap` times a function without changing what it
    does; `add` takes a time measured elsewhere (the segmenter's own timing_ms); `take` returns and resets."""

    def __init__(self, now: Callable[[], float] = time.perf_counter):
        self._now = now
        self._ms: dict[str, float] = {}

    def add(self, stage: str, ms: float) -> None:
        self._ms[stage] = self._ms.get(stage, 0.0) + ms

    def wrap(self, fn: Callable, stage: str) -> Callable:
        def timed(*args, **kwargs):
            t0 = self._now()
            try:
                return fn(*args, **kwargs)
            finally:
                self.add(stage, (self._now() - t0) * 1e3)
        timed.__wrapped__ = fn
        return timed

    def take(self) -> dict[str, float]:
        ms, self._ms = self._ms, {}
        return ms


def stage_check(whole_ms: float, stages: dict[str, float]) -> dict:
    """How far the timed stages fall short of (or past) the whole path; ok within STAGE_TOL of the whole."""
    total = sum(stages.values())
    frac = (whole_ms - total) / whole_ms if whole_ms else 0.0
    return {"sum_ms": round(total, 3), "unexplained_ms": round(whole_ms - total, 3),
            "unexplained_frac": round(frac, 4), "ok": abs(frac) <= STAGE_TOL}


def _stage_ms(stages: dict[str, float], prefixes: tuple[str, ...]) -> float:
    return sum(v for k, v in stages.items() if k.startswith(prefixes))


def model_share(photos: list[dict]) -> dict:
    """Per photo (one request to each endpoint, as an upload makes them), the time in each model stage (the XL0
    encoder and decoder, summed over both segmenter passes, and DINOv3) against the photo's whole time. The
    median share of each, and of the three together; stop if that is under MODEL_SHARE_STOP."""
    rows = []
    for p in photos:
        whole = sum(p[_name(e)]["whole_ms"] for e in ENDPOINTS)
        ms = {s: sum(_stage_ms(p[_name(e)]["stages"], pre) for e in ENDPOINTS) for s, pre in MODEL_STAGES.items()}
        rows.append({"photo": p.get("photo"), "whole_ms": round(whole, 3),
                     **{f"{s}_ms": round(v, 3) for s, v in ms.items()},
                     "model_ms": round(sum(ms.values()), 3), "share": sum(ms.values()) / whole,
                     "shares": {s: v / whole for s, v in ms.items()}})
    med = statistics.median(r["share"] for r in rows)
    return {"photos": rows, "median_share": med,
            "stage_shares": {s: statistics.median(r["shares"][s] for r in rows) for s in MODEL_STAGES},
            "stop": med < MODEL_SHARE_STOP}


def read_fixtures(list_file: Path, photo_dir: Path) -> list[Path]:
    """Every photo of a fixtures.txt (`<role> <sha256> <repo path>` per line), found by basename in photo_dir,
    as the host stores them, and checked against its sha256."""
    photos = []
    for line in list_file.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _, sha, path = line.split()
        photo = photo_dir / Path(path).name
        if hashlib.sha256(photo.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"baseline_stages: {photo} does not match its sha256 in {list_file}")
        photos.append(photo)
    return photos


def compare_live(summary: dict, live: dict) -> dict:
    """Each endpoint's fixture median against the live record: inside its range (min..max), and its ratio to
    the live median."""
    out = {}
    for e in ENDPOINTS:
        lv = (live.get("endpoints") or {}).get(e, {}).get("latency_ms")
        if not lv or not lv["n"]:
            out[_name(e)] = {"live": None, "within_range": None}
            continue
        med = summary[_name(e)]["whole_ms"]["median"]
        out[_name(e)] = {"fixtures_median_ms": med, "live": lv, "within_range": lv["min"] <= med <= lv["max"],
                         "ratio_to_live_median": round(med / lv["median"], 3)}
    return out


# ---------------------------------------------------------------------------------------------- host sampling
def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _mem_available_mb() -> float | None:
    for line in (_read(Path("/proc/meminfo")) or "").splitlines():
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 1024)
    return None


def _peak_rss_mb() -> float:
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)   # KiB on Linux


class Sample(NamedTuple):
    t: float
    cpu_mhz: float | None          # mean over cores
    temp_c: float | None           # package
    mem_available_mb: float | None


class Sampler(threading.Thread):
    """CPU clock, package temperature and MemAvailable every `period` s (plan §4). The files are found once;
    a sample is a few small sysfs reads."""

    def __init__(self, period: float = 0.1):
        super().__init__(daemon=True)
        self.period, self.samples = period, []
        self._freqs = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
        self._temp = next((z / "temp" for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
                           if _read(z / "type") == "x86_pkg_temp"), None)
        self._halt = threading.Event()    # not `_stop`: threading.Thread has a method of that name

    def _sample(self) -> Sample:
        freqs = [int(v) / 1e3 for f in self._freqs if (v := _read(f)) is not None]
        t = _read(self._temp) if self._temp else None
        return Sample(time.perf_counter(), round(statistics.mean(freqs)) if freqs else None,
                      int(t) / 1e3 if t else None, _mem_available_mb())

    def run(self) -> None:
        while not self._halt.is_set():
            self.samples.append(self._sample())
            self._halt.wait(self.period)

    def stop(self) -> None:
        self._halt.set()
        self.join()

    def between(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if t0 <= x.t <= t1]
        mhz = [x.cpu_mhz for x in s if x.cpu_mhz is not None]
        temp = [x.temp_c for x in s if x.temp_c is not None]
        mem = [x.mem_available_mb for x in s if x.mem_available_mb is not None]
        return {"samples": len(s), "cpu_mhz_median": statistics.median(mhz) if mhz else None,
                "cpu_mhz_min": min(mhz) if mhz else None, "temp_c_max": max(temp) if temp else None,
                "mem_available_mb_min": min(mem) if mem else None}


# ---------------------------------------------------------------------------------------------------- stages
def _import_release(clock: StageClock, threads: int | None):
    """webapp.app from the cwd (a release directory), its boot timed by stage from before torch's import, as a
    worker pays it. The loaders are wrapped in coffeecv.infer before the app imports them, so the app binds the
    timed ones; they are put back after, since a request's cached segmenter lookup is not a load."""
    release = Path.cwd().resolve()
    sys.path.insert(0, str(release))
    t0 = time.perf_counter()
    import torch
    if threads is not None:
        torch.set_num_threads(threads)
    import coffeecv.infer as infer
    clock.add("imports", (time.perf_counter() - t0) * 1e3)
    loaders = {"load_model": "load_dino", "_segmenter": "load_segmenter", "load_ood_reference": "load_guard",
               "load_ood_probe": "load_guard"}
    originals = {name: getattr(infer, name) for name in loaders}
    for name, stage in loaders.items():
        setattr(infer, name, clock.wrap(originals[name], stage))
    import webapp.app as app
    boot_ms = (time.perf_counter() - t0) * 1e3
    for name, fn in originals.items():
        setattr(infer, name, fn)
    for mod in (infer, app):
        if not Path(mod.__file__).resolve().is_relative_to(release):
            raise SystemExit(f"baseline_stages: {mod.__name__} came from {mod.__file__}, not from {release}")
    return app, infer, boot_ms


def _instrument(app, infer, clock: StageClock) -> None:
    """Time each request's stages with the release's own functions, wrapped where its code looks them up."""
    from coffeecv import segment_beans

    app.load_rgb_image = clock.wrap(app.load_rgb_image, "decode")            # /crop
    app._mask_png_b64 = clock.wrap(app._mask_png_b64, "mask_png")
    infer.load_rgb_image = clock.wrap(infer.load_rgb_image, "decode")        # /classify
    infer.grayscale_like_training = clock.wrap(infer.grayscale_like_training, "pitch")
    infer.estimate_bean_pitch = clock.wrap(infer.estimate_bean_pitch, "pitch")
    infer.sample_masked_bean_unit_boxes = clock.wrap(infer.sample_masked_bean_unit_boxes, "sampling")
    infer.ood_scores = clock.wrap(infer.ood_scores, "guard")
    infer.probe_score = clock.wrap(infer.probe_score, "guard")

    build_eval_transform = infer.build_eval_transform
    infer.build_eval_transform = lambda *a, **k: clock.wrap(build_eval_transform(*a, **k), "transform")

    segment_and_crop = segment_beans.segment_and_crop

    def timed_segment(rgb, seg):          # the segmenter's own stages: resize, encoder, decoder, upsample, fill_crop
        crop = segment_and_crop(rgb, seg)
        for k, v in crop.info["timing_ms"].items():
            clock.add(k, v)
        return crop
    segment_beans.segment_and_crop = timed_segment

    starts: list[float] = []
    app.model.register_forward_pre_hook(lambda m, inp: starts.append(time.perf_counter()))

    def after(m, inp, out):
        clock.add(f"{DINO_STAGE}{inp[0].shape[0]}", (time.perf_counter() - starts.pop()) * 1e3)
    app.model.register_forward_hook(after)


class _Bench(NamedTuple):
    client: object
    clock: StageClock
    sampler: Sampler
    log_file: Path


def _post(bench: _Bench, endpoint: str, photo: Path) -> dict:
    import io
    data = photo.read_bytes()
    bench.clock.take()
    t0 = time.perf_counter()
    r = bench.client.post(endpoint, data={"photo": (io.BytesIO(data), photo.name)},
                          content_type="multipart/form-data")
    t1 = time.perf_counter()
    if r.status_code != 200:
        raise SystemExit(f"baseline_stages: {endpoint} answered {r.status_code} for {photo.name}: {r.data[:200]!r}")
    whole = (t1 - t0) * 1e3
    stages = {k: round(v, 3) for k, v in bench.clock.take().items()}
    logged = json.loads(bench.log_file.read_text().splitlines()[-1])
    if (logged.get("msg"), logged.get("endpoint"), logged.get("status")) != ("request", endpoint, 200):
        raise SystemExit(f"baseline_stages: the log's last line is not this {endpoint} request: {logged}")
    return {"whole_ms": round(whole, 3), "stages": stages, "check": stage_check(whole, stages),
            "log": {f: logged.get(f) for f in LOG_FIELDS}, "host": bench.sampler.between(t0, t1),
            "verdict": (r.get_json() or {}).get("verdict")}


def _summary(rows: list[dict]) -> dict:
    out = {}
    for e in map(_name, ENDPOINTS):
        stages = sorted({k for r in rows for k in r[e]["stages"]})
        out[e] = {"whole_ms": quantiles(r[e]["whole_ms"] for r in rows),
                  "stages_ms": {k: quantiles(r[e]["stages"].get(k, 0.0) for r in rows) for k in stages},
                  "unexplained_ms": quantiles(r[e]["check"]["unexplained_ms"] for r in rows),
                  "checks_failed": sum(not r[e]["check"]["ok"] for r in rows)}
    out["photo_ms"] = quantiles(sum(r[_name(e)]["whole_ms"] for e in ENDPOINTS) for r in rows)
    out["model_share"] = model_share(rows)
    return out


def _release_checkpoint() -> None:
    """COFFEE_CV_CHECKPOINT from the release's release.env, as the service unit sets it, unless already set."""
    env = Path("release.env")
    if "COFFEE_CV_CHECKPOINT" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k == "COFFEE_CV_CHECKPOINT":
                os.environ[k] = v


def run_stages(photos: list[Path], passes: int, threads: int | None, live: dict | None) -> dict:
    _release_checkpoint()
    log_dir = Path(tempfile.mkdtemp(prefix="ml7_p0_log_"))
    os.environ["COFFEE_CV_LOG_DIR"] = str(log_dir)       # never the service's log
    sampler = Sampler()
    sampler.start()
    try:
        return _run_stages(photos, passes, threads, live, sampler, log_dir)
    finally:
        sampler.stop()
        shutil.rmtree(log_dir, ignore_errors=True)


def _run_stages(photos: list[Path], passes: int, threads: int | None, live: dict | None, sampler: Sampler,
                log_dir: Path) -> dict:
    mem_before = _mem_available_mb()
    clock = StageClock()
    t_boot = time.perf_counter()
    app, infer, boot_ms = _import_release(clock, threads)
    import torch
    boot_stages = {k: round(v, 3) for k, v in clock.take().items()}
    boot = {"whole_ms": round(boot_ms, 3), "stages": boot_stages, "check": stage_check(boot_ms, boot_stages),
            "peak_rss_mb": _peak_rss_mb(), "mem_available_mb_before": mem_before,
            "mem_available_mb_after": _mem_available_mb(), "host": sampler.between(t_boot, time.perf_counter())}
    seg = app.segmenter
    print(f"model {app.CHECKPOINT.name}; segmenter "
          + (f"{seg.p.weights} {seg.weights_sha256[:12]}, decoder {(seg.decoder_sha256 or 'pretrained')[:12]}"
             if seg is not None else "none") + f"; torch {torch.__version__}, {torch.get_num_threads()} threads",
          flush=True)
    print(f"boot {boot_ms:.0f} ms  " + "  ".join(f"{k} {v:.0f}" for k, v in boot_stages.items())
          + f"  peak RSS {boot['peak_rss_mb']} MB", flush=True)

    _instrument(app, infer, clock)
    bench = _Bench(app.app.test_client(), clock, sampler, log_dir / "app.jsonl")
    rows = []
    for k in range(1, passes + 1):
        for photo in photos:
            row = {"photo": photo.name, "pass": k}
            for e in ENDPOINTS:
                row[_name(e)] = _post(bench, e, photo)
            rows.append(row)
            print(f"pass {k} {photo.name}  " + "  |  ".join(
                f"{e} {row[_name(e)]['whole_ms']:.0f} ms: "
                + " ".join(f"{s} {v:.0f}" for s, v in row[_name(e)]["stages"].items())
                + f" (unexplained {row[_name(e)]['check']['unexplained_frac']:+.1%})" for e in ENDPOINTS),
                flush=True)

    warm = [r for r in rows if r["pass"] > 1] or rows
    summary = _summary(warm)
    report = {
        "ticket": "ML-7 P0", "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "release": Path.cwd().resolve().name, "build": app.BUILD, "checkpoint": app.CHECKPOINT.name,
        "segmenter": None if seg is None else {"weights": str(seg.p.weights), "weights_sha256": seg.weights_sha256,
                                                "decoder_sha256": seg.decoder_sha256},
        "python": sys.version.split()[0], "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "cpu": next((ln.split(":", 1)[1].strip() for ln in (_read(Path("/proc/cpuinfo")) or "").splitlines()
                     if ln.startswith("model name")), None),
        "passes": passes, "photos": len(photos), "summary_over": "passes 2+" if warm is not rows else "all passes",
        "boot": boot, "requests": rows, "summary": summary, "peak_rss_mb": _peak_rss_mb(),
        "host": sampler.between(0.0, float("inf")),
    }
    if live is not None:
        report["live_model"] = live.get("model")
        report["live_comparison"] = compare_live(summary, live)
    _print_checks(report)
    return report


def _print_checks(report: dict) -> None:
    s = report["summary"]
    print("\nP0 checks")
    for e in map(_name, ENDPOINTS):
        print(f"  /{e}: median {s[e]['whole_ms']['median']:.0f} ms; stages within {STAGE_TOL:.0%} of the whole "
              f"path on {s[e]['whole_ms']['n'] - s[e]['checks_failed']}/{s[e]['whole_ms']['n']} requests")
    if report.get("live_model") not in (None, report["checkpoint"]):
        print(f"  WARNING: the live record is {report['live_model']}'s, not {report['checkpoint']}'s")
    for e, c in (report.get("live_comparison") or {}).items():
        if c["live"]:
            print(f"  /{e} vs live: fixtures {c['fixtures_median_ms']:.0f} ms, live median {c['live']['median']}"
                  f" p90 {c['live']['p90']} range {c['live']['min']}-{c['live']['max']} "
                  f"({'within' if c['within_range'] else 'OUTSIDE'} the range; "
                  f"{c['ratio_to_live_median']:.2f}x the live median)")
        else:
            print(f"  /{e} vs live: no live requests")
    m = s["model_share"]
    print(f"  models: {m['median_share']:.1%} of each photo's time (median)"
          + ("  -> STOP for the owner (under half)" if m["stop"] else "") + "; "
          + ", ".join(f"{k} {v:.1%}" for k, v in m["stage_shares"].items())
          + " (encoder and decoder over both passes; DINOv3 with its head)")


# ----------------------------------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log", help="the live request log's aggregates")
    lg.add_argument("files", nargs="+", type=Path)
    lg.add_argument("--model", default=None, help="only this checkpoint's requests (the log's `model` field)")
    lg.add_argument("--out", type=Path, required=True)
    st = sub.add_parser("stages", help="the release's code per stage (run from inside a release directory)")
    st.add_argument("--list", type=Path, required=True, help="fixtures.txt")
    st.add_argument("--dir", type=Path, required=True, help="the fixture photos, flat, by basename")
    st.add_argument("--passes", type=int, default=2)
    st.add_argument("--limit", type=int, default=None, help="first N photos only (a toy run)")
    st.add_argument("--threads", type=int, default=None,
                    help="torch.set_num_threads; omitted, the release's own default, as the service runs")
    st.add_argument("--live", type=Path, default=None, help="the `log` command's output, to compare against")
    st.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "log":
        bad: list[str] = []
        records = list(read_log(sorted(args.files), bad))
        mine = [r for r in records if args.model is None or r.get("model") == args.model]
        stamps = sorted(r["ts"] for r in mine if _is_model_request(r) and "ts" in r)
        report = {"ticket": "ML-7 P0", "files": [f.name for f in sorted(args.files)], "bad_lines": len(bad),
                  "model": args.model, "models_seen": models_seen(records),
                  "first": stamps[0] if stamps else None, "last": stamps[-1] if stamps else None,
                  "endpoints": summarise_requests(records, args.model)}
        print(json.dumps(report, indent=2))
    else:
        photos = read_fixtures(args.list, args.dir)[:args.limit]
        live = json.loads(args.live.read_text()) if args.live else None
        report = run_stages(photos, args.passes, args.threads, live)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
