"""ML-14 P4: the by-hand checks of a parallel strip stopped by a Ctrl-C or an error, each on a dozen copies of the test
suite's fixture (private tags included) in a temp folder. A Ctrl-C is SIGINT to the run's process group, as a
terminal sends it. Throwaway; how the P4 fixes were found and checked (2026-10-30).

    python analysis/ml14/ctrl_c_checks.py hang            # SIGINT after the first row; a hang dumps every stack
    python analysis/ml14/ctrl_c_checks.py twice           # two SIGINTs: stop at once, every row matches its photo
    python analysis/ml14/ctrl_c_checks.py early 0.6 1.0   # SIGINT N s after the start, while the workers spawn
    python analysis/ml14/ctrl_c_checks.py pipe            # stdout closed after the first row (`| head -1`)

`hang` found the concurrent.futures.wait hang (see wait_cancel.py), by faulthandler (sigint_debug/); `early` the
workers killed during their imports; `pipe` the photos stripped with no row after an error.
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from coffeecv.repo_files import sha256_file  # noqa: E402
from tests.test_strip_metadata import manifest_rows, write_fixture  # noqa: E402


def photos(n: int = 12) -> tuple[Path, str]:
    """A temp folder with `n` copies of the fixture under photos/, and the fixture's sha256."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "photos").mkdir()
    fixture = write_fixture(tmp / "fixture.jpg")
    for i in range(n):
        shutil.copyfile(fixture, tmp / "photos" / f"{i:02}.jpg")
    return tmp, sha256_file(fixture)


def start(tmp: Path, jobs: int, env: dict | None = None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "coffeecv.strip_metadata", str(tmp / "photos"), "--manifest", str(tmp / "m.csv"),
           "--jobs", str(jobs)]
    return subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True, env=env)


def verdict(tmp: Path, original: str) -> None:
    """Every stripped photo has its row, every row is its photo's bytes, and the temporary copies left."""
    folder = tmp / "photos"
    rows = {Path(r["path"]).name: r for r in manifest_rows(tmp / "m.csv")} if (tmp / "m.csv").exists() else {}
    changed = {p.name for p in folder.iterdir() if not p.name.startswith(".") and sha256_file(p) != original}
    print("rows", sorted(rows))
    print("stripped without a row:", sorted(changed - set(rows)))
    print("rows match their photos:", all(r["sha256_after"] == sha256_file(folder / n) for n, r in rows.items()))
    print("temporary copies:", sorted(p.name for p in folder.iterdir() if p.name.startswith(".")))


def hang() -> None:
    tmp, original = photos()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent / "sigint_debug")}
    proc = start(tmp, 2, env)
    print("first:", proc.stdout.readline().strip(), flush=True)
    os.killpg(proc.pid, signal.SIGINT)
    try:
        out, err = proc.communicate(timeout=20)
        print("exit", proc.returncode)
        print(out)
        print(err)
    except subprocess.TimeoutExpired:
        os.kill(proc.pid, signal.SIGUSR1)
        time.sleep(1)
        proc.kill()
        out, err = proc.communicate()
        print("HUNG; stdout:", out)
        print("stderr:", err[-6000:])
    verdict(tmp, original)
    shutil.rmtree(tmp)


def twice() -> None:
    tmp, original = photos()
    proc = start(tmp, 3)
    print("first:", proc.stdout.readline().strip(), flush=True)
    os.killpg(proc.pid, signal.SIGINT)
    time.sleep(0.05)
    os.killpg(proc.pid, signal.SIGINT)
    t = time.time()
    out, err = proc.communicate(timeout=30)
    print(f"exit {proc.returncode}, {time.time() - t:.2f} s after the second Ctrl-C")
    print(out[-600:])
    print(err)
    time.sleep(3)                                     # the terminated workers' exiftool processes finish
    verdict(tmp, original)
    shutil.rmtree(tmp)


def early(delays: list[float]) -> None:
    for delay in delays:
        tmp, original = photos()
        proc = start(tmp, 4)
        time.sleep(delay)
        os.killpg(proc.pid, signal.SIGINT)
        out, err = proc.communicate(timeout=60)
        print(f"delay {delay}: exit {proc.returncode}, FAIL rows {out.count('FAIL')}, "
              f"BrokenProcessPool {'BrokenProcessPool' in out + err}, traceback {'Traceback' in err}")
        shutil.rmtree(tmp)


def pipe() -> None:
    tmp, original = photos()
    cmd = (f"{sys.executable} -m coffeecv.strip_metadata {tmp / 'photos'} --manifest {tmp / 'm.csv'} --jobs 3 "
           f"| head -1")
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    time.sleep(3)
    print("stdout:", r.stdout.strip())
    print("stderr, last line:", r.stderr.strip().splitlines()[-1:])
    verdict(tmp, original)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "early":
        early([float(d) for d in sys.argv[2:]] or [0.6, 0.8, 1.0])
    elif what in ("hang", "twice", "pipe"):
        globals()[what]()
    else:
        sys.exit(__doc__)
