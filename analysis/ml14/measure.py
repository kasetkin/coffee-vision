"""ML-14 P5 (D11): wall time and the peak summed RSS of a command's process tree (the run, its workers and their
exiftool processes), sampled every 0.2 s. Shared library pages count once per process, so the peak is a little
high. Prints one tab-separated line: the label, the times, the memory available at the start, the exit code and the
run's summary lines.

    python analysis/ml14/measure.py LABEL -- CMD...

run_all.sh ran the six runs of the ticket's Activity table (2026-10-30) with it; results.txt is their output.
"""
import os
import subprocess
import sys
import time


def tree(root: int) -> list[int]:
    """`root` and every process below it."""
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(d))
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack += kids.get(p, [])
    return out


def rss_kb(pid: int) -> int:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    label, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
    avail = next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable:")) / 1e6
    t = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak = peak_n = 0
    while proc.poll() is None:
        pids = tree(proc.pid)
        s = sum(rss_kb(p) for p in pids)
        if s > peak:
            peak, peak_n = s, len(pids)
        time.sleep(0.2)
    wall = time.time() - t
    out = proc.stdout.read()
    summary = " / ".join(line for line in out.splitlines() if "photos checked" in line or "at once" in line)
    print(f"{label}\twall {wall:.1f} s\tpeak {peak / 1e6:.2f} GB ({peak_n} processes)\tavailable at start "
          f"{avail:.1f} GB\trc {proc.returncode}\t{summary or out[-300:]}", flush=True)
