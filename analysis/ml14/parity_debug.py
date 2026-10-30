"""ML-14 P4: why the first parity test saw different hashes at --jobs 1 and --jobs 3. Prints each row field and
file that differs. It found the inputs differing, not the runs: write_fixture's ICC profile carries its creation
time, so two fixture sets written a second apart differ; the test now copies one set. Throwaway.

    python analysis/ml14/parity_debug.py
"""
import difflib
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])      # the repo
sys.path.insert(0, ROOT)
from tests.test_strip_metadata import TestPool  # noqa: E402


def main() -> None:
    t = TestPool()
    t.setUp()
    for mode in ([], ["--dry-run"]):
        s = t.run_on(t.mixed_set(t.tmp / f"serial{len(mode)}"), "--jobs", "1", *mode)
        p = t.run_on(t.mixed_set(t.tmp / f"parallel{len(mode)}"), "--jobs", "3", *mode)
        print("mode", mode or "strip", "exit codes", s[0], p[0], "output equal", s[1] == p[1])
        for a, b in zip(s[2], p[2]):
            for k in a:
                if a[k] != b[k]:
                    print("  row", a["path"], k, a[k][:20], b[k][:20])
        for k in s[3]:
            if s[3][k] != p[3][k]:
                print("  file", k)
        if s[1] != p[1]:
            print("\n".join(difflib.unified_diff(s[1].splitlines(), p[1].splitlines(), lineterm="")))
    t.doCleanups()


if __name__ == "__main__":                            # spawn re-imports this file in each worker
    main()
