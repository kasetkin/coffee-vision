"""ML-14 P4: concurrent.futures.wait never counts a future that ProcessPoolExecutor.shutdown(cancel_futures=True)
cancelled as done (CPython 3.12): its state is CANCELLED, never CANCELLED_AND_NOTIFIED. Prints the futures'
states, then wait()'s count (3 not done, after a 3 s timeout), then that Future.exception() returns or raises on
every one. Why run_pool cancels its futures itself and waits on each.

    python analysis/ml14/wait_cancel.py
"""
import time
from concurrent.futures import CancelledError, ProcessPoolExecutor, wait

if __name__ == "__main__":
    pool = ProcessPoolExecutor(1)
    fs = [pool.submit(time.sleep, 0.5) for _ in range(6)]
    time.sleep(0.2)
    pool.shutdown(wait=False, cancel_futures=True)
    time.sleep(1.5)
    done, not_done = wait(fs, timeout=3)
    print("states:", [f._state for f in fs])
    print("wait(): done", len(done), "not done", len(not_done))
    ends = []
    for f in fs:
        try:
            ends.append(f.exception() is None)
        except CancelledError:
            ends.append("cancelled")
    print("Future.exception() on each:", ends)
