"""Put on PYTHONPATH by sigint_repro.py: every process of the run (main and spawned workers) dumps all its threads'
stacks on SIGUSR1. How the ML-14 P4 hang was found (py-spy was not available)."""
import faulthandler
import signal
import sys

faulthandler.register(signal.SIGUSR1, all_threads=True, file=sys.stderr)
