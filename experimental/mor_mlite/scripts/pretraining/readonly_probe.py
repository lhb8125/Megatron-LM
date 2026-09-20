"""Bounded transport retries for explicitly read-only campaign probes only."""

import json
import subprocess
import sys
import time


def run_readonly(command):
    """Never wrap submission, deletion, or a mixed read/write operation here."""
    for attempt, timeout in enumerate((60, 120, 180)):
        try:
            return subprocess.run(command, check=True, capture_output=True,
                                  text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            transient = isinstance(exc, subprocess.TimeoutExpired) or exc.returncode == 255
            if not transient or attempt == 2:
                raise
            print(json.dumps({"readonly_probe_retry": attempt + 2,
                              "error_type": type(exc).__name__}), file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))
