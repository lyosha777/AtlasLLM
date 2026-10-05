"""Hard wall-clock deadlines for standalone live smoke processes."""
import os
import sys
import threading


def run_with_deadline(main, seconds):
    def expire():
        print(f"Integration test exceeded {seconds}s; terminating the test process (exit 124).", file=sys.stderr, flush=True)
        # HTTP timeouts do not stop a backend worker or a stream sending heartbeats.
        # These standalone smokes own only in-process server/worker threads.
        os._exit(124)

    timer = threading.Timer(seconds, expire)
    timer.daemon = True
    timer.start()
    try:
        return main()
    finally:
        timer.cancel()
