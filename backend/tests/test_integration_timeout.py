import subprocess
import sys
import unittest

from backend.tests.integration_timeout import run_with_deadline


class IntegrationTimeoutTests(unittest.TestCase):
    def test_success_returns_without_a_live_watchdog(self):
        self.assertEqual(run_with_deadline(lambda: "complete", 1), "complete")

    def test_hard_deadline_terminates_a_stalled_test(self):
        result = subprocess.run([sys.executable, "-c",
            "import time; from backend.tests.integration_timeout import run_with_deadline; "
            "run_with_deadline(lambda: time.sleep(30), 0.1)"], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 124)
        self.assertIn("Integration test exceeded", result.stderr)
