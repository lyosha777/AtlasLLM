import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import dev


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.python = self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.vite = self.root / "frontend/node_modules/vite/bin/vite.js"
        self.npm = self.root / ("npm.cmd" if os.name == "nt" else "npm")
        for path in (self.python, self.vite, self.root / "node_modules/npm/bin/npm-cli.js"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.enterContext(patch.object(dev, "ROOT", self.root))
        self.enterContext(patch.object(dev, "log"))
        self.enterContext(patch.dict(os.environ, {"OLLAMA_HOST": "127.0.0.1:11434"}))
        self.enterContext(patch.object(dev.shutil, "which", side_effect=lambda name: str(self.root / "node") if name == "node" else str(self.npm)))
        self.runner = self.enterContext(patch.object(dev, "run_check", side_effect=lambda command, **kwargs:
            json.dumps({"version": [3, 13, 2], "models": ["atlas-model"]}) if "-c" in command else "version"))
        self.url = self.enterContext(patch.object(dev, "get_url", return_value=b'{"models":[{"name":"atlas-model:latest"}]}'))
        self.ports = self.enterContext(patch.object(dev, "check_port"))

    def test_ready_checks_all_dependencies_and_ports_without_installing(self):
        python, _, vite = dev.check_environment()
        self.assertEqual(python, self.python)
        self.assertEqual(vite, self.vite)
        self.url.assert_called_once_with("http://127.0.0.1:11434/api/tags")
        self.assertEqual([call.args[0] for call in self.ports.call_args_list], [8000, 5173])
        commands = [call.args[0] for call in self.runner.call_args_list]
        self.assertTrue(any("ls" in command for command in commands))
        self.assertFalse(any("install" in command or "pull" in command for command in commands))

    def test_missing_venv_is_actionable(self):
        self.python.unlink()
        with self.assertRaisesRegex(dev.LaunchError, "python -m venv"):
            dev.check_environment()

    def test_broken_backend_environment_is_actionable(self):
        self.runner.side_effect = dev.LaunchError("Missing backend modules: uvicorn")
        with self.assertRaisesRegex(dev.LaunchError, "pip install -r requirements.txt"):
            dev.check_environment()

    def test_missing_frontend_dependencies_is_actionable(self):
        self.vite.unlink()
        with self.assertRaisesRegex(dev.LaunchError, "npm.cmd ci"):
            dev.check_environment()

    def test_ollama_unavailable_is_actionable(self):
        self.url.side_effect = OSError("Connection refused")
        with self.assertRaisesRegex(dev.LaunchError, "ollama serve"):
            dev.check_environment()

    def test_missing_model_is_actionable(self):
        self.url.return_value = b'{"models":[]}'
        with self.assertRaisesRegex(dev.LaunchError, "ollama pull atlas-model"):
            dev.check_environment()


class LifecycleTests(unittest.TestCase):
    def tearDown(self):
        dev.STOP.clear()

    def test_occupied_port_is_reported(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(dev.LaunchError, "Port .* unavailable"):
                dev.check_port(listener.getsockname()[1])

    def test_check_does_not_start_services_or_browser(self):
        with patch.object(dev, "check_environment", return_value=("python", "node", "vite")), patch.object(dev, "Services") as services, patch.object(dev.webbrowser, "open") as browser, patch.object(dev.signal, "signal"), patch.object(dev, "log"):
            self.assertEqual(dev.main(["--check"]), 0)
            services.assert_not_called()
            browser.assert_not_called()

    def test_early_process_exit_is_detected_and_cleanup_always_runs(self):
        services = Mock()
        with patch.object(dev, "check_environment", return_value=("python", "node", "vite")), patch.object(dev, "Services", return_value=services), patch.object(dev, "wait_ready", side_effect=dev.LaunchError("Backend exited")), patch.object(dev.signal, "signal"), patch.object(dev, "log"):
            self.assertEqual(dev.main([]), 1)
            self.assertEqual(services.start.call_count, 2)
            services.close.assert_called_once()

    def test_open_runs_only_after_both_services_are_ready(self):
        services = Mock()
        events = []
        stop = Mock()
        stop.is_set.return_value = False
        stop.wait.return_value = True
        with patch.object(dev, "check_environment", return_value=("python", "node", "vite")), patch.object(dev, "Services", return_value=services), patch.object(dev, "STOP", stop), patch.object(dev, "wait_ready", side_effect=lambda _: events.append("ready")), patch.object(dev.webbrowser, "open", side_effect=lambda _: events.append("browser") or True), patch.object(dev.signal, "signal"), patch.object(dev, "log"):
            self.assertEqual(dev.main(["--open"]), 0)
            self.assertEqual(events, ["ready", "browser"])
            services.close.assert_called_once()

    def test_unexpected_child_exit_and_cleanup_with_real_process(self):
        services = dev.Services()
        try:
            services.start("failing", [sys.executable, "-c", "raise SystemExit(7)"], dev.ROOT, sys.executable)
            deadline = time.monotonic() + 5
            while services.processes[0][1].poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            with self.assertRaisesRegex(dev.LaunchError, "exited with code 7"):
                services.check_alive()
        finally:
            services.close()
        self.assertIsNotNone(services.processes[0][1].poll())


if __name__ == "__main__":
    unittest.main()
