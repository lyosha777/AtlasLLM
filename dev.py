"""Run Atlas's separate backend and frontend for local development."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser


ROOT = Path(__file__).resolve().parent
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
ATLAS_URL = "http://127.0.0.1:5173"
BACKEND_URL = "http://127.0.0.1:8000/api/health"
STOP = threading.Event()
PRINT_LOCK = threading.Lock()


class LaunchError(Exception):
    pass


def log(message, prefix="atlas"):
    with PRINT_LOCK:
        print(f"[{prefix}] {message}", flush=True)


def run_check(command, cwd=ROOT, timeout=90):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LaunchError(f"Could not run {command[0]}: {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise LaunchError(detail or f"Command exited with code {result.returncode}.")
    return result.stdout.strip()


def get_url(url, timeout=3):
    # Local services should not depend on system HTTP proxy configuration.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.read()


def check_port(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            if os.name == "nt":
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(("127.0.0.1", port))
    except OSError as error:
        raise LaunchError(f"Port {port} is unavailable. Stop the service using 127.0.0.1:{port} and retry.") from error


def check_environment():
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise LaunchError("Project .venv is missing. Run: python -m venv .venv, then install requirements.txt with its Python.")
    log("Checking project Python and backend imports...")
    probe = """
import importlib.util, json, sys
required = ['uvicorn', 'fastapi', 'ollama', 'chromadb', 'sentence_transformers',
            'pypdf', 'python_multipart', 'ddgs', 'trafilatura']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    sys.exit('Missing backend modules: ' + ', '.join(missing))
import backend.api
from backend.core.config import CHAT_MODEL, PLANNER_MODEL, GENERATION_MODEL
print(json.dumps({'version': list(sys.version_info[:3]),
                  'models': list(dict.fromkeys([CHAT_MODEL, PLANNER_MODEL, GENERATION_MODEL]))}))
"""
    try:
        backend = json.loads(run_check([str(python), "-c", probe]).splitlines()[-1])
    except (LaunchError, ValueError, IndexError) as error:
        raise LaunchError(f"Backend environment is not ready: {error}\nRun: {python} -m pip install -r requirements.txt") from error
    log(f"Python {'.'.join(map(str, backend['version']))}: project .venv; backend imports OK")

    node = shutil.which("node")
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not node or not npm:
        raise LaunchError("Node.js/npm are missing from PATH. Install a Node.js version supported by frontend/package.json and reopen PowerShell.")
    npm_command = [npm]
    if os.name == "nt":
        npm_cli = Path(npm).parent / "node_modules/npm/bin/npm-cli.js"
        if not npm_cli.is_file():
            raise LaunchError(f"Cannot find npm's CLI beside {npm}. Repair the Node.js/npm installation.")
        npm_command = [node, str(npm_cli)]
    log(f"Node {run_check([node, '--version'])}; npm {run_check([*npm_command, '--version'])}")
    frontend = ROOT / "frontend"
    vite = frontend / "node_modules/vite/bin/vite.js"
    if not vite.is_file():
        raise LaunchError("Frontend dependencies are missing. Run: cd frontend; npm.cmd ci")
    try:
        engine_probe = """
const {createRequire} = require('node:module');
const fs = require('node:fs');
const semver = createRequire(process.argv[1])('semver');
const engine = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')).engines.node;
if (!semver.satisfies(process.versions.node, engine)) {
    console.error(`Installed Vite requires Node ${engine}; found ${process.versions.node}. Update Node.js.`);
    process.exit(1);
}
"""
        npm_entry = Path(npm_command[-1]).resolve()
        run_check([node, "-e", engine_probe, str(npm_entry), str(vite.parents[1] / "package.json")], cwd=frontend)
        run_check([*npm_command, "ls", "--depth=0"], cwd=frontend, timeout=30)
        run_check([node, str(vite), "--version"], cwd=frontend, timeout=15)
    except LaunchError as error:
        raise LaunchError(f"Frontend environment is not ready: {error}\nRun: cd frontend; npm.cmd ci") from error
    log("Frontend dependencies OK")

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if "://" not in host:
        host = "http://" + host
    try:
        models = json.loads(get_url(host + "/api/tags"))["models"]
    except (OSError, ValueError, KeyError) as error:
        raise LaunchError(f"Ollama is unavailable at {host}. Start the Ollama app or run 'ollama serve', then retry. Details: {error}") from error
    available = {model.get("name", model.get("model", "")) for model in models}
    missing = [model for model in backend["models"] if (model if ":" in model else model + ":latest") not in available]
    if missing:
        commands = "; ".join(f"ollama pull {model}" for model in missing)
        raise LaunchError(f"Configured Ollama model(s) missing: {', '.join(missing)}. Run: {commands}")
    log("Ollama reachable; configured models available: " + ", ".join(backend["models"]))
    for port in (BACKEND_PORT, FRONTEND_PORT):
        check_port(port)
    log(f"Ports {BACKEND_PORT} and {FRONTEND_PORT} available")
    return python, node, vite


class WindowsJob:
    """Kill all owned descendants if the launcher exits, even unexpectedly."""

    def __init__(self):
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class Services:
    def __init__(self):
        self.processes = []
        self.readers = []
        self.job = WindowsJob() if os.name == "nt" else None

    def start(self, name, command, cwd, python):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"}
        options = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if self.job:
            # Gate child creation until the wrapper belongs to our Job Object.
            wrapper = ("import signal,subprocess,sys; "
                       "signal.signal(signal.SIGBREAK,signal.SIG_IGN); "
                       "sys.stdin.buffer.read(1); "
                       "sys.exit(subprocess.Popen(sys.argv[1:],stdin=subprocess.DEVNULL).wait())")
            command = [str(python), "-u", "-c", wrapper, *command]
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE if self.job else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", **options)
        self.processes.append((name, process))
        if self.job:
            try:
                self.job.assign(process)
                process.stdin.write("\n")
                process.stdin.close()
            except BaseException:
                process.kill()
                process.wait()
                raise
        reader = threading.Thread(target=self.read_logs, args=(name, process), daemon=True)
        self.readers.append(reader)
        reader.start()
        log(f"Starting {name} (PID {process.pid})")

    @staticmethod
    def read_logs(name, process):
        for line in process.stdout:
            log(line.rstrip(), name)

    def check_alive(self):
        for name, process in self.processes:
            code = process.poll()
            if code is not None:
                raise LaunchError(f"{name.capitalize()} exited with code {code}. See its prefixed logs above.")

    def close(self):
        log("Stopping backend and frontend...")
        for _, process in self.processes:
            try:
                if process.poll() is None:
                    if self.job:
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = time.monotonic() + 8
        while any(process.poll() is None for _, process in self.processes) and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.job:
            self.job.close()
        else:
            for _, process in self.processes:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for _, process in self.processes:
            process.wait(timeout=5)
        for reader in self.readers:
            reader.join(timeout=2)
        for _, process in self.processes:
            process.stdout.close()
        log("Atlas stopped.")


def wait_ready(services, timeout=120):
    pending = {"backend": BACKEND_URL, "frontend": ATLAS_URL}
    deadline = time.monotonic() + timeout
    while pending and not STOP.is_set():
        services.check_alive()
        for name, url in list(pending.items()):
            try:
                body = get_url(url, timeout=0.5)
                if name == "backend" and json.loads(body).get("status") != "ok":
                    continue
                if name == "frontend" and b"/@vite/client" not in body:
                    continue
                del pending[name]
                log(f"{name.capitalize()} ready")
            except (OSError, ValueError):
                pass
        if time.monotonic() >= deadline:
            raise LaunchError("Startup timed out waiting for " + ", ".join(pending) + ". See the service logs above.")
        STOP.wait(0.2)
    services.check_alive()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Check and launch Atlas's local development services.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--open", action="store_true", help="Open Atlas in your default browser after startup")
    mode.add_argument("--check", action="store_true", help="Check readiness without starting services")
    args = parser.parse_args(argv)
    STOP.clear()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: STOP.set())
    services = None
    try:
        python, node, vite = check_environment()
        if STOP.is_set():
            return 0
        if args.check:
            log("Ready. Run: python dev.py (or python dev.py --open)")
            return 0
        services = Services()
        services.start("backend", [str(python), "-m", "uvicorn", "backend.api:app", "--reload",
                                   "--reload-dir", str(ROOT / "backend"), "--host", "127.0.0.1", "--port", str(BACKEND_PORT)], ROOT, python)
        services.start("frontend", [node, str(vite), "--host", "127.0.0.1", "--port", str(FRONTEND_PORT), "--strictPort"], ROOT / "frontend", python)
        wait_ready(services)
        if not STOP.is_set():
            log(f"Atlas ready: {ATLAS_URL}  |  API docs: http://127.0.0.1:{BACKEND_PORT}/docs")
            log("Press Ctrl+C to stop both services.")
            if args.open and not webbrowser.open(ATLAS_URL):
                log("Could not open your browser; open the Atlas URL above manually.")
        while not STOP.wait(0.3):
            services.check_alive()
        return 0
    except (LaunchError, OSError) as error:
        if STOP.is_set():
            return 0
        log(str(error), "error")
        return 1
    finally:
        if services:
            services.close()


if __name__ == "__main__":
    sys.exit(main())
