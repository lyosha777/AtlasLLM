"""Exercise real Windows launcher startup, Ctrl+C, and forced-exit cleanup."""

import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import psutil


ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def ctrl_c(pid):
    sender = """
import ctypes, sys, time
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.FreeConsole()
if not kernel.AttachConsole(int(sys.argv[1])):
    raise ctypes.WinError(ctypes.get_last_error())
kernel.SetConsoleCtrlHandler(None, True)
if not kernel.GenerateConsoleCtrlEvent(0, 0):
    raise ctypes.WinError(ctypes.get_last_error())
time.sleep(0.3)
kernel.FreeConsole()
"""
    subprocess.run([sys.executable, "-c", sender, str(pid)], check=True)


def exercise(force_exit):
    backend_port, frontend_port = free_port(), free_port()
    while backend_port == frontend_port:
        frontend_port = free_port()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        temporary = Path(directory)
        shutil.copytree(ROOT / "backend", temporary / "backend", ignore=shutil.ignore_patterns("__pycache__", "tests"))
        helper = temporary / "launch.py"
        helper.write_text(f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
import dev
dev.ROOT = Path({directory!r})
dev.BACKEND_PORT = {backend_port}
dev.FRONTEND_PORT = {frontend_port}
dev.BACKEND_URL = 'http://127.0.0.1:{backend_port}/api/health'
dev.ATLAS_URL = 'http://127.0.0.1:{frontend_port}'
dev.check_environment = lambda: (Path({sys.executable!r}), {shutil.which('node')!r}, Path({str(ROOT / 'frontend/node_modules/vite/bin/vite.js')!r}))
original_start = dev.Services.start
def start(self, name, command, cwd, python):
    if name == 'frontend':
        cwd = Path({str(ROOT / 'frontend')!r})
    original_start(self, name, command, cwd, python)
dev.Services.start = start
sys.exit(dev.main([]))
""", encoding="utf-8")
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        process = subprocess.Popen([sys.executable, "-u", str(helper)], cwd=temporary,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", startupinfo=startup,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
        lines = queue.Queue()

        def read():
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                lines.put(line)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        children = []
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise AssertionError(f"Launcher exited early: {process.returncode}")
                if time.monotonic() >= deadline:
                    raise AssertionError("Launcher readiness timed out")
                try:
                    if "Atlas ready:" in lines.get(timeout=0.2):
                        break
                except queue.Empty:
                    pass
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{backend_port}/api/health") as response:
                assert response.status == 200
            with urllib.request.urlopen(f"http://127.0.0.1:{frontend_port}/src/App.tsx") as response:
                assert response.status == 200
            children = psutil.Process(process.pid).children(recursive=True)
            assert len(children) >= 4, children
            if force_exit:
                process.kill()
            else:
                ctrl_c(process.pid)
            code = process.wait(timeout=20)
            if not force_exit:
                assert code == 0, code
            deadline = time.monotonic() + 5
            while any(child.is_running() for child in children) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not any(child.is_running() for child in children), "Launcher left an owned descendant running"
            for port in (backend_port, frontend_port):
                with socket.socket() as listener:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    listener.bind(("127.0.0.1", port))
            print(f"{'Forced exit' if force_exit else 'Ctrl+C'}: all descendants stopped; ports released", flush=True)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            reader.join(timeout=3)
            process.stdout.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if os.name != "nt":
        raise SystemExit("This smoke test checks Windows console/process cleanup.")
    exercise(force_exit=False)
    exercise(force_exit=True)
    print("Windows launcher smoke checks passed; existing Atlas servers and data were not changed.")
