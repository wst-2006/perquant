from __future__ import annotations

import socket
import subprocess
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.config import settings  # noqa: E402


def assert_port_available() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((settings.flask.host, settings.flask.port))
        except OSError as exc:
            raise SystemExit(
                f"Canonical port {settings.flask.host}:{settings.flask.port} is already occupied. "
                "Stop the existing site before starting another instance."
            ) from exc


def main() -> None:
    assert_port_available()
    worker = subprocess.Popen([sys.executable, "scripts/run_web_worker.py"], cwd=PROJECT_ROOT)
    web = subprocess.Popen([sys.executable, "scripts/run_web.py"], cwd=PROJECT_ROOT)
    print(f"Site: http://{settings.flask.host}:{settings.flask.port}")
    print(f"Web PID={web.pid}; Worker PID={worker.pid}")
    try:
        while True:
            if web.poll() is not None:
                raise SystemExit(f"Web process exited with code {web.returncode}")
            if worker.poll() is not None:
                raise SystemExit(f"Worker process exited with code {worker.returncode}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (web, worker):
            if process.poll() is None:
                process.terminate()
        for process in (web, worker):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
