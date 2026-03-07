"""vLLM server lifecycle management.

``start(config, resolved_model_path)`` spawns the vLLM OpenAI-compatible
server subprocess and polls health endpoints until ready.
``stop()`` terminates the server via PID file.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Optional

import requests

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

_PID_FILENAME = ".vllm.pid"
_server_process: Optional[subprocess.Popen] = None


def _pid_file_path() -> str:
    from src.config import PROJECT_DIR

    return os.path.join(PROJECT_DIR, _PID_FILENAME)


def _port_in_use(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _probe_health(base_url: str, endpoint: str) -> bool:
    try:
        resp = requests.get(f"{base_url}{endpoint}", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def start(config: Config, resolved_model_path: str) -> None:
    """Spawn the vLLM server subprocess and wait until healthy."""
    global _server_process

    host = config.SERVER_HOST
    port = config.SERVER_PORT
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    if config.CUDA_VISIBLE_DEVICES:
        env["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

    if _port_in_use("localhost", port):
        if _probe_health(base_url, config.SERVER_HEALTH_ENDPOINT) or _probe_health(base_url, "/v1/models"):
            logger.warning(
                "Port %d already in use by a healthy server — reusing.", port,
            )
            return
        raise RuntimeError(
            f"Port {port} is in use but the health check failed. "
            "Kill the existing process or choose a different port."
        )

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", resolved_model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(config.SERVER_TP_SIZE),
        "--dtype", config.SERVER_DTYPE,
        "--max-model-len", str(config.SERVER_CONTEXT_LENGTH),
        "--gpu-memory-utilization", config.VLLM_GPU_MEMORY_UTILIZATION,
    ]
    if config.VLLM_EXTRA_ARGS:
        import shlex
        cmd.extend(shlex.split(config.VLLM_EXTRA_ARGS))

    logger.info("Launching vLLM server: %s", " ".join(cmd))

    _server_process = subprocess.Popen(
        cmd,
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )

    pid_path = _pid_file_path()
    with open(pid_path, "w") as f:
        f.write(str(_server_process.pid))
    logger.debug("Wrote PID %d to %s", _server_process.pid, pid_path)

    atexit.register(stop)

    primary_endpoint = config.SERVER_HEALTH_ENDPOINT
    fallback_endpoint = "/v1/models"
    interval = config.SERVER_HEALTH_INTERVAL
    timeout = config.SERVER_HEALTH_TIMEOUT

    deadline = time.monotonic() + timeout
    use_fallback = False

    while time.monotonic() < deadline:
        if _server_process.poll() is not None:
            _dump_diagnostics(cmd, env, _server_process)
            raise RuntimeError(
                f"vLLM server exited prematurely with code {_server_process.returncode}."
            )

        endpoint = fallback_endpoint if use_fallback else primary_endpoint
        if _probe_health(base_url, endpoint):
            label = "fallback" if use_fallback else "primary"
            logger.info(
                "vLLM server ready on port %d (probe: %s %s)",
                port, label, endpoint,
            )
            return

        if not use_fallback:
            try:
                resp = requests.get(f"{base_url}{primary_endpoint}", timeout=5)
                if resp.status_code == 404:
                    logger.debug(
                        "Primary endpoint %s returned 404, switching to fallback %s",
                        primary_endpoint, fallback_endpoint,
                    )
                    use_fallback = True
            except Exception:
                pass

        logger.debug(
            "Health check pending (%.0fs remaining)...",
            deadline - time.monotonic(),
        )
        time.sleep(interval)

    _server_process.kill()
    _dump_diagnostics(cmd, env, _server_process)
    raise RuntimeError(
        f"vLLM server failed to become healthy within {timeout}s. "
        "See logs above for diagnostics."
    )


def stop() -> None:
    """Terminate the vLLM server using the PID file."""
    global _server_process

    pid_path = _pid_file_path()
    if not os.path.isfile(pid_path):
        return

    try:
        with open(pid_path, "r") as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        logger.warning("Could not read PID file %s — skipping stop.", pid_path)
        _remove_pid_file(pid_path)
        return

    logger.info("Stopping vLLM server (PID %d)...", pid)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        logger.debug("PID %d already exited.", pid)
        _remove_pid_file(pid_path)
        return

    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        logger.warning("PID %d did not exit after SIGTERM; sending SIGKILL.", pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _remove_pid_file(pid_path)
    _server_process = None
    logger.info("vLLM server stopped.")


def _dump_diagnostics(
    cmd: list[str],
    env: dict[str, str],
    proc: subprocess.Popen,
) -> None:
    logger.error("=== vLLM Server Diagnostics ===")
    logger.error("Launch command: %s", " ".join(cmd))

    for var in ("HF_HOME", "CUDA_VISIBLE_DEVICES", "TRANSFORMERS_CACHE"):
        logger.error("  %s=%s", var, env.get(var, "<not set>"))

    if proc.stderr:
        try:
            raw = proc.stderr.read()
            if raw:
                lines = raw.decode("utf-8", errors="replace").splitlines()
                tail = lines[-50:]
                logger.error("Last %d lines of stderr:", len(tail))
                for line in tail:
                    logger.error("  %s", line)
        except Exception as exc:
            logger.error("Could not read stderr: %s", exc)


def _remove_pid_file(pid_path: str) -> None:
    try:
        os.remove(pid_path)
    except OSError:
        pass


if __name__ == "__main__":
    from src.config import build_arg_parser, load_config

    parser = build_arg_parser(description="vLLM server lifecycle")
    parser.add_argument(
        "--action",
        choices=["start", "stop"],
        required=True,
        help="Start or stop the vLLM server.",
    )
    args = parser.parse_args()

    if args.action == "stop":
        stop()
    else:
        cfg = load_config(args.config, overrides=args.override)
        from src.logging_setup import setup_logging

        setup_logging(cfg)
        from src.download_models import resolve_model_path

        model_path = resolve_model_path(cfg)
        start(cfg, model_path)
        logger.info("Server is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop()
