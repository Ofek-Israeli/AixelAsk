"""Tests for src.vllm_server — subprocess launch, health check, PID lifecycle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest import mock

import pytest


@dataclass
class _StubConfig:
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 30000
    SERVER_TP_SIZE: int = 1
    SERVER_DTYPE: str = "float16"
    SERVER_CONTEXT_LENGTH: int = 8192
    SERVER_HEALTH_TIMEOUT: int = 5
    SERVER_HEALTH_INTERVAL: int = 1
    SERVER_HEALTH_ENDPOINT: str = "/health"
    VLLM_GPU_MEMORY_UTILIZATION: str = "0.85"
    VLLM_EXTRA_ARGS: str = ""
    CUDA_VISIBLE_DEVICES: str = ""


class TestVllmServerStart:

    def test_builds_correct_launch_command(self):
        from src import vllm_server

        cfg = _StubConfig(SERVER_PORT=31000, SERVER_TP_SIZE=2, SERVER_DTYPE="bfloat16")

        with mock.patch("src.vllm_server._port_in_use", return_value=False), \
             mock.patch("src.vllm_server.subprocess.Popen") as mock_popen, \
             mock.patch("src.vllm_server._probe_health", return_value=True), \
             mock.patch("src.vllm_server.time.sleep"), \
             mock.patch("builtins.open", mock.mock_open()):

            mock_proc = mock.MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            vllm_server._server_process = None
            vllm_server.start(cfg, "/models/Mistral")

            cmd = mock_popen.call_args[0][0]
            assert "--model" in cmd
            assert "/models/Mistral" in cmd
            assert "--port" in cmd
            assert "31000" in cmd
            assert "--tensor-parallel-size" in cmd
            assert "2" in cmd
            assert "--dtype" in cmd
            assert "bfloat16" in cmd

    def test_extra_args_appended(self):
        from src import vllm_server

        cfg = _StubConfig(VLLM_EXTRA_ARGS="--enforce-eager --disable-log-requests")

        with mock.patch("src.vllm_server._port_in_use", return_value=False), \
             mock.patch("src.vllm_server.subprocess.Popen") as mock_popen, \
             mock.patch("src.vllm_server._probe_health", return_value=True), \
             mock.patch("src.vllm_server.time.sleep"), \
             mock.patch("builtins.open", mock.mock_open()):

            mock_proc = mock.MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            vllm_server._server_process = None
            vllm_server.start(cfg, "/models/Mistral")

            cmd = mock_popen.call_args[0][0]
            assert "--enforce-eager" in cmd
            assert "--disable-log-requests" in cmd

    def test_reuses_healthy_existing_server(self):
        from src import vllm_server

        cfg = _StubConfig()

        with mock.patch("src.vllm_server._port_in_use", return_value=True), \
             mock.patch("src.vllm_server._probe_health", return_value=True), \
             mock.patch("src.vllm_server.subprocess.Popen") as mock_popen:

            vllm_server.start(cfg, "/models/Mistral")
            mock_popen.assert_not_called()


class TestVllmServerStop:

    def test_stop_removes_pid_file(self, tmp_path):
        from src import vllm_server

        pid_file = tmp_path / ".vllm.pid"
        pid_file.write_text("99999")

        with mock.patch("src.vllm_server._pid_file_path", return_value=str(pid_file)), \
             mock.patch("os.kill", side_effect=ProcessLookupError):

            vllm_server.stop()

        assert not pid_file.exists()
