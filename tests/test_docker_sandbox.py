"""Tests for the DockerSandbox OS-level isolation boundary.

Two layers:
  * "Dry" tests (always run, no daemon needed): monkeypatch `subprocess.run`
    to capture the `docker run` command and assert it is fully locked down
    (--network none, --read-only, --cap-drop ALL, non-root, limits…).
  * "Live" tests (skipped unless a Docker daemon is reachable): actually run
    a container and prove the network is cut and the FS is read-only.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import autoagent.sandbox as sb
from autoagent.sandbox import DockerSandbox, SubprocessSandbox, docker_available, make_sandbox


def _write_tool(tmp_path: Path, body: str, name: str = "t.py") -> Path:
    indented = "\n".join("    " + line for line in body.strip().splitlines())
    path = tmp_path / name
    path.write_text(f"def run(args, context):\n{indented}\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Dry tests — prove the docker run command is locked down (no daemon needed)
# ---------------------------------------------------------------------------


def _capture_docker_cmd(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if cmd[:3] == ["docker", "image", "inspect"]:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return types.SimpleNamespace(
            returncode=0, stdout=json.dumps({"ok": True, "result": {"ran": True}}), stderr=""
        )

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    return captured


class TestDockerCommandLockdown:
    def test_command_has_all_lockdown_flags(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured = _capture_docker_cmd(monkeypatch)
        tool = _write_tool(tmp_path, "return {'ok': True}")
        out = DockerSandbox().run_python_tool(tool, {"x": 1}, context={"u": 1}, allow_network=False)
        assert out["ok"] is True

        cmd = captured["cmd"]
        assert cmd[:4] == ["docker", "run", "--rm", "-i"]
        assert "--read-only" in cmd
        assert "--cap-drop" in cmd and "ALL" in cmd
        assert "no-new-privileges" in cmd
        assert "65534:65534" in cmd            # non-root user
        assert "--memory" in cmd and "--cpus" in cmd and "--pids-limit" in cmd
        assert any(tok.startswith("/tmp:") for tok in cmd)  # tmpfs scratch
        assert "-I" in cmd and "-S" in cmd      # isolated python
        # the tool SOURCE travels via stdin (no volume mount)
        assert "def run(args, context)" in captured["input"]
        assert "-v" not in cmd                  # no host path mounted

    def test_network_cut_without_permission(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured = _capture_docker_cmd(monkeypatch)
        tool = _write_tool(tmp_path, "return 1")
        DockerSandbox().run_python_tool(tool, {}, allow_network=False)
        cmd = captured["cmd"]
        assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"

    def test_network_open_with_permission(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured = _capture_docker_cmd(monkeypatch)
        tool = _write_tool(tmp_path, "return 1")
        DockerSandbox().run_python_tool(tool, {}, allow_network=True)
        # network permission granted -> we do NOT add `--network none`
        assert "--network" not in captured["cmd"]


class TestSandboxSelection:
    def test_make_sandbox_matches_docker_availability(self) -> None:
        sandbox = make_sandbox()
        if docker_available():
            assert isinstance(sandbox, DockerSandbox)
        else:
            assert isinstance(sandbox, SubprocessSandbox)


# ---------------------------------------------------------------------------
# Live tests — real container; skipped unless a Docker daemon is reachable
# ---------------------------------------------------------------------------

requires_docker = pytest.mark.skipif(not docker_available(), reason="Docker daemon not reachable")


@requires_docker
class TestDockerLiveIsolation:
    @pytest.mark.timeout(120)
    def test_normal_tool_runs_in_container(self, tmp_path: Path) -> None:
        tool = _write_tool(tmp_path, "return {'doubled': args['x'] * 2}")
        out = DockerSandbox(timeout=30).run_python_tool(tool, {"x": 21})
        assert out["ok"] is True
        assert out["result"]["doubled"] == 42

    @pytest.mark.timeout(120)
    def test_network_none_blocks_socket(self, tmp_path: Path) -> None:
        # Run raw (bypassing AST validation) to prove the CONTAINER itself
        # cuts the network — defence in depth even if validation were bypassed.
        tool = _write_tool(
            tmp_path,
            "import socket\n"
            "s = socket.socket()\n"
            "s.settimeout(5)\n"
            "s.connect(('1.1.1.1', 53))\n"
            "return {'connected': True}",
        )
        out = DockerSandbox(timeout=30).run_python_tool(tool, {}, allow_network=False)
        assert out["ok"] is False
        assert "unreachable" in out["error"].lower() or "errno" in out["error"].lower()

    @pytest.mark.timeout(120)
    def test_read_only_fs_blocks_write(self, tmp_path: Path) -> None:
        tool = _write_tool(tmp_path, "open('/evil.txt', 'w').write('x')\nreturn {'wrote': True}")
        out = DockerSandbox(timeout=30).run_python_tool(tool, {})
        assert out["ok"] is False
        assert "read-only" in out["error"].lower() or "permission" in out["error"].lower()
