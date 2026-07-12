"""End-to-end tests for the host-function bridge.

A SANDBOXED tool (no host objects, and under Docker `--network none`) reaches
the host ONLY through whitelisted callbacks it invokes as
``context["call_host"](name, args)``. The bridge rides the child's stdio
pipes, so it works without any network and the host keeps its credentials.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from autoagent.approval import ToolManifest, load_tools
from autoagent.registry import ToolRegistry
from autoagent.sandbox import DockerSandbox, SubprocessSandbox, docker_available
from autoagent.schema import ToolCall

# --- a "database" living on the HOST: the secret never leaves this process ---
_DB_SECRET = "super-secret-db-password"
_FAKE_ROWS = {"customers": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}


def fake_db(query: str) -> list[dict]:
    assert _DB_SECRET  # the credential is used HERE, host-side, never returned
    return _FAKE_ROWS.get(query, [])


HOST_FUNCTIONS = {"fake_db": fake_db}


def _write_tool(tmp_path: Path, name: str, body: str, schema_props: str = "{}") -> Path:
    run_body = "\n".join("    " + line for line in textwrap.dedent(body).strip().splitlines())
    src = (
        "TOOL = {\n"
        f'    "name": "{name}",\n'
        '    "description": "bridge test tool",\n'
        f'    "input_schema": {{"type": "object", "properties": {schema_props}}},\n'
        '    "permissions": [],\n'
        "}\n\n"
        "def run(args, context):\n"
        f"{run_body}\n"
    )
    path = tmp_path / f"{name}.py"
    path.write_text(src, encoding="utf-8")
    return path


class _FakeAgent:
    def __init__(self) -> None:
        self.registry = ToolRegistry()


# ---------------------------------------------------------------------------
# SubprocessSandbox bridge
# ---------------------------------------------------------------------------


def test_subprocess_bridge_returns_host_data(tmp_path: Path) -> None:
    tool = _write_tool(
        tmp_path,
        "db_reader",
        "rows = context['call_host']('fake_db', {'query': args.get('q', 'customers')})\n"
        "return {'count': len(rows), 'rows': rows}",
    )
    out = SubprocessSandbox(timeout=15).run_python_tool(
        tool, {"q": "customers"}, host_functions=HOST_FUNCTIONS
    )
    assert out["ok"] is True
    assert out["result"]["count"] == 2
    # the host credential never crossed into the sandbox
    assert _DB_SECRET not in json.dumps(out)


def test_bridge_refuses_non_whitelisted_function(tmp_path: Path) -> None:
    tool = _write_tool(tmp_path, "sneaky", "return context['call_host']('rm_rf', {})")
    out = SubprocessSandbox(timeout=15).run_python_tool(tool, {}, host_functions=HOST_FUNCTIONS)
    assert out["ok"] is False
    assert "not allowed" in out["error"]


def test_bridge_surfaces_host_function_error(tmp_path: Path) -> None:
    def boom() -> None:
        raise ValueError("db down")

    tool = _write_tool(tmp_path, "caller", "return context['call_host']('boom', {})")
    out = SubprocessSandbox(timeout=15).run_python_tool(tool, {}, host_functions={"boom": boom})
    assert out["ok"] is False
    assert "db down" in out["error"]


# ---------------------------------------------------------------------------
# Full path: load_tools → sandbox handler → bridge → host function
# ---------------------------------------------------------------------------


def test_load_tools_wires_bridge_for_sandboxed_tool(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_tool(
        tools,
        "db_reader",
        "rows = context['call_host']('fake_db', {'query': args.get('q', 'customers')})\n"
        "return {'count': len(rows)}",
        schema_props='{"q": {"type": "string"}}',
    )
    manifest = ToolManifest.load(tmp_path / "approved_tools.json")  # empty → sandboxed
    agent = _FakeAgent()
    modes = dict(
        load_tools(
            agent,
            tools,
            manifest,
            sandbox=SubprocessSandbox(timeout=15),
            sandbox_host_functions=HOST_FUNCTIONS,
        )
    )
    assert modes["db_reader"] == "sandbox"
    result = agent.registry.execute(
        ToolCall(id="t", name="db_reader", arguments={"q": "customers"}), context={}
    )
    assert result.ok, result.error
    assert result.result["count"] == 2


# ---------------------------------------------------------------------------
# Docker bridge — works even under --network none (the killer demo)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not docker_available(), reason="Docker daemon not reachable")
@pytest.mark.timeout(120)
def test_docker_bridge_works_under_network_none(tmp_path: Path) -> None:
    tool = _write_tool(
        tmp_path,
        "db_reader",
        "rows = context['call_host']('fake_db', {'query': 'customers'})\n"
        "return {'count': len(rows), 'rows': rows}",
    )
    # allow_network defaults False -> container runs with --network none, yet
    # the tool still reaches the host DB via the stdio bridge.
    out = DockerSandbox(timeout=30).run_python_tool(tool, {}, host_functions=HOST_FUNCTIONS)
    assert out["ok"] is True
    assert out["result"]["count"] == 2
    assert _DB_SECRET not in json.dumps(out)
