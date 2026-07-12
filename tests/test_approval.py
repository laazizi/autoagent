"""Tests for human-in-the-loop tool promotion (autoagent.approval).

Pins the trust model:
  * a tool runs NATIVE only if its current source hash is in the manifest;
  * one byte changed → approval void → back to the sandbox;
  * native tools receive the host_context (real handles); sandboxed tools do not;
  * an unapproved tool that fails validation is skipped, never registered.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from autoagent.approval import (
    ToolManifest,
    approve_tool,
    load_tools,
    review_card,
    sha256_of,
)
from autoagent.registry import ToolRegistry
from autoagent.sandbox import SubprocessSandbox
from autoagent.schema import ToolCall


class _FakeAgent:
    def __init__(self) -> None:
        self.registry = ToolRegistry()


def _tool_src(name: str, body: str, permissions: str = "[]") -> str:
    return textwrap.dedent(
        f'''
        TOOL = {{
            "name": "{name}",
            "description": "test tool {name}",
            "input_schema": {{"type": "object", "properties": {{}}}},
            "permissions": {permissions},
        }}

        def run(args, context):
        {textwrap.indent(textwrap.dedent(body).strip(), "    ")}
        '''
    ).strip() + "\n"


def _write(tmp_path: Path, name: str, src: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(src, encoding="utf-8")
    return path


def _run_tool(agent: _FakeAgent, name: str, args: dict | None = None):
    result = agent.registry.execute(ToolCall(id="t", name=name, arguments=args or {}), context={})
    assert result.ok, result.error
    return result.result


# ---------------------------------------------------------------------------
# Hashing + manifest
# ---------------------------------------------------------------------------


def test_sha256_distinguishes_code() -> None:
    assert sha256_of("a") == sha256_of("a")
    assert sha256_of("a") != sha256_of("a ")  # one byte differs


def test_manifest_roundtrip(tmp_path: Path) -> None:
    m = ToolManifest.load(tmp_path / "approved_tools.json")
    digest = m.approve("CODE", name="t", permissions=["network"], approved_by="mo", approved_at="2026-01-01")
    assert m.contains(digest)
    # reload from disk → still there
    m2 = ToolManifest.load(tmp_path / "approved_tools.json")
    assert m2.contains(digest)
    assert m2.entries[digest]["approved_by"] == "mo"
    m2.revoke(digest)
    assert not ToolManifest.load(tmp_path / "approved_tools.json").contains(digest)


def test_approval_voided_by_one_byte_change(tmp_path: Path) -> None:
    path = _write(tmp_path, "t", _tool_src("t", "return {'ok': True}"))
    manifest = ToolManifest.load(tmp_path / "approved_tools.json")
    digest = approve_tool(path, manifest, approved_by="mo")
    assert manifest.contains(digest)

    # Tamper with the file: the OLD hash no longer matches its content.
    path.write_text(path.read_text(encoding="utf-8") + "# sneaky\n", encoding="utf-8")
    assert not manifest.contains(sha256_of(path.read_text(encoding="utf-8")))


def test_review_card_fields(tmp_path: Path) -> None:
    path = _write(tmp_path, "rc", _tool_src("rc", "return 1", permissions='["network"]'))
    card = review_card(path)
    assert card["name"] == "rc"
    assert card["permissions"] == ["network"]
    assert card["sha256"] == sha256_of(path.read_text(encoding="utf-8"))
    assert "def run" in card["code"]


# ---------------------------------------------------------------------------
# The 2-way loader
# ---------------------------------------------------------------------------


def test_load_tools_routes_native_vs_sandbox_vs_invalid(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    # tool_a: will be APPROVED → native, must see the host secret.
    _write(tools, "tool_a", _tool_src("tool_a", "return {'secret_seen': context.get('secret', 'NONE')}"))
    # tool_b: NOT approved → sandbox, must NOT see the host secret.
    _write(tools, "tool_b", _tool_src("tool_b", "return {'secret_seen': context.get('secret', 'NONE')}"))
    # tool_bad: NOT approved AND invalid (imports os) → skipped entirely.
    _write(
        tools,
        "tool_bad",
        "TOOL = {'name': 'tool_bad', 'description': 'x', 'input_schema': {'type': 'object', 'properties': {}}}\n"
        "import os\n"
        "def run(args, context):\n    return os.listdir('.')\n",
    )

    manifest = ToolManifest.load(tmp_path / "approved_tools.json")
    approve_tool(tools / "tool_a.py", manifest, approved_by="mo")

    agent = _FakeAgent()
    modes = dict(
        load_tools(
            agent,
            tools,
            manifest,
            host_context={"secret": 42},
            sandbox=SubprocessSandbox(timeout=15),  # force Subprocess (no Docker dependency)
        )
    )

    assert modes == {"tool_a": "native", "tool_b": "sandbox", "tool_bad": "invalid"}
    # the invalid tool was never registered
    names = {spec.name for spec in agent.registry.specs()}
    assert "tool_bad" not in names and {"tool_a", "tool_b"} <= names

    # native tool gets the real host handle…
    assert _run_tool(agent, "tool_a") == {"secret_seen": 42}
    # …sandboxed tool is isolated from host objects.
    assert _run_tool(agent, "tool_b") == {"secret_seen": "NONE"}


def test_byte_change_drops_tool_back_to_sandbox(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    path = _write(tools, "promo", _tool_src("promo", "return {'v': 1}"))
    manifest = ToolManifest.load(tmp_path / "approved_tools.json")
    approve_tool(path, manifest, approved_by="mo")

    agent = _FakeAgent()
    modes = dict(load_tools(agent, tools, manifest, sandbox=SubprocessSandbox(timeout=15)))
    assert modes["promo"] == "native"

    # Edit the approved tool → it must no longer be trusted.
    path.write_text(path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    agent2 = _FakeAgent()
    modes2 = dict(load_tools(agent2, tools, manifest, sandbox=SubprocessSandbox(timeout=15)))
    assert modes2["promo"] == "sandbox"
