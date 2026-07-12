"""MCPClient — stdio transport, tool adaptation, registry integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from autoagent.errors import MCPError, ToolError
from autoagent.mcp import MCPClient
from autoagent.registry import ToolRegistry
from autoagent.schema import ToolCall

FAKE_SERVER = Path(__file__).with_name("fake_mcp_server.py")


@pytest.fixture()
def client():
    with MCPClient([sys.executable, str(FAKE_SERVER)], timeout=15.0) as mcp:
        yield mcp


def test_handshake_populates_server_info(client):
    assert client.server_info["name"] == "fake-mcp"
    assert client.protocol_version is not None
    assert "tools" in client.server_capabilities
    assert client.alive


def test_list_tools_follows_pagination_and_ignores_interleaved_traffic(client):
    # page 1 arrives AFTER a server-initiated ping and a notification
    names = [t["name"] for t in client.list_tools()]
    assert names == ["echo", "add", "boom"]


def test_call_tool_returns_text_payload(client):
    assert client.call_tool("echo", {"text": "salut"}) == {"text": "echo: salut"}


def test_call_tool_prefers_structured_content(client):
    assert client.call_tool("add", {"a": 2, "b": 3}) == {"sum": 5}


def test_call_tool_iserror_raises_toolerror(client):
    with pytest.raises(ToolError, match="kaboom"):
        client.call_tool("boom")


def test_unknown_tool_jsonrpc_error_raises_mcperror(client):
    with pytest.raises(MCPError, match="-32602"):
        client.call_tool("nope")


def test_tools_carry_autoagent_specs_with_prefix_and_filters(client):
    handlers = client.tools(prefix="mcp_", exclude={"boom"})
    specs = {h.__autoagent_tool_spec__.name: h.__autoagent_tool_spec__ for h in handlers}
    assert set(specs) == {"mcp_echo", "mcp_add"}
    assert specs["mcp_echo"].description == "Echo text back."
    assert specs["mcp_echo"].input_schema["required"] == ["text"]

    only = client.tools(include={"add"})
    assert [h.__autoagent_tool_spec__.name for h in only] == ["add"]


def test_mount_on_registry_executes_through_validation(client):
    registry = ToolRegistry()
    names = client.mount(registry, prefix="mcp_")
    assert names == ["mcp_echo", "mcp_add", "mcp_boom"]

    ok = registry.execute(ToolCall(id="1", name="mcp_echo", arguments={"text": "hi"}))
    assert ok.ok and ok.result == {"text": "echo: hi"}

    # invalid args are rejected by the registry's JSON-Schema validation
    # BEFORE anything reaches the server
    bad = registry.execute(ToolCall(id="2", name="mcp_echo", arguments={}))
    assert not bad.ok and "ValidationError" in bad.error

    # isError surfaces as a tool error, not a crash
    boom = registry.execute(ToolCall(id="3", name="mcp_boom", arguments={}))
    assert not boom.ok and "kaboom" in boom.error


def test_calls_after_close_raise_mcperror(client):
    client.close()
    assert not client.alive
    with pytest.raises(MCPError):
        client.call_tool("echo", {"text": "x"})


def test_unlaunchable_server_raises_mcperror():
    with pytest.raises(MCPError, match="Cannot launch"):
        MCPClient(["definitely-not-a-real-binary-xyz"]).start()
