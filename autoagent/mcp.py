"""MCP (Model Context Protocol) client over stdio — zero dependency.

``MCPClient`` launches an MCP server as a subprocess and speaks JSON-RPC
2.0 with it over stdin/stdout (newline-delimited JSON, the MCP *stdio*
transport). The server's tools become ordinary autoagent tools: each one
is exposed as a handler carrying ``__autoagent_tool_spec__``, so it goes
through the same ``ToolRegistry`` path as a local ``@agent.tool`` —
JSON-Schema validation included (MCP ``inputSchema`` is standard JSON
Schema, exactly what the registry validates with).

Typical use::

    from autoagent.mcp import MCPClient

    with MCPClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]) as mcp:
        mcp.mount(agent, prefix="fs_")          # every server tool, prefixed
        agent.run("Liste les fichiers du projet.")

Scoped mounting::

    mcp.mount(agent, include={"read_file", "list_directory"})

Design notes:
  * Transport only covers stdio (local subprocess). HTTP/SSE servers are
    out of scope for now — stdio is where the tool ecosystem lives and it
    needs nothing beyond the standard library.
  * The client is thread-safe: requests are correlated by id, so it works
    under ``Agent(parallel_tool_calls=True)``.
  * A tool result with ``isError`` raises ``ToolError`` — through the
    registry that surfaces as a *tool error* the LLM can react to, never
    a crash.
  * ``structuredContent`` (when the server provides it) is returned as-is
    (a dict — what Gemini wants from tools); otherwise the text parts of
    ``content`` are joined and returned as ``{"text": ...}``.
  * Windows: pass the real executable name in the command list (``npx``
    is ``npx.cmd`` on Windows); pipes are forced to UTF-8 on both sides.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
from collections import deque
from typing import Any, Callable, Iterable, Mapping, Sequence

from .errors import MCPError, ToolError
from .logging import get_logger
from .schema import JsonDict, ToolSpec

__all__ = ["MCPClient", "MCP_PROTOCOL_VERSION"]

_log = get_logger(__name__)

MCP_PROTOCOL_VERSION = "2025-06-18"

_JSONRPC_METHOD_NOT_FOUND = -32601


class _Pending:
    """A response slot one request thread waits on."""

    __slots__ = ("event", "message")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.message: JsonDict | None = None


class MCPClient:
    """Client for one MCP server subprocess (stdio transport).

    Args:
        command: The server command — a list (recommended: exact argv) or
            a string (split with shlex; prefer the list form on Windows,
            backslash paths do not survive POSIX splitting).
        env: Extra environment variables, MERGED over ``os.environ``
            (an MCP server usually needs its own API key here).
        cwd: Working directory for the server process.
        timeout: Default seconds to wait for each response (handshake,
            tools/list, tools/call). Per-call override on ``call_tool``.
        client_name: Advertised in the MCP ``initialize`` handshake.
    """

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 60.0,
        client_name: str = "autoagent",
    ) -> None:
        if isinstance(command, str):
            command = shlex.split(command)
        self.command = list(command)
        self.env = dict(env) if env else None
        self.cwd = cwd
        self.timeout = timeout
        self.client_name = client_name

        self.server_info: JsonDict = {}
        self.server_capabilities: JsonDict = {}
        self.protocol_version: str | None = None

        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=30)
        self._pending: dict[Any, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._next_id = 0
        self._closed = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> "MCPClient":
        """Spawn the server process and run the MCP handshake."""
        if self._proc is not None:
            return self
        env = None
        if self.env is not None:
            env = {**os.environ, **self.env}
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.cwd,
                env=env,
            )
        except OSError as exc:
            raise MCPError(f"Cannot launch MCP server {self.command!r}: {exc}") from exc

        self._reader = threading.Thread(
            target=self._read_loop, name="autoagent-mcp-reader", daemon=True
        )
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._stderr_loop, name="autoagent-mcp-stderr", daemon=True
        )
        self._stderr_reader.start()

        try:
            from . import __version__ as _version
        except ImportError:  # pragma: no cover - defensive
            _version = "0"
        result = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": _version},
            },
        )
        self.protocol_version = result.get("protocolVersion")
        self.server_info = result.get("serverInfo") or {}
        self.server_capabilities = result.get("capabilities") or {}
        self._notify("notifications/initialized")
        return self

    def close(self) -> None:
        """Terminate the server process. Idempotent."""
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._fail_all_pending()

    def __enter__(self) -> "MCPClient":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ── MCP operations ───────────────────────────────────────────────────

    def list_tools(self) -> list[JsonDict]:
        """Return the server's tool definitions (all pages)."""
        tools: list[JsonDict] = []
        cursor: str | None = None
        while True:
            params: JsonDict = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params)
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(
        self,
        name: str,
        arguments: JsonDict | None = None,
        *,
        timeout: float | None = None,
    ) -> JsonDict:
        """Invoke one server tool and normalize its result for the LLM.

        Returns ``structuredContent`` verbatim when present, otherwise
        ``{"text": <joined text parts>}``. Raises ``ToolError`` when the
        server flags the result with ``isError`` (so the registry reports
        it as a tool error, not a success payload).
        """
        result = self._request(
            "tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout
        )
        text = _content_to_text(result.get("content") or [])
        if result.get("isError"):
            raise ToolError(text or f"MCP tool {name!r} reported an error")
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        if structured is not None:
            return {"result": structured}
        return {"text": text}

    # ── autoagent integration ────────────────────────────────────────────

    def tools(
        self,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        prefix: str = "",
    ) -> list[Callable[..., Any]]:
        """Build one autoagent-compatible handler per server tool.

        Each handler carries ``__autoagent_tool_spec__`` (name prefixed
        with ``prefix``, server description, server ``inputSchema``), so
        it can be passed straight to ``agent.add_tool`` / a registry's
        ``add_function``. ``include``/``exclude`` filter on the SERVER
        tool names (before prefixing).
        """
        include_set = set(include) if include is not None else None
        exclude_set = set(exclude) if exclude is not None else set()
        handlers: list[Callable[..., Any]] = []
        for tool_def in self.list_tools():
            server_name = tool_def.get("name")
            if not server_name:
                continue
            if include_set is not None and server_name not in include_set:
                continue
            if server_name in exclude_set:
                continue
            handlers.append(self._make_handler(tool_def, prefix))
        return handlers

    def mount(
        self,
        agent: Any,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        prefix: str = "",
    ) -> list[str]:
        """Register every (filtered) server tool on ``agent``.

        Returns the registered tool names. ``agent`` is anything with an
        ``add_tool(handler)`` — an ``Agent`` — or an ``add_function``
        (a bare ``ToolRegistry``).
        """
        add = getattr(agent, "add_tool", None) or getattr(agent, "add_function")
        names: list[str] = []
        for handler in self.tools(include=include, exclude=exclude, prefix=prefix):
            add(handler)
            names.append(handler.__autoagent_tool_spec__.name)  # type: ignore[attr-defined]
        return names

    def _make_handler(self, tool_def: JsonDict, prefix: str) -> Callable[..., Any]:
        server_name = str(tool_def["name"])
        exposed_name = f"{prefix}{server_name}"
        description = tool_def.get("description") or f"MCP tool {server_name}"
        input_schema = tool_def.get("inputSchema") or {"type": "object", "properties": {}}
        client_self = self

        def handler(**kwargs: Any) -> JsonDict:
            return client_self.call_tool(server_name, kwargs)

        handler.__name__ = exposed_name
        handler.__doc__ = description
        handler.__autoagent_tool_spec__ = ToolSpec(  # type: ignore[attr-defined]
            name=exposed_name,
            description=description,
            input_schema=input_schema,
        )
        return handler

    # ── JSON-RPC plumbing ────────────────────────────────────────────────

    def _request(
        self, method: str, params: JsonDict | None = None, *, timeout: float | None = None
    ) -> JsonDict:
        if self._proc is None:
            self.start()
        with self._id_lock:
            self._next_id += 1
            request_id = self._next_id
        pending = _Pending()
        with self._pending_lock:
            self._pending[request_id] = pending
        message: JsonDict = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            self._write(message)
            if not pending.event.wait(timeout if timeout is not None else self.timeout):
                raise MCPError(
                    f"MCP server did not answer {method!r} within "
                    f"{timeout if timeout is not None else self.timeout}s{self._stderr_hint()}"
                )
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
        response = pending.message or {}
        if "error" in response:
            error = response["error"] or {}
            raise MCPError(
                f"MCP {method!r} failed: [{error.get('code')}] {error.get('message')}"
            )
        if response.get("_transport_error"):
            raise MCPError(str(response.get("_transport_error")) + self._stderr_hint())
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    def _notify(self, method: str, params: JsonDict | None = None) -> None:
        message: JsonDict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._write(message)

    def _write(self, message: JsonDict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise MCPError(f"MCP server is not running{self._stderr_hint()}")
        line = json.dumps(message, ensure_ascii=False)
        with self._write_lock:
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except OSError as exc:
                raise MCPError(f"Cannot write to MCP server: {exc}{self._stderr_hint()}") from exc

    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                _log.debug("mcp: ignoring non-JSON line from server: %.200s", line)
                continue
            if not isinstance(message, dict):
                continue
            if "method" in message:
                self._handle_server_message(message)
            elif "id" in message:
                with self._pending_lock:
                    pending = self._pending.get(message["id"])
                if pending is not None:
                    pending.message = message
                    pending.event.set()
                else:
                    _log.debug("mcp: response for unknown id %r", message.get("id"))
        self._fail_all_pending()

    def _handle_server_message(self, message: JsonDict) -> None:
        method = message.get("method")
        if "id" not in message:
            _log.debug("mcp: notification %r ignored", method)
            return
        if method == "ping":
            reply: JsonDict = {"jsonrpc": "2.0", "id": message["id"], "result": {}}
        else:
            # Server-initiated features (sampling, roots, elicitation) are
            # not supported: answer method-not-found instead of hanging it.
            reply = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": _JSONRPC_METHOD_NOT_FOUND, "message": f"Method not found: {method}"},
            }
        try:
            self._write(reply)
        except MCPError:
            pass

    def _stderr_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self._stderr_tail.append(line)

    def _fail_all_pending(self) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot.message = {"_transport_error": "MCP server closed the connection"}
            slot.event.set()

    def _stderr_hint(self) -> str:
        if not self._stderr_tail:
            return ""
        return " — server stderr: " + " | ".join(list(self._stderr_tail)[-5:])


def _content_to_text(content: list[Any]) -> str:
    """Join an MCP content list into plain text for the LLM.

    Non-text parts (image, audio, resource) are summarized by type — the
    agent loop is text-first; binary payloads are not forwarded.
    """
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text", "")))
        elif kind == "resource":
            resource = item.get("resource") or {}
            inner = resource.get("text")
            parts.append(str(inner) if inner else f"[resource {resource.get('uri', '?')}]")
        else:
            parts.append(f"[{kind or 'unknown'} content]")
    return "\n".join(part for part in parts if part)
