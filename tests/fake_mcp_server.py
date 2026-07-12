"""Minimal MCP stdio server used by tests. Standard library only.

Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout. Exposes three
tools across TWO tools/list pages (to exercise pagination), sends a
server-initiated ping before the first page (to exercise interleaving),
and emits a stray notification (which clients must ignore).
"""

import json
import sys

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

PAGE_1 = [
    {
        "name": "echo",
        "description": "Echo text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]
PAGE_2 = [
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "boom",
        "description": "Always fails.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def result(msg, payload):
    send({"jsonrpc": "2.0", "id": msg["id"], "result": payload})


def error(msg, code, text):
    send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": code, "message": text}})


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")

    if method is None:
        continue  # response to our server-initiated ping — ignore

    if method == "initialize":
        result(msg, {
            "protocolVersion": msg["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "1.0"},
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        cursor = (msg.get("params") or {}).get("cursor")
        if not cursor:
            # interleave a server->client ping and a stray notification
            # BEFORE the answer: the client must handle both.
            send({"jsonrpc": "2.0", "id": "srv-ping-1", "method": "ping"})
            send({"jsonrpc": "2.0", "method": "notifications/message",
                  "params": {"level": "info", "data": "hello"}})
            result(msg, {"tools": PAGE_1, "nextCursor": "page-2"})
        else:
            result(msg, {"tools": PAGE_2})
    elif method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments") or {}
        if name == "echo":
            result(msg, {"content": [{"type": "text", "text": "echo: " + args.get("text", "")}]})
        elif name == "add":
            total = args["a"] + args["b"]
            result(msg, {
                "content": [{"type": "text", "text": str(total)}],
                "structuredContent": {"sum": total},
            })
        elif name == "boom":
            result(msg, {"content": [{"type": "text", "text": "kaboom"}], "isError": True})
        else:
            error(msg, -32602, f"Unknown tool: {name}")
    elif method == "ping":
        result(msg, {})
    else:
        if "id" in msg:
            error(msg, -32601, f"Method not found: {method}")
