"""Red test for `asyncio.run` inside a running event loop.

CURRENT BUG (autoagent/registry.py:43):
    RegisteredTool.execute() calls `asyncio.run(value)` when the handler
    returns an awaitable. This crashes with
        RuntimeError: asyncio.run() cannot be called from a running event loop
    whenever the agent is itself running inside an event loop — i.e. ANY
    modern async framework (FastAPI, Starlette, Jupyter, aiohttp,
    Discord bots, etc.).

This test creates a real running loop via pytest-asyncio and verifies the
registry can execute an async tool without crashing. RED until the bug
is fixed (e.g. by detecting an existing loop and using
`asyncio.run_coroutine_threadsafe` on a worker loop, or by making
`execute` natively awaitable with a sync wrapper).
"""

from __future__ import annotations

import asyncio

import pytest

from autoagent.registry import ToolRegistry
from autoagent.schema import ToolCall, ToolSpec


@pytest.mark.asyncio
async def test_async_tool_executes_inside_running_loop() -> None:
    registry = ToolRegistry()

    async def doubler(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    registry.add(ToolSpec(name="doubler", description="x*2"), doubler)

    # We are inside a running event loop here (pytest-asyncio).
    # The current implementation calls asyncio.run(...) which raises
    # RuntimeError. After the fix, this must succeed.
    result = registry.execute(ToolCall(id="1", name="doubler", arguments={"x": 21}))

    assert result.ok, f"Expected ok=True, got error: {result.error}"
    assert result.result == 42


@pytest.mark.asyncio
async def test_async_tool_failure_propagates_inside_running_loop() -> None:
    registry = ToolRegistry()

    async def fails() -> None:
        raise ValueError("boom")

    registry.add(ToolSpec(name="fails", description="fail"), fails)

    result = registry.execute(ToolCall(id="1", name="fails", arguments={}))

    assert not result.ok
    assert result.error is not None
    assert "boom" in result.error
