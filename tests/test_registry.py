import asyncio
from typing import Any

import pytest

from autoagent.errors import ToolError
from autoagent.registry import (
    ToolRegistry,
    ToolResult,
    schema_from_annotation,
    schema_from_callable,
    tool,
)
from autoagent.schema import ToolCall, ToolSpec

# ---------------------------------------------------------------------------
# schema_from_callable
# ---------------------------------------------------------------------------


class TestSchemaFromCallable:
    def test_simple_function(self) -> None:
        def f(a: str, b: int) -> str: ...

        schema = schema_from_callable(f)
        assert schema["type"] == "object"
        assert schema["properties"]["a"] == {"type": "string"}
        assert schema["properties"]["b"] == {"type": "integer"}
        assert set(schema["required"]) == {"a", "b"}
        assert schema["additionalProperties"] is False

    def test_with_defaults_not_required(self) -> None:
        def f(a: str, b: int = 42) -> None: ...

        schema = schema_from_callable(f)
        assert schema["required"] == ["a"]

    def test_pep563_stringized_annotations(self) -> None:
        # Sous `from __future__ import annotations`, les annotations arrivent
        # comme des CHAÎNES ("int", "float"…). schema_from_callable doit les
        # résoudre (get_type_hints), sinon tout retombe en {"type":"string"}
        # et la validation casse. Régression trouvée via le record/replay.
        import textwrap

        ns: dict = {}
        exec(textwrap.dedent("""
            from __future__ import annotations
            def f(n: int, x: float, ok: bool, nom: str): ...
        """), ns)
        schema = schema_from_callable(ns["f"])
        assert schema["properties"]["n"] == {"type": "integer"}
        assert schema["properties"]["x"] == {"type": "number"}
        assert schema["properties"]["ok"] == {"type": "boolean"}
        assert schema["properties"]["nom"] == {"type": "string"}

    def test_context_param_excluded(self) -> None:
        def f(a: str, context: dict[str, Any] | None = None) -> str: ...

        schema = schema_from_callable(f)
        assert "context" not in schema["properties"]
        assert schema["required"] == ["a"]

    def test_list_and_dict_types(self) -> None:
        def f(items: list[str], config: dict[str, int]) -> None: ...

        schema = schema_from_callable(f)
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"
        assert schema["properties"]["config"]["type"] == "object"

    def test_literal_enum(self) -> None:
        from typing import Literal

        def f(direction: Literal["north", "south", "east", "west"]) -> None: ...

        schema = schema_from_callable(f)
        prop = schema["properties"]["direction"]
        assert prop["type"] == "string"
        assert set(prop["enum"]) == {"north", "south", "east", "west"}

    def test_optional_type(self) -> None:
        def f(name: str | None = None) -> None: ...

        schema = schema_from_callable(f)
        prop = schema["properties"]["name"]
        assert prop["type"] == ["string", "null"]

    def test_no_annotations(self) -> None:
        def f(a, b):  # type: ignore[no-untyped-def]
            ...

        schema = schema_from_callable(f)
        assert schema["properties"]["a"] == {"type": "string"}
        assert schema["properties"]["b"] == {"type": "string"}


# ---------------------------------------------------------------------------
# schema_from_annotation edge cases
# ---------------------------------------------------------------------------


class TestSchemaFromAnnotation:
    def test_bool(self) -> None:
        assert schema_from_annotation(bool) == {"type": "boolean"}

    def test_float(self) -> None:
        assert schema_from_annotation(float) == {"type": "number"}

    def test_none(self) -> None:
        assert schema_from_annotation(type(None)) == {"type": "null"}

    def test_union_mixed(self) -> None:
        from typing import Union

        schema = schema_from_annotation(Union[str, int, None])
        assert "anyOf" in schema
        types = [s.get("type") for s in schema["anyOf"] if "type" in s]
        assert "string" in types
        assert "integer" in types
        assert "null" in types


# ---------------------------------------------------------------------------
# ToolRegistry: registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_register_via_decorator(self) -> None:
        registry = ToolRegistry()

        @registry.register(name="hello", description="Say hello")
        def hello(name: str) -> str:
            return f"Hello {name}"

        assert "hello" in registry
        spec = registry._tools["hello"].spec
        assert spec.name == "hello"
        assert spec.description == "Say hello"

    def test_register_via_add(self) -> None:
        registry = ToolRegistry()

        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}"

        spec = ToolSpec(name="greet", description="Greet someone")
        registry.add(spec, greet)
        assert "greet" in registry

    def test_duplicate_name_raises(self) -> None:
        registry = ToolRegistry()

        def one() -> str:
            return "one"

        def two() -> str:
            return "two"

        spec = ToolSpec(name="dup", description="First")
        registry.add(spec, one)

        with pytest.raises(ToolError, match="already registered"):
            registry.add(ToolSpec(name="dup", description="Second"), two)

    def test_replace_overwrites(self) -> None:
        registry = ToolRegistry()

        def original() -> str:
            return "old"

        def replacement() -> str:
            return "new"

        spec = ToolSpec(name="t1", description="Original")
        registry.add(spec, original)
        registry.replace(ToolSpec(name="t1", description="Replaced"), replacement)
        assert registry._tools["t1"].handler is replacement

    def test_specs_returns_all(self) -> None:
        registry = ToolRegistry()

        def a() -> None: ...

        def b() -> None: ...

        registry.add(ToolSpec(name="a", description="A"), a)
        registry.add(ToolSpec(name="b", description="B"), b)
        names = {s.name for s in registry.specs()}
        assert names == {"a", "b"}

    def test_auto_register_name_from_func(self) -> None:
        registry = ToolRegistry()

        @registry.register
        def my_tool(x: int) -> int:
            """My docstring."""
            return x * 2

        assert "my_tool" in registry


# ---------------------------------------------------------------------------
# RegisteredTool.execute
# ---------------------------------------------------------------------------


class TestToolExecution:
    def test_successful_execution(self) -> None:
        reg = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        reg.add(ToolSpec(name="add", description="Add"), add)
        result = reg.execute(ToolCall(id="1", name="add", arguments={"a": 2, "b": 3}))
        assert result.ok
        assert result.result == 5

    def test_execution_with_context(self) -> None:
        reg = ToolRegistry()

        def whoami(context: dict[str, Any] | None = None) -> str:
            ctx = context or {}
            return ctx.get("user", "anonymous")

        reg.add(ToolSpec(name="whoami", description="Who am I"), whoami)
        result = reg.execute(
            ToolCall(id="1", name="whoami", arguments={}),
            context={"user": "claude"},
        )
        assert result.ok
        assert result.result == "claude"

    def test_unknown_tool(self) -> None:
        reg = ToolRegistry()
        result = reg.execute(ToolCall(id="1", name="nope", arguments={}))
        assert not result.ok
        assert "Unknown tool" in result.error

    def test_tool_raises_exception(self) -> None:
        reg = ToolRegistry()

        def crash() -> None:
            raise ValueError("boom")

        reg.add(ToolSpec(name="crash", description="Boom"), crash)
        result = reg.execute(ToolCall(id="1", name="crash", arguments={}))
        assert not result.ok
        assert "ValueError: boom" in result.error

    def test_async_tool(self) -> None:
        reg = ToolRegistry()

        async def async_tool(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 3

        reg.add(ToolSpec(name="async_tool", description="Async"), async_tool)
        result = reg.execute(ToolCall(id="1", name="async_tool", arguments={"x": 7}))
        assert result.ok
        assert result.result == 21


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_ok_json(self) -> None:
        result = ToolResult(ok=True, result=42)
        msg = result.to_message_content()
        assert '"ok": true' in msg
        assert "42" in msg

    def test_error_json(self) -> None:
        result = ToolResult(ok=False, error="something went wrong")
        msg = result.to_message_content()
        assert '"ok": false' in msg
        assert "something went wrong" in msg


# ---------------------------------------------------------------------------
# standalone tool() decorator
# ---------------------------------------------------------------------------


class TestStandaloneToolDecorator:
    def test_sets_spec_attribute(self) -> None:
        @tool(name="standalone", description="A standalone tool")
        def standalone_func(x: int) -> int:
            return x

        spec = standalone_func.__autoagent_tool_spec__
        assert spec is not None
        assert spec.name == "standalone"
        assert spec.description == "A standalone tool"

    def test_add_function_uses_spec(self) -> None:
        @tool(name="pre_registered", description="Already has spec")
        def pre_registered_func(x: int) -> int:
            return x * 10

        registry = ToolRegistry()
        registry.add_function(pre_registered_func)
        result = registry.execute(ToolCall(id="1", name="pre_registered", arguments={"x": 5}))
        assert result.ok
        assert result.result == 50
