"""Red tests for JSON Schema validation of tool arguments.

CURRENT BUG (autoagent/registry.py:35-41):
    RegisteredTool.execute() passes `args` straight to the handler without
    validating against `spec.input_schema`. A misbehaving LLM can crash the
    handler (wrong type), pass extra fields, or omit required ones, and the
    resulting error message bubbles up as a generic Python exception
    instead of a structured validation error.

These tests are RED until validation is added. They will turn GREEN once
RegisteredTool.execute validates `args` with jsonschema before calling
the handler.
"""

from __future__ import annotations

from autoagent.registry import ToolRegistry
from autoagent.schema import ToolCall, ToolSpec


def _add_tool(registry: ToolRegistry) -> None:
    def add(a: int, b: int) -> int:
        return a + b

    spec = ToolSpec(
        name="add",
        description="Add two integers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )
    registry.add(spec, add)


class TestToolArgsValidation:
    def test_wrong_type_is_rejected_before_handler(self) -> None:
        """Handler must NOT run; error must mention schema validation."""
        calls: list[dict] = []

        def add(a: int, b: int) -> int:
            calls.append({"a": a, "b": b})
            return a + b

        registry = ToolRegistry()
        registry.add(
            ToolSpec(
                name="add",
                description="Add",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            ),
            add,
        )

        result = registry.execute(
            ToolCall(id="1", name="add", arguments={"a": "not-an-int", "b": 3}),
        )
        assert not result.ok
        # Strict invariant: handler must never observe invalid args.
        assert calls == [], (
            "Handler ran with invalid arguments — validation must happen " "before the handler is called"
        )

    def test_missing_required_field_is_rejected(self) -> None:
        calls: list[dict] = []

        def add(a: int, b: int) -> int:
            calls.append({"a": a, "b": b})
            return a + b

        registry = ToolRegistry()
        registry.add(
            ToolSpec(
                name="add",
                description="Add",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            ),
            add,
        )

        result = registry.execute(ToolCall(id="1", name="add", arguments={"a": 1}))
        assert not result.ok
        assert calls == [], "Handler must not run when required field is missing"

    def test_unknown_field_is_rejected_when_additional_properties_false(self) -> None:
        calls: list[dict] = []

        def add(a: int, b: int) -> int:
            calls.append({"a": a, "b": b})
            return a + b

        registry = ToolRegistry()
        registry.add(
            ToolSpec(
                name="add",
                description="Add",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            ),
            add,
        )

        result = registry.execute(
            ToolCall(id="1", name="add", arguments={"a": 1, "b": 2, "c": 99}),
        )
        assert not result.ok
        assert calls == [], "Handler must not run when extra field present"

    def test_valid_args_pass_through(self) -> None:
        registry = ToolRegistry()
        _add_tool(registry)

        result = registry.execute(
            ToolCall(id="1", name="add", arguments={"a": 2, "b": 3}),
        )
        assert result.ok
        assert result.result == 5

    def test_handler_is_not_called_when_validation_fails(self) -> None:
        calls: list[dict] = []

        def add(a: int, b: int) -> int:
            calls.append({"a": a, "b": b})
            return a + b

        spec = ToolSpec(
            name="add",
            description="Add",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        )
        registry = ToolRegistry()
        registry.add(spec, add)

        registry.execute(ToolCall(id="1", name="add", arguments={"a": "x", "b": 3}))

        assert calls == [], "Handler must not run when validation fails"
