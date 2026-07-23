from __future__ import annotations

import asyncio
import inspect
import json
import threading
import types
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union, get_args, get_origin

import jsonschema
from jsonschema import Draft202012Validator

from .errors import ToolError
from .schema import JsonDict, ToolCall, ToolSpec

__all__ = [
    "RegisteredTool",
    "ToolHandler",
    "ToolRegistry",
    "ToolResult",
    "schema_from_annotation",
    "schema_from_callable",
    "tool",
]

ToolHandler = Callable[..., Any]


@dataclass
class ToolResult:
    ok: bool
    result: Any = None
    error: str | None = None

    def to_message_content(self) -> str:
        return json.dumps(
            {"ok": self.ok, "result": self.result, "error": self.error},
            ensure_ascii=False,
            default=repr,
        )


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler

    def __post_init__(self) -> None:
        # Hot-path caches, built ONCE at registration: recompiling the JSON
        # Schema validator and re-introspecting the handler signature on
        # every execute() was pure waste (both are static for a tool's
        # lifetime). `_schema_error` keeps the old behaviour of reporting an
        # invalid input_schema at call time rather than raising here.
        self._validator: Draft202012Validator | None = None
        self._schema_error: str | None = None
        if self.spec.input_schema:
            try:
                self._validator = Draft202012Validator(self.spec.input_schema)
            except jsonschema.SchemaError as exc:
                self._schema_error = f"SchemaError: tool input_schema is invalid: {exc.message}"
        try:
            self._wants_context = "context" in inspect.signature(self.handler).parameters
        except (TypeError, ValueError):  # builtins / exotic callables
            self._wants_context = False

    def execute(self, args: JsonDict, context: JsonDict | None = None) -> ToolResult:
        if self._schema_error is not None:
            return ToolResult(ok=False, error=self._schema_error)
        if self._validator is not None:
            errors = sorted(self._validator.iter_errors(args), key=lambda e: list(e.absolute_path))
            if errors:
                parts = [
                    (".".join(str(p) for p in err.absolute_path) or "<root>") + f": {err.message}"
                    for err in errors
                ]
                return ToolResult(ok=False, error="ValidationError: " + "; ".join(parts))

        try:
            if self._wants_context:
                value = self.handler(**args, context=context or {})
            else:
                value = self.handler(**args)
            if inspect.isawaitable(value):
                value = _run_awaitable(value)
            return ToolResult(ok=True, result=value)
        except Exception as exc:
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


def _run_awaitable(awaitable: Any) -> Any:
    """Run an awaitable to completion from a synchronous context, whether
    or not there is already a running event loop on the current thread.

    Background: `asyncio.run` cannot be called from a thread that has a
    running event loop (FastAPI handlers, Jupyter, aiohttp, etc.). When we
    detect that situation we run the coroutine on a fresh loop in a
    dedicated worker thread and block until it finishes. When no loop is
    running, we keep the cheap path: `asyncio.run`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    container: dict[str, Any] = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            container["result"] = loop.run_until_complete(awaitable)
        except BaseException as exc:
            container["error"] = exc
        finally:
            loop.close()

    thread_obj = threading.Thread(target=_worker, name="autoagent-sync-await", daemon=True)
    thread_obj.start()
    thread_obj.join()
    if "error" in container:
        raise container["error"]
    return container.get("result")


def _validate_args(args: JsonDict, input_schema: JsonDict | None) -> str | None:
    """Return a human-readable validation error, or None if args are valid.

    Validation happens BEFORE the handler is called so a malformed LLM call
    cannot crash user code with a misleading Python exception. The returned
    string is what the LLM sees as the tool error, so it must clearly point
    to the invalid field.
    """
    if not input_schema:
        return None
    try:
        validator = Draft202012Validator(input_schema)
    except jsonschema.SchemaError as exc:
        return f"SchemaError: tool input_schema is invalid: {exc.message}"
    errors = sorted(validator.iter_errors(args), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    parts: list[str] = []
    for err in errors:
        location = ".".join(str(p) for p in err.absolute_path) or "<root>"
        parts.append(f"{location}: {err.message}")
    return "ValidationError: " + "; ".join(parts)


class ToolRegistry:
    """Thread-safe registry of tools the agent can invoke.

    The internal `_tools` dict is read by `specs()`, `execute()`,
    `__contains__` and written by `add()` / `replace()`. All accesses are
    guarded by an RLock so the registry stays consistent under concurrent
    use (e.g. one thread serving requests, another hot-swapping a tool).
    Tool *execution itself* is not serialized — only the dict lookup is.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._lock = threading.RLock()

    def register(
        self,
        func: ToolHandler | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: JsonDict | None = None,
        permissions: list[str] | None = None,
        untrusted: bool = False,
    ):
        def decorator(handler: ToolHandler) -> ToolHandler:
            spec = ToolSpec(
                name=name or handler.__name__,
                description=description or inspect.getdoc(handler) or handler.__name__,
                input_schema=input_schema or schema_from_callable(handler),
                permissions=permissions or [],
                untrusted=untrusted,
            )
            self.add(spec, handler)
            return handler

        if func is not None:
            return decorator(func)
        return decorator

    def add_function(self, func: ToolHandler) -> ToolHandler:
        spec = getattr(func, "__autoagent_tool_spec__", None)
        if spec is None:
            spec = ToolSpec(
                name=func.__name__,
                description=inspect.getdoc(func) or func.__name__,
                input_schema=schema_from_callable(func),
                permissions=[],
            )
        self.add(spec, func)
        return func

    def add(self, spec: ToolSpec, handler: ToolHandler) -> None:
        with self._lock:
            if spec.name in self._tools:
                raise ToolError(f"Tool already registered: {spec.name}")
            self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def replace(self, spec: ToolSpec, handler: ToolHandler) -> None:
        with self._lock:
            self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def specs(self) -> list[ToolSpec]:
        with self._lock:
            return [tool.spec for tool in self._tools.values()]

    def execute(self, call: ToolCall, context: JsonDict | None = None) -> ToolResult:
        with self._lock:
            tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(ok=False, error=f"Unknown tool: {call.name}")
        return tool.execute(call.arguments, context=context)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tools


def tool(
    func: ToolHandler | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: JsonDict | None = None,
    permissions: list[str] | None = None,
    untrusted: bool = False,
):
    def decorator(handler: ToolHandler) -> ToolHandler:
        spec = ToolSpec(
            name=name or handler.__name__,
            description=description or inspect.getdoc(handler) or handler.__name__,
            input_schema=input_schema or schema_from_callable(handler),
            permissions=permissions or [],
            untrusted=untrusted,
        )
        handler.__autoagent_tool_spec__ = spec  # type: ignore[attr-defined]
        return handler

    if func is not None:
        return decorator(func)
    return decorator


def schema_from_callable(func: ToolHandler) -> JsonDict:
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if name == "context":
            continue
        # *args / **kwargs are not real LLM-facing parameters — including
        # them would advertise bogus string properties named "args"/"kwargs".
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        properties[name] = schema_from_annotation(parameter.annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    schema: JsonDict = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


_PRIMITIVE_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def schema_from_annotation(annotation: Any) -> JsonDict:
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    if annotation in _PRIMITIVE_JSON_TYPES:
        return {"type": _PRIMITIVE_JSON_TYPES[annotation]}
    if annotation is dict:
        return {"type": "object"}
    if annotation is list:
        return {"type": "array", "items": {}}
    if annotation is type(None):
        return {"type": "null"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return _literal_schema(args)

    if origin is Union or origin is types.UnionType:
        return _union_schema(args)

    if origin in {list, tuple, set, frozenset}:
        item_schema = schema_from_annotation(args[0]) if args else {}
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return {"type": "object"}
    if origin is None and hasattr(annotation, "__members__"):
        return {"type": "string", "enum": list(annotation.__members__.keys())}
    return {"type": "string"}


def _literal_schema(values: tuple[Any, ...]) -> JsonDict:
    type_to_json = {bool: "boolean", int: "integer", float: "number", str: "string"}
    json_types = {type_to_json[type(v)] for v in values if type(v) in type_to_json}
    schema: JsonDict = {"enum": list(values)}
    if len(json_types) == 1:
        schema["type"] = next(iter(json_types))
    return schema


def _union_schema(args: tuple[Any, ...]) -> JsonDict:
    non_none = [arg for arg in args if arg is not type(None)]
    nullable = len(non_none) != len(args)

    if len(non_none) == 1:
        schema = schema_from_annotation(non_none[0])
        if nullable and "type" in schema:
            schema = dict(schema)
            current = schema["type"]
            schema["type"] = [current, "null"] if isinstance(current, str) else [*list(current), "null"]
        return schema

    sub_schemas = [schema_from_annotation(arg) for arg in non_none]
    if nullable:
        sub_schemas.append({"type": "null"})
    return {"anyOf": sub_schemas}
