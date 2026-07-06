from __future__ import annotations

import inspect
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .agent import Agent
from .errors import ToolError
from .pipeline import PipelineManager
from .registry import schema_from_callable
from .workspace import ProjectWorkspace

__all__ = [
    "EVOLUTION_CAPABILITIES",
    "EVOLUTION_SYSTEM_PROMPT",
    "EvolutionRuntime",
    "enable_software_evolution",
]

EVOLUTION_SYSTEM_PROMPT = """

You can evolve a target software workspace through controlled tools.
Inspect the existing state before changing files or pipeline slots.
Do not create or rewrite code unless the user request requires it.
Prefer adding focused modules/plugins over editing core code when the pipeline supports it.
After a change, run validation if a validation command is available.
If validation fails because of your change, rollback or make a smaller corrective change.
"""

EVOLUTION_CAPABILITIES: tuple[str, ...] = (
    "read",
    "write",
    "host_state",
    "host_call",
    "pipeline",
    "validate",
)


class EvolutionRuntime:
    """Controlled surface that lets an LLM agent modify a target application.

    The runtime wires a curated tool set onto an `Agent`. Each capability
    is OPT-IN via the `capabilities` argument of `register_tools`:

        * `read`        — list/read files, list pipeline slots, list changes.
        * `write`       — write/replace files and roll back changes.
        * `host_state`  — expose `state_reader` snapshot to the agent.
        * `host_call`   — call functions registered via
                          `register_host_function`.
        * `pipeline`    — replace pipeline slots (hot-swap of
                          {module, callable, config}).
        * `validate`    — run the configured validation command.

    Security boundaries:
        * Custom validation commands from the model are REJECTED unless
          `allow_custom_validation_command=True`. Default behaviour: the
          agent can only invoke the pre-configured `validation_command`.
        * `call_host_function` only accepts names that the host has
          explicitly registered via `register_host_function`.
        * All file mutations go through `ProjectWorkspace`, so every
          edit is rollback-able.

    Thread-safety:
        Not designed for concurrent `register_tools` calls on the same
        agent. `enable_software_evolution` IS idempotent on a given
        agent: calling it twice is a no-op.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        pipeline_path: str | None = None,
        validation_command: str | list[str] | None = None,
        allow_custom_validation_command: bool = False,
        allowed_write_extensions: set[str] | None = None,
        state_reader: Callable[[], Any] | None = None,
        max_write_chars: int = 200000,
    ) -> None:
        self.workspace = ProjectWorkspace(
            workspace_root,
            allowed_write_extensions=allowed_write_extensions,
            max_write_chars=max_write_chars,
        )
        self.pipeline = PipelineManager(self.workspace, pipeline_path or "pipeline.json")
        self.validation_command = validation_command
        self.allow_custom_validation_command = allow_custom_validation_command
        self.state_reader = state_reader
        self.host_functions: dict[str, Callable[..., Any]] = {}

    def register_host_function(self, name: str, func: Callable[..., Any]) -> None:
        if not name:
            raise ToolError("Host function name cannot be empty")
        self.host_functions[name] = func

    def register_tools(
        self,
        agent: Agent,
        capabilities: set[str] | None = None,
    ) -> None:
        active = self._resolve_capabilities(capabilities)
        # Idempotency: a host that calls `enable_software_evolution` twice
        # (e.g. on hot-reload) must not crash with "Tool already
        # registered". We temporarily wrap `agent.tool` so any name that
        # is already in the registry becomes a no-op decorator.
        existing = {spec.name for spec in agent.registry.specs()}
        original_tool = agent.tool

        def tool_or_skip(*args: Any, name: str | None = None, **kwargs: Any) -> Any:
            if name and name in existing:
                return lambda handler: handler
            if name:
                existing.add(name)
            return original_tool(*args, name=name, **kwargs)

        agent.tool = tool_or_skip  # type: ignore[method-assign]
        try:
            self._register_all(agent, active)
        finally:
            agent.tool = original_tool  # type: ignore[method-assign]

    def _register_all(self, agent: Agent, active: set[str]) -> None:
        if "read" in active:
            agent.tool(
                name="list_project_files",
                description="List files in the target software workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "default": "**/*"},
                        "max_files": {"type": "integer", "default": 200},
                    },
                    "additionalProperties": False,
                },
            )(self.list_project_files)
            agent.tool(
                name="read_project_file",
                description="Read a text file from the target software workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_chars": {"type": "integer", "default": 50000},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            )(self.read_project_file)
            agent.tool(
                name="list_pipeline_slots",
                description="List configurable pipeline slots for the target software.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            )(self.list_pipeline_slots)
            agent.tool(
                name="get_pipeline_slot",
                description="Read one pipeline slot configuration.",
                input_schema={
                    "type": "object",
                    "properties": {"slot": {"type": "string"}},
                    "required": ["slot"],
                    "additionalProperties": False,
                },
            )(self.get_pipeline_slot)
            agent.tool(
                name="list_changes",
                description="List workspace changes recorded by the evolution runtime.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            )(self.list_changes)

        if "write" in active:
            agent.tool(
                name="write_project_file",
                description="Create or replace one file in the target software workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                permissions=["filesystem.write"],
            )(self.write_project_file)
            agent.tool(
                name="replace_project_text",
                description="Replace exact text in one workspace file while recording a rollback change.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                        "count": {"type": "integer", "default": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["path", "old", "new"],
                    "additionalProperties": False,
                },
                permissions=["filesystem.write"],
            )(self.replace_project_text)
            agent.tool(
                name="rollback_change",
                description="Rollback a recorded change and all changes made after it.",
                input_schema={
                    "type": "object",
                    "properties": {"change_id": {"type": "string"}},
                    "required": ["change_id"],
                    "additionalProperties": False,
                },
                permissions=["filesystem.write"],
            )(self.rollback_change)
            agent.tool(
                name="rollback_last_change",
                description="Rollback the last recorded workspace change.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                permissions=["filesystem.write"],
            )(self.rollback_last_change)

        if "host_state" in active:
            agent.tool(
                name="get_runtime_state",
                description="Read live state exposed by the host application, if available.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            )(self.get_runtime_state)

        if "host_call" in active:
            agent.tool(
                name="list_host_functions",
                description="List host application functions exposed to the agent.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            )(self.list_host_functions)
            agent.tool(
                name="call_host_function",
                description="Call one exposed host application function with JSON arguments.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            )(self.call_host_function)

        if "pipeline" in active:
            agent.tool(
                name="replace_pipeline_slot",
                description="Replace one pipeline slot with a module/callable reference.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "slot": {"type": "string"},
                        "module": {"type": "string"},
                        "callable_name": {"type": "string", "default": "run"},
                        "config": {"type": "object"},
                        "reason": {"type": "string"},
                    },
                    "required": ["slot", "module"],
                    "additionalProperties": False,
                },
                permissions=["pipeline.write"],
            )(self.replace_pipeline_slot)

        if "validate" in active:
            agent.tool(
                name="run_validation",
                description="Run the configured validation command for the target software.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "default": 60},
                    },
                    "additionalProperties": False,
                },
            )(self.run_validation)

    @staticmethod
    def _resolve_capabilities(capabilities: set[str] | None) -> set[str]:
        if capabilities is None:
            return set(EVOLUTION_CAPABILITIES)
        unknown = set(capabilities) - set(EVOLUTION_CAPABILITIES)
        if unknown:
            raise ToolError(
                f"Unknown evolution capabilities: {sorted(unknown)}. "
                f"Valid: {list(EVOLUTION_CAPABILITIES)}"
            )
        return set(capabilities)

    def list_project_files(self, pattern: str = "**/*", max_files: int = 200) -> dict[str, Any]:
        return self.workspace.list_files(pattern=pattern, max_files=max_files)

    def read_project_file(self, path: str, max_chars: int = 50000) -> dict[str, Any]:
        return self.workspace.read_file(path, max_chars=max_chars)

    def write_project_file(self, path: str, content: str, reason: str = "") -> dict[str, Any]:
        return self.workspace.write_file(path, content, reason=reason)

    def replace_project_text(
        self,
        path: str,
        old: str,
        new: str,
        count: int = 1,
        reason: str = "",
    ) -> dict[str, Any]:
        return self.workspace.replace_text(path, old, new, count=count, reason=reason)

    def get_runtime_state(self) -> dict[str, Any]:
        if self.state_reader is None:
            return {"available": False, "state": None}
        return {"available": True, "state": self.state_reader()}

    def list_host_functions(self) -> dict[str, Any]:
        return {
            "functions": [
                self._describe_host_function(name, func) for name, func in sorted(self.host_functions.items())
            ]
        }

    def _describe_host_function(self, name: str, func: Callable[..., Any]) -> dict[str, Any]:
        try:
            input_schema = schema_from_callable(func)
            signature = str(inspect.signature(func))
        except (TypeError, ValueError):
            input_schema = {"type": "object", "properties": {}, "additionalProperties": True}
            signature = "(...)"
        return {
            "name": name,
            "signature": signature,
            "input_schema": input_schema,
            "description": inspect.getdoc(func) or "",
        }

    def call_host_function(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        func = self.host_functions.get(name)
        if func is None:
            raise ToolError(f"Unknown host function: {name}")
        result = func(**(arguments or {}))
        return {"ok": True, "result": result}

    def list_pipeline_slots(self) -> dict[str, Any]:
        return self.pipeline.list_slots()

    def get_pipeline_slot(self, slot: str) -> dict[str, Any]:
        return self.pipeline.get_slot(slot)

    def replace_pipeline_slot(
        self,
        slot: str,
        module: str,
        callable_name: str = "run",
        config: dict[str, Any] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        return self.pipeline.replace_slot(
            slot,
            module,
            callable_name=callable_name,
            config=config,
            reason=reason,
        )

    def run_validation(self, command: str | None = None, timeout: int = 60) -> dict[str, Any]:
        selected_command = self._validation_command(command)
        try:
            completed = subprocess.run(
                selected_command,
                cwd=str(self.workspace.root),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "timeout": True, "error": f"validation timed out after {timeout}s"}
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
        }

    def list_changes(self) -> dict[str, Any]:
        return self.workspace.list_changes()

    def rollback_change(self, change_id: str) -> dict[str, Any]:
        return self.workspace.rollback_change(change_id)

    def rollback_last_change(self) -> dict[str, Any]:
        return self.workspace.rollback_last_change()

    def _validation_command(self, command: str | None) -> list[str]:
        if command:
            if not self.allow_custom_validation_command:
                raise ToolError("Custom validation commands are disabled for this runtime")
            return shlex.split(command)
        if self.validation_command is None:
            raise ToolError("No validation command configured")
        if isinstance(self.validation_command, str):
            return shlex.split(self.validation_command)
        return list(self.validation_command)


def enable_software_evolution(
    agent: Agent,
    runtime: EvolutionRuntime,
    *,
    capabilities: set[str] | None = None,
) -> EvolutionRuntime:
    runtime.register_tools(agent, capabilities=capabilities)
    if EVOLUTION_SYSTEM_PROMPT.strip() not in agent.system_prompt:
        agent.system_prompt = f"{agent.system_prompt.rstrip()}{EVOLUTION_SYSTEM_PROMPT}"
    return runtime
