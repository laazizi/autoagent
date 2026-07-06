from __future__ import annotations

import ast
import json
import pprint
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ToolValidationError
from .providers.base import LLMProvider
from .sandbox import GeneratedPythonTool, SubprocessSandbox, load_generated_tool, validate_generated_tool_code
from .schema import LLMRequest, Message

__all__ = ["DynamicToolBuilder", "ToolBuildRequest"]


@dataclass
class ToolBuildRequest:
    capability: str
    tool_name: str | None = None
    input_schema: dict[str, Any] | None = None
    permissions: list[str] = field(default_factory=list)


class DynamicToolBuilder:
    """Generates a new Python tool by asking an LLM, validates it, and stores it.

    Flow per `build(request)`:
        1. Ask the provider for a single JSON object describing the new
           tool (metadata + Python source + self-tests).
        2. Sanitize the tool name and merge any caller-imposed
           name / input_schema / permissions overrides.
        3. Walk the AST via `validate_generated_tool_code` to reject
           dangerous calls, dangerous identifiers, and disallowed imports.
        4. Write the code to `tools_dir/<name>.py`.
        5. Execute the declared self-tests inside the sandbox.
        6. Return a `GeneratedPythonTool` that, when called, runs the
           code in an isolated subprocess (`SubprocessSandbox`).

    The host should treat dynamically generated tools as untrusted: even
    after passing the AST validator, they run in `-I -S` Python with an
    empty `env={}` and no shared filesystem cwd beyond the tools dir.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        tools_dir: str | Path = ".autoagent/tools",
        sandbox: SubprocessSandbox | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.provider = provider
        self.tools_dir = Path(tools_dir)
        self.sandbox = sandbox or SubprocessSandbox(timeout=timeout)

    def build(self, request: ToolBuildRequest) -> GeneratedPythonTool:
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        response = self.provider.complete(
            LLMRequest(
                messages=[
                    Message(role="system", content=self._system_prompt()),
                    Message(role="user", content=self._user_prompt(request)),
                ],
                temperature=0,
                # 8192 : laisse de la marge aux modèles « thinking » (Gemini 2.5)
                # dont la réflexion consomme des tokens avant le JSON de sortie.
                max_tokens=8192,
                tool_choice="none",
                # JSON mode natif quand le provider le supporte (OpenAI/
                # DeepSeek/Gemini) : supprime les balises ```json à la source.
                # _parse_json_object reste tolérant (Anthropic = best effort).
                response_format={"type": "json_object"},
            )
        )
        payload = _parse_json_object(response.content)
        tool_meta = payload.get("tool") or {}
        code = payload.get("code")
        tests = payload.get("self_tests") or []

        if not isinstance(code, str) or not code.strip():
            raise ToolValidationError("Tool builder did not return code")

        if request.tool_name:
            requested_name = _safe_tool_name(request.tool_name)
            tool_meta["name"] = requested_name
        if request.input_schema:
            tool_meta["input_schema"] = request.input_schema
        if request.permissions:
            tool_meta["permissions"] = request.permissions

        tool_meta["name"] = _safe_tool_name(tool_meta["name"])
        tool_meta.setdefault("permissions", [])
        code = _upsert_tool_metadata(code, tool_meta)
        permissions = tool_meta.get("permissions") or []
        validate_generated_tool_code(code, permissions=permissions)

        name = tool_meta["name"]
        file_path = self.tools_dir / f"{name}.py"
        file_path.write_text(code, encoding="utf-8")

        generated_tool = load_generated_tool(file_path, sandbox=self.sandbox)
        self._run_self_tests(generated_tool, tests)
        return generated_tool

    def _run_self_tests(self, tool: GeneratedPythonTool, tests: list[dict[str, Any]]) -> None:
        for index, test in enumerate(tests[:5]):
            args = test.get("args") or {}
            result = tool(**args)
            if "expect_equals" in test and result != test["expect_equals"]:
                raise ToolValidationError(
                    f"Self-test {index} failed: expected {test['expect_equals']!r}, got {result!r}"
                )
            if "expect_contains" in test and test["expect_contains"] not in str(result):
                raise ToolValidationError(
                    f"Self-test {index} failed: {test['expect_contains']!r} not in {result!r}"
                )

    def _system_prompt(self) -> str:
        return (
            "You create small Python tools for an AI agent. Return only one JSON object. "
            "No markdown. The JSON must have keys: tool, code, self_tests. "
            "tool must contain name, description, input_schema, permissions. "
            "code must be a complete Python module defining TOOL = {...} and "
            "def run(args, context): ... . The run function must return JSON-serializable data. "
            "Do not use network, filesystem, shell, eval, exec, or subprocess unless the requested "
            "permissions explicitly allow it. Prefer pure Python and standard library only."
        )

    def _user_prompt(self, request: ToolBuildRequest) -> str:
        return json.dumps(
            {
                "capability": request.capability,
                "preferred_tool_name": request.tool_name,
                "input_schema": request.input_schema,
                "permissions": request.permissions,
                "self_test_format": [
                    {"args": {"example": "value"}, "expect_contains": "value"},
                    {"args": {"example": "value"}, "expect_equals": {"ok": True}},
                ],
            },
            ensure_ascii=False,
        )


def _strip_code_fences(text: str) -> str:
    """LLMs often wrap JSON in a ```json … ``` fence despite being told not to.
    Strip a leading ```/```json line and a trailing ``` so json.loads can parse."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n?", "", t)
        t = re.sub(r"\r?\n?[ \t]*```$", "", t)
    return t.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed: dict[str, Any] = json.loads(text)
        return parsed
    except json.JSONDecodeError:
        pass

    # Markdown-fenced JSON (```json … ```) is the most common deviation — strip
    # the fence and retry strict parsing before the brace-walking fallback.
    stripped = _strip_code_fences(text)
    if stripped != text.strip():
        try:
            parsed_fenced: dict[str, Any] = json.loads(stripped)
            return parsed_fenced
        except json.JSONDecodeError:
            pass

    candidate = _extract_first_json_object(stripped)
    if candidate is None:
        raise ToolValidationError("Tool builder response did not contain JSON")
    try:
        parsed_candidate: dict[str, Any] = json.loads(candidate)
        return parsed_candidate
    except json.JSONDecodeError as exc:
        raise ToolValidationError(f"Tool builder response is not valid JSON: {exc}") from exc


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced top-level {...} block in text, or None.

    Walks the string tracking brace depth while skipping over string literals
    so braces inside JSON strings don't throw off the count.

    Quotes that appear in surrounding prose (before any `{` was seen) are
    treated as plain text, not as the start of a JSON string. Otherwise a
    quote in the LLM's natural-language preamble would cause the parser to
    miss the actual JSON block that follows.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, char in enumerate(text):
        if depth == 0:
            if char == "{":
                start = i
                depth = 1
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def _safe_tool_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()
    if not safe:
        raise ToolValidationError("Tool name is empty")
    if safe[0].isdigit():
        safe = f"tool_{safe}"
    return safe[:64]


def _upsert_tool_metadata(code: str, tool_meta: dict[str, Any]) -> str:
    required = {"name", "description", "input_schema"}
    missing = required - set(tool_meta)
    if missing:
        raise ToolValidationError(f"Tool metadata missing keys: {sorted(missing)}")
    metadata = pprint.pformat(tool_meta, sort_dicts=False, width=100)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f"TOOL = {metadata}\n\n{code}"

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOL":
                    lines = code.splitlines()
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", node.lineno)
                    replacement = f"TOOL = {metadata}".splitlines()
                    updated = lines[:start] + replacement + lines[end:]
                    return "\n".join(updated) + "\n"
    return f"TOOL = {metadata}\n\n{code}"
