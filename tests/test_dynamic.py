"""Tests for the dynamic tool builder.

DynamicToolBuilder asks an LLM to emit a JSON payload describing a new
tool (metadata + Python code + self-tests). The output goes through:

    parse JSON -> sanitize name -> upsert TOOL metadata into code -> validate
    code with the sandbox AST walker -> write file -> load + self-test

These tests pin each step individually plus an end-to-end build using a
fake provider, because the live LLM cannot be relied on in CI.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from autoagent.dynamic import (
    DynamicToolBuilder,
    ToolBuildRequest,
    _extract_first_json_object,
    _parse_json_object,
    _safe_tool_name,
    _upsert_tool_metadata,
)
from autoagent.errors import ToolValidationError
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedProvider:
    """Minimal LLMProvider replacement that returns a pre-baked response."""

    def __init__(self, content: str) -> None:
        self.config = ModelConfig(provider="fake", model="fake")
        self.content = content
        self.last_request: LLMRequest | None = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(content=self.content, model="fake")


# ---------------------------------------------------------------------------
# _safe_tool_name
# ---------------------------------------------------------------------------


class TestSafeToolName:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("My Tool", "my_tool"),
            ("calc-sum-v2", "calc_sum_v2"),
            ("  leading and trailing  ", "leading_and_trailing"),
            ("read-FILE", "read_file"),
        ],
    )
    def test_sanitizes_to_snake_case(self, raw: str, expected: str) -> None:
        assert _safe_tool_name(raw) == expected

    def test_digit_prefix_gets_tool_prefix(self) -> None:
        assert _safe_tool_name("123abc").startswith("tool_")

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ToolValidationError, match="empty"):
            _safe_tool_name("   !!!  ")

    def test_truncates_at_64_chars(self) -> None:
        assert len(_safe_tool_name("a" * 200)) == 64


# ---------------------------------------------------------------------------
# _extract_first_json_object — the LLM-output parser
# ---------------------------------------------------------------------------


class TestExtractFirstJsonObject:
    def test_returns_full_object_when_text_is_pure_json(self) -> None:
        text = '{"a": 1, "b": 2}'
        assert _extract_first_json_object(text) == text

    def test_skips_prose_around_json(self) -> None:
        text = 'Here is the result:\n```json\n{"a": 1}\n```\nThanks.'
        assert _extract_first_json_object(text) == '{"a": 1}'

    def test_brace_inside_string_does_not_close_block(self) -> None:
        text = 'noise {"msg": "has } inside"} tail'
        assert _extract_first_json_object(text) == '{"msg": "has } inside"}'

    def test_escaped_quote_inside_string(self) -> None:
        text = 'pre {"msg": "she said \\"hi\\""} post'
        assert _extract_first_json_object(text) == '{"msg": "she said \\"hi\\""}'

    def test_no_braces_returns_none(self) -> None:
        assert _extract_first_json_object("just some text") is None

    def test_unbalanced_braces_returns_none(self) -> None:
        assert _extract_first_json_object("noise {open without close") is None


# ---------------------------------------------------------------------------
# _parse_json_object
# ---------------------------------------------------------------------------


class TestParseJsonObject:
    def test_pure_json(self) -> None:
        assert _parse_json_object('{"a": 1}') == {"a": 1}

    def test_extracts_embedded_json(self) -> None:
        result = _parse_json_object('prose...\n{"a": 1, "b": [2,3]}\ntrailing')
        assert result == {"a": 1, "b": [2, 3]}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ToolValidationError, match="did not contain JSON"):
            _parse_json_object("not json at all")

    def test_malformed_extracted_json_raises(self) -> None:
        # First-pass loads fails; extractor finds something that *looks* like
        # JSON but is invalid — must surface a clear error.
        with pytest.raises(ToolValidationError, match="not valid JSON"):
            _parse_json_object('prose {"a": 1,}')


# ---------------------------------------------------------------------------
# _upsert_tool_metadata
# ---------------------------------------------------------------------------


class TestUpsertToolMetadata:
    def test_prepends_tool_when_absent(self) -> None:
        code = "def run(args, context):\n    return None\n"
        out = _upsert_tool_metadata(
            code,
            {"name": "x", "description": "y", "input_schema": {"type": "object"}},
        )
        assert out.startswith("TOOL = ")
        assert "def run" in out

    def test_replaces_existing_tool_block(self) -> None:
        code = (
            textwrap.dedent("""
            TOOL = {"name": "old", "description": "old", "input_schema": {}}

            def run(args, context):
                return None
        """).strip()
            + "\n"
        )
        out = _upsert_tool_metadata(
            code,
            {"name": "new_name", "description": "new desc", "input_schema": {"type": "object"}},
        )
        assert "old" not in out
        assert "new_name" in out
        assert "def run" in out

    def test_missing_required_metadata_keys_rejected(self) -> None:
        with pytest.raises(ToolValidationError, match="missing keys"):
            _upsert_tool_metadata("def run(args, context): return None\n", {"name": "x"})

    def test_handles_syntax_error_by_prepending(self) -> None:
        # If the code can't be parsed, we still prepend the TOOL block.
        out = _upsert_tool_metadata(
            "def run(args, context):\n    return [1, 2,\n",
            {"name": "x", "description": "y", "input_schema": {}},
        )
        assert out.startswith("TOOL = ")


# ---------------------------------------------------------------------------
# DynamicToolBuilder.build — end-to-end happy path
# ---------------------------------------------------------------------------


def _good_builder_payload() -> str:
    return json.dumps(
        {
            "tool": {
                "name": "double",
                "description": "double a number",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
                "permissions": [],
            },
            "code": textwrap.dedent("""
                def run(args, context):
                    return {"out": args["x"] * 2}
            """).strip(),
            "self_tests": [
                {"args": {"x": 5}, "expect_equals": {"out": 10}},
                {"args": {"x": 0}, "expect_contains": "out"},
            ],
        }
    )


class TestBuildEndToEnd:
    @pytest.mark.timeout(30)
    def test_build_writes_file_and_returns_tool(self, tmp_path: Path) -> None:
        provider = _ScriptedProvider(_good_builder_payload())
        builder = DynamicToolBuilder(provider, tools_dir=tmp_path, timeout=15)

        tool = builder.build(ToolBuildRequest(capability="double a number"))

        assert tool.spec.name == "double"
        assert tool.file_path.exists()
        assert tool.file_path.parent == tmp_path
        content = tool.file_path.read_text(encoding="utf-8")
        assert "TOOL = " in content
        assert "def run" in content

    def test_build_rejects_payload_without_code(self, tmp_path: Path) -> None:
        bad = json.dumps({"tool": {"name": "x", "description": "y", "input_schema": {}}, "code": ""})
        builder = DynamicToolBuilder(_ScriptedProvider(bad), tools_dir=tmp_path)
        with pytest.raises(ToolValidationError, match="did not return code"):
            builder.build(ToolBuildRequest(capability="x"))

    @pytest.mark.timeout(30)
    def test_build_failing_self_test_raises(self, tmp_path: Path) -> None:
        payload = json.loads(_good_builder_payload())
        payload["self_tests"] = [{"args": {"x": 5}, "expect_equals": {"out": 999}}]
        builder = DynamicToolBuilder(_ScriptedProvider(json.dumps(payload)), tools_dir=tmp_path, timeout=15)
        with pytest.raises(ToolValidationError, match="Self-test"):
            builder.build(ToolBuildRequest(capability="double"))

    def test_build_rejects_code_with_banned_call(self, tmp_path: Path) -> None:
        payload: dict[str, Any] = json.loads(_good_builder_payload())
        payload["code"] = "def run(args, context):\n    return eval(args['x'])\n"
        builder = DynamicToolBuilder(_ScriptedProvider(json.dumps(payload)), tools_dir=tmp_path)
        with pytest.raises(ToolValidationError, match="not allowed: eval"):
            builder.build(ToolBuildRequest(capability="x"))

    def test_build_overrides_name_when_requested(self, tmp_path: Path) -> None:
        payload: dict[str, Any] = json.loads(_good_builder_payload())
        # Drop self_tests to keep this fast; we only verify the override path.
        payload["self_tests"] = []
        builder = DynamicToolBuilder(_ScriptedProvider(json.dumps(payload)), tools_dir=tmp_path, timeout=15)
        tool = builder.build(ToolBuildRequest(capability="x", tool_name="custom-name"))
        assert tool.spec.name == "custom_name"
