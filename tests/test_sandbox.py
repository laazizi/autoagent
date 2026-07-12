"""Tests for the sandbox security boundary.

The sandbox is the critical trust line between agent-generated code and
the host process. These tests pin the invariants that protect the host:

  * `validate_generated_tool_code` must REJECT eval/exec/compile/subprocess/
    ctypes/multiprocessing/signal/__import__/input/breakpoint and any
    network or filesystem access not explicitly granted by permissions.
  * `extract_tool_metadata` must reject TOOL blocks that are missing
    required keys or are not dicts.
  * `GeneratedPythonTool` must execute the generated `run(args, context)`
    in a subprocess and return the parsed result.

Any test failure here is a security regression and must block release.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from autoagent.errors import ToolError, ToolValidationError
from autoagent.sandbox import (
    GeneratedPythonTool,
    SubprocessSandbox,
    extract_tool_metadata,
    load_generated_tool,
    validate_generated_tool_code,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_run(body: str) -> str:
    """Wrap a snippet inside a minimal `run(args, context)` function so
    the AST walker can reach the calls/imports under test."""
    body_lines = textwrap.dedent(body).strip().splitlines() or ["pass"]
    indented = "\n".join("    " + line for line in body_lines)
    return f"def run(args, context):\n{indented}\n"


# ---------------------------------------------------------------------------
# validate_generated_tool_code — always-banned calls
# ---------------------------------------------------------------------------


class TestAlwaysBannedCalls:
    @pytest.mark.parametrize("call_name", ["eval", "exec", "compile", "__import__", "input", "breakpoint"])
    def test_banned_call_is_rejected(self, call_name: str) -> None:
        code = _wrap_run(f"{call_name}('payload')")
        with pytest.raises(ToolValidationError, match=f"not allowed: {call_name}"):
            validate_generated_tool_code(code)

    def test_shell_call_rejected(self) -> None:
        code = _wrap_run("os.system('rm -rf /')")
        with pytest.raises(ToolValidationError, match="Shell call is not allowed"):
            validate_generated_tool_code(code)


# ---------------------------------------------------------------------------
# validate_generated_tool_code — always-banned modules
# ---------------------------------------------------------------------------


class TestAlwaysBannedModules:
    @pytest.mark.parametrize(
        "module",
        ["subprocess", "ctypes", "multiprocessing", "signal"],
    )
    def test_banned_import_is_rejected(self, module: str) -> None:
        code = "import " + module + "\n" + _wrap_run("pass")
        with pytest.raises(ToolValidationError, match=f"not allowed: {module}"):
            validate_generated_tool_code(code)

    def test_submodule_of_banned_root_rejected(self) -> None:
        # `subprocess.run` import via from-import still uses the banned root.
        code = "from subprocess import run\n" + _wrap_run("pass")
        with pytest.raises(ToolValidationError, match="not allowed: subprocess"):
            validate_generated_tool_code(code)


# ---------------------------------------------------------------------------
# Network gating
# ---------------------------------------------------------------------------


class TestNetworkPermission:
    @pytest.mark.parametrize("module", ["socket", "urllib", "http", "requests"])
    def test_network_blocked_without_permission(self, module: str) -> None:
        code = "import " + module + "\n" + _wrap_run("pass")
        with pytest.raises(ToolValidationError, match="requires 'network' permission"):
            validate_generated_tool_code(code, permissions=[])

    def test_network_allowed_with_permission(self) -> None:
        code = "import urllib.request\n" + _wrap_run("pass")
        validate_generated_tool_code(code, permissions=["network"])  # must not raise


# ---------------------------------------------------------------------------
# Filesystem gating
# ---------------------------------------------------------------------------


class TestFilesystemPermission:
    @pytest.mark.parametrize("module", ["pathlib", "glob", "shutil", "tempfile"])
    def test_filesystem_module_blocked_without_permission(self, module: str) -> None:
        code = "import " + module + "\n" + _wrap_run("pass")
        with pytest.raises(ToolValidationError, match="requires a filesystem"):
            validate_generated_tool_code(code, permissions=[])

    def test_open_call_blocked_without_permission(self) -> None:
        code = _wrap_run("open('/etc/passwd')")
        with pytest.raises(ToolValidationError, match=r"open\(\) requires a filesystem"):
            validate_generated_tool_code(code, permissions=[])

    def test_open_allowed_with_filesystem_dot_perm(self) -> None:
        # The current convention is `filesystem.read` / `filesystem.write`.
        code = _wrap_run("open('x')")
        validate_generated_tool_code(code, permissions=["filesystem.read"])

    def test_filesystem_module_allowed_with_perm(self) -> None:
        code = "import pathlib\n" + _wrap_run("pass")
        validate_generated_tool_code(code, permissions=["filesystem.read"])


# ---------------------------------------------------------------------------
# Required structure
# ---------------------------------------------------------------------------


class TestStructure:
    def test_missing_run_function_rejected(self) -> None:
        code = "def helper(): return 1\n"
        with pytest.raises(ToolValidationError, match="must define run"):
            validate_generated_tool_code(code)

    def test_syntax_error_rejected(self) -> None:
        code = "def run(args, context):\n    return [1, 2,\n"
        with pytest.raises(ToolValidationError, match="invalid syntax"):
            validate_generated_tool_code(code)

    def test_clean_pure_python_code_accepted(self) -> None:
        code = _wrap_run("x = sum([1, 2, 3])\nreturn {'sum': x}")
        validate_generated_tool_code(code)


# ---------------------------------------------------------------------------
# extract_tool_metadata
# ---------------------------------------------------------------------------


class TestExtractToolMetadata:
    def test_valid_metadata_extracted(self) -> None:
        code = textwrap.dedent("""
            TOOL = {
                "name": "do_thing",
                "description": "Does a thing",
                "input_schema": {"type": "object", "properties": {}},
                "permissions": [],
            }

            def run(args, context):
                return None
        """).strip()
        meta = extract_tool_metadata(code)
        assert meta["name"] == "do_thing"
        assert meta["description"] == "Does a thing"
        assert meta["input_schema"]["type"] == "object"

    def test_missing_required_key_rejected(self) -> None:
        code = textwrap.dedent("""
            TOOL = {"name": "x", "description": "y"}

            def run(args, context):
                return None
        """).strip()
        with pytest.raises(ToolValidationError, match="missing key"):
            extract_tool_metadata(code)

    def test_non_dict_tool_rejected(self) -> None:
        code = "TOOL = ['not', 'a', 'dict']\n"
        with pytest.raises(ToolValidationError, match="must be a dict"):
            extract_tool_metadata(code)

    def test_no_tool_definition_rejected(self) -> None:
        code = "def run(args, context):\n    return None\n"
        with pytest.raises(ToolValidationError, match="must define TOOL metadata"):
            extract_tool_metadata(code)


# ---------------------------------------------------------------------------
# Subprocess sandbox roundtrip
# ---------------------------------------------------------------------------


@pytest.fixture
def generated_tool_file(tmp_path: Path) -> Path:
    code = textwrap.dedent("""
        TOOL = {
            "name": "echo_double",
            "description": "Return x*2",
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            "permissions": [],
        }

        def run(args, context):
            return {"doubled": args["x"] * 2, "ctx_keys": sorted((context or {}).keys())}
    """).strip()
    path = tmp_path / "echo_double.py"
    path.write_text(code, encoding="utf-8")
    return path


class TestSubprocessSandbox:
    @pytest.mark.timeout(30)
    def test_runs_tool_in_subprocess_and_returns_result(self, generated_tool_file: Path) -> None:
        sandbox = SubprocessSandbox(timeout=15)
        result = sandbox.run_python_tool(generated_tool_file, {"x": 21}, context={"user": "claude"})
        assert result["ok"] is True
        assert result["result"]["doubled"] == 42
        assert "user" in result["result"]["ctx_keys"]

    @pytest.mark.timeout(30)
    def test_load_generated_tool_returns_callable_wrapper(self, generated_tool_file: Path) -> None:
        tool = load_generated_tool(generated_tool_file)
        assert isinstance(tool, GeneratedPythonTool)
        assert tool.spec.name == "echo_double"
        out = tool(x=5)
        assert out["doubled"] == 10

    @pytest.mark.timeout(30)
    def test_runtime_error_in_generated_tool_surfaces(self, tmp_path: Path) -> None:
        code = textwrap.dedent("""
            TOOL = {"name": "crash", "description": "boom", "input_schema": {"type": "object"}}

            def run(args, context):
                raise RuntimeError("crashed inside generated tool")
        """).strip()
        path = tmp_path / "crash.py"
        path.write_text(code, encoding="utf-8")
        tool = load_generated_tool(path)
        with pytest.raises(ToolError, match="crashed inside generated tool"):
            tool()
