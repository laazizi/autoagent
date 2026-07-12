"""Tests for the EvolutionRuntime.

EvolutionRuntime exposes a controlled surface that lets the agent
**modify a running application**: read/write files, hot-swap pipeline
slots, call host functions, run validation commands. Each capability is
opt-in.

Critical invariants under test:

  * Unknown capability names are rejected at wire-up time so a typo
    cannot silently strip permissions.
  * `run_validation` refuses caller-supplied commands unless
    `allow_custom_validation_command=True` — otherwise an LLM could
    inject `rm -rf /` via the `command` argument.
  * `call_host_function` rejects unknown names so the agent cannot call
    arbitrary attributes.
  * `register_tools` wires the documented tool set per capability.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoagent.agent import Agent
from autoagent.errors import ToolError
from autoagent.evolution import EVOLUTION_CAPABILITIES, EvolutionRuntime
from autoagent.schema import LLMResponse

from .conftest import FakeLLMProvider


@pytest.fixture
def runtime(tmp_path: Path) -> EvolutionRuntime:
    return EvolutionRuntime(workspace_root=tmp_path)


@pytest.fixture
def agent() -> Agent:
    return Agent(FakeLLMProvider([LLMResponse(content="ok", model="fake")]))


# ---------------------------------------------------------------------------
# Capability resolution
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_defaults_to_all_capabilities(self) -> None:
        assert EvolutionRuntime._resolve_capabilities(None) == set(EVOLUTION_CAPABILITIES)

    def test_explicit_subset_kept(self) -> None:
        assert EvolutionRuntime._resolve_capabilities({"read"}) == {"read"}

    def test_unknown_capability_rejected(self) -> None:
        with pytest.raises(ToolError, match="Unknown evolution capabilities"):
            EvolutionRuntime._resolve_capabilities({"read", "rrrread"})


# ---------------------------------------------------------------------------
# register_tools wiring
# ---------------------------------------------------------------------------


class TestRegisterToolsWiring:
    def test_read_only_registers_read_tools(self, runtime: EvolutionRuntime, agent: Agent) -> None:
        runtime.register_tools(agent, capabilities={"read"})
        names = {spec.name for spec in agent.registry.specs()}
        assert "list_project_files" in names
        assert "read_project_file" in names
        # Write tools must stay out when only `read` is requested.
        assert "write_project_file" not in names
        assert "rollback_change" not in names

    def test_write_adds_write_and_rollback(self, runtime: EvolutionRuntime, agent: Agent) -> None:
        runtime.register_tools(agent, capabilities={"write"})
        names = {spec.name for spec in agent.registry.specs()}
        assert "write_project_file" in names
        assert "replace_project_text" in names
        assert "rollback_change" in names
        assert "rollback_last_change" in names

    def test_validate_only_adds_run_validation(self, runtime: EvolutionRuntime, agent: Agent) -> None:
        runtime.register_tools(agent, capabilities={"validate"})
        names = {spec.name for spec in agent.registry.specs()}
        assert names == {"run_validation"}


# ---------------------------------------------------------------------------
# run_validation: shell-injection guard
# ---------------------------------------------------------------------------


class TestRunValidationGuard:
    def test_no_command_configured_raises(self, tmp_path: Path) -> None:
        rt = EvolutionRuntime(workspace_root=tmp_path)
        with pytest.raises(ToolError, match="No validation command"):
            rt.run_validation()

    def test_custom_command_blocked_by_default(self, tmp_path: Path) -> None:
        rt = EvolutionRuntime(workspace_root=tmp_path, validation_command="echo ok")
        with pytest.raises(ToolError, match="disabled for this runtime"):
            rt.run_validation(command="rm -rf /")

    def test_custom_command_allowed_when_flag_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the flag is True, the guard must pass the command on to
        subprocess.run instead of raising. We mock subprocess so the test
        is portable across OSes."""
        from autoagent import evolution as evo_mod

        captured: dict[str, object] = {}

        class _FakeCompleted:
            returncode = 0
            stdout = "fake stdout"
            stderr = ""

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _FakeCompleted()

        monkeypatch.setattr(evo_mod.subprocess, "run", fake_run)
        rt = EvolutionRuntime(
            workspace_root=tmp_path,
            validation_command="default cmd",
            allow_custom_validation_command=True,
        )
        result = rt.run_validation(command="my-script --flag value")
        assert result["ok"] is True
        # Confirm shlex.split was used and our custom command reached subprocess.
        assert captured["args"] == ["my-script", "--flag", "value"]

    def test_configured_command_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from autoagent import evolution as evo_mod

        captured: dict[str, object] = {}

        class _FakeCompleted:
            returncode = 0
            stdout = "configured ran"
            stderr = ""

        def fake_run(args, **kwargs):
            captured["args"] = args
            return _FakeCompleted()

        monkeypatch.setattr(evo_mod.subprocess, "run", fake_run)
        rt = EvolutionRuntime(
            workspace_root=tmp_path,
            validation_command=["python", "-c", "print('configured')"],
        )
        result = rt.run_validation()
        assert result["ok"] is True
        assert captured["args"] == ["python", "-c", "print('configured')"]


# ---------------------------------------------------------------------------
# host_functions
# ---------------------------------------------------------------------------


class TestHostFunctions:
    def test_register_empty_name_rejected(self, runtime: EvolutionRuntime) -> None:
        with pytest.raises(ToolError, match="cannot be empty"):
            runtime.register_host_function("", lambda: None)

    def test_unknown_host_function_rejected(self, runtime: EvolutionRuntime) -> None:
        with pytest.raises(ToolError, match="Unknown host function"):
            runtime.call_host_function("does_not_exist")

    def test_registered_function_callable(self, runtime: EvolutionRuntime) -> None:
        def greeter(name: str) -> str:
            return f"hello {name}"

        runtime.register_host_function("greet", greeter)
        result = runtime.call_host_function("greet", {"name": "claude"})
        assert result == {"ok": True, "result": "hello claude"}

    def test_list_host_functions_describes_signature(self, runtime: EvolutionRuntime) -> None:
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        runtime.register_host_function("add", add)
        listing = runtime.list_host_functions()
        names = [item["name"] for item in listing["functions"]]
        assert "add" in names
        add_info = next(item for item in listing["functions"] if item["name"] == "add")
        assert add_info["description"] == "Add two numbers."
        assert add_info["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# State reader
# ---------------------------------------------------------------------------


class TestStateReader:
    def test_no_reader_returns_unavailable(self, runtime: EvolutionRuntime) -> None:
        result = runtime.get_runtime_state()
        assert result == {"available": False, "state": None}

    def test_reader_invoked(self, tmp_path: Path) -> None:
        rt = EvolutionRuntime(workspace_root=tmp_path, state_reader=lambda: {"count": 7})
        result = rt.get_runtime_state()
        assert result == {"available": True, "state": {"count": 7}}


# ---------------------------------------------------------------------------
# File operations route through the workspace (so rollback works)
# ---------------------------------------------------------------------------


class TestFileOpsRouteThroughWorkspace:
    def test_write_then_rollback(self, runtime: EvolutionRuntime) -> None:
        runtime.write_project_file("hello.txt", "first", reason="t1")
        runtime.write_project_file("hello.txt", "second", reason="t2")
        assert runtime.read_project_file("hello.txt")["content"] == "second"

        result = runtime.rollback_last_change()
        assert result["ok"] is True
        assert runtime.read_project_file("hello.txt")["content"] == "first"


# ---------------------------------------------------------------------------
# Tool wrappers — each method below is a single entry point the LLM can
# call. The body is mostly a passthrough, so a regression here would not
# be a logic bug but a wiring bug (wrong argument forwarded, error eaten,
# etc.). We test each at least once.
# ---------------------------------------------------------------------------


class TestToolWrappers:
    def test_list_project_files_returns_filtered_list(self, runtime: EvolutionRuntime) -> None:
        runtime.write_project_file("src/a.py", "pass")
        runtime.write_project_file("src/b.py", "pass")
        runtime.write_project_file("README.md", "# hi")
        result = runtime.list_project_files(pattern="src/*.py", max_files=10)
        assert set(result["files"]) == {"src/a.py", "src/b.py"}

    def test_read_project_file_returns_content(self, runtime: EvolutionRuntime) -> None:
        runtime.write_project_file("note.txt", "hello world")
        result = runtime.read_project_file("note.txt", max_chars=1000)
        assert result["content"] == "hello world"

    def test_replace_project_text_edits_in_place(self, runtime: EvolutionRuntime) -> None:
        runtime.write_project_file("config.py", "DEBUG = False")
        result = runtime.replace_project_text("config.py", "False", "True", count=1, reason="enable debug")
        assert result["ok"] is True
        assert result["replaced"] == 1
        assert runtime.read_project_file("config.py")["content"] == "DEBUG = True"

    def test_list_changes_reflects_writes(self, runtime: EvolutionRuntime) -> None:
        runtime.write_project_file("a.txt", "A")
        runtime.write_project_file("b.txt", "B")
        result = runtime.list_changes()
        paths = {c["path"] for c in result["changes"]}
        assert paths == {"a.txt", "b.txt"}

    def test_rollback_change_by_id(self, runtime: EvolutionRuntime) -> None:
        first = runtime.write_project_file("x.txt", "v1")
        runtime.write_project_file("x.txt", "v2")
        change_id = first["change"]["id"]
        runtime.rollback_change(change_id)
        # Rolling back the first write deletes the file (it had no `before`).
        from autoagent.errors import ToolError as _TE

        with pytest.raises(_TE):
            runtime.read_project_file("x.txt")

    def test_list_pipeline_slots_and_replace(self, runtime: EvolutionRuntime) -> None:
        # Initially empty.
        assert runtime.list_pipeline_slots()["slots"] == {}
        runtime.replace_pipeline_slot(
            "parser", "myapp.parser", callable_name="run", config={"strict": True}, reason="init"
        )
        # Now visible via list/get.
        assert "parser" in runtime.list_pipeline_slots()["slots"]
        got = runtime.get_pipeline_slot("parser")
        assert got["slot"] == "parser"
        assert got["value"]["module"] == "myapp.parser"

    def test_enable_software_evolution_appends_system_prompt(
        self, runtime: EvolutionRuntime, agent: Agent
    ) -> None:
        from autoagent.evolution import EVOLUTION_SYSTEM_PROMPT, enable_software_evolution

        base = agent.system_prompt
        enable_software_evolution(agent, runtime)
        assert EVOLUTION_SYSTEM_PROMPT.strip() in agent.system_prompt
        assert base.rstrip() in agent.system_prompt

    def test_enable_software_evolution_is_idempotent(self, runtime: EvolutionRuntime, agent: Agent) -> None:
        """Calling `enable_software_evolution` twice on the same agent
        must be a no-op: the same tools stay registered, no exception."""
        from autoagent.evolution import enable_software_evolution

        enable_software_evolution(agent, runtime)
        before = {spec.name for spec in agent.registry.specs()}

        # Second call must NOT raise.
        enable_software_evolution(agent, runtime)
        after = {spec.name for spec in agent.registry.specs()}

        assert before == after, "Re-enabling must not change the tool set"
