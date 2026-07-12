"""Tests for the PipelineManager.

The pipeline file is a JSON document with a `slots` mapping. Slots can
be hot-swapped at runtime: the agent picks a slot and writes a new
{module, callable, config} entry. Every edit goes through
`workspace.write_file`, which records a ChangeRecord — so slot edits are
rollback-able like any other workspace change.

Invariants under test:
  * Missing pipeline file -> empty default spec (no crash).
  * Invalid JSON -> PipelineError.
  * Unknown slot lookup -> PipelineError.
  * Empty `slot` / `module` rejected.
  * `replace_slot` returns the `before` value and routes through the
    workspace so it shows up in `list_changes`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoagent.pipeline import PipelineError, PipelineManager
from autoagent.workspace import ProjectWorkspace


@pytest.fixture
def workspace(tmp_path: Path) -> ProjectWorkspace:
    return ProjectWorkspace(root=tmp_path)


@pytest.fixture
def manager(workspace: ProjectWorkspace) -> PipelineManager:
    return PipelineManager(workspace=workspace, path="pipeline.json")


# ---------------------------------------------------------------------------
# load() — missing file & invalid JSON
# ---------------------------------------------------------------------------


class TestLoad:
    def test_missing_file_returns_empty_spec(self, manager: PipelineManager) -> None:
        spec = manager.load()
        assert spec == {"name": "pipeline", "slots": {}}

    def test_invalid_json_raises(self, workspace: ProjectWorkspace, manager: PipelineManager) -> None:
        workspace.write_file("pipeline.json", "not json at all")
        with pytest.raises(PipelineError, match="Invalid pipeline JSON"):
            manager.load()

    def test_non_object_root_rejected(self, workspace: ProjectWorkspace, manager: PipelineManager) -> None:
        workspace.write_file("pipeline.json", "[1, 2, 3]")
        with pytest.raises(PipelineError, match="must be a JSON object"):
            manager.load()

    def test_slots_not_object_rejected(self, workspace: ProjectWorkspace, manager: PipelineManager) -> None:
        workspace.write_file("pipeline.json", json.dumps({"slots": ["not", "an", "object"]}))
        with pytest.raises(PipelineError, match="'slots' must be an object"):
            manager.load()

    def test_slots_default_populated_when_absent(
        self, workspace: ProjectWorkspace, manager: PipelineManager
    ) -> None:
        workspace.write_file("pipeline.json", json.dumps({"name": "p"}))
        spec = manager.load()
        assert spec["slots"] == {}


# ---------------------------------------------------------------------------
# list_slots / get_slot
# ---------------------------------------------------------------------------


class TestReadSlots:
    def test_list_slots_returns_full_spec(
        self, workspace: ProjectWorkspace, manager: PipelineManager
    ) -> None:
        workspace.write_file(
            "pipeline.json",
            json.dumps(
                {
                    "name": "ingest",
                    "slots": {"parser": {"module": "m.parser", "callable": "run", "config": {}}},
                }
            ),
        )
        result = manager.list_slots()
        assert result["name"] == "ingest"
        assert "parser" in result["slots"]

    def test_get_unknown_slot_raises(self, manager: PipelineManager) -> None:
        with pytest.raises(PipelineError, match="Unknown pipeline slot"):
            manager.get_slot("does-not-exist")

    def test_get_existing_slot(self, workspace: ProjectWorkspace, manager: PipelineManager) -> None:
        workspace.write_file(
            "pipeline.json",
            json.dumps({"slots": {"parser": {"module": "m.p", "callable": "run", "config": {}}}}),
        )
        result = manager.get_slot("parser")
        assert result["slot"] == "parser"
        assert result["value"]["module"] == "m.p"


# ---------------------------------------------------------------------------
# replace_slot
# ---------------------------------------------------------------------------


class TestReplaceSlot:
    def test_empty_slot_rejected(self, manager: PipelineManager) -> None:
        with pytest.raises(PipelineError, match="slot cannot be empty"):
            manager.replace_slot("", "m.x")

    def test_empty_module_rejected(self, manager: PipelineManager) -> None:
        with pytest.raises(PipelineError, match="module cannot be empty"):
            manager.replace_slot("parser", "")

    def test_replace_creates_slot_and_records_change(
        self,
        workspace: ProjectWorkspace,
        manager: PipelineManager,
    ) -> None:
        result = manager.replace_slot(
            "parser",
            "myapp.parsers.v2",
            callable_name="parse",
            config={"strict": True},
            reason="upgrade to v2",
        )
        assert result["ok"] is True
        assert result["before"] is None
        assert result["after"]["module"] == "myapp.parsers.v2"
        assert result["after"]["callable"] == "parse"

        # The edit must go through the workspace so it appears in changes.
        changes = workspace.list_changes()["changes"]
        assert any(c["path"] == "pipeline.json" for c in changes)

        # And the on-disk pipeline must reflect the new slot.
        reloaded = manager.load()
        assert reloaded["slots"]["parser"]["module"] == "myapp.parsers.v2"

    def test_replace_returns_previous_value_as_before(
        self,
        manager: PipelineManager,
    ) -> None:
        manager.replace_slot("parser", "m.v1")
        result = manager.replace_slot("parser", "m.v2", reason="upgrade")
        assert result["before"]["module"] == "m.v1"
        assert result["after"]["module"] == "m.v2"

    def test_replace_is_rollback_able(
        self,
        workspace: ProjectWorkspace,
        manager: PipelineManager,
    ) -> None:
        manager.replace_slot("parser", "m.v1")
        manager.replace_slot("parser", "m.v2")
        # Roll back the v2 write; v1 must be restored.
        workspace.rollback_last_change()
        reloaded = manager.load()
        assert reloaded["slots"]["parser"]["module"] == "m.v1"
