from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import ToolError
from .workspace import ProjectWorkspace

__all__ = ["PipelineError", "PipelineManager"]


class PipelineError(ToolError):
    """Raised when a pipeline operation is invalid."""


@dataclass
class PipelineManager:
    workspace: ProjectWorkspace
    path: str = "pipeline.json"

    def list_slots(self) -> dict[str, Any]:
        spec = self.load()
        return {
            "path": self.path,
            "name": spec.get("name"),
            "slots": spec.get("slots", {}),
        }

    def get_slot(self, slot: str) -> dict[str, Any]:
        spec = self.load()
        slots = spec.get("slots", {})
        if slot not in slots:
            raise PipelineError(f"Unknown pipeline slot: {slot}")
        return {"slot": slot, "value": slots[slot]}

    def replace_slot(
        self,
        slot: str,
        module: str,
        *,
        callable_name: str = "run",
        config: dict[str, Any] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        if not slot:
            raise PipelineError("slot cannot be empty")
        if not module:
            raise PipelineError("module cannot be empty")
        spec = self.load()
        spec.setdefault("name", "pipeline")
        slots = spec.setdefault("slots", {})
        before = slots.get(slot)
        slots[slot] = {
            "module": module,
            "callable": callable_name or "run",
            "config": config or {},
        }
        result = self.workspace.write_file(
            self.path,
            json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
            reason=reason or f"replace pipeline slot {slot}",
        )
        return {"ok": True, "slot": slot, "before": before, "after": slots[slot], "change": result["change"]}

    def load(self) -> dict[str, Any]:
        try:
            data = self.workspace.read_file(self.path)
        except ToolError:
            return {"name": "pipeline", "slots": {}}
        try:
            spec = json.loads(data["content"])
        except json.JSONDecodeError as exc:
            raise PipelineError(f"Invalid pipeline JSON in {self.path}: {exc}") from exc
        if not isinstance(spec, dict):
            raise PipelineError(f"Pipeline spec must be a JSON object: {self.path}")
        spec.setdefault("slots", {})
        if not isinstance(spec["slots"], dict):
            raise PipelineError("Pipeline 'slots' must be an object")
        return spec
