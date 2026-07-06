from __future__ import annotations

import fnmatch
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ToolError

__all__ = ["ChangeRecord", "ProjectWorkspace", "WorkspaceError"]


class WorkspaceError(ToolError):
    """Raised when a workspace operation is refused or fails."""


@dataclass
class ChangeRecord:
    id: str
    action: str
    path: str
    reason: str
    timestamp: float
    before: str | None
    after: str | None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "path": self.path,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "created": self.before is None and self.after is not None,
            "deleted": self.before is not None and self.after is None,
        }


class ProjectWorkspace:
    """Path-confined read/write surface with a built-in change history.

    Every file edit is recorded as a `ChangeRecord` (before/after pair)
    so any change is later reversible via `rollback_change(id)` or
    `rollback_last_change()`. This is what lets the evolution runtime
    safely modify a target application.

    Path validation:
        * Absolute paths are rejected.
        * Any path that resolves outside `root` is rejected (no `../`
          escapes).
        * Paths that hit an entry of `ignored_dirs` (default: `.git`,
          `.autoagent`, `__pycache__`, `.pytest_cache`) are rejected.
        * If `allowed_write_extensions` is set, write operations on
          other extensions are rejected.
        * Writes larger than `max_write_chars` are rejected.

    Thread-safety:
        All mutating operations (`write_file`, `replace_text`,
        rollbacks) and the history readers serialize on a per-instance
        `threading.RLock`. Concurrent writes to the SAME path keep a
        coherent change history (every `before` matches an earlier
        `after`). The lock does NOT cover other processes touching the
        same directory — that case requires an external file lock.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        allowed_write_extensions: set[str] | None = None,
        ignored_dirs: set[str] | None = None,
        max_read_chars: int = 50000,
        max_write_chars: int = 200000,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_write_extensions = allowed_write_extensions
        self.ignored_dirs = ignored_dirs or {".git", ".autoagent", "__pycache__", ".pytest_cache"}
        self.max_read_chars = max_read_chars
        self.max_write_chars = max_write_chars
        self._changes: list[ChangeRecord] = []
        # Serializes every (read-before, write, record-change) sequence so
        # concurrent writers cannot interleave and produce an inconsistent
        # change history. Mutations of self._changes and rollbacks share
        # this same lock.
        self._lock = threading.RLock()

    def list_files(self, pattern: str = "**/*", max_files: int = 200) -> dict[str, Any]:
        self._validate_pattern(pattern)
        files: list[str] = []
        for path in self.root.glob(pattern):
            if len(files) >= max_files:
                break
            if not path.is_file() or self._is_ignored(path):
                continue
            files.append(path.relative_to(self.root).as_posix())
        return {"root": str(self.root), "pattern": pattern, "files": sorted(files)}

    def read_file(self, path: str, max_chars: int | None = None) -> dict[str, Any]:
        resolved = self.resolve(path)
        if not resolved.exists():
            raise WorkspaceError(f"File does not exist: {path}")
        if not resolved.is_file():
            raise WorkspaceError(f"Path is not a file: {path}")

        limit = min(max_chars or self.max_read_chars, self.max_read_chars)
        content = resolved.read_text(encoding="utf-8")
        truncated = len(content) > limit
        return {
            "path": path,
            "content": content[:limit],
            "truncated": truncated,
            "chars": len(content),
        }

    def write_file(self, path: str, content: str, *, reason: str = "") -> dict[str, Any]:
        resolved = self.resolve(path)
        self._validate_write_path(resolved)
        self._validate_write_size(len(content))
        with self._lock:
            before = resolved.read_text(encoding="utf-8") if resolved.exists() else None
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            change = self._record_change(
                action="write_file",
                path=path,
                before=before,
                after=content,
                reason=reason,
            )
        return {"ok": True, "change": change.summary()}

    def replace_text(
        self,
        path: str,
        old: str,
        new: str,
        *,
        count: int = 1,
        reason: str = "",
    ) -> dict[str, Any]:
        if not old:
            raise WorkspaceError("old text cannot be empty")
        resolved = self.resolve(path)
        self._validate_write_path(resolved)
        with self._lock:
            if not resolved.exists():
                raise WorkspaceError(f"File does not exist: {path}")
            before = resolved.read_text(encoding="utf-8")
            occurrences = before.count(old)
            if occurrences == 0:
                raise WorkspaceError(f"Text to replace was not found in {path}")
            limit = count if count > 0 else occurrences
            after = before.replace(old, new, limit)
            self._validate_write_size(len(after))
            resolved.write_text(after, encoding="utf-8")
            change = self._record_change(
                action="replace_text",
                path=path,
                before=before,
                after=after,
                reason=reason,
            )
        return {"ok": True, "replaced": min(occurrences, limit), "change": change.summary()}

    def list_changes(self) -> dict[str, Any]:
        with self._lock:
            return {"changes": [change.summary() for change in self._changes]}

    def rollback_last_change(self) -> dict[str, Any]:
        with self._lock:
            if not self._changes:
                raise WorkspaceError("No changes to rollback")
            return self.rollback_change(self._changes[-1].id)

    def rollback_change(self, change_id: str) -> dict[str, Any]:
        with self._lock:
            index = next(
                (i for i, change in enumerate(self._changes) if change.id == change_id),
                None,
            )
            if index is None:
                raise WorkspaceError(f"Unknown change id: {change_id}")

            rolled_back = self._changes[index:]
            for change in reversed(rolled_back):
                resolved = self.resolve(change.path)
                if change.before is None:
                    if resolved.exists():
                        resolved.unlink()
                else:
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    resolved.write_text(change.before, encoding="utf-8")

            del self._changes[index:]
            return {"ok": True, "rolled_back": [change.summary() for change in rolled_back]}

    def resolve(self, path: str) -> Path:
        if not path or "\x00" in path:
            raise WorkspaceError("Invalid path")
        raw_path = Path(path)
        if raw_path.is_absolute():
            raise WorkspaceError("Absolute paths are not allowed")
        resolved = (self.root / raw_path).resolve()
        if not resolved.is_relative_to(self.root):
            raise WorkspaceError(f"Path escapes workspace: {path}")
        if self._is_ignored(resolved):
            raise WorkspaceError(f"Path is ignored by workspace policy: {path}")
        return resolved

    def _record_change(
        self,
        *,
        action: str,
        path: str,
        before: str | None,
        after: str | None,
        reason: str,
    ) -> ChangeRecord:
        change = ChangeRecord(
            id=uuid.uuid4().hex,
            action=action,
            path=path,
            reason=reason,
            timestamp=time.time(),
            before=before,
            after=after,
        )
        self._changes.append(change)
        return change

    def _validate_pattern(self, pattern: str) -> None:
        if not pattern or pattern.startswith("/") or ".." in Path(pattern).parts:
            raise WorkspaceError(f"Invalid glob pattern: {pattern}")

    def _validate_write_path(self, path: Path) -> None:
        if self.allowed_write_extensions is None:
            return
        if path.suffix not in self.allowed_write_extensions:
            allowed = ", ".join(sorted(self.allowed_write_extensions))
            raise WorkspaceError(
                f"Writes to {path.suffix or '<no extension>'} files are blocked. Allowed: {allowed}"
            )

    def _validate_write_size(self, size: int) -> None:
        if size > self.max_write_chars:
            raise WorkspaceError(
                f"Write rejected: {size} chars exceeds max_write_chars={self.max_write_chars}"
            )

    def _is_ignored(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return True
        parts = set(relative.parts)
        if parts.intersection(self.ignored_dirs):
            return True
        return any(fnmatch.fnmatch(part, "*.pyc") for part in relative.parts)
