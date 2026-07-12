"""Red tests for thread-safety of shared mutable state.

CURRENT BUGS:
    * autoagent/registry.py: `ToolRegistry._tools` dict is mutated by
      `add` / `replace` and iterated by `specs` and `execute` without any
      lock. Concurrent registration + iteration can raise
      `RuntimeError: dictionary changed size during iteration`.
    * autoagent/workspace.py: `ProjectWorkspace._changes` list and the
      `_change_index` mapping are mutated by `write_file`, `replace_text`,
      `rollback_change` without locking. Concurrent writes corrupt the
      change history (lost updates, mis-ordered rollback).

These tests stress the lib from many threads. They are RED today and
will turn GREEN once a Lock / RLock is added around the critical
sections.

We use `pytest-timeout` to fail fast if a deadlock is introduced by a
naive fix (e.g. nested non-reentrant lock).
"""

from __future__ import annotations

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from autoagent.registry import ToolRegistry
from autoagent.schema import ToolCall, ToolSpec
from autoagent.workspace import ProjectWorkspace

# ---------------------------------------------------------------------------
# ToolRegistry concurrency
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
def test_registry_concurrent_add_and_iterate() -> None:
    """Many threads register tools while others read specs(). Must not raise."""
    registry = ToolRegistry()
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                # Iterate specs while other threads add. Without a lock this
                # can raise RuntimeError: dictionary changed size during iteration
                names = [s.name for s in registry.specs()]
                _ = len(names)
        except BaseException as exc:
            errors.append(exc)

    def writer(start: int, count: int) -> None:
        try:
            for i in range(start, start + count):

                def handler(value: int = i) -> int:
                    return value

                registry.add(ToolSpec(name=f"tool_{i}", description="t"), handler)
        except BaseException as exc:
            errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    writers = [threading.Thread(target=writer, args=(i * 50, 50)) for i in range(4)]
    for t in readers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()

    assert not errors, f"Concurrency errors: {errors!r}"
    # All 200 tools should be registered.
    assert len(registry.specs()) == 200


@pytest.mark.timeout(10)
def test_registry_concurrent_execute() -> None:
    """Many threads call execute() on the same tool concurrently. Each call
    must return its own correct result."""
    registry = ToolRegistry()

    def square(x: int) -> int:
        return x * x

    registry.add(ToolSpec(name="square", description="x*x"), square)

    def call(x: int) -> int:
        result = registry.execute(ToolCall(id=str(x), name="square", arguments={"x": x}))
        assert result.ok, f"execute failed: {result.error}"
        return result.result

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(call, x) for x in range(200)]
        results = sorted(f.result() for f in as_completed(futures))

    assert results == sorted(x * x for x in range(200))


# ---------------------------------------------------------------------------
# ProjectWorkspace concurrency
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace() -> ProjectWorkspace:
    return ProjectWorkspace(root=Path(tempfile.mkdtemp(prefix="autoagent_concur_")))


@pytest.mark.timeout(10)
def test_workspace_concurrent_writes_record_all_changes(
    tmp_workspace: ProjectWorkspace,
) -> None:
    """N threads each write a distinct file. After joining, the change log
    must contain exactly N entries — no lost updates from racing on
    `_changes.append`."""
    n = 50

    def write(i: int) -> None:
        tmp_workspace.write_file(f"file_{i}.txt", f"content-{i}", reason=f"write {i}")

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(write, range(n)))

    changes = tmp_workspace.list_changes()["changes"]
    assert len(changes) == n, (
        f"Expected {n} recorded changes, got {len(changes)} — likely a race "
        f"in ProjectWorkspace._changes.append()"
    )
    # All change IDs must be unique.
    ids = [c["id"] for c in changes]
    assert len(set(ids)) == n, "Duplicate change IDs — _change_index is racy"


@pytest.mark.timeout(10)
def test_workspace_concurrent_writes_same_file_consistent_history(
    tmp_workspace: ProjectWorkspace,
) -> None:
    """When two threads write the same file concurrently, the change log
    must remain coherent: for every recorded ChangeRecord, `before` must
    equal the `after` of an earlier record (or None for the very first
    write). Otherwise rolling back chronologically would yield garbage.

    Without a lock around (read before -> write -> append record),
    interleavings break this invariant.
    """
    path = "shared.txt"
    n = 30

    def write(i: int) -> None:
        tmp_workspace.write_file(path, f"v{i}", reason=f"v{i}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(write, range(n)))

    records = [c for c in tmp_workspace.list_changes()["changes"] if c["path"] == path]
    assert len(records) == n

    # Build the chronological chain of (before, after) and verify every
    # `before` was produced by some earlier `after`. This is what a
    # serialized history would guarantee.
    # Note: list_changes summary hides before/after; we read internals here
    # to enforce the invariant.
    raw = tmp_workspace._changes  # type: ignore[attr-defined]
    afters: set[str | None] = {None}
    for rec in raw:
        if rec.path != path:
            continue
        assert rec.before in afters, (
            f"Inconsistent change history for {path!r}: a write recorded "
            f"before={rec.before!r} but no earlier write produced that value. "
            f"This indicates a race between read-modify-append in write_file()."
        )
        afters.add(rec.after)
