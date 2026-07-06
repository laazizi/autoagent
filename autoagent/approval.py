"""Human-in-the-loop promotion for dynamically generated tools.

Trust lifecycle:

    generated  ──▶  SANDBOX (DockerSandbox/Subprocess, JSON-only context)
        │
        │  human reviews code + permissions, runs `approve`
        ▼
    approved   ──▶  NATIVE (in-process, receives real host handles via context)

The trust gate is a content **hash** pinned in a manifest (``approved_tools.json``,
committed to git). A tool is run native ONLY if the sha256 of its current source
is in the manifest — change one byte and it drops back to the sandbox until a
human re-approves. This closes the swap-after-approval (TOCTOU) hole.

`load_tools()` is the single wiring point: it registers every tool in a
directory onto an Agent, choosing the execution path per tool from the manifest.
The LLM calls a tool by name identically either way — only the execution path
(and what `context` carries) differs.

CLI:
    python -m autoagent.approval list    <tools_dir> [--manifest approved_tools.json]
    python -m autoagent.approval show    <tool_file>
    python -m autoagent.approval approve <tool_file> [--manifest …] [--by NAME]
    python -m autoagent.approval reject  <tool_file>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .errors import ToolError, ToolValidationError
from .sandbox import (
    extract_tool_metadata,
    load_generated_tool,
    make_sandbox,
    validate_generated_tool_code,
)
from .schema import ToolSpec

__all__ = [
    "ToolManifest",
    "load_tools",
    "approve_tool",
    "reject_tool",
    "review_card",
    "sha256_of",
]


def sha256_of(code: str) -> str:
    """Content hash that the trust manifest is pinned to."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Trust manifest
# ---------------------------------------------------------------------------


@dataclass
class ToolManifest:
    """The allowlist of human-approved tool versions, keyed by source hash."""

    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ToolManifest":
        path = Path(path)
        entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("approved"), dict):
                    entries = data["approved"]
            except json.JSONDecodeError as exc:
                raise ToolError(f"Corrupt manifest {path}: {exc}") from exc
        return cls(path=path, entries=entries)

    def contains(self, digest: str) -> bool:
        return digest in self.entries

    def approve(
        self,
        code: str,
        *,
        name: str,
        permissions: list[str] | None = None,
        approved_by: str = "unknown",
        approved_at: str | None = None,
    ) -> str:
        digest = sha256_of(code)
        self.entries[digest] = {
            "name": name,
            "permissions": permissions or [],
            "approved_by": approved_by,
            "approved_at": approved_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.save()
        return digest

    def revoke(self, digest: str) -> None:
        self.entries.pop(digest, None)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"approved": self.entries}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Review + approve/reject
# ---------------------------------------------------------------------------


def review_card(file_path: str | Path) -> dict[str, Any]:
    """Everything a human needs to decide, in one structure: name,
    description, requested permissions, source hash, and the code."""
    path = Path(file_path)
    code = path.read_text(encoding="utf-8")
    meta = extract_tool_metadata(code)
    return {
        "name": meta["name"],
        "description": meta.get("description", ""),
        "permissions": meta.get("permissions") or [],
        "sha256": sha256_of(code),
        "path": str(path),
        "code": code,
    }


def approve_tool(
    file_path: str | Path,
    manifest: ToolManifest,
    *,
    approved_by: str = "unknown",
    approved_at: str | None = None,
) -> str:
    """Validate the tool statically, then pin its hash in the manifest.
    Returns the approved hash. Re-approving after any edit creates a NEW
    hash entry (the old one no longer matches the file)."""
    path = Path(file_path)
    code = path.read_text(encoding="utf-8")
    meta = extract_tool_metadata(code)
    validate_generated_tool_code(code, permissions=meta.get("permissions") or [])
    return manifest.approve(
        code,
        name=meta["name"],
        permissions=meta.get("permissions") or [],
        approved_by=approved_by,
        approved_at=approved_at,
    )


def reject_tool(file_path: str | Path, *, rejected_dir: str | Path | None = None) -> None:
    """Remove a tool (default) or move it to `rejected_dir` for the record."""
    path = Path(file_path)
    if rejected_dir is not None:
        dest_dir = Path(rejected_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path.replace(dest_dir / path.name)
    else:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The 2-way loader — the single wiring point onto an Agent
# ---------------------------------------------------------------------------


def _import_native(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"autoagent_approved_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise ToolError(f"Cannot import approved tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise ToolError(f"Approved tool {path.name} defines no run(args, context)")
    return module


def _make_native_handler(
    run: Callable[..., Any],
    host_context: dict[str, Any],
    host_functions: dict[str, Callable[..., Any]] | None = None,
) -> Callable[..., Any]:
    host_functions = host_functions or {}

    def _call_host(name: str, args: dict[str, Any] | None = None) -> Any:
        fn = host_functions.get(name)
        if fn is None:
            raise RuntimeError(f"host function not allowed: {name}")
        return fn(**(args or {}))

    def handler(context: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        # Trusted: gets the REAL host handles (db, config…) plus runtime context.
        # `call_host` is provided too (in-process dispatch) so a tool written
        # against the bridge keeps working unchanged once promoted to native.
        merged = dict(host_context)
        if context:
            merged.update(context)
        merged["call_host"] = _call_host
        return run(kwargs, merged)

    return handler


def _make_sandbox_handler(
    generated: Any, host_functions: dict[str, Callable[..., Any]] | None
) -> Callable[..., Any]:
    def handler(context: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        # Untrusted: NO host objects in context (isolation). Controlled access
        # to the host is ONLY via the whitelisted host_functions bridge, which
        # the tool reaches as context["call_host"](name, args) — the bridge
        # rides stdio so it works even under --network none.
        return generated(context={}, host_functions=host_functions, **kwargs)

    return handler


def load_tools(
    agent: Any,
    tools_dir: str | Path,
    manifest: ToolManifest,
    *,
    host_context: dict[str, Any] | None = None,
    sandbox: Any | None = None,
    sandbox_host_functions: dict[str, Callable[..., Any]] | None = None,
) -> list[tuple[str, str]]:
    """Register every ``*.py`` tool in ``tools_dir`` onto ``agent``, choosing
    the execution path per tool from ``manifest``:

      * hash in manifest  → **native** (in-process, receives ``host_context``).
      * otherwise          → **sandbox** (DockerSandbox/Subprocess, no host
        objects).

    Returns a list of ``(tool_name, mode)`` where mode is
    ``"native"`` / ``"sandbox"`` / ``"invalid"`` (the last for an unapproved
    tool that fails validation — it is skipped, never registered).
    """
    host_context = host_context or {}
    tools_dir = Path(tools_dir)
    sandbox = sandbox or make_sandbox()
    registered: list[tuple[str, str]] = []
    if not tools_dir.is_dir():
        return registered

    for file in sorted(tools_dir.glob("*.py")):
        code = file.read_text(encoding="utf-8")
        try:
            meta = extract_tool_metadata(code)
        except ToolValidationError:
            continue  # not a tool module — skip silently
        spec = ToolSpec(
            name=meta["name"],
            description=meta.get("description", meta["name"]),
            input_schema=meta.get("input_schema") or {"type": "object", "properties": {}},
            permissions=meta.get("permissions") or [],
        )
        digest = sha256_of(code)
        if manifest.contains(digest):
            handler = _make_native_handler(
                _import_native(file).run, host_context, sandbox_host_functions
            )
            mode = "native"
        else:
            try:
                generated = load_generated_tool(file, sandbox=sandbox)  # validates internally
            except ToolValidationError:
                registered.append((spec.name, "invalid"))
                continue
            handler = _make_sandbox_handler(generated, sandbox_host_functions)
            mode = "sandbox"
        agent.registry.replace(spec, handler)
        registered.append((spec.name, mode))
    return registered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:
    manifest = ToolManifest.load(args.manifest)
    tools_dir = Path(args.tools_dir)
    if not tools_dir.is_dir():
        print(f"No such tools dir: {tools_dir}", file=sys.stderr)
        return 2
    for file in sorted(tools_dir.glob("*.py")):
        code = file.read_text(encoding="utf-8")
        try:
            meta = extract_tool_metadata(code)
        except ToolValidationError:
            continue
        digest = sha256_of(code)
        status = "APPROVED (native)" if manifest.contains(digest) else "pending (sandbox)"
        perms = ",".join(meta.get("permissions") or []) or "-"
        print(f"  {meta['name']:30} {digest[:12]}  perms={perms:15} {status}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    card = review_card(args.tool_file)
    print(f"name        : {card['name']}")
    print(f"description : {card['description']}")
    print(f"permissions : {card['permissions'] or '-'}")
    print(f"sha256      : {card['sha256']}")
    print("--- code " + "-" * 60)
    print(card["code"])
    print("-" * 69)
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    manifest = ToolManifest.load(args.manifest)
    digest = approve_tool(args.tool_file, manifest, approved_by=args.by)
    print(f"Approved {Path(args.tool_file).name} -> {digest[:12]} (manifest: {manifest.path})")
    print("It will now run NATIVE. Any future edit voids this approval.")
    return 0


def _cmd_reject(args: argparse.Namespace) -> int:
    reject_tool(args.tool_file, rejected_dir=args.to)
    where = f" -> {args.to}" if args.to else " (deleted)"
    print(f"Rejected {Path(args.tool_file).name}{where}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoagent.approval", description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List tools + approval status.")
    p_list.add_argument("tools_dir")
    p_list.add_argument("--manifest", default="approved_tools.json")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Print a tool's review card (code + perms + hash).")
    p_show.add_argument("tool_file")
    p_show.set_defaults(func=_cmd_show)

    p_app = sub.add_parser("approve", help="Pin a tool's hash → it runs native.")
    p_app.add_argument("tool_file")
    p_app.add_argument("--manifest", default="approved_tools.json")
    p_app.add_argument("--by", default="unknown", help="Who approved (recorded in the manifest).")
    p_app.set_defaults(func=_cmd_approve)

    p_rej = sub.add_parser("reject", help="Delete a tool (or move it with --to).")
    p_rej.add_argument("tool_file")
    p_rej.add_argument("--to", default=None, help="Move here instead of deleting.")
    p_rej.set_defaults(func=_cmd_reject)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
