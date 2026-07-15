from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import ToolError, ToolValidationError
from .schema import JsonDict, ToolSpec

__all__ = [
    "ALWAYS_BANNED_CALLS",
    "ALWAYS_BANNED_MODULES",
    "DANGEROUS_NAMES",
    "DockerSandbox",
    "FILESYSTEM_MODULES",
    "GeneratedPythonTool",
    "NETWORK_MODULES",
    "PROCESS_SPAWN_CALLS",
    "SubprocessSandbox",
    "docker_available",
    "extract_tool_metadata",
    "load_generated_tool",
    "make_sandbox",
    "validate_generated_tool_code",
]

RUNNER_CODE = r"""
import contextlib
import importlib.util
import io
import json
import sys
import traceback

path = sys.argv[1]
payload = json.loads(sys.stdin.read() or "{}")
args = payload.get("args", {})
context = payload.get("context", {})

try:
    spec = importlib.util.spec_from_file_location("generated_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result = module.run(args, context)
    print(json.dumps({"ok": True, "result": result, "stdout": stdout.getvalue()}, default=repr))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=8),
    }))
"""

# ---------------------------------------------------------------------------
# Host-function bridge — lets a SANDBOXED tool call back into whitelisted host
# functions (e.g. a read-only DB query) over the child's stdio pipes. Works
# even with `--network none` (it rides stdin/stdout, not the network) and on
# Windows/macOS/Linux (no extra FDs, no unix sockets). Line-delimited JSON:
#   child→host : {"t":"call","name":...,"args":{...}}   (a host_function call)
#   host→child : {"ok":true,"result":...}               (the answer)
#   child→host : {"t":"result","ok":...,"result":...}   (the final return)
# ---------------------------------------------------------------------------

_BRIDGE_RUNNER_CODE = r"""
import contextlib
import io
import json
import sys
import traceback

_REAL_STDOUT = sys.stdout  # saved before run() redirects stdout to a buffer


def _emit(obj):
    _REAL_STDOUT.write(json.dumps(obj, default=repr) + "\n")
    _REAL_STDOUT.flush()


def _call_host(name, args=None):
    _emit({"t": "call", "name": name, "args": args or {}})
    resp = json.loads(sys.stdin.readline())
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error") or ("host function failed: " + str(name)))
    return resp.get("result")


init = json.loads(sys.stdin.readline() or "{}")
code = init.get("code", "")
args = init.get("args", {})
context = dict(init.get("context") or {})
context["call_host"] = _call_host

try:
    namespace = {}
    exec(compile(code, "generated_tool", "exec"), namespace)
    run = namespace.get("run")
    if run is None:
        raise RuntimeError("generated tool defines no run(args, context)")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run(args, context)
    _emit({"t": "result", "ok": True, "result": result, "stdout": buf.getvalue()})
except Exception as exc:
    _emit({
        "t": "result",
        "ok": False,
        "error": "%s: %s" % (type(exc).__name__, exc),
        "traceback": traceback.format_exc(limit=8),
    })
"""


def _drive_bridge(
    cmd: list[str],
    init_payload: dict[str, Any],
    host_functions: dict[str, Callable[..., Any]],
    timeout: float,
    on_timeout: Callable[[], None] | None = None,
) -> JsonDict:
    """Run the interactive runner and service host_function calls until the
    tool returns. Each call name MUST be in ``host_functions`` or it is
    refused — so a tool can only reach the callbacks the host whitelisted."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    stderr_chunks: list[str] = []
    drainer = threading.Thread(target=lambda: stderr_chunks.extend(proc.stderr or []), daemon=True)
    drainer.start()
    timed_out = {"value": False}

    def _kill() -> None:
        timed_out["value"] = True
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        if on_timeout:
            on_timeout()

    timer = threading.Timer(timeout, _kill)
    timer.start()
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(init_payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                if timed_out["value"]:
                    raise ToolError(f"Generated tool timed out after {timeout}s")
                raise ToolError(
                    f"Sandbox ended without a result. stderr={''.join(stderr_chunks)[:400]!r}"
                )
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ToolError(f"Sandbox protocol error on line {line[:200]!r}") from exc
            kind = msg.get("t")
            if kind == "call":
                name = msg.get("name")
                fn = host_functions.get(name)
                if fn is None:
                    resp: JsonDict = {"ok": False, "error": f"host function not allowed: {name}"}
                else:
                    try:
                        resp = {"ok": True, "result": fn(**(msg.get("args") or {}))}
                    except Exception as exc:  # noqa: BLE001
                        resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                proc.stdin.write(json.dumps(resp, ensure_ascii=False, default=repr) + "\n")
                proc.stdin.flush()
            elif kind == "result":
                return {key: value for key, value in msg.items() if key != "t"}
            else:
                raise ToolError(f"Unknown sandbox protocol message: {kind!r}")
    finally:
        timer.cancel()
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


ALWAYS_BANNED_CALLS = {"eval", "exec", "compile", "__import__", "input", "breakpoint"}

# Process-spawning / OS-exec call names. Reachable through ``os.*`` / ``posix.*``
# (which also bypass the subprocess ban) and via attribute access on arbitrary
# objects, so we deny the call *name* directly — defence in depth on top of the
# banned modules below.
PROCESS_SPAWN_CALLS = {
    "fork", "forkpty", "kill", "startfile", "putenv",
    "execl", "execle", "execlp", "execlpe",
    "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "posix_spawn", "posix_spawnp",
}

# ``os``/``posix``/``nt`` expose exec/spawn/fork AND unrestricted file ops that
# would bypass the filesystem-permission gate; ``sys`` exposes ``sys.modules``
# (a live handle to already-imported modules) and ``settrace``. None of these
# belong in a generated tool — use ``pathlib`` (filesystem.* permission) for
# file access instead.
ALWAYS_BANNED_MODULES = {
    "subprocess", "ctypes", "multiprocessing", "signal", "importlib",
    "os", "posix", "nt", "sys", "pty",
}
NETWORK_MODULES = {"socket", "urllib", "http", "ftplib", "smtplib", "imaplib", "poplib", "requests"}
FILESYSTEM_MODULES = {"pathlib", "glob", "shutil", "tempfile"}

# Identifiers that, when referenced by name *anywhere* in the AST, allow
# trivial bypass of the call/import filters above. We deny them outright:
# generated tools have no legitimate reason to introspect builtins,
# importlib, or globals/locals/vars dictionaries.
DANGEROUS_NAMES = {
    "__builtins__",
    "__import__",
    "__loader__",
    "__spec__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "importlib",
    # Object-graph introspection — the classic CPython sandbox escape
    # `().__class__.__bases__[0].__subclasses__()` reaches dangerous classes
    # (subprocess.Popen, os funcs…) without importing anything. Function
    # `__globals__` / `__code__` leak the module namespace just as badly.
    "__class__",
    "__bases__",
    "__base__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__dict__",
    "__getattribute__",
    "__subclasshook__",
    "__code__",
    "__closure__",
}


@dataclass
class SubprocessSandbox:
    timeout: float = 10.0

    def run_python_tool(
        self,
        file_path: str | Path,
        args: JsonDict,
        context: JsonDict | None = None,
        *,
        allow_network: bool = False,
        host_functions: dict[str, Callable[..., Any]] | None = None,
    ) -> JsonDict:
        # `allow_network` is accepted for signature-parity with DockerSandbox
        # but a plain subprocess CANNOT isolate the network — here the AST
        # `network` permission gate is the only control. Real network
        # isolation requires DockerSandbox.
        del allow_network
        path = Path(file_path).resolve()
        if host_functions:
            code = path.read_text(encoding="utf-8")
            cmd = [sys.executable, "-X", "utf8", "-I", "-S", "-u", "-c", _BRIDGE_RUNNER_CODE]
            return _drive_bridge(
                cmd, {"code": code, "args": args, "context": context or {}}, host_functions, self.timeout
            )
        payload = json.dumps({"args": args, "context": context or {}}, ensure_ascii=False)
        try:
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "-I", "-S", "-c", RUNNER_CODE, str(path)],
                input=payload,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout,
                cwd=str(path.parent),
                env=_child_env(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"Generated tool timed out after {self.timeout}s") from exc

        if completed.returncode != 0:
            raise ToolError(
                "Generated tool runner failed: "
                f"stdout={completed.stdout[:500]!r} stderr={completed.stderr[:500]!r}"
            )
        try:
            parsed: dict[str, Any] = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ToolError(f"Generated tool returned invalid JSON: {completed.stdout[:500]}") from exc
        return parsed


@dataclass
class GeneratedPythonTool:
    spec: ToolSpec
    file_path: Path
    sandbox: "SubprocessSandbox | DockerSandbox"

    def __call__(
        self,
        context: JsonDict | None = None,
        *,
        host_functions: dict[str, Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        allow_network = "network" in (self.spec.permissions or [])
        result = self.sandbox.run_python_tool(
            self.file_path,
            kwargs,
            context=context,
            allow_network=allow_network,
            host_functions=host_functions,
        )
        if not result.get("ok"):
            raise ToolError(result.get("error") or "Generated tool failed")
        return result.get("result")


# ---------------------------------------------------------------------------
# DockerSandbox — OS-level isolation (the REAL boundary; SubprocessSandbox is
# only a denylist). Runs each tool in a fresh, locked, ephemeral container.
# ---------------------------------------------------------------------------

# Runs INSIDE the container. The tool SOURCE travels via stdin (not a mounted
# volume) so there are zero host-path-mount pitfalls on Windows/macOS/Linux.
_DOCKER_RUNNER_CODE = r"""
import contextlib
import io
import json
import sys
import traceback

payload = json.loads(sys.stdin.read() or "{}")
code = payload.get("code", "")
args = payload.get("args", {})
context = payload.get("context", {})

try:
    namespace = {}
    exec(compile(code, "generated_tool", "exec"), namespace)
    run = namespace.get("run")
    if run is None:
        raise RuntimeError("generated tool defines no run(args, context)")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = run(args, context)
    print(json.dumps({"ok": True, "result": result, "stdout": out.getvalue()}, default=repr))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=8),
    }))
"""

_DOCKER_STATE: dict[str, bool] = {}


def _child_env() -> dict[str, str]:
    """Environnement MINIMAL du sous-processus sandboxé.

    Le but est l'isolation : aucun secret de l'hôte (clés API…) ne doit
    fuiter dans l'environnement du tool. Sous POSIX, un env vide suffit.
    Sous Windows, un bloc d'environnement VIDE fait échouer CreateProcess
    (``OSError: [WinError 87]`` sur les Python ≤ 3.11 — trouvé par la CI)
    et le Python enfant a besoin de ``SystemRoot`` : on ne passe que des
    variables système inoffensives, jamais le reste de ``os.environ``.
    """
    if os.name != "nt":
        return {}
    keep = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT",
            "TEMP", "TMP", "NUMBER_OF_PROCESSORS")
    return {var: os.environ[var] for var in keep if var in os.environ}


def docker_available() -> bool:
    """True if a working Docker daemon running LINUX containers is
    reachable (cached once). Lets the host fall back to SubprocessSandbox
    when Docker is absent — or useless: a daemon in *Windows containers*
    mode (Docker Desktop switched, GitHub windows runners) answers ``info``
    happily but can neither pull ``python:*-slim`` (linux image) nor honor
    ``--read-only`` — found by CI."""
    if "available" not in _DOCKER_STATE:
        exe = shutil.which("docker")
        if not exe:
            _DOCKER_STATE["available"] = False
        else:
            try:
                done = subprocess.run(
                    [exe, "version", "--format", "{{.Server.Os}}"],
                    capture_output=True, timeout=20,
                )
                _DOCKER_STATE["available"] = (
                    done.returncode == 0
                    and done.stdout.strip().lower() == b"linux"
                )
            except Exception:  # noqa: BLE001
                _DOCKER_STATE["available"] = False
    return _DOCKER_STATE["available"]


@dataclass
class DockerSandbox:
    """Runs a generated tool in a fresh, locked, ephemeral container.

    Per call: ``docker run --rm`` from a standard prebuilt image (default
    ``python:3.11-slim``, pulled once) with the root FS read-only, all
    capabilities dropped, a non-root user, CPU/memory/pid limits, and —
    unless the tool holds the ``network`` permission — ``--network none``.
    The tool SOURCE is piped via stdin (no volume mount → portable across
    OSes); the container ``exec``s it and returns JSON on stdout.

    Same ``run_python_tool(...)`` contract as :class:`SubprocessSandbox`, so
    the two are interchangeable behind :func:`make_sandbox`.
    """

    image: str = "python:3.11-slim"
    timeout: float = 10.0
    memory: str = "256m"
    cpus: str = "1.0"
    pids_limit: int = 128

    def _ensure_image(self) -> None:
        key = f"image:{self.image}"
        if _DOCKER_STATE.get(key):
            return
        inspect = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True)
        if inspect.returncode != 0:
            pull = subprocess.run(
                ["docker", "pull", self.image], capture_output=True, text=True, timeout=600
            )
            if pull.returncode != 0:
                raise ToolError(f"docker pull {self.image} failed: {pull.stderr[:300]}")
        _DOCKER_STATE[key] = True

    def run_python_tool(
        self,
        file_path: str | Path,
        args: JsonDict,
        context: JsonDict | None = None,
        *,
        allow_network: bool = False,
        host_functions: dict[str, Callable[..., Any]] | None = None,
    ) -> JsonDict:
        path = Path(file_path).resolve()
        code = path.read_text(encoding="utf-8")
        self._ensure_image()
        name = f"autoagent-tool-{uuid.uuid4().hex[:12]}"
        base = [
            "docker", "run", "--rm", "-i", "--name", name,
            "--read-only",
            "--tmpfs", "/tmp:size=32m",
            "--memory", self.memory,
            "--cpus", str(self.cpus),
            "--pids-limit", str(self.pids_limit),
            "--user", "65534:65534",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "-e", "HOME=/tmp",
        ]
        if not allow_network:
            base += ["--network", "none"]

        init = {"code": code, "args": args, "context": context or {}}
        if host_functions:
            # The bridge rides the docker CLI's stdio pipes, so DB/host access
            # works even with `--network none` still in force.
            cmd = base + [self.image, "python", "-X", "utf8", "-I", "-S", "-u", "-c", _BRIDGE_RUNNER_CODE]
            return _drive_bridge(
                cmd,
                init,
                host_functions,
                self.timeout,
                on_timeout=lambda: subprocess.run(["docker", "rm", "-f", name], capture_output=True),
            )

        cmd = base + [self.image, "python", "-X", "utf8", "-I", "-S", "-c", _DOCKER_RUNNER_CODE]
        try:
            completed = subprocess.run(
                cmd, input=json.dumps(init, ensure_ascii=False), text=True, encoding="utf-8",
                capture_output=True, timeout=self.timeout + 20,
            )
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            raise ToolError(f"Generated tool timed out after {self.timeout}s (docker)") from exc

        if completed.returncode != 0:
            raise ToolError(
                "Docker sandbox runner failed: "
                f"stdout={completed.stdout[:500]!r} stderr={completed.stderr[:500]!r}"
            )
        try:
            parsed: dict[str, Any] = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ToolError(f"Docker sandbox returned invalid JSON: {completed.stdout[:500]}") from exc
        return parsed


def make_sandbox(
    *, prefer_docker: bool = True, timeout: float = 10.0, image: str = "python:3.11-slim"
) -> "SubprocessSandbox | DockerSandbox":
    """Return a DockerSandbox when a Docker daemon is available (real
    OS-level isolation), else the hardened SubprocessSandbox as a fallback."""
    if prefer_docker and docker_available():
        return DockerSandbox(image=image, timeout=timeout)
    return SubprocessSandbox(timeout=timeout)


def load_generated_tool(
    file_path: str | Path, sandbox: "SubprocessSandbox | DockerSandbox | None" = None
) -> GeneratedPythonTool:
    path = Path(file_path)
    code = path.read_text(encoding="utf-8")
    metadata = extract_tool_metadata(code)
    spec = ToolSpec(
        name=metadata["name"],
        description=metadata["description"],
        input_schema=metadata.get("input_schema") or {"type": "object", "properties": {}},
        permissions=metadata.get("permissions") or [],
    )
    validate_generated_tool_code(code, permissions=spec.permissions)
    return GeneratedPythonTool(spec=spec, file_path=path, sandbox=sandbox or SubprocessSandbox())


def extract_tool_metadata(code: str) -> JsonDict:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOL":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise ToolValidationError("TOOL metadata must be a dict")
                    for key in ("name", "description", "input_schema"):
                        if key not in value:
                            raise ToolValidationError(f"TOOL metadata missing key: {key}")
                    return value
    raise ToolValidationError("Generated tool must define TOOL metadata")


def validate_generated_tool_code(code: str, permissions: list[str] | None = None) -> None:
    permissions = permissions or []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ToolValidationError(f"Generated tool has invalid syntax: {exc}") from exc

    has_run = any(isinstance(node, ast.FunctionDef) and node.name == "run" for node in tree.body)
    if not has_run:
        raise ToolValidationError("Generated tool must define run(args, context)")

    allow_network = "network" in permissions
    allow_filesystem = any(permission.startswith("filesystem.") for permission in permissions)

    for node in ast.walk(tree):
        # Catch any reference to a dangerous identifier — by Name node
        # (e.g. `__builtins__`), by attribute access (`x.__import__`),
        # or as a function name. This closes bypasses that hide eval/
        # importlib behind getattr/globals/__builtins__.
        if isinstance(node, ast.Name) and node.id in DANGEROUS_NAMES:
            raise ToolValidationError(f"Reference to dangerous identifier is not allowed: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in DANGEROUS_NAMES:
            raise ToolValidationError(f"Reference to dangerous attribute is not allowed: {node.attr}")

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names = _imported_module_names(node)
            for module in module_names:
                root = module.split(".", 1)[0]
                if root in ALWAYS_BANNED_MODULES:
                    raise ToolValidationError(f"Import is not allowed: {module}")
                if not allow_network and root in NETWORK_MODULES:
                    raise ToolValidationError(f"Network import requires 'network' permission: {module}")
                if not allow_filesystem and root in FILESYSTEM_MODULES:
                    raise ToolValidationError(
                        f"Filesystem import requires a filesystem.* permission: {module}"
                    )
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in ALWAYS_BANNED_CALLS:
                raise ToolValidationError(f"Call is not allowed: {call_name}")
            if call_name == "open" and not allow_filesystem:
                raise ToolValidationError("open() requires a filesystem.* permission")
            if call_name in {"system", "popen"}:
                raise ToolValidationError(f"Shell call is not allowed: {call_name}")
            if call_name in PROCESS_SPAWN_CALLS:
                raise ToolValidationError(f"Process-spawning call is not allowed: {call_name}")


def _imported_module_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.module:
        return [node.module]
    return []


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
