from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import ToolError, ToolValidationError
from .gate import active_gate
from .logging import get_logger, warn_once
from .schema import JsonDict, ToolSpec

_log = get_logger("sandbox")

__all__ = [
    "ALWAYS_BANNED_CALLS",
    "ALWAYS_BANNED_MODULES",
    "ATTRIBUTE_BANNED_CALLS",
    "DANGEROUS_NAMES",
    "DockerSandbox",
    "FILESYSTEM_MODULES",
    "GeneratedPythonTool",
    "NETWORK_MODULES",
    "PROCESS_SPAWN_CALLS",
    "SandboxLimits",
    "SubprocessSandbox",
    "discard_generated_tool",
    "docker_available",
    "extract_tool_metadata",
    "load_generated_tool",
    "make_sandbox",
    "safe_environment",
    "validate_generated_tool_code",
]

RUNNER_CODE = r"""
import contextlib
import io
import json
import sys
import traceback
import types

path = sys.argv[1]
payload = json.loads(sys.stdin.read() or "{}")
args = payload.get("args", {})
context = payload.get("context", {})

try:
    # Le SOURCE est compilé à chaque exécution — jamais via importlib, donc
    # jamais via __pycache__ : un fichier réécrit avec la même taille dans la
    # même seconde exécutait le bytecode PÉRIMÉ de la version précédente.
    with open(path, encoding="utf-8") as fh:
        code = fh.read()
    module = types.ModuleType("generated_tool")
    module.__file__ = path
    exec(compile(code, path, "exec"), module.__dict__)
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


# ---------------------------------------------------------------------------
# Plafonds de ressources du sous-processus (0.23.1) — OPT-IN, Linux seulement.
# Le `SubprocessSandbox` n'en avait AUCUN : un outil pouvait allouer des
# centaines de Mo, écrire des Go ou tenir un cœur jusqu'au délai mur. Les
# plafonds sont posés par le RUNNER lui-même, avant de charger le code de
# l'outil (un court préambule collé devant le runner, sans `preexec_fn` : la doc
# Python le déclare « NOT SAFE » en présence de threads). Sans `limits=`, le
# préambule est la chaîne vide : le runner par défaut reste octet pour octet
# celui d'avant.
# ---------------------------------------------------------------------------

_MO = 1024 * 1024
_LIMITE_MAX = 2**62         # sous le plus grand `long` C : au-delà, `resource.setrlimit` lève OverflowError


@dataclass(frozen=True)
class SandboxLimits:
    """Plafonds de ressources d'un :class:`SubprocessSandbox` (0.23.1, opt-in, Linux).

    * ``memory_mb`` : espace d'adressage du processus (``RLIMIT_AS``). Au-delà,
      l'allocation échoue : l'outil reçoit ``MemoryError``. C'est de la mémoire
      VIRTUELLE : un outil qui lance des threads réserve plus qu'il n'utilise —
      garde de la marge. Le PLANCHER dépend de la compilation de Python (mesuré,
      Linux : un outil trivial tient dès 16 Mo avec le python3 3.12.3 d'Ubuntu,
      mais il en faut 64 avec les 3.12.15 et 3.13.16 d'``uv``, 32 avec le 3.10.22
      du même ``uv``) : pars de 128 Mo ou plus, et mesure sur TON interpréteur.
    * ``cpu_s`` : secondes de CPU du processus (``RLIMIT_CPU``). Au-delà, le
      processus est tué (SIGKILL, limite souple = limite dure). C'est du temps
      CPU, pas du temps mur — pour celui-ci, ``timeout``. Refusé avec
      ``warm=True`` : le plafond compterait tous les appels du worker. Il compte en
      secondes ENTIÈRES (≥ 1 ; 1,5 s s'applique comme 2 s, et `isolation()` le dit).
    * ``fsize_mb`` : taille maximale d'un fichier écrit (``RLIMIT_FSIZE``). Au-delà,
      l'écriture échoue (``OSError: File too large``).

    Posés par le runner avant le code de l'outil, limite souple = limite dure :
    un processus non-root ne peut pas les relever. Un processus ROOT le peut
    (CAP_SYS_RESOURCE) : ce sont des garde-fous contre un outil qui s'emballe,
    PAS une frontière de sécurité — la frontière, c'est ``DockerSandbox``. Non
    couverts, dits : le nombre de processus (``RLIMIT_NPROC`` est sans effet en
    root, où tourne souvent la prod), le volume de sortie standard, le réseau et
    le système de fichiers. Hors Linux, ``SubprocessSandbox(limits=...)``
    REFUSE de se construire plutôt que de tourner sans plafond."""

    memory_mb: float | None = None
    cpu_s: float | None = None
    fsize_mb: float | None = None

    def __post_init__(self) -> None:
        brut = (("memory_mb", self.memory_mb), ("cpu_s", self.cpu_s), ("fsize_mb", self.fsize_mb))
        fixes = {nom: valeur for nom, valeur in brut if valeur is not None}
        if not fixes:
            raise ValueError("SandboxLimits() sets no limit: give at least one of memory_mb, cpu_s, fsize_mb")
        for nom, valeur in fixes.items():
            if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
                raise ValueError(f"SandboxLimits.{nom} must be a finite number > 0, got {valeur!r}")
            try:
                fini = math.isfinite(valeur)
            except OverflowError:                       # un entier trop grand pour un flottant
                fini = False
            if not fini or valeur <= 0:
                raise ValueError(f"SandboxLimits.{nom} must be a finite number > 0, got {valeur!r}")
        # Ce que le noyau accepte (relecture : des valeurs « valides » échouaient à CHAQUE appel, à l'exécution) :
        # `RLIMIT_CPU` compte en secondes ENTIÈRES, `RLIMIT_AS` ne descend pas sous 1 Mo, et aucune valeur ne dépasse
        # un `long` C — au-delà `setrlimit` lève `OverflowError`.
        if self.memory_mb is not None and self.memory_mb < 1:
            raise ValueError(f"SandboxLimits.memory_mb must be >= 1 MB (an interpreter cannot run below), got {self.memory_mb!r}")
        if self.cpu_s is not None and self.cpu_s < 1:
            raise ValueError(f"SandboxLimits.cpu_s must be >= 1 (RLIMIT_CPU counts whole seconds), got {self.cpu_s!r}")
        for nom, fixe, facteur in (("memory_mb", self.memory_mb, _MO), ("fsize_mb", self.fsize_mb, _MO),
                                   ("cpu_s", self.cpu_s, 1)):
            if fixe is not None and fixe * facteur > _LIMITE_MAX:
                raise ValueError(f"SandboxLimits.{nom}={fixe!r} is out of range")

    def as_dict(self) -> dict[str, float]:
        """Seuls les plafonds FIXÉS, par nom — ce que `isolation()` déclare. `cpu_s` est déclaré tel qu'il est
        APPLIQUÉ : `RLIMIT_CPU` arrondit à la seconde au-dessus (1,5 → 2)."""
        brut = (("memory_mb", self.memory_mb), ("cpu_s", self.cpu_s), ("fsize_mb", self.fsize_mb))
        fixes = {nom: valeur for nom, valeur in brut if valeur is not None}
        if "cpu_s" in fixes:
            fixes["cpu_s"] = math.ceil(fixes["cpu_s"])
        return fixes


def _plafonds_applicables() -> bool:
    """Les rlimit sont appliqués ET mesurés sous Linux. Ailleurs on refuse : Windows n'a pas le module
    `resource`, et `RLIMIT_AS` n'est pas fiable sous macOS (non mesuré ici)."""
    return sys.platform.startswith("linux")


def _amorce_plafonds(limits: SandboxLimits | None) -> str:
    """Le préambule à coller DEVANT un runner : pose les rlimit puis s'efface. Chaîne vide sans plafond.

    Seuls des entiers calculés ici y sont écrits (aucune donnée de l'outil, aucune injection possible). Si
    `setrlimit` échoue, le préambule lève AVANT le code de l'outil : le processus sort en erreur, l'outil
    ne tourne jamais sans le plafond demandé (fail-closed)."""
    if limits is None:
        return ""
    reglages: list[tuple[str, int]] = []
    if limits.memory_mb is not None:
        reglages.append(("RLIMIT_AS", int(limits.memory_mb * _MO)))
    if limits.cpu_s is not None:
        reglages.append(("RLIMIT_CPU", max(1, math.ceil(limits.cpu_s))))
    if limits.fsize_mb is not None:
        reglages.append(("RLIMIT_FSIZE", max(1, int(limits.fsize_mb * _MO))))
    lignes = ["import resource as _resource"]
    lignes += [f"_resource.setrlimit(_resource.{nom}, ({valeur}, {valeur}))" for nom, valeur in reglages]
    lignes.append("del _resource")
    return "\n".join(lignes) + "\n"


def _indice_plafonds(limits: SandboxLimits | None) -> str:
    """Ce qu'on ajoute à une erreur du bac à sable quand des plafonds étaient en vigueur : sans lui, un
    `stdout='' stderr=''` (plafond CPU) ou un amorçage d'interpréteur coupé (plafond mémoire trop bas) ne laissent
    pas deviner la cause. Chaîne vide SANS plafond : les messages par défaut ne bougent pas d'un caractère."""
    if limits is None:
        return ""
    return (f" [sandbox limits in force: {limits.as_dict()} — a limit may be what stopped it: cpu_s kills the "
            "process, a memory_mb that is too low stops the interpreter from even starting (the floor depends on "
            "the Python build, 16 to 64 MB measured: start at 128 MB or more)]")


def _echec_du_runner(completed: "subprocess.CompletedProcess[str]", limits: SandboxLimits | None) -> str:
    """Le message d'un runner sorti en erreur. Avec des plafonds, il nomme ceux qui étaient en vigueur, dit si un
    signal a tué le processus, et montre la FIN de stderr (la ligne qui dit pourquoi), pas son amorçage."""
    sorties = f"stdout={completed.stdout[:500]!r} stderr={completed.stderr[:500]!r}"
    if limits is not None and completed.returncode < 0:
        try:
            nom = signal.Signals(-completed.returncode).name
        except ValueError:
            nom = f"signal {-completed.returncode}"
        return (
            f"Generated tool runner was killed by {nom}{_indice_plafonds(limits)}: a limit was most likely hit "
            f"(cpu_s kills the process; memory_mb and fsize_mb normally surface as MemoryError / OSError inside "
            f"the tool). {sorties}"
        )
    if limits is not None:
        return (f"Generated tool runner failed{_indice_plafonds(limits)}: stdout={completed.stdout[:500]!r} "
                f"stderr(tail)={completed.stderr[-500:]!r}")
    return f"Generated tool runner failed: {sorties}"


# ---------------------------------------------------------------------------
# Warm worker (0.22.0) — one persistent Python process per tool, so the ~115 ms
# interpreter start-up is paid once, not at every call. Opt-in
# (`SubprocessSandbox(warm=True)`). Line-delimited JSON over the child's stdio:
#   host→child : {"code":..., "path":...}               (once, at spawn)
#   child→host : {"t":"ready"} | {"t":"init_error",...}
#   host→child : {"args":{...},"context":{...}}         (one line per call)
#   child→host : {"t":"result","ok":...,"result":...}
# ---------------------------------------------------------------------------

_WARM_RUNNER_CODE = r"""
import contextlib
import io
import json
import sys
import traceback
import types

_REAL_STDOUT = sys.stdout  # saved before run() redirects stdout to a buffer


def _emit(obj):
    _REAL_STDOUT.write(json.dumps(obj, default=repr) + "\n")
    _REAL_STDOUT.flush()


init = json.loads(sys.stdin.readline() or "{}")
module = types.ModuleType("generated_tool")
try:
    exec(compile(init.get("code", ""), init.get("path", "generated_tool"), "exec"), module.__dict__)
    if not callable(getattr(module, "run", None)):
        raise RuntimeError("generated tool defines no run(args, context)")
    _emit({"t": "ready"})
except Exception as exc:
    _emit({"t": "init_error", "error": "%s: %s" % (type(exc).__name__, exc)})
    sys.exit(0)

for line in sys.stdin:
    if not line.strip():
        continue
    request = json.loads(line)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = module.run(request.get("args", {}), request.get("context", {}))
        _emit({"t": "result", "ok": True, "result": result, "stdout": buf.getvalue()})
    except Exception as exc:
        _emit({
            "t": "result",
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(limit=8),
        })
"""


class _WarmWorker:
    """Un processus Python persistant qui sert les appels d'UN outil, un à la fois.

    Recyclé après ``max_calls`` appels (l'état des variables globales de l'outil
    survit d'un appel à l'autre dans un même worker : on borne cette durée), et
    TUÉ puis relancé au dépassement du délai — un outil qui boucle ne bloque
    jamais le suivant. Environnement épuré comme en mode normal."""

    def __init__(
        self, path: Path, code: str, sha: str, timeout: float, max_calls: int,
        limits: SandboxLimits | None = None,
    ) -> None:
        self.path, self.code, self.sha = path, code, sha
        self.timeout, self.max_calls = timeout, max_calls
        self.limits = limits
        self.calls = 0
        self.proc: subprocess.Popen[str] | None = None
        self.verrou = threading.Lock()
        self._stderr: list[str] = []

    # -- cycle de vie -------------------------------------------------------

    def _lire_ligne(self, proc: subprocess.Popen[str]) -> str:
        """Une ligne du worker, sous délai : au dépassement le processus est tué."""
        minuteur = threading.Timer(self.timeout, proc.kill)
        minuteur.start()
        try:
            assert proc.stdout is not None
            return proc.stdout.readline()
        finally:
            minuteur.cancel()

    def _demarrer(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-I", "-S", "-B", "-u", "-c", _amorce_plafonds(self.limits) + _WARM_RUNNER_CODE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
            env=_child_env(), cwd=str(self.path.parent),
        )
        self._stderr = []
        threading.Thread(target=lambda: self._stderr.extend(proc.stderr or []), daemon=True).start()
        self.proc, self.calls = proc, 0
        try:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps({"code": self.code, "path": str(self.path)}, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            ligne = self._lire_ligne(proc)
            msg = json.loads(ligne) if ligne.strip() else {}
        except (OSError, ValueError) as exc:
            self.arreter()
            raise ToolError(f"Warm worker failed to start: {exc}{_indice_plafonds(self.limits)}") from exc
        if msg.get("t") != "ready":
            erreur = msg.get("error") or f"no handshake. stderr={''.join(self._stderr)[:300]!r}"
            self.arreter()
            raise ToolError(f"Generated tool failed to load: {erreur}{_indice_plafonds(self.limits)}")

    def arreter(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        for flux in (proc.stdout, proc.stderr):
            try:
                if flux:
                    flux.close()
            except OSError:
                pass

    # -- un appel -----------------------------------------------------------

    def appeler(self, args: JsonDict, context: JsonDict) -> JsonDict:
        with self.verrou:
            if self.proc is None or self.proc.poll() is not None or self.calls >= self.max_calls:
                self.arreter()
                self._demarrer()
            proc = self.proc
            assert proc is not None and proc.stdin is not None
            requete = json.dumps({"args": args, "context": context}, ensure_ascii=False)
            try:
                proc.stdin.write(requete + "\n")
                proc.stdin.flush()
            except OSError as exc:
                self.arreter()
                raise ToolError(f"Warm worker is gone ({exc}); the next call starts a fresh one") from exc
            ligne = self._lire_ligne(proc)
            self.calls += 1
            if not ligne:
                # Délai dépassé (processus tué) ou mort : le suivant repart à neuf.
                self.arreter()
                raise ToolError(
                    f"Generated tool timed out after {self.timeout}s (or ended without a result). "
                    f"stderr={''.join(self._stderr)[:300]!r}{_indice_plafonds(self.limits)}"
                )
            try:
                msg = json.loads(ligne)
            except json.JSONDecodeError as exc:
                self.arreter()
                raise ToolError(f"Sandbox protocol error on line {ligne[:200]!r}") from exc
            return {k: v for k, v in msg.items() if k != "t"}


def _fermer_workers(workers: dict[str, _WarmWorker]) -> None:
    for worker in list(workers.values()):
        worker.arreter()
    workers.clear()


def _drive_bridge(
    cmd: list[str],
    init_payload: dict[str, Any],
    host_functions: dict[str, Callable[..., Any]],
    timeout: float,
    on_timeout: Callable[[], Any] | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    diagnostic: str = "",
) -> JsonDict:
    """Run the interactive runner and service host_function calls until the
    tool returns. Each call name MUST be in ``host_functions`` or it is
    refused — so a tool can only reach the callbacks the host whitelisted.

    ``env``/``cwd`` : le mode pont du `SubprocessSandbox` héritait de TOUT
    l'environnement de l'hôte (clés API comprises) et de son répertoire courant,
    alors que le mode normal les retire déjà (0.22.0). Le chemin Docker passe
    ``env=None`` : c'est la CLI docker qui en a besoin, le conteneur, lui, ne
    reçoit que ses `-e` explicites."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env,
        cwd=cwd,
    )
    stderr_chunks: list[str] = []
    drainer = threading.Thread(target=lambda: stderr_chunks.extend(proc.stderr or []), daemon=True)
    drainer.start()
    timed_out = {"value": False}

    def _kill() -> None:
        timed_out["value"] = True
        try:
            proc.kill()
        except Exception:
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
                    f"Sandbox ended without a result. stderr={''.join(stderr_chunks)[:400]!r}{diagnostic}"
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
                    # LA porte (D1, gate.py) AVANT l'exécution : du code écrit par le modèle
                    # appelle ici une fonction de l'hôte — avec la politique, la garde
                    # trifecta et la teinte du run qui l'a lancé. Hors d'un run d'agent
                    # (`active_gate()` est None), rien ne change.
                    gate = active_gate()
                    refus = None if gate is None else gate.decide(
                        name, msg.get("args") or {},
                        spec=getattr(fn, "__autoagent_tool_spec__", None))
                    if refus is not None:
                        resp = {"ok": False, "error": f"ToolPolicyDenied: {refus}"}
                    else:
                        try:
                            resp = {"ok": True, "result": fn(**(msg.get("args") or {}))}
                        except Exception as exc:
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
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


ALWAYS_BANNED_CALLS = {"eval", "exec", "compile", "__import__", "input", "breakpoint"}

# Les mêmes noms, mais atteints par ATTRIBUT (``x.compile(...)``). ``compile``
# en est RETIRÉ : ``re.compile`` est l'usage le plus courant d'un outil légitime
# (regex précompilée) et était refusé à tort (0.22.0). Le ``compile`` NU reste
# interdit (il est toujours dans ALWAYS_BANNED_CALLS, testé sur un ``ast.Name``),
# et l'accès au vrai ``compile`` du runtime passe par ``builtins`` /
# ``__builtins__``, tous deux bannis — l'attribut ``.compile`` d'un autre objet
# n'est donc pas une porte vers l'exécution de code.
ATTRIBUTE_BANNED_CALLS = ALWAYS_BANNED_CALLS - {"compile"}

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
    # ``import builtins`` rendait ``builtins.exec`` / ``.eval`` / ``.__import__``
    # alors que ``exec`` nu était refusé — contournement trivial (0.22.0).
    "builtins",
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
    """Exécute un outil généré dans un sous-processus Python ``-I -S``, env épuré.

    ``warm=True`` (0.22.0, opt-in) : UN worker persistant par outil au lieu d'un
    processus par appel — le démarrage de l'interpréteur (~115 ms mesurés) n'est
    payé qu'une fois. Contreparties, dites : les variables GLOBALES d'un outil
    survivent d'un appel à l'autre dans un même worker (borné par
    ``warm_max_calls`` : le worker est recyclé ensuite) ; les appels d'un même
    outil sont sérialisés ; avec ``host_functions`` (pont) on garde le processus
    par appel. Un dépassement de délai TUE le worker, le suivant repart à neuf.
    ``warm_max_workers`` plafonne les processus vivants (le plus ancien est fermé).
    Appelle ``close()`` (ou utilise ``with``) pour fermer les workers ; sinon ils
    sont fermés à la sortie du processus.

    ``limits=SandboxLimits(...)`` (0.23.1, opt-in, Linux) pose des plafonds de
    mémoire, de CPU et de taille de fichier ; sans eux, RIEN n'est borné hormis le
    délai mur. Ce ne sont pas une frontière — voir :class:`SandboxLimits` et
    :meth:`isolation`, qui dit ce que ce bac à sable fait réellement respecter.
    """

    timeout: float = 10.0
    warm: bool = False
    warm_max_calls: int = 200
    warm_max_workers: int = 8
    limits: SandboxLimits | None = None
    _workers: dict[str, _WarmWorker] = field(default_factory=dict, init=False, repr=False, compare=False)
    _verrou: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.limits is not None:
            if not isinstance(self.limits, SandboxLimits):
                raise TypeError(f"limits must be a SandboxLimits (or None), got {type(self.limits).__name__}")
            if not _plafonds_applicables():
                raise ToolError(
                    f"SubprocessSandbox(limits=...): resource limits are applied (and measured) on Linux only; on "
                    f"this platform ({sys.platform}) the tool would run with NO limit, so this is refused. Drop "
                    "limits=, or use DockerSandbox (memory, cpus, pids_limit)."
                )
            if self.warm and self.limits.cpu_s is not None:
                raise ValueError(
                    "limits.cpu_s does not apply to a warm worker: RLIMIT_CPU counts the CPU time of the whole "
                    "process, which serves many calls. Use warm=False, or drop cpu_s (memory_mb and fsize_mb "
                    "stay allowed)."
                )
        # Fermeture garantie à la sortie, sans référencer `self` (sinon il ne
        # serait jamais ramassé) : on ne passe que le dictionnaire des workers.
        weakref.finalize(self, _fermer_workers, self._workers)

    def isolation(self) -> dict[str, Any]:
        """Ce que ce bac à sable fait RÉELLEMENT respecter (0.23.1) — à lire avant de lui confier du
        code qu'on ne maîtrise pas. Un sous-processus ``-I -S`` à environnement épuré derrière une liste
        d'interdits AST : PAS une frontière (``os_boundary`` est faux), ni réseau ni système de fichiers
        isolés (``allow_network`` n'est pas appliqué ici). ``limits`` ne liste que les plafonds RÉELS : le
        délai mur, plus ceux de ``SandboxLimits`` s'ils sont posés."""
        plafonds: dict[str, float] = {"timeout_s": float(self.timeout)}
        if self.limits is not None:
            plafonds.update(self.limits.as_dict())
        return {
            "kind": "subprocess",
            "os_boundary": False,
            "network_isolated": False,
            "filesystem_isolated": False,
            "env_scrubbed": True,
            "limits": plafonds,
        }

    def close(self) -> None:
        """Ferme tous les workers chauds (sans effet hors ``warm=True``)."""
        with self._verrou:
            _fermer_workers(self._workers)

    def __enter__(self) -> "SubprocessSandbox":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _run_warm(self, path: Path, args: JsonDict, context: JsonDict) -> JsonDict:
        code = path.read_text(encoding="utf-8")
        sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        cle = str(path)
        a_fermer: list[_WarmWorker] = []
        with self._verrou:
            # Retrait PUIS réinsertion dans la même section critique : aucun autre
            # thread ne voit l'outil « absent » et n'en crée un second ; réinséré, le
            # plus récemment utilisé passe en queue (l'éviction prend la tête).
            worker = self._workers.pop(cle, None)
            if worker is not None and worker.sha != sha:
                a_fermer.append(worker)            # fichier réécrit : on ne sert pas l'ancien code
                worker = None
            if worker is None:
                while len(self._workers) >= max(1, self.warm_max_workers):
                    a_fermer.append(self._workers.pop(next(iter(self._workers))))
                worker = _WarmWorker(path, code, sha, self.timeout, max(1, self.warm_max_calls), self.limits)
            self._workers[cle] = worker
        for ancien in a_fermer:                    # hors du verrou global ; attend l'appel en cours
            with ancien.verrou:
                ancien.arreter()
        return worker.appeler(args, context)

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
            cmd = [sys.executable, "-X", "utf8", "-I", "-S", "-u", "-c", _amorce_plafonds(self.limits) + _BRIDGE_RUNNER_CODE]
            return _drive_bridge(
                cmd, {"code": code, "args": args, "context": context or {}}, host_functions, self.timeout,
                env=_child_env(), cwd=str(path.parent), diagnostic=_indice_plafonds(self.limits),
            )
        if self.warm:
            return self._run_warm(path, args, context or {})
        payload = json.dumps({"args": args, "context": context or {}}, ensure_ascii=False)
        try:
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "-I", "-S", "-B", "-c", _amorce_plafonds(self.limits) + RUNNER_CODE,
                 str(path)],
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
            raise ToolError(_echec_du_runner(completed, self.limits))
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
    # Crochet d'observation (0.22.0) : `observer(nom, ok, erreur)` après CHAQUE
    # appel — la bibliothèque persistante y compte les appels et les erreurs.
    # Observabilité = fail-open : un crochet qui plante ne casse jamais l'appel.
    observer: "Callable[[str, bool, str | None], None] | None" = None
    # Fonctions de l'hôte rendues appelables depuis l'outil (pont, 0.22.0) ; un
    # `host_functions=` passé à l'appel est prioritaire.
    host_functions: "dict[str, Callable[..., Any]] | None" = None

    # Lu par le registre : un outil écrit par le modèle ne reçoit PAS le `context`
    # du run (0.22.0). Il partait tel quel dans le bac à sable — l'id utilisateur,
    # un jeton, un handle de base — et un handle non sérialisable faisait échouer
    # TOUS les outils générés (`TypeError: not JSON serializable`). C'est le
    # modèle de confiance déjà écrit dans `approval.py` : en bac à sable, aucun
    # objet de l'hôte ; l'accès passe par les `host_functions` en liste blanche.
    __autoagent_sandboxed__ = True

    def __call__(
        self,
        context: JsonDict | None = None,
        *,
        host_functions: dict[str, Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        allow_network = "network" in (self.spec.permissions or [])
        try:
            result = self.sandbox.run_python_tool(
                self.file_path,
                kwargs,
                context=context,
                allow_network=allow_network,
                host_functions=host_functions if host_functions is not None else self.host_functions,
            )
            if not result.get("ok"):
                raise ToolError(result.get("error") or "Generated tool failed")
        except Exception as exc:
            self._observer_dit(False, f"{type(exc).__name__}: {exc}")
            raise
        self._observer_dit(True, None)
        return result.get("result")

    def _observer_dit(self, ok: bool, erreur: str | None) -> None:
        if self.observer is None:
            return
        try:
            self.observer(self.spec.name, ok, erreur)
        except Exception:                                   # fail-open
            pass


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


# Variables qu'un processus tiers peut hériter sans risque : la liste EXACTE du
# SDK MCP officiel (`mcp.client.stdio.DEFAULT_INHERITED_ENV_VARS`). Tout le
# reste — clés API, mots de passe de base, jetons — n'a rien à faire chez un
# serveur MCP tiers ni dans une commande qui exécute du code écrit par le modèle.
_SAFE_INHERITED_ENV = (
    ("APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "PATHEXT",
     "PROCESSOR_ARCHITECTURE", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "USERNAME", "USERPROFILE")
    if os.name == "nt"
    else ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER")
)


def safe_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """L'environnement minimal hérité, plus ``extra`` (0.22.0).

    Même règle que le SDK MCP officiel : on ne garde que les variables
    système inoffensives, en écartant les fonctions bash exportées (valeurs
    commençant par ``()``), puis on ajoute ce que l'hôte passe EXPLICITEMENT.
    """
    env = {k: v for k in _SAFE_INHERITED_ENV
           if (v := os.environ.get(k)) is not None and not v.startswith("()")}
    if extra:
        env.update(extra)
    return env


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
            except Exception:
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

    def isolation(self) -> dict[str, Any]:
        """Ce que ce bac à sable fait RÉELLEMENT respecter (0.23.1) : une vraie frontière de l'OS (conteneur
        éphémère, racine en lecture seule, capacités retirées, utilisateur non-root). ``network_isolated`` :
        le réseau est COUPÉ (``--network none``) tant que l'outil n'a pas la permission ``network``."""
        return {
            "kind": "docker",
            "os_boundary": True,
            "network_isolated": True,
            "filesystem_isolated": True,
            "env_scrubbed": True,
            "limits": {
                "timeout_s": float(self.timeout), "memory": self.memory, "cpus": str(self.cpus),
                "pids": self.pids_limit,
            },
        }

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


def discard_generated_tool(tool: GeneratedPythonTool | Path) -> None:
    """Un outil refusé ne doit pas rester chargeable : on le retire du disque —
    son bytecode aussi, sinon un essai suivant de même nom, même taille et même
    seconde exécuterait le code REFUSÉ (vu en CI). Partagé par la synthèse et
    `create_python_tool` (0.22.0).

    Accepte un ``GeneratedPythonTool`` OU un ``Path`` : un outil peut être refusé
    AVANT d'être chargé (self-tests qui plantent, chargement qui échoue), où seul
    le chemin existe (0.22.0)."""
    path = tool if isinstance(tool, Path) else tool.file_path
    try:
        path.unlink(missing_ok=True)
        cache = path.parent / "__pycache__"
        if cache.is_dir():
            for pyc in cache.glob(f"{path.stem}.*.pyc"):
                pyc.unlink(missing_ok=True)
    except OSError:
        pass


def make_sandbox(
    *, prefer_docker: bool = True, timeout: float = 10.0, image: str = "python:3.11-slim",
    require_docker: bool = False,
) -> "SubprocessSandbox | DockerSandbox":
    """Return a DockerSandbox when a Docker daemon is available (real
    OS-level isolation), else the hardened SubprocessSandbox as a fallback.

    The fallback is NOT an isolation boundary (an AST denylist behind ``-I -S``),
    so it is no longer silent (0.23.1) : a warning, once per process, says so.
    ``require_docker=True`` refuses instead of falling back — raises ``ToolError``
    when no usable Docker daemon is reachable. Pass ``prefer_docker=False`` to
    choose the subprocess knowingly (no warning). ``sandbox.isolation()`` tells
    what the returned sandbox really enforces."""
    if require_docker and not prefer_docker:
        raise ValueError("require_docker=True contradicts prefer_docker=False")
    if prefer_docker and docker_available():
        return DockerSandbox(image=image, timeout=timeout)
    if require_docker:
        raise ToolError(
            "make_sandbox(require_docker=True): no usable Docker daemon (Linux containers) is reachable, and "
            "the fallback SubprocessSandbox is not an isolation boundary, so nothing is run. Start Docker, or "
            "drop require_docker=True to accept the weaker sandbox."
        )
    if prefer_docker:
        warn_once(
            _log, "sandbox.make_sandbox_fallback",
            "make_sandbox() found no usable Docker daemon and fell back to SubprocessSandbox (AST denylist + "
            "`-I -S`, not an isolation boundary). Pass require_docker=True to refuse instead of falling back, "
            "or prefer_docker=False to choose the subprocess knowingly.",
        )
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
        # Un module interdit à l'IMPORT restait joignable par ATTRIBUT : presque
        # tout module de la stdlib importe `os` et l'expose (`logging.os`,
        # `platform.os`, `random._os`…). `logging.os.remove(...)` effaçait un
        # fichier sans aucune permission, alors que `import os` était refusé
        # (0.22.0). Même règle pour les modules réseau / fichiers sans la
        # permission correspondante. Le validateur reste une liste d'interdits —
        # la vraie frontière est le DockerSandbox.
        if isinstance(node, ast.Attribute):
            attr = node.attr.lstrip("_")
            if attr in ALWAYS_BANNED_MODULES:
                raise ToolValidationError(f"Access to module `{node.attr}` through an attribute is not allowed")
            if not allow_network and attr in NETWORK_MODULES:
                raise ToolValidationError(f"Network module `{node.attr}` requires 'network' permission")
            if not allow_filesystem and attr in FILESYSTEM_MODULES:
                raise ToolValidationError(f"Filesystem module `{node.attr}` requires a filesystem.* permission")

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
            # Un appel NU (``compile(...)``) : toute la liste s'applique. Un appel
            # par ATTRIBUT (``re.compile(...)``) : ``compile`` est toléré, le reste
            # non. _call_name ne distingue pas les deux ; on teste le type du nœud.
            if isinstance(node.func, ast.Name) and call_name in ALWAYS_BANNED_CALLS:
                raise ToolValidationError(f"Call is not allowed: {call_name}")
            if isinstance(node.func, ast.Attribute) and call_name in ATTRIBUTE_BANNED_CALLS:
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
