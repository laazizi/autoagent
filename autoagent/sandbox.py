from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import sysconfig
import threading
import time
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
    "Confinement",
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
# Confinement par le noyau (0.25.0) — Linux x86_64 ; posé d'office quand l'hôte ne choisit pas de bac
# (`_bac_par_defaut`), exigé avec `Agent(require_boundary=True)` (`_bac_avec_frontiere`).
# La liste d'interdits AST se contourne, et un outil qui la contourne avait les
# droits du processus hôte : ses fichiers, le réseau, ses clés (lisibles par
# `/proc/<pid>/environ` alors que l'environnement de l'enfant est épuré), ses
# voisins. `SubprocessSandbox(confine=Confinement())` fait poser par le RUNNER,
# avant le code de l'outil, deux mécanismes que le noyau applique quoi que le code
# écrive ensuite :
#   * Landlock : le processus se retire l'accès aux fichiers, sauf une liste
#     blanche (irrévocable, hérité par ses enfants, vaut aussi pour root) ;
#   * seccomp : un filtre d'appels système pour ce que Landlock ne couvre pas —
#     le réseau, le lancement de programmes, les autres processus, les droits et
#     dates des fichiers, les espaces de noms.
# Un refus rend une ERREUR au code (`PermissionError`), pas la mort du processus :
# l'outil échoue, le modèle voit une erreur d'outil. Même forme que les plafonds :
# un préambule collé devant le runner, sans `preexec_fn`, chaîne vide sans
# `confine=` (le runner par défaut ne bouge pas d'un octet), et fail-closed — si
# le confinement ne se pose pas, l'outil ne tourne pas. Le noyau reste PARTAGÉ :
# une faille du noyau traverse cette frontière, comme elle traverse Docker.
# ---------------------------------------------------------------------------

_LANDLOCK_ABI_MIN = 3                # noyau 6.2 : en dessous, vider un fichier (`truncate`) échappe à Landlock
_LL_TOUS = (1 << 15) - 1             # les quinze droits « fichiers » des ABI 1 à 3 : tout ce qui n'est pas accordé est refusé
_LL_ECRIRE_FICHIER, _LL_LIRE_FICHIER, _LL_LIRE_DOSSIER = 1 << 1, 1 << 2, 1 << 3
_LL_RETIRER_DOSSIER, _LL_RETIRER_FICHIER, _LL_CREER_DOSSIER, _LL_CREER_FICHIER = 1 << 4, 1 << 5, 1 << 7, 1 << 8
_LL_VIDER = 1 << 14
# Ni EXECUTE (lancer un programme) ni REFER (lier ou déplacer d'un dossier à l'autre) ne sont jamais accordés.
_LL_LECTURE = _LL_LIRE_FICHIER | _LL_LIRE_DOSSIER
_LL_ECRITURE = (_LL_LECTURE | _LL_ECRIRE_FICHIER | _LL_VIDER | _LL_CREER_FICHIER | _LL_CREER_DOSSIER
                | _LL_RETIRER_FICHIER | _LL_RETIRER_DOSSIER)

# Ce que l'interpréteur doit pouvoir LIRE pour qu'un outil ordinaire tourne (imports paresseux de la
# bibliothèque standard, bibliothèques partagées, fuseaux horaires). Rien ici n'est un secret.
_DOSSIERS_SYSTEME = ("/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/share/zoneinfo")
_PERIPHERIQUES = (("/dev/null", _LL_LIRE_FICHIER | _LL_ECRIRE_FICHIER), ("/dev/urandom", _LL_LIRE_FICHIER))
# Les tables de types MIME que `mimetypes` ouvre au premier appel (`mimetypes.knownfiles`) : sans elles,
# `mimetypes.guess_type()` levait PermissionError sous confinement (trouvé par le banc des extraits légitimes).
_FICHIERS_SYSTEME = ("/etc/mime.types", "/etc/httpd/mime.types", "/etc/httpd/conf/mime.types",
                     "/etc/apache/mime.types", "/etc/apache2/mime.types", "/usr/local/etc/httpd/conf/mime.types",
                     "/usr/local/lib/netscape/mime.types", "/usr/local/etc/mime.types")
# Réseau rouvert (`Confinement(network=True)` ET outil porteur de la permission) : la résolution des noms
# et les certificats se lisent dans ces fichiers.
_FICHIERS_RESEAU = ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/host.conf", "/etc/gai.conf",
                    "/etc/services", "/etc/protocols")
_DOSSIERS_RESEAU = ("/etc/ssl", "/etc/pki", "/etc/ca-certificates", "/usr/share/ca-certificates",
                    "/usr/local/share/ca-certificates")

# Numéros d'appels système x86_64 (`asm/unistd_64.h` ; `tests/test_confinement.py` les recoupe avec les
# en-têtes du noyau quand ils sont installés). Chaque appel ci-dessous rend EPERM.
_APPELS_RESEAU = {"socket": 41, "connect": 42, "accept": 43, "bind": 49, "listen": 50, "accept4": 288}
_APPELS_REFUSES = {
    # lancer un programme
    "execve": 59, "execveat": 322,
    # agir sur un AUTRE processus (kill, tgkill et prlimit64 sont filtrés sur leur cible, plus bas)
    "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311, "kcmp": 312, "pidfd_open": 434,
    "pidfd_getfd": 438, "pidfd_send_signal": 424, "process_madvise": 440, "process_mrelease": 448,
    "tkill": 200, "rt_sigqueueinfo": 129, "rt_tgsigqueueinfo": 297, "setpriority": 141,
    "sched_setparam": 142, "sched_setscheduler": 144, "sched_setaffinity": 203, "sched_setattr": 314,
    "ioprio_set": 251, "migrate_pages": 256, "move_pages": 279,
    # droits, propriétaire, attributs et dates d'un fichier : Landlock ne les couvre pas
    "chmod": 90, "fchmod": 91, "fchmodat": 268, "fchmodat2": 452, "chown": 92, "fchown": 93, "lchown": 94,
    "fchownat": 260, "setxattr": 188, "lsetxattr": 189, "fsetxattr": 190, "removexattr": 197,
    "lremovexattr": 198, "fremovexattr": 199, "utime": 132, "utimes": 235, "futimesat": 261, "utimensat": 280,
    # io_uring exécute des opérations sans repasser par ce filtre
    "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    # montages, espaces de noms, noyau
    "mount": 165, "umount2": 166, "pivot_root": 155, "chroot": 161, "unshare": 272, "setns": 308,
    "open_tree": 428, "move_mount": 429, "fsopen": 430, "fsconfig": 431, "fsmount": 432, "fspick": 433,
    "mount_setattr": 442, "swapon": 167, "swapoff": 168, "reboot": 169, "kexec_load": 246,
    "kexec_file_load": 320, "init_module": 175, "finit_module": 313, "delete_module": 176, "bpf": 321,
    "perf_event_open": 298, "userfaultfd": 323, "add_key": 248, "request_key": 249, "keyctl": 250,
    "acct": 163, "settimeofday": 164, "clock_settime": 227, "adjtimex": 159, "clock_adjtime": 305,
    "sethostname": 170, "setdomainname": 171, "iopl": 172, "ioperm": 173, "quotactl": 179,
    "quotactl_fd": 443, "name_to_handle_at": 303, "open_by_handle_at": 304, "personality": 135,
    "syslog": 103, "vhangup": 153, "modify_ldt": 154, "fanotify_init": 300, "fanotify_mark": 301,
    "lsm_set_self_attr": 460,
    # mémoire et files partagées entre processus (System V, POSIX)
    "shmget": 29, "shmat": 30, "shmctl": 31, "semget": 64, "semop": 65, "semctl": 66, "semtimedop": 220,
    "msgget": 68, "msgsnd": 69, "msgrcv": 70, "msgctl": 71, "mq_open": 240, "mq_unlink": 241,
    "mq_timedsend": 242, "mq_timedreceive": 243, "mq_notify": 244, "mq_getsetattr": 245,
}
_DERNIER_APPEL_CONNU = 462           # mseal (noyau 6.10). Au-delà : ENOSYS — un appel ajouté par un futur noyau est refusé par défaut
_PID_A_POSER = 0xFFFFFFFF            # remplacé dans l'enfant par son propre pid (jamais un pid ni un numéro d'appel réels)
_CLONE_ESPACES = (0x00000080 | 0x00020000 | 0x02000000 | 0x04000000 | 0x08000000 | 0x10000000
                  | 0x20000000 | 0x40000000)     # CLONE_NEWTIME, NEWNS, NEWCGROUP, NEWUTS, NEWIPC, NEWUSER, NEWPID, NEWNET


def _filtre_seccomp(reseau_ouvert: bool) -> tuple[bytes, tuple[int, ...]]:
    """Le programme BPF du filtre, et les positions (en octets) où l'enfant écrit son propre pid.

    Fonction PURE (aucun appel au noyau) : testée instruction par instruction sur toute plateforme. Dans
    l'ordre : une autre architecture que x86_64 TUE le processus (la table 32 bits a d'autres numéros) ; un
    numéro au-delà du dernier connu rend ENOSYS, ce qui couvre aussi la table x32 ; `clone3` rend ENOSYS (ses
    drapeaux sont derrière un pointeur, illisibles ici : la libc retombe sur `clone`, dont on lit les
    drapeaux) ; chaque appel refusé rend EPERM ; `kill` / `tgkill` ne passent que vers soi, `prlimit64` que
    pour soi, `clone` que sans nouvel espace de noms ; tout le reste passe."""
    permis, eperm, enosys, tuer = 0x7FFF0000, 0x00050000 | 1, 0x00050000 | 38, 0x80000000
    instructions: list[tuple[int, int, int, int]] = []
    positions: list[int] = []

    def charger(decalage: int) -> None:
        instructions.append((0x20, 0, 0, decalage))            # BPF_LD | BPF_W | BPF_ABS

    def si_egal(valeur: int, si_vrai: int, si_faux: int) -> None:
        instructions.append((0x15, si_vrai, si_faux, valeur))  # BPF_JMP | BPF_JEQ | BPF_K

    def rendre(action: int) -> None:
        instructions.append((0x06, 0, 0, action))              # BPF_RET | BPF_K

    def si_moi(si_vrai: int, si_faux: int) -> None:
        positions.append(len(instructions) * 8 + 4)
        si_egal(_PID_A_POSER, si_vrai, si_faux)

    charger(4)                                                 # seccomp_data.arch
    si_egal(0xC000003E, 1, 0)                                  # AUDIT_ARCH_X86_64
    rendre(tuer)
    charger(0)                                                 # seccomp_data.nr
    instructions.append((0x25, 0, 1, _DERNIER_APPEL_CONNU))    # BPF_JGT : au-delà du dernier appel connu
    rendre(enosys)
    si_egal(435, 0, 1)                                         # clone3
    rendre(enosys)
    refuses = dict(_APPELS_REFUSES) if reseau_ouvert else {**_APPELS_REFUSES, **_APPELS_RESEAU}
    for numero in sorted(set(refuses.values())):
        si_egal(numero, 0, 1)
        rendre(eperm)
    for numero in (62, 234):                                   # kill(pid, …) et tgkill(tgid, …) : vers soi seulement
        si_egal(numero, 0, 4)
        charger(16)                                            # premier argument, mot bas : le pid que lit le noyau
        si_moi(0, 1)
        rendre(permis)
        rendre(eperm)
    si_egal(302, 0, 5)                                         # prlimit64(pid, …) : 0 ou soi
    charger(16)
    si_egal(0, 1, 0)
    si_moi(0, 1)
    rendre(permis)
    rendre(eperm)
    si_egal(56, 0, 4)                                          # clone(drapeaux, …) : un fil ou un enfant, pas un espace de noms
    charger(16)
    instructions.append((0x45, 0, 1, _CLONE_ESPACES))          # BPF_JSET
    rendre(eperm)
    rendre(permis)
    rendre(permis)
    return b"".join(struct.pack("<HBBI", *i) for i in instructions), tuple(positions)


def _poser_pid(programme: bytes, positions: tuple[int, ...], pid: int) -> bytes:
    """Le filtre prêt pour le processus `pid` (ce que fait l'enfant, avec le sien, avant de l'installer)."""
    tampon = bytearray(programme)
    for position in positions:
        struct.pack_into("<I", tampon, position, pid)
    return bytes(tampon)


# Le préambule exécuté par l'ENFANT, avant le runner. Tout échec sort AVANT le code de l'outil (fail-closed).
_AMORCE_CONFINEMENT = r'''
def _confiner():
    import ctypes, os, struct, sys
    libc = ctypes.CDLL(None, use_errno=True)
    appel = libc.syscall
    appel.restype = ctypes.c_long
    L = ctypes.c_long

    def echec(quoi):
        numero = ctypes.get_errno()
        raise OSError(numero, "%s: %s" % (quoi, os.strerror(numero)))

    if libc.prctl(L(38), L(1), L(0), L(0), L(0)) != 0:                     # PR_SET_NO_NEW_PRIVS
        echec("no_new_privs")
    if appel(L(444), None, L(0), L(1)) < @ABI_MIN@:                          # landlock_create_ruleset(VERSION)
        raise OSError("Landlock ABI @ABI_MIN@ is not available in this process")
    droits = struct.pack("Q", @TOUS@)
    jeu = appel(L(444), droits, L(len(droits)), L(0))
    if jeu < 0:
        echec("landlock_create_ruleset")
    regles = list(@REGLES@)
    if len(sys.argv) > 1:
        regles.append((sys.argv[1], @LIRE_FICHIER@))                         # le fichier de l'outil, et lui seul
    for chemin, accorde in regles:
        fd = os.open(chemin, os.O_PATH | os.O_CLOEXEC)
        try:
            if appel(L(445), L(jeu), L(1), struct.pack("=Qi", accorde, fd), L(0)) != 0:
                echec("landlock_add_rule(%s)" % chemin)
        finally:
            os.close(fd)
    if appel(L(446), L(jeu), L(0)) != 0:
        echec("landlock_restrict_self")
    os.close(jeu)

    filtre = bytearray(bytes.fromhex("@FILTRE@"))
    for position in @POSITIONS@:
        struct.pack_into("<I", filtre, position, os.getpid())
    tampon = ctypes.create_string_buffer(bytes(filtre), len(filtre))

    class Programme(ctypes.Structure):
        _fields_ = [("longueur", ctypes.c_ushort), ("instructions", ctypes.c_void_p)]

    programme = Programme(len(filtre) // 8, ctypes.addressof(tampon))
    if libc.prctl(L(22), L(2), ctypes.byref(programme), L(0), L(0)) != 0:   # PR_SET_SECCOMP, MODE_FILTER
        echec("seccomp")


try:
    _confiner()
except BaseException as _exc:
    raise SystemExit("autoagent confinement could not be applied (%s: %s): the tool was NOT run"
                     % (type(_exc).__name__, _exc))
del _confiner
'''


@dataclass(frozen=True)
class Confinement:
    """Ce que le noyau laisse au code d'un :class:`SubprocessSandbox` confiné (opt-in, Linux x86_64).

    ``Confinement()`` : aucun fichier hors de l'interpréteur, pas de réseau, aucun programme lancé, aucun
    signal vers un autre processus. Restent lisibles sans rien accorder : la bibliothèque standard, les
    bibliothèques partagées du système, les fuseaux horaires, et le fichier de l'outil lui-même. Rien en
    écriture.

    * ``read`` / ``write`` : dossiers accordés par l'HÔTE (jamais par le modèle), chemins absolus de dossiers
      existants. ``write`` donne aussi la lecture. On y crée, écrit, vide et supprime ; on n'y change ni les
      droits ni les dates (``shutil.copy`` échoue, ``shutil.copyfile`` passe), on n'y crée pas de lien.
    * ``network`` : ``False`` (défaut) coupe le réseau même pour un outil qui porte la permission
      ``network``. ``True`` le rouvre pour ces outils-là seulement, comme ``DockerSandbox``.

    Non couverts, dits : le noyau est partagé ; mémoire, CPU et taille de fichier relèvent de
    ``limits=`` ; le nombre de processus n'est pas borné (un enfant hérite du confinement, il n'en sort
    pas) ; l'existence et la taille d'un fichier restent visibles (``stat``), pas son contenu ni la liste
    de son dossier ; ce qui passe par le pont ``call_host`` relève de la porte de décision."""

    read: tuple[str, ...] = ()
    write: tuple[str, ...] = ()
    network: bool = False

    def __post_init__(self) -> None:
        for nom in ("read", "write"):
            valeur = getattr(self, nom)
            if isinstance(valeur, (str, bytes, os.PathLike)):
                raise TypeError(f"Confinement.{nom} must be a list of directories, not a single path: {valeur!r}")
            chemins: list[str] = []
            for chemin in valeur:
                if not isinstance(chemin, (str, os.PathLike)):
                    raise TypeError(f"Confinement.{nom}: each directory must be a path, got {chemin!r}")
                texte = os.fspath(chemin)
                if not isinstance(texte, str) or not texte.startswith("/"):
                    raise ValueError(f"Confinement.{nom}: {texte!r} is not an absolute path")
                chemins.append(texte)
            object.__setattr__(self, nom, tuple(chemins))
        if not isinstance(self.network, bool):
            raise TypeError(f"Confinement.network must be True or False, got {self.network!r}")


def _landlock_abi() -> int:
    """La version de Landlock que le noyau offre à CE processus (0 : absent, désactivé ou filtré)."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        abi = libc.syscall(ctypes.c_long(444), None, ctypes.c_long(0), ctypes.c_long(1))
    except Exception:
        return 0
    return int(abi) if abi > 0 else 0


def _seccomp_disponible() -> bool:
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        return bool(libc.prctl(ctypes.c_long(21), ctypes.c_long(0), ctypes.c_long(0), ctypes.c_long(0),
                               ctypes.c_long(0)) >= 0)                       # PR_GET_SECCOMP
    except Exception:
        return False


def _confinement_impossible() -> str | None:
    """Pourquoi le confinement ne peut PAS être posé sur cette machine — ou `None` s'il le peut. Mesuré
    sous Linux x86_64 (Landlock ABI 3). Ailleurs on refuse plutôt que de prétendre : la table d'appels
    système dépend de l'architecture, et Windows comme macOS n'ont ni Landlock ni seccomp."""
    if not sys.platform.startswith("linux"):
        return f"Linux only (this platform: {sys.platform})"
    machine = platform.machine()
    if machine != "x86_64":
        return f"x86_64 only for now (this machine: {machine}): the system-call filter is written per architecture"
    abi = _landlock_abi()
    if abi < _LANDLOCK_ABI_MIN:
        return (f"Landlock ABI {_LANDLOCK_ABI_MIN} or later is required (kernel 6.2+, Landlock enabled and not "
                f"filtered out by an outer sandbox); this process sees {abi or 'none'}")
    if not _seccomp_disponible():
        return "seccomp filters are not available to this process"
    return None


def _regles_du_runtime() -> list[tuple[str, int]]:
    """Ce que tout outil confiné peut lire : l'interpréteur, les bibliothèques du système, les fuseaux."""
    candidats = [sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib"),
                 os.path.join(sys.base_prefix, "lib"), os.path.join(sys.base_exec_prefix, "lib"),
                 *_DOSSIERS_SYSTEME]
    regles: dict[str, int] = {}
    for candidat in candidats:
        if candidat and os.path.isdir(candidat):
            regles.setdefault(os.path.realpath(candidat), _LL_LECTURE)
    for chemin, droits in _PERIPHERIQUES:
        if os.path.exists(chemin):
            regles.setdefault(chemin, droits)
    for chemin in _FICHIERS_SYSTEME:
        if os.path.isfile(chemin):
            regles.setdefault(os.path.realpath(chemin), _LL_LIRE_FICHIER)
    return list(regles.items())


def _regles_du_reseau() -> list[tuple[str, int]]:
    regles: dict[str, int] = {}
    for chemin in _FICHIERS_RESEAU:
        if os.path.isfile(chemin):
            regles.setdefault(os.path.realpath(chemin), _LL_LIRE_FICHIER)
    for chemin in _DOSSIERS_RESEAU:
        if os.path.isdir(chemin):
            regles.setdefault(os.path.realpath(chemin), _LL_LECTURE)
    return list(regles.items())


def _amorce_confinement(regles: list[tuple[str, int]], reseau_ouvert: bool) -> str:
    """Le préambule à coller devant un runner : pose Landlock puis seccomp, et s'efface. Seules des données
    calculées par l'hôte y sont écrites (chemins accordés par l'hôte, entiers, filtre) — rien de l'outil."""
    programme, positions = _filtre_seccomp(reseau_ouvert)
    return (_AMORCE_CONFINEMENT
            .replace("@ABI_MIN@", str(_LANDLOCK_ABI_MIN))
            .replace("@TOUS@", str(_LL_TOUS))
            .replace("@LIRE_FICHIER@", str(_LL_LIRE_FICHIER))
            .replace("@REGLES@", repr(regles))
            .replace("@FILTRE@", programme.hex())
            .replace("@POSITIONS@", repr(positions)))


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
        amorce_confinement: str = "",
    ) -> None:
        self.path, self.code, self.sha = path, code, sha
        self.timeout, self.max_calls = timeout, max_calls
        self.limits = limits
        self.amorce_confinement = amorce_confinement
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
            [sys.executable, "-X", "utf8", "-I", "-S", "-B", "-u", "-c",
             _amorce_plafonds(self.limits) + self.amorce_confinement + _WARM_RUNNER_CODE],
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

    def _fin_sans_resultat() -> ToolError:
        """L'enfant est sorti sans rendre de résultat. On laisse au fil qui lit stderr le temps de finir :
        c'est là qu'est la raison (un préambule de plafonds ou de confinement qui a refusé de démarrer)."""
        if timed_out["value"]:
            return ToolError(f"Generated tool timed out after {timeout}s")
        drainer.join(timeout=2)
        return ToolError(f"Sandbox ended without a result. stderr={''.join(stderr_chunks)[:400]!r}{diagnostic}")

    def _ecrire(ligne: str) -> None:
        """Un enfant mort avant de lire (préambule en échec) faisait remonter une `BrokenPipeError` BRUTE,
        selon qui des deux processus allait le plus vite. C'est une erreur du bac à sable : `ToolError`."""
        assert proc.stdin is not None
        try:
            proc.stdin.write(ligne + "\n")
            proc.stdin.flush()
        except OSError as exc:
            raise _fin_sans_resultat() from exc

    # Combien d'appels à une fonction de l'hôte ont été REFUSÉS (la porte) ou ont ÉCHOUÉ (la fonction a levé).
    # Compté ici, par l'hôte : le texte d'une erreur est choisi par l'outil, ce compteur non. Un nom hors de la
    # liste blanche n'y entre pas : appeler une fonction qui n'existe pas est un défaut de l'outil.
    echecs_hote = 0
    timer = threading.Timer(timeout, _kill)
    timer.start()
    try:
        assert proc.stdin is not None and proc.stdout is not None
        _ecrire(json.dumps(init_payload, ensure_ascii=False))
        while True:
            line = proc.stdout.readline()
            if not line:
                raise _fin_sans_resultat()
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
                        echecs_hote += 1
                    else:
                        try:
                            resp = {"ok": True, "result": fn(**(msg.get("args") or {}))}
                        except Exception as exc:
                            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                            echecs_hote += 1
                _ecrire(json.dumps(resp, ensure_ascii=False, default=repr))
            elif kind == "result":
                sortie = {key: value for key, value in msg.items() if key != "t"}
                sortie["host_failures"] = echecs_hote      # posé APRÈS lecture : l'outil ne peut pas l'écrire
                return sortie
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

    ``confine=Confinement(...)`` (opt-in, Linux x86_64) fait de ce sous-processus une
    vraie frontière, appliquée par le noyau : fichiers hors liste blanche, réseau,
    lancement de programmes et autres processus sont refusés quoi que le code écrive
    — voir :class:`Confinement`. Là où il ne peut pas être posé, la construction est
    REFUSÉE plutôt que de tourner sans frontière.
    """

    timeout: float = 10.0
    warm: bool = False
    warm_max_calls: int = 200
    warm_max_workers: int = 8
    limits: SandboxLimits | None = None
    confine: Confinement | None = None
    _workers: dict[str, _WarmWorker] = field(default_factory=dict, init=False, repr=False, compare=False)
    _verrou: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _confine_abi: int = field(default=0, init=False, repr=False, compare=False)
    _confine_regles: list[tuple[str, int]] = field(default_factory=list, init=False, repr=False, compare=False)
    _confine_lecture: list[str] = field(default_factory=list, init=False, repr=False, compare=False)
    _confine_ecriture: list[str] = field(default_factory=list, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.confine is not None:
            if not isinstance(self.confine, Confinement):
                raise TypeError(f"confine must be a Confinement (or None), got {type(self.confine).__name__}")
            raison = _confinement_impossible()
            if raison is not None:
                raise ToolError(
                    f"SubprocessSandbox(confine=...): kernel confinement cannot be applied here — {raison}. The "
                    "tool would run with NO boundary, so this is refused. Drop confine=, or use DockerSandbox."
                )
            regles = dict(_regles_du_runtime())
            for nom, droits, declares in (("read", _LL_LECTURE, self._confine_lecture),
                                          ("write", _LL_ECRITURE, self._confine_ecriture)):
                for chemin in getattr(self.confine, nom):
                    if not os.path.isdir(chemin):
                        raise ValueError(f"Confinement.{nom}: {chemin!r} is not an existing directory")
                    reel = os.path.realpath(chemin)
                    declares.append(reel)
                    regles[reel] = regles.get(reel, 0) | droits
            self._confine_regles = list(regles.items())
            self._confine_abi = _landlock_abi()
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
        if self.confine is not None:
            # Confiné : c'est le noyau qui applique les règles (Landlock + seccomp). `kind` reste distinct de
            # "docker" — le noyau est partagé, et un hôte qui exige un conteneur peut continuer à l'exiger.
            # `network_isolated` : coupé, sauf `network=True` (rouvert alors pour les outils qui en ont la permission).
            return {
                "kind": "subprocess+confine",
                "os_boundary": True,
                "network_isolated": not self.confine.network,
                "filesystem_isolated": True,
                "env_scrubbed": True,
                "confinement": {
                    "landlock_abi": self._confine_abi, "seccomp": "denylist", "arch": "x86_64",
                    "read": list(self._confine_lecture), "write": list(self._confine_ecriture),
                    "network": self.confine.network,
                },
                "limits": plafonds,
            }
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

    def _amorce_du_confinement(self, reseau_ouvert: bool) -> str:
        if self.confine is None:
            return ""
        regles = self._confine_regles + (_regles_du_reseau() if reseau_ouvert else [])
        return _amorce_confinement(regles, reseau_ouvert)

    def _amorce(self, reseau_ouvert: bool) -> str:
        """Ce qui précède le runner : les plafonds, puis le confinement. Chaîne vide sans l'un ni l'autre."""
        return _amorce_plafonds(self.limits) + self._amorce_du_confinement(reseau_ouvert)

    def _run_warm(self, path: Path, args: JsonDict, context: JsonDict, reseau_ouvert: bool = False) -> JsonDict:
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
                worker = _WarmWorker(
                    path, code, sha, self.timeout, max(1, self.warm_max_calls), self.limits,
                    amorce_confinement=self._amorce_du_confinement(reseau_ouvert),
                )
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
        # isolation requires DockerSandbox, or `confine=` : confiné, le réseau
        # ne se rouvre que si l'HÔTE l'a dit (`Confinement(network=True)`) ET
        # que l'outil porte la permission.
        reseau_ouvert = bool(self.confine is not None and self.confine.network and allow_network)
        del allow_network
        path = Path(file_path).resolve()
        amorce = self._amorce(reseau_ouvert)
        if host_functions:
            code = path.read_text(encoding="utf-8")
            cmd = [sys.executable, "-X", "utf8", "-I", "-S", "-u", "-c", amorce + _BRIDGE_RUNNER_CODE]
            return _drive_bridge(
                cmd, {"code": code, "args": args, "context": context or {}}, host_functions, self.timeout,
                env=_child_env(), cwd=str(path.parent), diagnostic=_indice_plafonds(self.limits),
            )
        if self.warm:
            return self._run_warm(path, args, context or {}, reseau_ouvert)
        payload = json.dumps({"args": args, "context": context or {}}, ensure_ascii=False)
        try:
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "-I", "-S", "-B", "-c", amorce + RUNNER_CODE,
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


# Le début de TOUS les messages de délai dépassé que les bacs lèvent eux-mêmes (sous-processus, worker chaud,
# pont, Docker). Une erreur venue du code de l'outil commence par le nom de son exception, jamais par ceci.
_DELAI_DEPASSE = "Generated tool timed out"

# Le REFUS DÉCLARÉ (0.28.0) : un outil qui lève `ValueError("invalid input: …")` dit qu'il refuse cette
# entrée — ce n'est pas un plantage de son code. Le début de l'erreur telle que le runner la rend, en minuscules.
# C'est la déclaration de l'outil sur LUI-MÊME : elle reste visible (carnet, fiche, erreur rendue à l'appelant),
# l'hôte peut la juger à tort, et elle ne masque jamais un échec de l'hôte (compté de son côté).
_REFUS_DECLARE = "valueerror: invalid input"


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
    # Contrôle d'admission AVANT chaque appel : `admission(nom)` rend `None` (autorisé) ou la
    # raison d'un refus. La bibliothèque persistante y refuse un outil en quarantaine hors du
    # run qui l'a construit. Sécurité = fail-CLOSED : un contrôle qui plante refuse.
    admission: "Callable[[str], str | None] | None" = None
    # Crochet de SURVEILLANCE (0.28.0) : `recorder(appel)` après CHAQUE appel, avec ce que l'outil a
    # reçu et rendu — `{"name", "ok", "kind", "seconds", "args", "result", "error"}`. `kind` est l'issue classée
    # par ce que l'HÔTE a vu, jamais par le texte de l'erreur (que l'outil choisit) : `ok`, `exception` (l'outil
    # a planté), `timeout`, `host` (une fonction de l'hôte a refusé ou échoué), `sandbox` (le bac lui-même a
    # échoué), `rejected` (le schéma d'entrée a refusé l'appel avant que l'outil tourne), `refused` (l'outil a
    # refusé l'entrée : `ValueError("invalid input: …")`). La bibliothèque
    # persistante y tient ses compteurs et son carnet. Fail-open comme `observer`.
    recorder: "Callable[[dict[str, Any]], None] | None" = None
    # D'où viennent les TESTS de cet outil (0.28.0) — `{"kind": "none"}` (aucun), `"self_tests"` (écrits
    # par le modèle qui a écrit l'outil), `"examples"` (des cas de l'hôte, dont une part cachée au modèle).
    # `None` : inconnu (un outil chargé hors de la bibliothèque persistante).
    verified_by: "dict[str, Any] | None" = None

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
        if self.admission is not None:
            try:
                refus = self.admission(self.spec.name)
            except Exception as exc:
                refus = f"admission check failed ({type(exc).__name__}: {exc}); call refused (fail-closed)"
            if refus is not None:
                raise ToolError(refus)
        _exiger_la_frontiere(self.sandbox)
        allow_network = "network" in (self.spec.permissions or [])
        debut = time.monotonic()
        issue = "sandbox"               # tant que le bac n'a rien rendu, l'échec est le sien
        try:
            result = self.sandbox.run_python_tool(
                self.file_path,
                kwargs,
                context=context,
                allow_network=allow_network,
                host_functions=host_functions if host_functions is not None else self.host_functions,
            )
            if not result.get("ok"):
                declare = str(result.get("error") or "").lower().startswith(_REFUS_DECLARE)
                issue = "host" if result.get("host_failures") else "refused" if declare else "exception"
                raise ToolError(result.get("error") or "Generated tool failed")
        except Exception as exc:
            if issue == "sandbox" and str(exc).startswith(_DELAI_DEPASSE):
                issue = "timeout"
            erreur = f"{type(exc).__name__}: {exc}"
            self._observer_dit(False, erreur)
            self._recorder_dit(issue, kwargs, None, erreur, debut)
            raise
        self._observer_dit(True, None)
        self._recorder_dit("ok", kwargs, result.get("result"), None, debut)
        return result.get("result")

    def __autoagent_rejected__(self, args: Any, erreur: str) -> None:
        """Appelé par le registre quand le SCHÉMA d'entrée refuse un appel avant que l'outil tourne : noté à
        part (`kind: "rejected"`). Ni le modèle qui appelle mal ni un schéma trop strict ne se voyaient."""
        self._recorder_dit("rejected", args if isinstance(args, dict) else {"value": args}, None, erreur,
                           time.monotonic())

    def _recorder_dit(self, issue: str, args: dict[str, Any], resultat: Any, erreur: str | None, debut: float) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder({
                "name": self.spec.name, "ok": issue == "ok", "kind": issue,
                "seconds": time.monotonic() - debut, "args": args, "result": resultat, "error": erreur,
            })
        except Exception:                                   # fail-open
            pass

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


def _bac_par_defaut(permissions: Any = (), *, timeout: float = 10.0) -> "SubprocessSandbox":
    """Le bac d'un outil quand l'HÔTE n'en a pas choisi (0.25.0).

    Règle : le noyau fait respecter ce que les permissions disent déjà. Un outil sans permission
    ``filesystem.*`` n'a pas de fichiers ; sans permission ``network``, pas de réseau — là où le
    confinement peut être posé (Linux x86_64, Landlock ABI 3). Deux cas rendent le sous-processus
    SIMPLE d'avant, qui n'est pas une frontière et le dit :

    * la machine ne permet pas le confinement (Windows, macOS, ARM64, noyau trop ancien) ;
    * l'outil porte une permission ``filesystem.*`` : l'hôte a accordé des fichiers sans dire
      LESQUELS, et une liste blanche ne s'invente pas. Pour le confiner, passer
      ``sandbox=SubprocessSandbox(confine=Confinement(read=[...], write=[...]))``.

    Pour retrouver le bac d'avant partout : ``sandbox=SubprocessSandbox()``."""
    demande_des_fichiers = any(str(p).startswith("filesystem.") for p in (permissions or ()))
    if not demande_des_fichiers and _confinement_impossible() is None:
        try:
            return SubprocessSandbox(timeout=timeout, confine=Confinement(network=True))
        except (ToolError, ValueError, OSError):
            _log.exception("kernel confinement could not be set up; falling back to the plain subprocess")
    return SubprocessSandbox(timeout=timeout)


def _a_une_frontiere(sandbox: Any) -> bool:
    """Le bac à sable est-il une frontière que l'OS fait respecter ? C'est `isolation()` qui le dit :
    un conteneur Docker, ou un sous-processus confiné par le noyau (`SubprocessSandbox(confine=...)`).
    Un bac qui ne sait pas le dire compte pour « non »."""
    try:
        return bool(sandbox.isolation().get("os_boundary"))
    except Exception:
        return False


def _frontiere_exigee() -> bool:
    """Le run en cours exige-t-il une frontière (`Agent(require_boundary=True)`, 0.28.0) ? Lu sur l'agent de la
    porte active : le sien, ou celui d'un parent qui l'exige et lui a délégué la tâche (`_frontiere_heritee`).
    Hors de tout run (l'hôte qui appelle un outil à la main) : non, comme la porte."""
    porte = active_gate()
    agent = getattr(porte, "agent", None) if porte is not None else None
    return bool(getattr(agent, "require_boundary", False) or getattr(agent, "_frontiere_heritee", False))


def _exiger_la_frontiere(sandbox: Any) -> None:
    """Fail-closed, À L'EXÉCUTION : dans un run qui exige une frontière, du code écrit par le modèle ne part
    jamais dans un bac qui n'en est pas une — même si le bac a été remplacé après la configuration, si l'outil
    a été chargé à la main, ou si le run est celui d'un sous-agent délégué par un agent qui l'exige."""
    if _frontiere_exigee() and not _a_une_frontiere(sandbox):
        try:
            nature = str(sandbox.isolation().get("kind", type(sandbox).__name__))
        except Exception:
            nature = type(sandbox).__name__
        raise ToolError(
            f"require_boundary=True: this sandbox ({nature}) is not an OS-enforced isolation boundary, so the "
            "model-written code was not run. Use kernel confinement (Linux x86_64) or DockerSandbox; a sub-agent "
            "called by an agent that requires a boundary needs require_boundary=True too."
        )


def _bac_avec_frontiere(permissions: Any = (), *, timeout: float = 10.0) -> "SubprocessSandbox | DockerSandbox":
    """Le meilleur bac à VRAIE frontière de cette machine (0.28.0), pour `Agent(require_boundary=True)` : le
    confinement par le noyau (rapide, sans démon) ; sinon Docker, si un démon à conteneurs Linux répond ; sinon
    ``ToolError`` — jamais le sous-processus simple. Un outil qui porte une permission ``filesystem.*`` sans
    dossiers nommés ne peut pas être confiné (une liste blanche ne s'invente pas) : Docker, ou refus."""
    fichiers = any(str(p).startswith("filesystem.") for p in (permissions or ()))
    cause = _confinement_impossible()
    if not fichiers and cause is None:
        try:
            return SubprocessSandbox(timeout=timeout, confine=Confinement(network=True))
        except (ToolError, ValueError, OSError) as exc:
            cause = f"kernel confinement could not be set up: {exc}"
    if docker_available():
        return DockerSandbox(timeout=timeout)
    raison = ("the tool holds a filesystem.* permission and no directory was named, so kernel confinement "
              "cannot apply" if fichiers else (cause or "no kernel confinement"))
    raise ToolError(
        f"require_boundary=True: no isolation boundary is available here ({raison}), and no Docker daemon "
        "(Linux containers) is reachable, so model-written code will not run. Run on Linux x86_64 (kernel "
        "confinement), or start Docker; or drop require_boundary=True to accept the plain subprocess."
    )


def make_sandbox(
    *, prefer_docker: bool = True, timeout: float = 10.0, image: str = "python:3.11-slim",
    require_docker: bool = False, require_boundary: bool = False,
) -> "SubprocessSandbox | DockerSandbox":
    """Return the strongest sandbox available: a DockerSandbox when a Docker
    daemon is reachable, else a kernel-confined SubprocessSandbox where the
    kernel allows it (Linux x86_64, Landlock ABI 3 — 0.25.0), else
    the plain SubprocessSandbox.

    The plain subprocess is NOT an isolation boundary (an AST denylist behind
    ``-I -S``), so falling back to it is not silent (0.23.1) : a warning, once
    per process, says so. ``require_docker=True`` refuses anything but Docker;
    ``require_boundary=True`` accepts Docker or the confined subprocess and
    refuses the plain one — both raise ``ToolError``. Pass ``prefer_docker=False``
    to choose the plain subprocess knowingly (no warning). The confined fallback
    behaves like the container for a tool's permissions: no host file, network
    only for a tool holding ``network``. ``sandbox.isolation()`` tells what the
    returned sandbox really enforces."""
    if (require_docker or require_boundary) and not prefer_docker:
        raise ValueError("require_docker=True / require_boundary=True contradict prefer_docker=False")
    if prefer_docker and docker_available():
        return DockerSandbox(image=image, timeout=timeout)
    if require_docker:
        raise ToolError(
            "make_sandbox(require_docker=True): no usable Docker daemon (Linux containers) is reachable, and "
            "the fallback SubprocessSandbox is not an isolation boundary, so nothing is run. Start Docker, or "
            "drop require_docker=True to accept the weaker sandbox."
        )
    if prefer_docker:
        confine = _bac_par_defaut(timeout=timeout)
        if confine.confine is not None:
            return confine
    if require_boundary:
        raise ToolError(
            "make_sandbox(require_boundary=True): neither a Docker daemon nor kernel confinement is available "
            f"here ({_confinement_impossible() or 'confinement could not be set up'}), and the plain "
            "SubprocessSandbox is not an isolation boundary, so nothing is run."
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
    return GeneratedPythonTool(spec=spec, file_path=path, sandbox=sandbox or _bac_par_defaut(spec.permissions))


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
