from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
import pprint
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._fichiers import atomic_write_text, quarantine
from .errors import ToolError, ToolValidationError
from .logging import get_logger, warn_once
from .providers.base import LLMProvider, loads_tolerant
from .sandbox import (
    DockerSandbox,
    GeneratedPythonTool,
    SubprocessSandbox,
    discard_generated_tool,
    load_generated_tool,
    validate_generated_tool_code,
)
from .schema import LLMRequest, Message, TokenUsage

__all__ = ["DynamicToolBuilder", "PythonRunner", "ToolBuildRequest"]

_log = get_logger("dynamic")

_CATALOGUE = "catalogue.json"


class _PermissionRefusee(ToolValidationError):
    """Refus du PLAFOND de permissions de l'hôte : jamais repris par `max_repairs`
    (c'est une décision de l'hôte, pas un bug que le modèle doit contourner en boucle)."""


def _lignes_fonctions(host_functions: dict[str, Callable[..., Any]]) -> list[str]:
    """Une ligne par fonction de l'hôte : nom, signature, première ligne de docstring."""
    lignes = []
    for nom, fonction in sorted(host_functions.items()):
        try:
            signature = str(inspect.signature(fonction))
        except (TypeError, ValueError):
            signature = "(...)"
        doc = (inspect.getdoc(fonction) or "").strip().splitlines()
        lignes.append(f"- {nom}{signature}" + (f": {doc[0]}" if doc else ""))
    return lignes


def _egal(attendu: Any, obtenu: Any, rel_tol: float) -> bool:
    """Égalité de deux valeurs JSON, nombres comparés à ``rel_tol`` près (récursif).

    Mesuré : un outil de haversine rendait ``10007.543398010288`` quand le modèle
    attendait ``10007.543398010286`` — UN bit de flottant — et la création était
    rejetée. 10 échecs sur 15 tentatives, alors que le code et l'attendu étaient
    justes. Seuls les NOMBRES sont tolérants : les booléens suivent l'égalité de
    Python (comme avant), les chaînes et les clés restent exactes. ``rel_tol=0``
    redonne l'égalité stricte d'avant.
    """
    if isinstance(attendu, bool) or isinstance(obtenu, bool):
        return bool(attendu == obtenu)
    if isinstance(attendu, (int, float)) and isinstance(obtenu, (int, float)):
        if attendu == obtenu:
            return True
        return rel_tol > 0 and math.isclose(attendu, obtenu, rel_tol=rel_tol, abs_tol=1e-12)
    if isinstance(attendu, dict) and isinstance(obtenu, dict):
        return attendu.keys() == obtenu.keys() and all(_egal(v, obtenu[k], rel_tol) for k, v in attendu.items())
    if isinstance(attendu, (list, tuple)) and isinstance(obtenu, (list, tuple)):
        return len(attendu) == len(obtenu) and all(_egal(a, b, rel_tol) for a, b in zip(attendu, obtenu, strict=True))
    return bool(attendu == obtenu)


def _somme(usages: list[TokenUsage | None]) -> TokenUsage | None:
    """Somme de plusieurs usages ; ``None`` si aucun n'a été rapporté (jamais un zéro inventé)."""
    reels = [u for u in usages if u is not None]
    if not reels:
        return None
    caches = [u.cached_tokens for u in reels if u.cached_tokens is not None]
    return TokenUsage(
        input_tokens=sum(u.input_tokens or 0 for u in reels),
        output_tokens=sum(u.output_tokens or 0 for u in reels),
        cached_tokens=sum(caches) if caches else None,
    )


@dataclass
class ToolBuildRequest:
    capability: str
    tool_name: str | None = None
    input_schema: dict[str, Any] | None = None
    permissions: list[str] = field(default_factory=list)


class PythonRunner:
    """Exécute UN extrait de code Python écrit par le modèle — validé, sandboxé, jamais gardé (0.22.0).

    Le pendant éphémère de `create_python_tool` : pas de constructeur, pas
    d'outil enregistré, pas de fichier qui reste. Le modèle écrit
    ``def run(args, context): ...``, le validateur AST le juge, le bac à sable
    l'exécute dans un dossier temporaire supprimé ensuite. Sans permission
    (ni réseau, ni fichiers) sauf celles que l'HÔTE accorde ici — jamais le modèle.

    Même honnêteté que le reste du bac à sable : le ``SubprocessSandbox`` est une
    liste d'interdits, pas une frontière. Pour du code que tu ne maîtrises pas,
    passe ``sandbox=DockerSandbox(...)``. Un sandbox « chaud » est ignoré ici (un
    fichier neuf à chaque appel : un worker par appel n'aurait aucun sens).
    """

    def __init__(
        self,
        sandbox: SubprocessSandbox | DockerSandbox | None = None,
        *,
        timeout: float = 10.0,
        permissions: Iterable[str] | None = None,
        max_code_chars: int = 20_000,
        host_functions: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        """``host_functions`` : callbacks de l'hôte appelables par l'extrait via
        ``context["call_host"]("nom", {...})`` (le pont du bac à sable). C'est ce qui
        permet au modèle d'écrire UN programme qui appelle plusieurs fois des outils
        de l'hôte, au lieu d'un appel d'outil par tour. Mêmes règles que pour
        `DynamicToolBuilder` : liste blanche, exécutées chez l'hôte avec ses droits sur
        des arguments choisis par le modèle. Depuis D1 elles passent par LA porte de
        décision de l'agent qui lance l'extrait (`gate.py`) : `tool_policy` (avec
        `ctx.source == "host_function"`), garde trifecta et teinte — une approbation
        humaine y REFUSE, un programme ne pouvant pas être mis en pause. Pour qu'une
        fonction compte comme `egress` ou `untrusted`, décore-la avec `autoagent.tool(...)`.
        Hors d'un run d'agent (le runner appelé à la main), rien n'est consulté."""
        sandbox = sandbox or SubprocessSandbox(timeout=timeout)
        if isinstance(sandbox, SubprocessSandbox) and sandbox.warm:
            sandbox = dataclasses.replace(sandbox, warm=False)
        self.sandbox = sandbox
        self.permissions = sorted(set(permissions or []))
        self.max_code_chars = max_code_chars
        self.host_functions: dict[str, Callable[..., Any]] | None = dict(host_functions) if host_functions else None
        if not isinstance(self.sandbox, DockerSandbox):
            warn_once(
                _log, "dynamic.run_python_not_docker",
                "run_python executes model-written code in a SubprocessSandbox (AST denylist, "
                "not an isolation boundary). For untrusted input pass sandbox=DockerSandbox(...) "
                "and govern the tool with tool_policy.",
            )

    def describe_host_functions(self) -> str:
        """Texte pour la description de l'outil `run_python` (vide si aucune fonction)."""
        if not self.host_functions:
            return ""
        return (
            " The host also exposes these functions, callable inside run() as "
            "context['call_host'](name, {argument: value, ...}). Each call returns EXACTLY the "
            "value the function returns (a dict stays a dict: it is NOT wrapped in "
            "{'result': ...}) and raises if it fails:\n" + "\n".join(_lignes_fonctions(self.host_functions))
        )

    def __call__(self, code: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(code, str) or not code.strip():
            raise ToolValidationError("code is empty: define def run(args, context): ...")
        if len(code) > self.max_code_chars:
            raise ToolValidationError(f"code is {len(code)} characters; the limit is {self.max_code_chars}")
        if args is not None and not isinstance(args, dict):
            raise ToolValidationError(f"args must be an object, got {type(args).__name__}")
        validate_generated_tool_code(code, permissions=self.permissions)
        with tempfile.TemporaryDirectory(prefix="autoagent_run_python_", ignore_cleanup_errors=True) as dossier:
            fichier = Path(dossier) / "run_python.py"
            fichier.write_text(code, encoding="utf-8")
            sortie = self.sandbox.run_python_tool(
                fichier, args or {}, context={}, allow_network="network" in self.permissions,
                host_functions=self.host_functions,
            )
        if not sortie.get("ok"):
            raise ToolError(sortie.get("error") or "Python snippet failed")
        return {"result": sortie.get("result"), "stdout": (sortie.get("stdout") or "")[:4000]}


class DynamicToolBuilder:
    """Generates a new Python tool by asking an LLM, validates it, and stores it.

    Flow per `build(request)`:
        1. Ask the provider for a single JSON object describing the new
           tool (metadata + Python source + self-tests).
        2. Sanitize the tool name and merge any caller-imposed
           name / input_schema / permissions overrides.
        3. Walk the AST via `validate_generated_tool_code` to reject
           dangerous calls, dangerous identifiers, and disallowed imports.
        4. Write the code to `tools_dir/<name>.py`.
        5. Execute the declared self-tests inside the sandbox.
        6. Return a `GeneratedPythonTool` that, when called, runs the
           code in an isolated subprocess (`SubprocessSandbox`).

    The host should treat dynamically generated tools as untrusted: even
    after passing the AST validator, they run in `-I -S` Python with an
    empty `env={}` and no shared filesystem cwd beyond the tools dir.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        tools_dir: str | Path = ".autoagent/tools",
        sandbox: SubprocessSandbox | None = None,
        timeout: float = 10.0,
        allowed_permissions: Iterable[str] | None = None,
        max_repairs: int = 0,
        persist: bool = False,
        retire_after_errors: int = 3,
        self_test_rel_tol: float = 1e-6,
        host_functions: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        """``host_functions`` (0.22.0) : callbacks de l'HÔTE que les outils générés
        peuvent appeler — ``context["call_host"]("nom", {"arg": valeur})`` — par le
        pont du bac à sable (il passe par stdin/stdout : marche même sans réseau).
        Liste blanche : un nom absent est refusé, et chaque fonction s'exécute DANS
        le processus de l'hôte, avec ses droits, sur des arguments choisis par le
        modèle — valide-les comme n'importe quelle entrée. Le modèle constructeur en
        voit les noms, signatures et première ligne de docstring. Les self-tests
        tournent SANS ces fonctions (le build n'a aucun effet de bord sur l'hôte) :
        le prompt demande donc une liste vide à un outil qui s'en sert. ``None``
        (défaut) = rien n'est exposé.

        ``self_test_rel_tol`` (0.22.0) : tolérance RELATIVE des nombres dans
        ``expect_equals`` (récursif). L'égalité exacte rejetait des outils justes
        pour un bit de flottant (mesuré : haversine ``…286`` attendu, ``…288`` rendu).
        Un self-test sert à repérer un code FAUX, pas à certifier des décimales.
        ``0`` redonne l'égalité stricte d'avant.

        ``max_repairs`` (0.22.0) : combien de fois le constructeur peut REPRENDRE
        un outil refusé — JSON illisible, code refusé par le validateur, self-test
        faux, plantage pendant le test. Le refus lui est renvoyé tel quel
        (« expected 392.4, got 392.2 ») et il rend un JSON complet corrigé : c'est
        lui qui sait si le code ou l'attendu était faux. ``0`` (défaut) = comme
        avant, un refus est un échec. Chaque reprise est un appel PAYÉ, compté dans
        ``last_build_usage`` donc dans ``token_budget``. Un refus de PERMISSION (le
        plafond ci-dessous) n'est jamais repris : c'est une décision, pas un bug.

        ``persist`` (0.22.0) : bibliothèque persistante. Chaque outil accepté est
        inscrit dans ``tools_dir/catalogue.json`` (sha256 du fichier, date, nombre
        d'appels et d'erreurs) ; ``Agent.enable_dynamic_tools`` le recharge au
        démarrage au lieu de repayer le constructeur à chaque run. Un outil dont
        le fichier a changé depuis sa validation, ou dont les permissions
        dépassent le plafond actuel, n'est PAS rechargé. ``retire_after_errors``
        erreurs de suite RETIRENT l'outil de la bibliothèque (0 = jamais). Un seul
        processus écrivain par ``tools_dir``. Défaut ``False`` : rien n'est écrit,
        rien n'est rechargé.

        ``allowed_permissions`` (0.22.0) : le PLAFOND fixé par l'hôte.

        Les permissions d'un outil généré (``network``, ``filesystem.*``)
        étaient choisies par le modèle lui-même — dans l'appel à
        ``create_python_tool`` ou dans la réponse du modèle constructeur. Le
        validateur AST et Docker (``--network none`` levé si ``network``) les
        appliquaient fidèlement : le modèle s'ouvrait le réseau tout seul.
        Avec un plafond, toute permission hors liste est REFUSÉE
        (``ToolValidationError``, que le modèle voit et peut contourner en
        demandant moins). ``"filesystem.*"`` couvre toutes les permissions
        fichiers. ``None`` (défaut historique) = pas de plafond.
        """
        self.provider = provider
        self.tools_dir = Path(tools_dir)
        self.sandbox = sandbox or SubprocessSandbox(timeout=timeout)
        self.allowed_permissions = None if allowed_permissions is None else set(allowed_permissions)
        self.max_repairs = max(0, int(max_repairs))
        self.self_test_rel_tol = max(0.0, float(self_test_rel_tol))
        self.host_functions: dict[str, Callable[..., Any]] | None = dict(host_functions) if host_functions else None
        self.persist = bool(persist)
        self.retire_after_errors = max(0, int(retire_after_errors))
        self._verrou = threading.RLock()        # lecture-modification-écriture du catalogue
        # Dépense du dernier `build()` (0.22.0). L'appel au modèle constructeur
        # coûtait des jetons INVISIBLES à `token_budget` : le modèle pouvait
        # construire outil sur outil sous un plafond aveugle. `create_python_tool`
        # lit `last_build_usage` et le reverse dans la comptabilité du run. PAR
        # THREAD (comme `_CanalDepense`) : deux créations en `parallel_tool_calls`
        # ne s'écrasent pas leur dépense.
        self._local = threading.local()
        if not isinstance(self.sandbox, DockerSandbox):
            # Le SubprocessSandbox est un validateur AST + un `-I -S` : une liste
            # d'interdits, PAS une frontière d'isolation (son propre code le dit).
            # Du code écrit par le modèle y reste contournable. On le dit une fois
            # pour que l'hôte choisisse Docker en connaissance de cause.
            warn_once(
                _log, "dynamic.sandbox_not_docker",
                "DynamicToolBuilder runs generated tools in a SubprocessSandbox "
                "(AST denylist, not an isolation boundary). For untrusted "
                "capabilities pass sandbox=DockerSandbox(...) or make_sandbox().",
            )

    @property
    def last_build_usage(self) -> TokenUsage | None:
        """Jetons de l'appel au modèle constructeur du dernier `build()` DE CE THREAD
        — rempli même quand le build échoue ensuite (l'appel est payé quoi qu'il
        arrive). ``None`` si le fournisseur ne déclare pas d'usage."""
        return getattr(self._local, "usage", None)

    @last_build_usage.setter
    def last_build_usage(self, usage: TokenUsage | None) -> None:
        self._local.usage = usage

    def _check_permissions(self, permissions: list[str]) -> None:
        if self.allowed_permissions is None:
            # Pas de plafond (défaut historique, gardé pour la production) : on
            # ne refuse rien, mais on le dit dès qu'un outil s'accorde un droit.
            if permissions:
                warn_once(_log, "dynamic.allowed_permissions",
                          f"DynamicToolBuilder: a generated tool granted itself {permissions} with no "
                          "host ceiling. Pass allowed_permissions=set() (or the permissions you "
                          "accept, e.g. {'filesystem.*'}) to decide who grants what.")
            return
        plafond = self.allowed_permissions
        refusees = [
            p for p in permissions
            if p not in plafond and not (p.startswith("filesystem.") and "filesystem.*" in plafond)
        ]
        if refusees:
            raise _PermissionRefusee(
                f"Permissions not allowed by the host: {refusees}. "
                f"Allowed: {sorted(plafond) or 'none'}. Build the tool without them."
            )

    def build(self, request: ToolBuildRequest) -> GeneratedPythonTool:
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.last_build_usage = None
        messages = [
            Message(role="system", content=self._system_prompt()),
            Message(role="user", content=self._user_prompt(request)),
        ]
        reprises = 0
        while True:
            response = self.provider.complete(
                LLMRequest(
                    messages=messages,
                    temperature=0,
                    # 8192 : laisse de la marge aux modèles « thinking » (Gemini 2.5)
                    # dont la réflexion consomme des tokens avant le JSON de sortie.
                    max_tokens=8192,
                    tool_choice="none",
                    # JSON mode natif quand le provider le supporte (OpenAI/
                    # DeepSeek/Gemini) : supprime les balises ```json à la source.
                    # _parse_json_object reste tolérant (Anthropic = best effort).
                    response_format={"type": "json_object"},
                )
            )
            # Chaque appel est PAYÉ, accepté ou non : on additionne avant de juger.
            self.last_build_usage = _somme([self.last_build_usage, response.usage])
            try:
                generated_tool = self._construire(request, response.content)
                break
            except _PermissionRefusee:
                raise
            except (ToolValidationError, ToolError) as exc:
                if reprises >= self.max_repairs:
                    raise
                reprises += 1
                _log.info("generated tool refused (%s); repair %d/%d", exc, reprises, self.max_repairs)
                # Le constructeur revoit SA réponse et la raison du refus : c'est lui
                # qui sait si le code ou l'attendu du self-test était faux.
                messages = [
                    *messages,
                    Message(role="assistant", content=response.content),
                    Message(role="user", content=self._retour(exc)),
                ]
        # Après les self-tests : le build ne doit avoir aucun effet de bord sur l'hôte.
        generated_tool.host_functions = self.host_functions
        if self.persist:
            self._noter_construction(generated_tool)
            generated_tool.observer = self._observer
        return generated_tool

    def _construire(self, request: ToolBuildRequest, contenu: str) -> GeneratedPythonTool:
        """Une tentative : réponse du constructeur → outil validé, écrit, chargé, testé."""
        payload = _parse_json_object(contenu)
        if not isinstance(payload, dict):
            raise ToolValidationError(f"Tool builder response must be a JSON object, got {type(payload).__name__}")
        tool_meta = payload.get("tool") or {}
        if not isinstance(tool_meta, dict):
            raise ToolValidationError("'tool' must be a JSON object with name, description, input_schema, permissions")
        code = payload.get("code")
        tests = payload.get("self_tests") or []

        if not isinstance(code, str) or not code.strip():
            raise ToolValidationError("Tool builder did not return code")

        if request.tool_name:
            requested_name = _safe_tool_name(request.tool_name)
            tool_meta["name"] = requested_name
        if request.input_schema:
            tool_meta["input_schema"] = request.input_schema
        if request.permissions:
            tool_meta["permissions"] = request.permissions

        if not isinstance(tool_meta.get("name"), str):
            raise ToolValidationError("Tool metadata missing keys: ['name']")
        tool_meta["name"] = _safe_tool_name(tool_meta["name"])
        tool_meta.setdefault("permissions", [])
        permissions = tool_meta.get("permissions") or []
        if not isinstance(permissions, list) or not all(isinstance(p, str) for p in permissions):
            raise ToolValidationError("'permissions' must be a list of strings")
        if "input_schema" in tool_meta and not isinstance(tool_meta["input_schema"], dict):
            raise ToolValidationError("'input_schema' must be a JSON-schema object")
        code = _upsert_tool_metadata(code, tool_meta)
        self._check_permissions(list(permissions))       # AVANT d'écrire quoi que ce soit
        validate_generated_tool_code(code, permissions=permissions)

        name = tool_meta["name"]
        file_path = self.tools_dir / f"{name}.py"
        file_path.write_text(code, encoding="utf-8")

        # Un outil qui ne se charge pas ou qui échoue à ses self-tests ne doit pas
        # rester sur le disque : chargeable tel quel au prochain démarrage (0.22.0),
        # et son bytecode périmé rejouerait du code refusé (le bug `.pyc` vu en CI).
        try:
            generated_tool = load_generated_tool(file_path, sandbox=self.sandbox)
            self._run_self_tests(generated_tool, tests)
        except Exception:
            discard_generated_tool(file_path)
            raise
        return generated_tool

    @staticmethod
    def _retour(erreur: Exception) -> str:
        """Message de reprise : la raison EXACTE du refus, puis la consigne."""
        texte = str(erreur)
        if "not valid JSON" in texte:
            # « Expecting ',' delimiter: column 2749 » ne dit rien au modèle : mesuré,
            # il refaisait la MÊME faute à chaque reprise (jusqu'à 16 000 jetons brûlés).
            # La cause réelle était presque toujours une EXPRESSION à la place d'un
            # nombre (`6371 * math.pi / 2`) — on la nomme.
            indice = (
                "Your answer was not valid JSON. The usual cause: a value written as an "
                "expression (6371 * math.pi / 2) instead of a plain number (10007.54), in "
                "self_tests. Write every value as a literal; or drop the self-tests."
            )
        else:
            indice = (
                "If a self-test expectation disagrees with what the code returns, decide which "
                "one is wrong: fix the code if the code is wrong, otherwise correct the "
                "expectation or drop that self-test."
            )
        return (
            f"Your tool was rejected: {texte[:1500]}\n"
            "Return the COMPLETE JSON object again (tool, code, self_tests) with that problem "
            f"fixed, keeping the same tool name. {indice}"
        )

    # ── Bibliothèque persistante ────────────────────────────────────────────

    def _lire_catalogue(self) -> dict[str, dict[str, Any]]:
        chemin = self.tools_dir / _CATALOGUE
        if not chemin.is_file():
            return {}
        try:
            outils = json.loads(chemin.read_text(encoding="utf-8"))["tools"]
            if not isinstance(outils, dict):
                raise ValueError("catalogue 'tools' is not an object")
            return {nom: dict(e) for nom, e in outils.items() if isinstance(e, dict)}
        except (OSError, ValueError, KeyError, TypeError):
            # Mis de côté, jamais écrasé : sans catalogue la bibliothèque repart
            # vide, mais les fichiers d'outils et l'ancien catalogue restent là.
            _log.exception("tool catalogue unreadable (%s); starting empty", chemin)
            quarantine(chemin)
            return {}

    def _ecrire_catalogue(self, outils: dict[str, dict[str, Any]]) -> None:
        try:
            atomic_write_text(
                self.tools_dir / _CATALOGUE,
                json.dumps({"version": 1, "tools": outils}, ensure_ascii=False, indent=1),
            )
        except OSError:
            _log.exception("could not write the tool catalogue (the run goes on)")   # fail-open

    def catalogue(self) -> dict[str, dict[str, Any]]:
        """Copie du catalogue de la bibliothèque : par nom d'outil, `sha256`, `created`,
        `description`, `permissions`, `calls`, `errors`, `consecutive_errors`,
        `retired` (+ `retired_reason`, `last_error`). Vide sans `persist=True`."""
        with self._verrou:
            copie: dict[str, dict[str, Any]] = json.loads(json.dumps(self._lire_catalogue()))
            return copie

    def _noter_construction(self, outil: GeneratedPythonTool) -> None:
        with self._verrou:
            outils = self._lire_catalogue()
            outils[outil.spec.name] = {
                "sha256": hashlib.sha256(outil.file_path.read_bytes()).hexdigest(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "description": outil.spec.description,
                "permissions": list(outil.spec.permissions or []),
                "calls": 0, "errors": 0, "consecutive_errors": 0, "retired": False,
            }
            self._ecrire_catalogue(outils)

    def _observer(self, nom: str, ok: bool, erreur: str | None) -> None:
        """Compte chaque appel d'un outil de la bibliothèque ; retire celui qui échoue de suite."""
        with self._verrou:
            outils = self._lire_catalogue()
            entree = outils.get(nom)
            if entree is None:
                return
            entree["calls"] = int(entree.get("calls") or 0) + 1
            if ok:
                entree["consecutive_errors"] = 0
            else:
                entree["errors"] = int(entree.get("errors") or 0) + 1
                entree["consecutive_errors"] = int(entree.get("consecutive_errors") or 0) + 1
                entree["last_error"] = (erreur or "")[:300]
                limite = self.retire_after_errors
                if limite and entree["consecutive_errors"] >= limite and not entree.get("retired"):
                    entree["retired"] = True
                    entree["retired_reason"] = f"{entree['consecutive_errors']} consecutive errors"
                    _log.warning("generated tool %r retired from the library (%s)", nom, entree["retired_reason"])
            self._ecrire_catalogue(outils)

    def retire(self, name: str, reason: str = "retired by the host") -> bool:
        """Retire un outil de la bibliothèque : il ne sera plus rechargé. ``True`` si connu."""
        with self._verrou:
            outils = self._lire_catalogue()
            if name not in outils:
                return False
            outils[name]["retired"] = True
            outils[name]["retired_reason"] = reason
            self._ecrire_catalogue(outils)
            return True

    def load_library(self) -> list[GeneratedPythonTool]:
        """Recharge les outils déjà acceptés lors de runs précédents (``persist=True``).

        Rechargé SEULEMENT si : l'outil n'est pas retiré, son fichier existe et son
        sha256 est celui noté à la validation (un fichier modifié depuis n'est pas
        rechargé, jamais « re-validé en silence »), le validateur AST l'accepte
        encore, et ses permissions tiennent sous le plafond ACTUEL de l'hôte. Les
        outils sans entrée au catalogue (provenance inconnue) sont ignorés. Ce
        garde-fou protège de la dérive et des écritures partielles ; il n'est pas
        une défense contre quelqu'un qui peut écrire dans ``tools_dir`` (il pourrait
        aussi réécrire le catalogue) — pour cela, l'approbation par manifeste.
        """
        if not self.persist:
            return []
        with self._verrou:
            outils = self._lire_catalogue()
        charges: list[GeneratedPythonTool] = []
        for nom, entree in sorted(outils.items()):
            if entree.get("retired"):
                continue
            chemin = self.tools_dir / f"{nom}.py"
            try:
                octets = chemin.read_bytes()
            except OSError:
                _log.warning("library tool %r: file missing, skipped", nom)
                continue
            if hashlib.sha256(octets).hexdigest() != entree.get("sha256"):
                _log.warning("library tool %r changed on disk since it was validated; NOT loaded", nom)
                continue
            try:
                outil = load_generated_tool(chemin, sandbox=self.sandbox)       # revalide l'AST
                self._check_permissions(list(outil.spec.permissions or []))
            except (ToolValidationError, OSError, SyntaxError, ValueError) as exc:
                _log.warning("library tool %r not loaded: %s", nom, exc)
                continue
            if outil.spec.name != nom:
                _log.warning("library tool %r declares another name (%r); skipped", nom, outil.spec.name)
                continue
            outil.observer = self._observer
            outil.host_functions = self.host_functions
            charges.append(outil)
        return charges

    def _run_self_tests(self, tool: GeneratedPythonTool, tests: list[dict[str, Any]]) -> None:
        if not isinstance(tests, list):
            raise ToolValidationError(f"self_tests must be a list, got {type(tests).__name__}")
        for index, test in enumerate(tests[:5]):
            if not isinstance(test, dict):
                raise ToolValidationError(f"Self-test {index} must be an object, got {type(test).__name__}")
            args = test.get("args") or {}
            if not isinstance(args, dict):
                raise ToolValidationError(
                    f"Self-test {index}: 'args' must be an object matching input_schema, "
                    f"got {type(args).__name__}"
                )
            result = tool(**args)
            if "expect_equals" in test and not _egal(test["expect_equals"], result, self.self_test_rel_tol):
                raise ToolValidationError(
                    f"Self-test {index} failed: expected {test['expect_equals']!r}, got {result!r}"
                )
            if "expect_contains" in test and test["expect_contains"] not in str(result):
                raise ToolValidationError(
                    f"Self-test {index} failed: {test['expect_contains']!r} not in {result!r}"
                )

    def _system_prompt(self) -> str:
        return (
            "You create small Python tools for an AI agent. Return only one JSON object. "
            "No markdown. The JSON must have keys: tool, code, self_tests. "
            "tool must contain name, description, input_schema, permissions. "
            "code must be a complete Python module defining TOOL = {...} and "
            "def run(args, context): ... . The run function must return JSON-serializable data. "
            "Do not use network, filesystem, shell, eval, exec, or subprocess unless the requested "
            "permissions explicitly allow it. Prefer pure Python and standard library only. "
            "Your whole answer must be ONE valid JSON document: every value is a plain literal — "
            "a number such as 10007.54, NEVER an expression such as 6371 * 3.14159 / 2 or "
            "math.pi — and the code string escapes its double quotes and newlines. "
            "self_tests are optional: only use inputs whose result you are certain of (simple, "
            "round cases). Numbers in expect_equals are compared with a relative tolerance of "
            f"{self.self_test_rel_tol:g}, so do not assert digits you are unsure of. A wrong "
            "expectation makes the whole tool be rejected, so return an empty list rather than guess."
            + self._host_prompt()
        )

    def _host_prompt(self) -> str:
        """Paragraphe sur les fonctions de l'hôte (vide si aucune n'est exposée)."""
        if not self.host_functions:
            return ""
        lignes = _lignes_fonctions(self.host_functions)
        return (
            " The host exposes these functions, callable from run() as "
            "context['call_host'](name, {argument: value, ...}); the call returns the function's "
            "result and raises if it fails. They are the ONLY way to reach host data; use no "
            "other name. Self-tests run WITHOUT the host, so a tool that calls them must return "
            "an empty self_tests list. Available functions:\n" + "\n".join(lignes)
        )

    def _user_prompt(self, request: ToolBuildRequest) -> str:
        return json.dumps(
            {
                "capability": request.capability,
                "preferred_tool_name": request.tool_name,
                "input_schema": request.input_schema,
                "permissions": request.permissions,
                # Des descriptions, PAS des valeurs : l'ancien exemple littéral
                # (`expect_equals: {"ok": True}`) était recopié tel quel par les
                # modèles, et un self-test faux fait échouer la création entière.
                "self_test_format": [
                    {
                        "args": "<an object valid for input_schema>",
                        "expect_equals": "<the JSON value run() returns for those args, as a plain literal>",
                    },
                    {
                        "args": "<an object valid for input_schema>",
                        "expect_contains": "<a substring of the result's text form>",
                    },
                ],
            },
            ensure_ascii=False,
        )


def _strip_code_fences(text: str) -> str:
    """LLMs often wrap JSON in a ```json … ``` fence despite being told not to.
    Strip a leading ```/```json line and a trailing ``` so json.loads can parse."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n?", "", t)
        t = re.sub(r"\r?\n?[ \t]*```$", "", t)
    return t.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    # `loads_tolerant` = `json.loads`, puis une seconde chance SI le texte est
    # refusé : échappements invalides réparés (`\d` dans une regex du champ `code`)
    # et retours à la ligne bruts acceptés (0.22.0). Un JSON valide n'est jamais touché.
    try:
        parsed: dict[str, Any] = loads_tolerant(text)
        return parsed
    except json.JSONDecodeError:
        pass

    # Markdown-fenced JSON (```json … ```) is the most common deviation — strip
    # the fence and retry strict parsing before the brace-walking fallback.
    stripped = _strip_code_fences(text)
    if stripped != text.strip():
        try:
            parsed_fenced: dict[str, Any] = loads_tolerant(stripped)
            return parsed_fenced
        except json.JSONDecodeError:
            pass

    candidate = _extract_first_json_object(stripped)
    if candidate is None:
        raise ToolValidationError("Tool builder response did not contain JSON")
    try:
        parsed_candidate: dict[str, Any] = loads_tolerant(candidate)
        return parsed_candidate
    except json.JSONDecodeError as exc:
        raise ToolValidationError(f"Tool builder response is not valid JSON: {exc}") from exc


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced top-level {...} block in text, or None.

    Walks the string tracking brace depth while skipping over string literals
    so braces inside JSON strings don't throw off the count.

    Quotes that appear in surrounding prose (before any `{` was seen) are
    treated as plain text, not as the start of a JSON string. Otherwise a
    quote in the LLM's natural-language preamble would cause the parser to
    miss the actual JSON block that follows.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, char in enumerate(text):
        if depth == 0:
            if char == "{":
                start = i
                depth = 1
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def _safe_tool_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()
    if not safe:
        raise ToolValidationError("Tool name is empty")
    if safe[0].isdigit():
        safe = f"tool_{safe}"
    return safe[:64]


def _upsert_tool_metadata(code: str, tool_meta: dict[str, Any]) -> str:
    required = {"name", "description", "input_schema"}
    missing = required - set(tool_meta)
    if missing:
        raise ToolValidationError(f"Tool metadata missing keys: {sorted(missing)}")
    metadata = pprint.pformat(tool_meta, sort_dicts=False, width=100)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f"TOOL = {metadata}\n\n{code}"

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOL":
                    lines = code.splitlines()
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", node.lineno)
                    replacement = f"TOOL = {metadata}".splitlines()
                    updated = lines[:start] + replacement + lines[end:]
                    return "\n".join(updated) + "\n"
    return f"TOOL = {metadata}\n\n{code}"
