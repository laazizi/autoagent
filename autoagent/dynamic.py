from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
import pprint
import re
import statistics
import tempfile
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._fichiers import atomic_write_text, quarantine
from .errors import ToolError, ToolValidationError
from .logging import get_logger, warn_once
from .providers.base import LLMProvider, loads_tolerant
from .sandbox import (
    _REFUS_DECLARE,
    DockerSandbox,
    GeneratedPythonTool,
    SubprocessSandbox,
    _a_une_frontiere,
    _bac_avec_frontiere,
    _bac_par_defaut,
    _exiger_la_frontiere,
    discard_generated_tool,
    load_generated_tool,
    validate_generated_tool_code,
)
from .schema import LLMRequest, Message, TokenUsage
from .validation import format_errors, validate

__all__ = ["DynamicToolBuilder", "ImprovementResult", "PythonRunner", "ToolBuildRequest", "ToolHealth"]

_log = get_logger("dynamic")

_CATALOGUE = "catalogue.json"

# Les issues qui comptent CONTRE l'outil (série d'erreurs, retrait). `host` (une fonction de l'hôte a refusé ou
# échoué) et `sandbox` (le bac lui-même a échoué) ne disent rien du code de l'outil : comptées à part.
_ISSUES_DE_L_OUTIL = frozenset({"exception", "timeout"})


def _json_canonique(valeur: Any) -> str:
    """La forme sous laquelle une valeur est gardée et empreintée au carnet : clés triées, sans espaces."""
    return json.dumps(valeur, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=repr)


_ABSENT: Any = object()                                    # « argument non donné », quand `None` est une valeur
_VARIABLE = re.compile(r"'[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?")   # ce qui change d'une erreur à l'autre


def _signature(erreur: str) -> str:
    """Deux erreurs qui ne diffèrent que par une valeur (« negatif : -1 », « negatif : -22 ») sont la même."""
    premiere = erreur.splitlines()[0] if erreur else ""
    return _VARIABLE.sub("…", premiere)[:160]


def _jugement(appel: dict[str, Any]) -> bool | None:
    """Cet appel montre-t-il un DÉFAUT de l'outil ? `True` oui, `False` non, `None` : il ne dit rien de l'outil.
    Le verdict de l'hôte prime ; sans lui, l'issue : `ok` non, `exception` / `timeout` oui, le reste (`host`,
    `sandbox`, `rejected`, `refused`) ne dit rien."""
    verdict = appel.get("verdict")
    if isinstance(verdict, dict) and isinstance(verdict.get("correct"), bool):
        return not verdict["correct"]
    issue = appel.get("kind")
    if issue == "ok":
        return False
    return True if issue in _ISSUES_DE_L_OUTIL else None


def _origine_des_tests(verified_by: dict[str, Any]) -> str:
    sorte = verified_by.get("kind")
    if sorte == "none":
        return "jamais vérifié"
    if sorte == "self_tests":
        n = int(verified_by.get("cases") or 0)
        return f"{n} auto-test écrit par le modèle" if n == 1 else f"{n} auto-tests écrits par le modèle"
    if sorte == "examples":
        montres, caches = int(verified_by.get("shown") or 0), int(verified_by.get("hidden") or 0)
        return f"{montres + caches} exemples de l'hôte, dont {caches} cachés au modèle"
    if sorte == "replay":
        return _preuve_par_rejeu(int(verified_by.get("fixed") or 0), int(verified_by.get("checked") or 0),
                                 int(verified_by.get("kept") or 0), int(verified_by.get("extended") or 0))
    return "vérification inconnue"


def _preuve_par_rejeu(corriges: int, verifies: int, gardes: int, etendus: int = 0) -> str:
    """Ce qu'une version améliorée a prouvé, en clair (`improve`) — et ce qu'elle n'a PAS prouvé : une entrée
    auparavant refusée qu'elle lit maintenant est dite « non vérifiée »."""
    rates = f"{corriges} cas raté corrigé" if corriges == 1 else f"{corriges} cas ratés corrigés"
    if verifies:
        rates += f" (dont {verifies} contre le résultat attendu)"
    reussis = (f"{gardes} cas réussi rejoué à l'identique" if gardes == 1
               else f"{gardes} cas réussis rejoués à l'identique")
    if not etendus:
        return f"{rates}, {reussis}"
    lues = (f"{etendus} entrée auparavant refusée maintenant lue (non vérifiée)" if etendus == 1
            else f"{etendus} entrées auparavant refusées maintenant lues (non vérifiées)")
    return f"{reussis}, {lues}" if not corriges else f"{rates}, {reussis}, {lues}"


def _empreinte(valeur: Any) -> str:
    return hashlib.sha256(_json_canonique(valeur).encode("utf-8")).hexdigest()


def _verdict_de(appel: dict[str, Any]) -> dict[str, Any]:
    verdict = appel.get("verdict")
    return verdict if isinstance(verdict, dict) else {}


def _texte_du_cas(appel: dict[str, Any]) -> str:
    return str(_verdict_de(appel).get("reason") or appel.get("error") or "wrong result")


def _choisir_les_cas(montrables: list[dict[str, Any]], combien: int) -> list[dict[str, Any]]:
    """Les cas ratés MONTRÉS au modèle : d'abord un par sorte d'erreur (le plus récent), puis les plus récents."""
    par_sorte: dict[str, dict[str, Any]] = {}
    for appel in montrables:                                # du plus ancien au plus récent : le dernier gagne
        par_sorte[_signature(_texte_du_cas(appel))] = appel
    choisis = list(par_sorte.values())[-combien:]
    pris = {id(appel) for appel in choisis}
    for appel in reversed(montrables):
        if len(choisis) >= combien:
            break
        if id(appel) not in pris:
            choisis.append(appel)
            pris.add(id(appel))
    return choisis


def _decrire_le_cas(numero: int, appel: dict[str, Any]) -> str:
    args = json.dumps(appel.get("args"), ensure_ascii=False)[:600]
    verdict = _verdict_de(appel)
    attendu = (f" The right result is {json.dumps(verdict['expected'], ensure_ascii=False)[:400]}."
               if "expected" in verdict else "")
    if verdict.get("expected_refusal") is True:
        attendu = ' This input is INVALID: the tool must REFUSE it (raise ValueError("invalid input: <why>")).'
    raison = f" The host says: {str(verdict['reason'])[:200]}." if verdict.get("reason") else ""
    issue = appel.get("kind")
    if issue == "ok":
        rendu = json.dumps(appel["result"], ensure_ascii=False)[:300] if "result" in appel else "a result"
        return f"{numero}. args = {args} -> it returned {rendu}, which is WRONG.{attendu}{raison}"
    if issue == "rejected":
        return f"{numero}. args = {args} -> refused by the input_schema, but this input is VALID.{attendu}{raison}"
    valide = " This input is VALID: the tool must handle it." if verdict.get("correct") is False else ""
    return f"{numero}. args = {args} -> error: {str(appel.get('error'))[:300]}.{valide}{attendu}{raison}"


def _decrire_le_refus(numero: int, appel: dict[str, Any]) -> str:
    args = json.dumps(appel.get("args"), ensure_ascii=False)[:600]
    par = "refused by the input_schema" if appel.get("kind") == "rejected" else "refused"
    return f"{numero}. args = {args} -> {par}: {str(appel.get('error'))[:300]}"


def _demande_d_amelioration(description: str, code: str, montres: list[dict[str, Any]],
                            refus: list[dict[str, Any]]) -> str:
    """Ce que le modèle constructeur reçoit. Sans refus à montrer, le texte est MOT POUR MOT celui qui a été
    mesuré sur un vrai modèle : la rubrique des refus ne s'ajoute que si `learn_from_refusals` en donne."""
    cas = "\n".join(_decrire_le_cas(i, appel) for i, appel in enumerate(montres, 1))
    rates = f"It failed on these REAL calls:\n{cas}\n\n" if montres else ""
    refuses = ""
    if refus:
        lignes = "\n".join(_decrire_le_refus(i, appel) for i, appel in enumerate(refus, 1))
        refuses = (
            f"It REFUSED these REAL inputs, and nobody has said whether it was right to:\n{lignes}\n"
            "Decide for each one. If it is a valid request for this tool's purpose, only written another way, "
            "handle it and return the right result. If it is truly invalid, or if you cannot read it with "
            "certainty, keep refusing it: never guess a value.\n\n"
        )
    return (
        f"IMPROVE AN EXISTING TOOL - keep its name and its purpose: {description}\n\n"
        f"Its current code:\n{code[:20000]}\n\n"
        f"{rates}{refuses}"
        "Fix the CAUSE, not only these inputs. Other real calls that succeed today will be replayed against your "
        "new version, and you are not shown them: their results must not change.\n"
        "If an input is truly invalid for this tool, REFUSE it: let input_schema reject it (pattern, enum, minimum, "
        'minItems...) or raise ValueError("invalid input: <why>"). Never return an error value in place of a result.'
    )


def _cas_resolu(appel: dict[str, Any], issue: str, valeur: Any, tolerance: float) -> tuple[bool, bool]:
    """Un cas RATÉ est-il résolu par la nouvelle version ? Rend (résolu, vérifié contre l'attendu de l'hôte).

    L'hôte a dit qu'il fallait refuser l'entrée : il faut un refus (prouvé). L'hôte a donné l'attendu : il faut
    CE résultat (prouvé). L'hôte a jugé l'appel fautif sans donner l'attendu :
    l'entrée était valide, il faut UN résultat — et un autre que le résultat faux. Un plantage que personne n'a
    jugé : un résultat, un refus du schéma ou un refus déclaré le résolvent (l'outil ne plante plus)."""
    verdict = _verdict_de(appel)
    if verdict.get("expected_refusal") is True:
        return issue in _REFUS_NON_JUGEABLES, True           # l'hôte a dit : cette entrée devait être refusée
    if "expected" in verdict:
        return issue == "result" and _egal(verdict["expected"], valeur, tolerance), True
    if "expected_sha256" in verdict:
        return issue == "result" and _empreinte(valeur) == verdict["expected_sha256"], True
    if verdict.get("correct") is False:
        if issue != "result":
            return False, False
        return not (appel.get("kind") == "ok" and appel.get("result_sha256") == _empreinte(valeur)), False
    return issue in ("result", "rejected", "refused"), False


# Les issues qui sont un REFUS de l'entrée — par l'outil (`ValueError("invalid input: …")`) ou par son schéma.
_REFUS_NON_JUGEABLES = ("refused", "rejected")
# Combien d'entrées refusées « déjà examinées » le catalogue retient par outil (`learn_from_refusals`).
_MAX_REFUS_EXAMINES = 200


def _cle_du_refus(appel: dict[str, Any]) -> str:
    """L'entrée d'un appel refusé, par son empreinte (le début suffit) : ce que le catalogue retient."""
    return str(appel.get("args_sha256") or "")[:16]


@dataclass
class _Cas:
    """Les appels de la version courante d'un outil, triés pour `improve` (`_cas_du_carnet`)."""

    rates: list[dict[str, Any]] = field(default_factory=list)          # des défauts : à résoudre
    reussis: list[dict[str, Any]] = field(default_factory=list)        # à rejouer à l'identique
    refuses: list[dict[str, Any]] = field(default_factory=list)        # des refus que personne n'a jugés
    legitimes: list[dict[str, Any]] = field(default_factory=list)      # des échecs que l'hôte a jugés légitimes
    rates_inutilisables: int = 0
    refus_inutilisables: int = 0


@dataclass
class ImprovementResult:
    """Le compte rendu d'un `DynamicToolBuilder.improve` (main, non publié).

    Attributes:
        name: l'outil.
        status: `"improved"` (la nouvelle version a remplacé l'ancienne) ; `"rejected"` (elle n'a pas prouvé ce
            qu'il fallait : l'ancienne reste) ; `"build_failed"` (le modèle n'a pas rendu un outil valide) ;
            `"nothing_to_fix"` (aucun cas raté au carnet pour la version courante) ; `"no_usable_case"` (des cas
            ratés, mais aucun qu'on puisse montrer ou rejouer) ; `"not_replayable"` (l'outil appelle des fonctions
            de l'hôte) ; `"quarantined"` ; `"tainted_run"` (appelé depuis un run qui a lu du contenu non fiable) ;
            `"nothing_gained"` (`learn_from_refusals` : la nouvelle version ne corrige rien et refuse toujours les
            mêmes entrées — l'ancienne reste).
        reason: pourquoi, en une phrase.
        old_version, new_version: le début du sha256 du code avant, et après (`None` si rien n'a été installé).
        fixed, checked: cas ratés résolus ; dont vérifiés contre le résultat que l'hôte attendait.
        kept: cas réussis du carnet rejoués à l'identique — jamais montrés au modèle.
        extended, still_refused (`learn_from_refusals`) : entrées refusées que la nouvelle version LIT — ce qu'elle
            en rend n'est PAS vérifié, c'est le modèle qui a jugé l'entrée valide ; et celles qu'elle refuse toujours.
        extended_cases: les 20 premières entrées nouvellement lues, avec ce que l'outil en rend — `{"args", "got"}` :
            à relire par l'hôte, puisque personne ne les a vérifiées.
        broken_refusals: refus cassés — un refus propre devenu un plantage, ou une entrée que l'hôte avait jugée
            invalide et que la nouvelle version accepte — `{"args", "got" | "error", "why"}`.
        still_failing: cas ratés non résolus — `{"args", "got" | "error", "expected"?}`.
        regressions: cas réussis qui ne donnent plus le même résultat — `{"args", "expected", "got"}`.
        skipped: cas ratés inutilisables (masqués par `redact`, ou trop gros pour avoir été gardés).
        usage: jetons du modèle constructeur (`None` si aucun appel n'a été fait).
        tool: la nouvelle version, prête à l'emploi (`None` si rien n'a été installé).
    """

    name: str
    status: str = ""
    reason: str = ""
    old_version: str = ""
    new_version: str | None = None
    fixed: int = 0
    checked: int = 0
    kept: int = 0
    still_failing: list[dict[str, Any]] = field(default_factory=list)
    regressions: list[dict[str, Any]] = field(default_factory=list)
    skipped: int = 0
    usage: TokenUsage | None = None
    tool: GeneratedPythonTool | None = None
    extended: int = 0
    still_refused: int = 0
    extended_cases: list[dict[str, Any]] = field(default_factory=list)
    broken_refusals: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        if self.status == "improved":
            return (f"{self.name} — amélioré : version {self.old_version} remplacée par {self.new_version} · "
                    f"{_preuve_par_rejeu(self.fixed, self.checked, self.kept, self.extended)}. "
                    "L'ancienne est gardée (rollback).")
        if self.status == "rejected":
            rates, changes = len(self.still_failing), len(self.regressions)
            pourquoi = [f"{rates} cas raté non résolu" if rates == 1 else f"{rates} cas ratés non résolus"] if rates else []
            if changes:
                pourquoi.append(f"{changes} cas réussi changé" if changes == 1 else f"{changes} cas réussis changés")
            if self.broken_refusals:
                pourquoi.append(f"{len(self.broken_refusals)} refus cassé" + ("" if len(self.broken_refusals) == 1 else "s"))
            # Sans cas en cause (des permissions en plus), c'est `reason` qui dit pourquoi.
            return (f"{self.name} — nouvelle version REFUSÉE, l'ancienne reste : "
                    + (", ".join(pourquoi) + "." if pourquoi else self.reason))
        quoi = {
            "nothing_to_fix": "rien à corriger : aucun cas raté au carnet pour la version courante",
            "no_usable_case": "des cas ratés existent, mais aucun ne peut être montré au modèle ni rejoué",
            "not_replayable": "l'outil appelle des fonctions de l'hôte : ses appels ne se rejouent pas sans effets de bord",
            "quarantined": "l'outil est en quarantaine : le libérer d'abord (release)",
            "tainted_run": "refusé : appelé depuis un run qui a lu du contenu non fiable",
            "build_failed": "le modèle n'a pas rendu un outil valide ; l'ancienne version reste",
            "nothing_gained": "aucun gain : la nouvelle version ne corrige rien et refuse toujours les mêmes entrées ; "
                              "l'ancienne reste",
        }.get(self.status, self.status)
        return f"{self.name} — {quoi}." + (f" {self.reason}" if self.reason and self.status == "build_failed" else "")


@dataclass(frozen=True)
class ToolHealth:
    """La fiche de qualité d'un outil créé par le modèle (`DynamicToolBuilder.health`, main, non publié).

    Attributes:
        name, status: le nom ; `"active"`, `"retired"` ou `"quarantined"`.
        version, created: le début du sha256 du code courant ; sa date de construction.
        verified_by: d'où viennent ses tests — `{"kind": "none"}`, `"self_tests"` (+ `cases`), `"examples"`
            (+ `shown`, `hidden`), `"unknown"` (un catalogue d'avant).
        calls, errors, unattributed_failures, rejected_calls, refused_calls: les compteurs du catalogue, depuis
            la construction de la version courante (`errors` : les défauts de l'outil ; non attribués : hôte, bac ;
            `rejected` : refusés par le schéma avant de tourner ; `refused` : refusés par l'outil lui-même).
        basis: `"call log"` (les chiffres ci-dessous viennent du carnet) ou `"catalogue"` (pas de carnet).
        window: appels du carnet pris en compte — ceux de la VERSION COURANTE (0 sans carnet).
        previous_calls: appels du carnet faits sur une version précédente (remplacée par `improve`, ou
            reconstruite) : lisibles dans `versions` et dans `calls()`, jamais comptés dans les chiffres ci-dessous
            — la fiche parle du code qui tourne, pas des fautes qu'on vient de corriger.
        outcomes: appels de la fenêtre, par issue.
        judged, defects: appels qui disent quelque chose de l'outil, et ceux qui montrent un DÉFAUT — le verdict
            de l'hôte (`report`) prime ; sinon `exception` et `timeout`.
        defect_rate, defect_interval: `defects / judged` et son intervalle de Wilson à 95 % (`None` sans appel jugé).
        median_seconds: durée médiane d'une exécution, vue par l'hôte.
        errors_seen: les défauts regroupés — `{"error", "count", "example_args", "last_id"}`, du plus fréquent.
        verdicts: verdicts de l'hôte dans la fenêtre — `{"correct": n, "wrong": n}`.
        versions: par version vue dans la fenêtre — `{"judged": n, "defects": n}`.
    """

    name: str
    status: str
    version: str
    created: str
    verified_by: dict[str, Any]
    calls: int
    errors: int
    unattributed_failures: int
    rejected_calls: int
    refused_calls: int
    basis: str
    window: int
    outcomes: dict[str, int]
    judged: int
    defects: int
    defect_rate: float | None
    defect_interval: tuple[float, float] | None
    median_seconds: float | None
    errors_seen: list[dict[str, Any]]
    verdicts: dict[str, int]
    versions: dict[str, dict[str, int]]
    previous_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        statut = {"active": "actif", "retired": "retiré", "quarantined": "en quarantaine"}.get(self.status, self.status)
        lignes = [f"{self.name} — {statut} · version {self.version or '?'} · {_origine_des_tests(self.verified_by)}"]
        if self.judged and self.defect_rate is not None and self.defect_interval is not None:
            bas, haut = self.defect_interval
            mot = "défaut" if self.defects <= 1 else "défauts"
            juges = "appel jugé" if self.judged == 1 else "appels jugés"
            ligne = (f"  {self.defects} {mot} sur {self.judged} {juges} ({self.defect_rate * 100:.0f} %, intervalle à "
                     f"95 % : {bas * 100:.0f} % à {haut * 100:.0f} %)")
            if self.median_seconds is not None:
                ligne += f" · durée médiane {self.median_seconds:.2f} s"
            lignes.append(ligne)
        else:
            lignes.append("  aucun appel qui dise quelque chose de l'outil")
        source = "carnet" if self.basis == "call log" else "catalogue seul (pas de carnet)"
        if self.outcomes:
            detail = ", ".join(f"{n} {issue}" for issue, n in sorted(self.outcomes.items(), key=lambda p: (-p[1], p[0])))
            lignes.append(f"  issues ({source}, {self.window} appels) : {detail}")
        else:
            lignes.append(f"  source : {source} · {self.calls} exécutions, {self.unattributed_failures} échecs non "
                          f"attribués, {self.rejected_calls} appels refusés par le schéma, {self.refused_calls} refusés "
                          f"par l'outil")
        if self.previous_calls:
            mot = "appel d'une version précédente" if self.previous_calls == 1 else "appels de versions précédentes"
            lignes.append(f"  {self.previous_calls} {mot} au carnet, non comptés ici")
        if self.verdicts.get("correct") or self.verdicts.get("wrong"):
            lignes.append(f"  verdicts de l'hôte : {self.verdicts.get('correct', 0)} « correct », "
                          f"{self.verdicts.get('wrong', 0)} « faux »")
        for vue in self.errors_seen[:5]:
            exemple = "" if vue.get("example_args") is None else f"   (ex. {json.dumps(vue['example_args'], ensure_ascii=False)[:80]})"
            lignes.append(f"  {vue['count']} fois : {str(vue['error'])[:110]}{exemple}")
        return "\n".join(lignes)


def _teinte_du_run() -> bool | None:
    """Le run qui appelle l'outil a-t-il ingéré du contenu non fiable ? `None` hors de tout run. Une lecture
    qui plante : on suppose teinté (comme `_provenance_du_run`)."""
    from .gate import run_taint
    try:
        return run_taint()
    except Exception:
        return True


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


def _provenance_du_run() -> dict[str, Any]:
    """Le contexte du run à la construction, lu sur la porte active (`gate.py`). Hors d'un run
    d'agent (appel à la main) : pas de teinte connue, donc propre. Un sous-agent appelé par un
    run teinté est teinté (`lineage_tainted`) : sans cela, déléguer la création à un sous-agent
    propre blanchissait la teinte. Lecture qui plante : on suppose teinté (fail-closed)."""
    from .gate import active_gate
    porte = active_gate()
    if porte is None:
        return {"tainted": False, "context": "no_run"}
    try:
        # La délégation d'abord : le run du sous-agent naît teinté, mais la cause est le parent.
        if porte.lineage_tainted:
            return {"tainted": True, "context": "delegated_from_tainted_run"}
        if porte.tainted:
            return {"tainted": True, "context": "run"}
        return {"tainted": False, "context": "run"}
    except Exception:
        _log.exception("could not read the run taint; assuming tainted (fail-closed)")
        return {"tainted": True, "context": "run_unreadable"}


def _run_en_cours() -> Any:
    """L'objet qui identifie le run en cours (ses gardes de tour, un par run), ou `None`."""
    from .gate import active_gate
    porte = active_gate()
    return getattr(porte, "guards", None) if porte is not None else None


def _descendants(outils: dict[str, dict[str, Any]], name: str) -> list[str]:
    def seq(n: str) -> int:
        return int((outils[n].get("provenance") or {}).get("seq") or 0)

    if name not in outils:
        return []
    trouves: set[str] = set()
    courants = {name}
    while courants:
        suivants: set[str] = set()
        for nom, e in outils.items():
            if nom == name or nom in trouves:
                continue
            vus = set((e.get("provenance") or {}).get("visible_tools") or []) & set(outils)
            if any(c in vus and seq(nom) > seq(c) for c in courants):
                suivants.add(nom)
        trouves |= suivants
        courants = suivants
    return sorted(trouves)


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
        self.permissions = sorted(set(permissions or []))
        # Un bac choisi par l'HÔTE n'est jamais remplacé en silence (`Agent(require_boundary=True)` le refuse).
        self._bac_impose = sandbox is not None
        # Sans bac choisi par l'hôte : confiné par le noyau là où c'est possible, tant que le lanceur n'a pas
        # de permission de fichiers (`sandbox._bac_par_defaut` dit la règle). Sinon le sous-processus d'avant.
        sandbox = sandbox or _bac_par_defaut(self.permissions, timeout=timeout)
        if isinstance(sandbox, SubprocessSandbox) and sandbox.warm:
            sandbox = dataclasses.replace(sandbox, warm=False)
        self.sandbox = sandbox
        self.max_code_chars = max_code_chars
        self.host_functions: dict[str, Callable[..., Any]] | None = dict(host_functions) if host_functions else None
        if not _a_une_frontiere(self.sandbox):
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
        _exiger_la_frontiere(self.sandbox)
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
        sandbox: SubprocessSandbox | DockerSandbox | None = None,
        timeout: float = 10.0,
        allowed_permissions: Iterable[str] | None = None,
        max_repairs: int = 0,
        persist: bool = False,
        retire_after_errors: int = 3,
        self_test_rel_tol: float = 1e-6,
        host_functions: dict[str, Callable[..., Any]] | None = None,
        record_calls: int = 0,
        redact: Callable[[Any], Any] | None = None,
        max_record_chars: int = 4000,
        max_build_tokens: int = 8192,
    ) -> None:
        """``max_build_tokens`` (main, non publié) : le plafond de SORTIE de l'appel au modèle constructeur (8192,
        comme avant). Un modèle qui réfléchit le dépense avant d'écrire l'outil — mesuré sur Gemini, à une demande
        d'amélioration : 7 861 jetons de réflexion sur 8 192, réponse coupée. Une réponse coupée par ce plafond
        lève ``ToolValidationError`` en le DISANT, et n'est jamais reprise par ``max_repairs`` : au même plafond,
        la reprise rendrait la même réponse coupée. `improve` garde ce plafond. Un fournisseur peut refuser une
        valeur plus haute que la limite de son modèle.

        ``record_calls`` (main, non publié) : le CARNET d'appels. Les ``N`` derniers appels de chaque outil de
        la bibliothèque sont gardés dans ``tools_dir/<nom>.calls.jsonl`` — ce qu'il a reçu, l'issue (``kind``),
        la cause, la durée, la version du code, la teinte du run — et relus par ``calls(nom)``. ``0`` (défaut) :
        rien n'est écrit. Demande ``persist=True``. Le carnet garde ce que l'outil REÇOIT : il peut porter des
        données personnelles. ``redact`` (une fonction ``valeur -> valeur``) passe AVANT toute écriture sur les
        arguments, le résultat et le texte de l'erreur (aussi le ``last_error`` du catalogue) ; s'il plante, ni
        valeur ni empreinte ne sont écrites. ``max_record_chars`` : au-delà, une valeur n'est gardée que par son
        empreinte. Observabilité : un carnet qui ne s'écrit pas ne casse jamais un appel.

        ``host_functions`` (0.22.0) : callbacks de l'HÔTE que les outils générés
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
        # Sans bac choisi par l'hôte, le choix se fait OUTIL PAR OUTIL (`_bac_pour`) : confiné par le noyau
        # là où c'est possible, sauf pour un outil qui porte une permission de fichiers.
        self._bac_impose = sandbox is not None
        self.sandbox: SubprocessSandbox | DockerSandbox = sandbox or _bac_par_defaut(timeout=timeout)
        self._bac_sans_frontiere: SubprocessSandbox | None = None
        # Posé par `Agent(require_boundary=True)` (0.28.0) : plus jamais le sous-processus simple.
        self._frontiere_requise = False
        self.allowed_permissions = None if allowed_permissions is None else set(allowed_permissions)
        self.max_repairs = max(0, int(max_repairs))
        if isinstance(max_build_tokens, bool) or not isinstance(max_build_tokens, int) or max_build_tokens <= 0:
            raise ValueError(f"max_build_tokens must be an integer > 0, got {max_build_tokens!r}")
        self.max_build_tokens = max_build_tokens
        self.self_test_rel_tol = max(0.0, float(self_test_rel_tol))
        self.host_functions: dict[str, Callable[..., Any]] | None = dict(host_functions) if host_functions else None
        self.persist = bool(persist)
        self.retire_after_errors = max(0, int(retire_after_errors))
        if isinstance(record_calls, bool) or not isinstance(record_calls, int) or record_calls < 0:
            raise ValueError(f"record_calls must be an integer >= 0, got {record_calls!r}")
        if record_calls and not self.persist:
            raise ValueError("record_calls needs persist=True: the call log belongs to the persistent library")
        if isinstance(max_record_chars, bool) or not isinstance(max_record_chars, int) or max_record_chars <= 0:
            raise ValueError(f"max_record_chars must be an integer > 0, got {max_record_chars!r}")
        if redact is not None and not callable(redact):
            raise TypeError("redact must be a callable (value -> value) or None")
        self.record_calls = record_calls
        self.redact = redact
        self.max_record_chars = max_record_chars
        self._lignes_du_carnet: dict[str, int] = {}     # par outil : lignes du fichier (pour borner sans le relire)
        self._verrou = threading.RLock()        # lecture-modification-écriture du catalogue
        # Provenance : les outils de la bibliothèque que le modèle VOIT dans ce processus
        # (rechargés par `load_library` + construits ici). Un outil en quarantaine n'y est jamais.
        self._visibles: set[str] = set()
        # Quarantaine connue de CE processus, en plus du catalogue : un catalogue perdu ou
        # corrompu ne doit pas libérer un outil déjà chargé. Et, pour chaque outil mis en
        # quarantaine à sa construction, le run qui l'a construit (référence faible) : c'est
        # le seul où il reste appelable.
        self._en_quarantaine: set[str] = set()
        self._run_de_construction: dict[str, weakref.ref[Any]] = {}
        # Dépense du dernier `build()` (0.22.0). L'appel au modèle constructeur
        # coûtait des jetons INVISIBLES à `token_budget` : le modèle pouvait
        # construire outil sur outil sous un plafond aveugle. `create_python_tool`
        # lit `last_build_usage` et le reverse dans la comptabilité du run. PAR
        # THREAD (comme `_CanalDepense`) : deux créations en `parallel_tool_calls`
        # ne s'écrasent pas leur dépense.
        self._local = threading.local()
        if not _a_une_frontiere(self.sandbox):
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

    def _imposer_la_frontiere(self) -> None:
        """Appelé par `Agent.enable_dynamic_tools` quand l'agent exige une frontière (0.28.0). Un bac par défaut
        sans frontière devient le meilleur bac à frontière de la machine ; un bac choisi par l'hôte sans frontière
        est refusé (``ToolError``) — jamais remplacé en silence. Rien ne change si le remplacement échoue."""
        if not _a_une_frontiere(self.sandbox):
            if self._bac_impose:
                raise ToolError(
                    "require_boundary=True: the sandbox given to DynamicToolBuilder is not an OS-enforced "
                    "isolation boundary. Pass a DockerSandbox or a confined SubprocessSandbox, or none to let "
                    "the agent pick the best boundary available."
                )
            self.sandbox = _bac_avec_frontiere(timeout=self.sandbox.timeout)
        self._frontiere_requise = True

    def _bac_pour(self, permissions: Iterable[str] | None) -> SubprocessSandbox | DockerSandbox:
        """Le bac d'UN outil. Celui de l'hôte s'il en a choisi un. Sinon le bac par défaut (confiné là où le
        noyau le permet) — sauf pour un outil qui porte une permission ``filesystem.*`` : des fichiers ont
        été accordés sans dire lesquels, il tourne donc comme avant, dans le sous-processus simple, et on le
        dit une fois. Sous `require_boundary=True` (0.28.0), cet outil-là va dans Docker, ou il est refusé."""
        if self._bac_impose or not _a_une_frontiere(self.sandbox):
            return self.sandbox
        if not any(str(p).startswith("filesystem.") for p in (permissions or ())):
            return self.sandbox
        if self._frontiere_requise:
            if isinstance(self.sandbox, DockerSandbox):
                return self.sandbox
            return _bac_avec_frontiere(permissions, timeout=self.sandbox.timeout)
        with self._verrou:
            if self._bac_sans_frontiere is None:
                self._bac_sans_frontiere = SubprocessSandbox(timeout=self.sandbox.timeout)
        warn_once(
            _log, "dynamic.outil_fichiers_sans_frontiere",
            "A generated tool holds a filesystem.* permission and no sandbox was chosen: it runs in a plain "
            "SubprocessSandbox (AST denylist, not an isolation boundary), because no directory was named. "
            "Pass sandbox=SubprocessSandbox(confine=Confinement(read=[...], write=[...])) to confine it.",
        )
        return self._bac_sans_frontiere

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
                    # 8192 par défaut : de la marge pour les modèles qui réfléchissent, dont la
                    # réflexion consomme des jetons avant le JSON de sortie. Pas toujours assez :
                    # voir `max_build_tokens`.
                    max_tokens=self.max_build_tokens,
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
                if response.finish_reason == "length":
                    # La réponse a été COUPÉE par le plafond de sortie (mesuré sur Gemini : 7 861 jetons de
                    # réflexion sur 8 192). Ce n'est pas un outil mal écrit, et une reprise au même plafond
                    # rendrait la même réponse coupée : on le dit, sans la payer une seconde fois.
                    raise ToolValidationError(
                        f"Tool builder ran out of output tokens (max_tokens={self.max_build_tokens}) before it "
                        "finished its answer: a reasoning model can spend them thinking. Raise "
                        "DynamicToolBuilder(max_build_tokens=...) or lower the model's reasoning effort."
                    ) from exc
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
            self._noter_construction(generated_tool, _provenance_du_run())
            generated_tool.recorder = self._noter_appel
            generated_tool.admission = self._admission
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
            generated_tool = load_generated_tool(file_path, sandbox=self._bac_pour(permissions))
            self._run_self_tests(generated_tool, tests)
        except Exception:
            discard_generated_tool(file_path)
            raise
        # L'origine des tests est ÉCRITE : aucun, ou ceux que le modèle a fournis lui-même (5 exécutés au plus).
        executes = len(tests[:5]) if isinstance(tests, list) else 0
        generated_tool.verified_by = {"kind": "self_tests", "cases": executes} if executes else {"kind": "none"}
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
        `description`, `permissions`, `calls`, `errors` (les défauts de l'OUTIL : il a planté ou dépassé son
        délai), `consecutive_errors`, `unattributed_failures` (les échecs qui ne disent rien de son code : une
        fonction de l'hôte refusée, un bac en panne), `retired` (+ `retired_reason`, `last_error`).
        Vide sans `persist=True`."""
        with self._verrou:
            copie: dict[str, dict[str, Any]] = json.loads(json.dumps(self._lire_catalogue()))
            return copie

    def _noter_construction(self, outil: GeneratedPythonTool, provenance: dict[str, Any]) -> None:
        """Inscrit l'outil au catalogue AVEC sa provenance (0.25.0) : le run était-il teinté
        (du contenu non fiable avait-il été ingéré) quand le modèle l'a fait écrire, et
        quels outils de la bibliothèque étaient visibles. Un outil construit dans un run
        teinté est mis en QUARANTAINE : jamais rechargé tant qu'un humain ne l'a pas
        libéré (`release`). Fail-closed : une teinte illisible quarantaine."""
        with self._verrou:
            outils = self._lire_catalogue()
            nom = outil.spec.name
            suite = max([int((e.get("provenance") or {}).get("seq") or 0) for e in outils.values()] + [0]) + 1
            provenance = {**provenance, "seq": suite, "visible_tools": sorted(self._visibles - {nom})}
            entree: dict[str, Any] = {
                "sha256": hashlib.sha256(outil.file_path.read_bytes()).hexdigest(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "description": outil.spec.description,
                "permissions": list(outil.spec.permissions or []),
                "calls": 0, "errors": 0, "consecutive_errors": 0, "unattributed_failures": 0,
                "rejected_calls": 0, "refused_calls": 0, "retired": False,
                "verified_by": dict(outil.verified_by or {"kind": "none"}),
                "provenance": provenance,
            }
            if provenance.get("tainted"):
                entree["quarantined"] = True
                entree["quarantine_reason"] = "built in a run that had ingested untrusted content"
                _log.warning("generated tool %r built in a TAINTED run: quarantined (not reloaded until released)", nom)
                self._visibles.discard(nom)
                self._en_quarantaine.add(nom)
                run = _run_en_cours()
                if run is not None:
                    self._run_de_construction[nom] = weakref.ref(run)
                else:
                    self._run_de_construction.pop(nom, None)
            else:
                self._visibles.add(nom)
                self._en_quarantaine.discard(nom)
                self._run_de_construction.pop(nom, None)
            outils[nom] = entree
            self._ecrire_catalogue(outils)

    def _noter_appel(self, appel: dict[str, Any]) -> None:
        """Après CHAQUE appel d'un outil de la bibliothèque (`GeneratedPythonTool.recorder`) : les compteurs du
        catalogue, puis le carnet. Seul un défaut de l'OUTIL (`exception`, `timeout`) allonge la série d'erreurs
        et peut le faire retirer : avant, une fonction de l'hôte refusée par la politique ou un bac en panne
        (un démon Docker arrêté) retiraient un outil sain au bout de `retire_after_errors` appels."""
        nom, ok, issue = str(appel.get("name")), bool(appel.get("ok")), str(appel.get("kind"))
        masque = self._masquer(appel)
        with self._verrou:
            outils = self._lire_catalogue()
            entree = outils.get(nom)
            if entree is None:
                return
            if issue == "rejected":
                # Le schéma d'entrée a refusé l'appel : l'outil n'a pas tourné. Ni une exécution, ni un défaut
                # de son code, ni une réussite : la série d'erreurs ne bouge pas.
                entree["rejected_calls"] = int(entree.get("rejected_calls") or 0) + 1
            elif issue == "refused":
                # L'outil a tourné et a REFUSÉ l'entrée (`ValueError("invalid input: …")`) : ni un défaut de son
                # code, ni une réussite. Compté à part ; la série d'erreurs ne bouge pas. Avant, trois mauvaises
                # entrées de suite faisaient retirer un outil juste (constaté sur un vrai modèle).
                entree["calls"] = int(entree.get("calls") or 0) + 1
                entree["refused_calls"] = int(entree.get("refused_calls") or 0) + 1
            elif ok:
                entree["calls"] = int(entree.get("calls") or 0) + 1
                entree["consecutive_errors"] = 0
            elif issue in _ISSUES_DE_L_OUTIL:
                entree["calls"] = int(entree.get("calls") or 0) + 1
                entree["errors"] = int(entree.get("errors") or 0) + 1
                entree["consecutive_errors"] = int(entree.get("consecutive_errors") or 0) + 1
                entree["last_error"] = (masque["error"] or "")[:300]
                limite = self.retire_after_errors
                if limite and entree["consecutive_errors"] >= limite and not entree.get("retired"):
                    entree["retired"] = True
                    entree["retired_reason"] = f"{entree['consecutive_errors']} consecutive errors"
                    _log.warning("generated tool %r retired from the library (%s)", nom, entree["retired_reason"])
            else:
                entree["calls"] = int(entree.get("calls") or 0) + 1
                entree["unattributed_failures"] = int(entree.get("unattributed_failures") or 0) + 1
            self._ecrire_catalogue(outils)
            if self.record_calls:
                self._ecrire_au_carnet(nom, self._ligne_du_carnet(appel, masque, str(entree.get("sha256") or "")[:12]))

    def _masquer(self, appel: dict[str, Any]) -> dict[str, Any]:
        """Ce qui peut porter une donnée — arguments, résultat, texte de l'erreur — passé par `redact`.
        Un `redact` qui plante ne laisse RIEN passer (fail-closed) : ni valeur, ni empreinte, ni texte."""
        args, resultat, erreur = appel.get("args"), appel.get("result"), appel.get("error")
        if self.redact is None:
            return {"args": args, "result": resultat, "error": erreur, "redacted": None}
        try:
            return {
                "args": self.redact(args),
                "result": None if resultat is None else self.redact(resultat),
                "error": None if erreur is None else str(self.redact(erreur)),
                "redacted": True,
            }
        except Exception as exc:
            # Le type seulement : le message d'une exception de `redact` peut citer la donnée qu'il devait masquer.
            _log.warning("redact raised %s; the call is recorded without its data", type(exc).__name__)
            return {"args": None, "result": None, "error": None, "redacted": "failed"}

    def _ligne_du_carnet(self, appel: dict[str, Any], masque: dict[str, Any], version: str) -> str:
        entree: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": version,
            "ok": bool(appel.get("ok")),
            "kind": str(appel.get("kind")),
            "seconds": round(float(appel.get("seconds") or 0.0), 3),
            "tainted": _teinte_du_run(),
        }
        if masque["redacted"] is not None:
            entree["redacted"] = masque["redacted"]
        if masque["redacted"] != "failed":
            champs = [("args", masque["args"])] + ([("result", masque["result"])] if entree["ok"] else [])
            tronques: list[str] = []
            for cle, valeur in champs:
                texte = _json_canonique(valeur)
                entree[f"{cle}_sha256"] = hashlib.sha256(texte.encode("utf-8")).hexdigest()
                if len(texte) <= self.max_record_chars:
                    entree[cle] = json.loads(texte)         # gardé sous la forme empreintée
                else:
                    tronques.append(cle)
            if tronques:
                entree["truncated"] = tronques
            if not entree["ok"] and masque["error"]:
                entree["error"] = str(masque["error"])[:300]
        return json.dumps(entree, ensure_ascii=False)

    def _chemin_du_carnet(self, nom: str) -> Path:
        return self.tools_dir / f"{nom}.calls.jsonl"

    def _ecrire_au_carnet(self, nom: str, ligne: str) -> None:
        """Ajoute une ligne ; au-delà de 2 N lignes, ne garde que les N dernières (réécriture atomique).
        Appelé sous le verrou. Fail-open : un carnet qui ne s'écrit pas ne casse jamais un appel."""
        chemin = self._chemin_du_carnet(nom)
        try:
            lignes = self._lignes_du_carnet.get(nom)
            if lignes is None:
                lignes = len(chemin.read_text(encoding="utf-8").splitlines()) if chemin.is_file() else 0
            with chemin.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(ligne + "\n")
            lignes += 1
            if lignes > 2 * self.record_calls:
                gardees = chemin.read_text(encoding="utf-8").splitlines()[-self.record_calls:]
                atomic_write_text(chemin, "\n".join(gardees) + "\n")
                lignes = len(gardees)
            self._lignes_du_carnet[nom] = lignes
        except (OSError, ValueError):
            warn_once(_log, f"dynamic.carnet.{nom}",
                      f"could not write the call log of generated tool {nom!r} ({chemin}); calls go on unrecorded")

    def calls(self, name: str, *, failed: bool | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Le carnet d'un outil de la bibliothèque, du plus ancien au plus récent (`record_calls=N`).

        Chaque appel : `id`, `ts`, `version` (début du sha256 du code qui a tourné), `ok`, `kind` (`ok`,
        `exception`, `timeout`, `host`, `sandbox`), `seconds`, `tainted` (le run avait-il lu du contenu non
        fiable ; `None` hors d'un run), `args` + `args_sha256`, `result` + `result_sha256` (appel réussi),
        `error` (appel raté), `truncated` (les champs trop gros, gardés par leur seule empreinte), `redacted`
        (`True` : passé par `redact` ; `"failed"` : `redact` a planté, aucune donnée n'a été écrite).
        `failed=True` / `False` ne rend que les ratés / les réussis ; `limit` les derniers. Un nom qui n'est
        pas au catalogue rend une liste vide."""
        with self._verrou:
            if name not in self._lire_catalogue():
                return []
            try:
                lignes = self._chemin_du_carnet(name).read_text(encoding="utf-8").splitlines()
            except (OSError, ValueError):
                return []
        appels: list[dict[str, Any]] = []
        for ligne in lignes:
            try:
                entree = json.loads(ligne)
            except ValueError:
                continue                                    # une ligne coupée par un arrêt brutal
            if isinstance(entree, dict):
                appels.append(entree)
        if self.record_calls:
            appels = appels[-self.record_calls:]
        if failed is not None:
            appels = [a for a in appels if (not a.get("ok")) is bool(failed)]
        return appels[-limit:] if limit else appels

    def descendants(self, name: str) -> list[str]:
        """Les outils construits APRÈS ``name`` dans un run où il était visible (transitif).

        EVOMAL (arXiv 2608.25776) : un agent qui écrit ses outils en imitant sa bibliothèque
        recopie les outils piégés, et supprimer l'original ne supprime pas les copies. Cette
        liste est ce qu'il faut regarder avant de retirer ou de mettre un outil en quarantaine.
        Se fie à la provenance du catalogue ; un outil sans provenance (avant 0.25.0) n'a pas
        de descendants connus. Vide si ``name`` est inconnu."""
        with self._verrou:
            outils = self._lire_catalogue()
        return _descendants(outils, name)

    def _marquer(self, name: str, champ: str, champ_raison: str, raison: str, cascade: bool) -> list[str] | None:
        with self._verrou:
            outils = self._lire_catalogue()
            if name not in outils:
                return None
            cibles = [name] + (_descendants(outils, name) if cascade else [])
            for nom in cibles:
                outils[nom][champ] = True
                outils[nom][champ_raison] = raison if nom == name else f"{raison} (built after {name!r}, which it could see)"
                self._visibles.discard(nom)
                if champ == "quarantined":
                    # Posée par l'hôte : plus aucun run, pas même celui qui l'a construit.
                    self._en_quarantaine.add(nom)
                    self._run_de_construction.pop(nom, None)
            self._ecrire_catalogue(outils)
            return cibles

    def retire(self, name: str, reason: str = "retired by the host", *, cascade: bool = False) -> bool:
        """Retire un outil de la bibliothèque : il ne sera plus rechargé. ``True`` si connu.
        ``cascade=True`` retire aussi ses `descendants` (0.25.0)."""
        return self._marquer(name, "retired", "retired_reason", reason, cascade) is not None

    def quarantine(self, name: str, reason: str = "quarantined by the host", *, cascade: bool = False) -> bool:
        """Met un outil en quarantaine (0.25.0) : il n'est plus rechargé, reste inspectable et
        peut être libéré par `release`. ``cascade=True`` y met aussi ses `descendants`."""
        return self._marquer(name, "quarantined", "quarantine_reason", reason, cascade) is not None

    def reinstate(self, name: str) -> bool:
        """Remet dans la bibliothèque un outil RETIRÉ (par sa série d'erreurs, ou par `retire`) : il sera
        rechargé au prochain `load_library`. ``True`` si l'outil est connu. La bibliothèque retire toute seule ;
        seul l'hôte remet — par exemple quand la fiche (`health`) montre que les échecs étaient des entrées
        invalides, refusées à raison. Ne lève PAS une quarantaine : c'est une décision de sécurité (`release`)."""
        with self._verrou:
            outils = self._lire_catalogue()
            entree = outils.get(name)
            if entree is None:
                return False
            entree["retired"] = False
            entree["consecutive_errors"] = 0
            entree.pop("retired_reason", None)
            self._ecrire_catalogue(outils)
            return True

    # ── Apprendre des erreurs ───────────────────────────────────────────────

    def _cas_du_carnet(self, name: str, version: str) -> _Cas:
        """Les appels de la VERSION COURANTE, le dernier par entrée, triés (`_Cas`). Raté : `_jugement` dit
        défaut. Réussi : un appel `ok` que personne n'a jugé faux, avec son résultat (ou son empreinte). Refusé :
        un refus (de l'outil ou de son schéma) que PERSONNE n'a jugé. Légitime : un échec que l'hôte a jugé
        légitime (`report(correct=True)`) — l'entrée est invalide. Inutilisable : sans ses arguments (trop gros)
        ou passé par `redact` (les arguments gardés ne sont plus ceux que l'outil a reçus)."""
        par_entree: dict[str, dict[str, Any]] = {}
        for appel in self.calls(name):
            if appel.get("version") != version:
                continue
            cle = str(appel.get("args_sha256") or appel.get("id"))
            par_entree.pop(cle, None)                       # le dernier appel sur cette entrée prend sa place
            par_entree[cle] = appel
        cas = _Cas()
        for appel in par_entree.values():
            jugement = _jugement(appel)
            utilisable = "args" in appel and not appel.get("redacted")
            refus = appel.get("kind") in _REFUS_NON_JUGEABLES
            if jugement is True:
                if utilisable:
                    cas.rates.append(appel)
                else:
                    cas.rates_inutilisables += 1
            elif jugement is None and refus:
                if utilisable:
                    cas.refuses.append(appel)
                else:
                    cas.refus_inutilisables += 1
            elif not utilisable or jugement is None:
                continue
            elif appel.get("kind") != "ok":
                cas.legitimes.append(appel)             # un échec, et l'hôte a dit : c'était la bonne réponse
            elif "result" in appel or "result_sha256" in appel:
                cas.reussis.append(appel)
        return cas

    @staticmethod
    def _rejouer(outil: GeneratedPythonTool, args: dict[str, Any]) -> tuple[str, Any]:
        """UN appel rejoué sur une version candidate, comme le ferait le registre : le schéma d'entrée d'abord.
        Rend (`"result"`, la valeur) · (`"rejected"`, le message du schéma) · (`"refused"` | `"error"`, l'erreur)."""
        if outil.spec.input_schema:
            erreurs = validate(args, outil.spec.input_schema)
            if erreurs:
                return "rejected", format_errors(erreurs)
        try:
            return "result", outil(**args)
        except ToolError as exc:
            texte = str(exc)
            return ("refused" if texte.lower().startswith(_REFUS_DECLARE) else "error"), texte

    def _noter_l_essai(self, name: str, statut: str, raison: str, examines: list[str] | None = None) -> None:
        with self._verrou:
            outils = self._lire_catalogue()
            entree = outils.get(name)
            if entree is None:
                return
            entree["improvement_attempts"] = int(entree.get("improvement_attempts") or 0) + 1
            entree["last_improvement"] = {"status": statut, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          "reason": raison[:200]}
            if examines is not None:
                entree["refusals_examined"] = examines
            self._ecrire_catalogue(outils)

    def improve(self, name: str, *, max_cases: int = 5, learn_from_refusals: bool = False) -> ImprovementResult:
        """APPRENDRE des erreurs d'un outil de la bibliothèque : une nouvelle version, gardée seulement si du CODE
        le prouve (`ImprovementResult`, `.summary()`).

        1. Le modèle constructeur reçoit le code actuel et jusqu'à ``max_cases`` VRAIS cas ratés du carnet
           (`record_calls`) — l'entrée, l'erreur ou le résultat faux, ce que l'hôte attendait s'il l'a dit
           (`report`). Jamais les cas qui réussissent ; jamais un cas venu d'un run qui avait lu du contenu non
           fiable (ses arguments pourraient porter un ordre) — celui-là est tout de même rejoué.
        2. La version qu'il rend est jugée dans le bac, PAR DU CODE : chaque cas raté doit être résolu — contre le
           résultat attendu quand l'hôte l'a donné ; sinon l'outil ne plante plus (un résultat, ou un refus : son
           schéma, ou `ValueError("invalid input: …")`) — et chaque cas réussi du carnet, rejoué sans que le
           modèle l'ait vu, doit donner le même résultat qu'avant. Elle ne peut pas demander plus de permissions.
        3. Si tout passe, elle remplace l'ancienne, qui est gardée (`rollback`) ; ses compteurs repartent de zéro,
           un outil retiré est remis, et `verified_by` dit ce qui a été prouvé. Sinon RIEN ne change sur le
           disque, et l'essai est noté au catalogue.

        Ne fait rien, et n'appelle pas le modèle, quand il n'y a rien à corriger, quand aucun cas n'est utilisable,
        quand l'outil appelle des fonctions de l'hôte (les rejouer referait des effets de bord), quand l'outil est
        en quarantaine, ou depuis un run qui a lu du contenu non fiable. Un seul appel au modèle par `improve`
        (plus les reprises de `max_repairs`) ; sa dépense est dans `.usage` et dans `last_build_usage`.
        ``KeyError`` si l'outil est inconnu.

        ``learn_from_refusals=True`` — apprendre aussi des REFUS, sans rien demander à l'hôte. Mesuré au banc :
        l'outil d'un vrai modèle refuse proprement l'écriture qu'il ne connaît pas (« 1:30 ») au lieu de planter,
        et un refus que personne n'a jugé n'est pas un défaut — sans l'hôte, il n'y avait donc rien à corriger.
        Avec l'option, le modèle voit aussi les entrées refusées que personne n'a jugées, et décide : une demande
        valide écrite autrement se traite, une entrée invalide reste refusée. Le code prouve que chaque cas réussi
        donne le même résultat et qu'un refus propre ne devient pas un plantage. Il ne prouve PAS que la nouvelle
        lecture est juste : elle est comptée à part (`.extended`, `.extended_cases` à relire) et dite « non
        vérifiée » dans la fiche — le niveau de preuve d'une première construction, pas plus. Une version qui ne
        lit rien de plus n'est pas installée (`"nothing_gained"`).

        Toujours, option ou pas : un échec que l'hôte a jugé LÉGITIME (`report(correct=True)`) doit rester un
        échec — une version qui se met à répondre à une entrée que l'hôte a dite invalide est refusée."""
        if not isinstance(learn_from_refusals, bool):
            raise TypeError(f"learn_from_refusals must be True or False, got {learn_from_refusals!r}")
        with self._verrou:
            entree = self._lire_catalogue().get(name)
        if entree is None:
            raise KeyError(name)
        ancienne = str(entree.get("sha256") or "")
        rendu = ImprovementResult(name=name, old_version=ancienne[:12])

        def fin(statut: str, raison: str, *, noter: bool = False) -> ImprovementResult:
            rendu.status, rendu.reason = statut, raison
            if noter:
                self._noter_l_essai(name, statut, raison)
            return rendu

        if entree.get("quarantined"):
            return fin("quarantined", "the tool is quarantined; release it first")
        if _provenance_du_run()["tainted"]:
            return fin("tainted_run", "improve() was called from a run that has ingested untrusted content")
        chemin = self.tools_dir / f"{name}.py"
        try:
            octets = chemin.read_bytes()
        except OSError as exc:
            return fin("build_failed", f"the tool file cannot be read: {exc}")
        if hashlib.sha256(octets).hexdigest() != ancienne:
            return fin("build_failed", "the tool file changed on disk since it was validated")
        cas = self._cas_du_carnet(name, ancienne[:12])
        rates, reussis = cas.rates, cas.reussis
        refuses = cas.refuses if learn_from_refusals else []
        # Une entrée refusée que le modèle a DÉJÀ examinée et laissée refusée n'est pas reproposée : la question
        # a eu sa réponse, et la reposer se paierait à chaque appel (mesuré : 4 appels sur 8 payés pour un « ??? »).
        deja = {str(cle) for cle in (entree.get("refusals_examined") or [])}
        nouveaux = [appel for appel in refuses if _cle_du_refus(appel) not in deja]
        rendu.skipped = cas.rates_inutilisables + (cas.refus_inutilisables if learn_from_refusals else 0)
        if not rates and not nouveaux:
            if rendu.skipped:
                return fin("no_usable_case", "failing calls exist, but their arguments were redacted or too large to keep")
            if refuses:
                return fin("nothing_to_fix", "every refused input has already been examined and kept refused")
            return fin("nothing_to_fix", "no failing call of the current version in the call log")
        code = octets.decode("utf-8")
        if "call_host" in code:
            return fin("not_replayable", "the tool calls host functions: replaying its calls would repeat side effects")
        # Un cas venu d'un run teinté n'est JAMAIS montré au modèle (ses arguments pourraient porter un ordre) ;
        # il est tout de même rejoué. Même règle pour un refus.
        montrables = [appel for appel in rates if appel.get("tainted") is not True]
        refus_montrables = [appel for appel in nouveaux if appel.get("tainted") is not True]
        if not montrables and not refus_montrables:
            return fin("no_usable_case", "every failing call comes from a run that had ingested untrusted content")
        permissions = [str(p) for p in (entree.get("permissions") or [])]
        combien = max(1, int(max_cases))
        refus_montres = _choisir_les_cas(refus_montrables, combien)
        demande = _demande_d_amelioration(str(entree.get("description") or name), code,
                                          _choisir_les_cas(montrables, combien), refus_montres)

        with tempfile.TemporaryDirectory(prefix="autoagent_improve_", ignore_cleanup_errors=True) as brouillon:
            # La candidate se construit À CÔTÉ : tant qu'elle n'a rien prouvé, la bibliothèque ne bouge pas.
            # Dans le bac de CE constructeur, avec sa règle de choix : du code écrit par le modèle ne tourne jamais
            # ailleurs que là où tournait l'ancienne version (aucun autre bac n'est créé).
            essai = DynamicToolBuilder(self.provider, tools_dir=brouillon, sandbox=self.sandbox,
                                       allowed_permissions=self.allowed_permissions, max_repairs=self.max_repairs,
                                       self_test_rel_tol=self.self_test_rel_tol, max_build_tokens=self.max_build_tokens)
            essai._bac_impose, essai._frontiere_requise = self._bac_impose, self._frontiere_requise
            try:
                candidat = essai.build(ToolBuildRequest(capability=demande, tool_name=name, permissions=permissions))
            except (ToolValidationError, ToolError) as exc:
                rendu.usage = self.last_build_usage = essai.last_build_usage
                return fin("build_failed", f"{type(exc).__name__}: {exc}"[:300], noter=True)
            rendu.usage = self.last_build_usage = essai.last_build_usage
            en_plus = sorted(set(candidat.spec.permissions or []) - set(permissions))
            if en_plus:
                return fin("rejected", f"the new version asks for more permissions than the old one: {en_plus}", noter=True)

            for appel in rates:
                issue, valeur = self._rejouer(candidat, appel["args"])
                resolu, verifie = _cas_resolu(appel, issue, valeur, self.self_test_rel_tol)
                if resolu:
                    rendu.fixed += 1
                    rendu.checked += int(verifie)
                    continue
                reste: dict[str, Any] = {"args": appel["args"]}
                reste["got" if issue == "result" else "error"] = valeur if issue == "result" else str(valeur)[:300]
                if "expected" in _verdict_de(appel):
                    reste["expected"] = _verdict_de(appel)["expected"]
                rendu.still_failing.append(reste)
            for appel in reussis:
                issue, valeur = self._rejouer(candidat, appel["args"])
                identique = issue == "result" and (
                    _egal(appel["result"], valeur, self.self_test_rel_tol) if "result" in appel
                    else appel.get("result_sha256") == _empreinte(valeur))
                if identique:
                    rendu.kept += 1
                    continue
                obtenu = valeur if issue == "result" else (
                    f"refused by the input schema: {valeur}" if issue == "rejected" else f"error: {str(valeur)[:300]}")
                rendu.regressions.append({"args": appel["args"], "got": obtenu,
                                          "expected": appel["result"] if "result" in appel
                                          else {"sha256": appel.get("result_sha256")}})
            for appel in cas.legitimes:
                # L'hôte a dit : cette entrée est invalide, l'échec était la bonne réponse. Elle doit le rester.
                issue, valeur = self._rejouer(candidat, appel["args"])
                if issue == "result":
                    rendu.broken_refusals.append({
                        "args": appel["args"], "got": valeur,
                        "why": "the host judged this failure legitimate: the input is invalid"})
            encore_refuses: set[str] = set()
            for appel in refuses:
                # Un refus que personne n'a jugé : la nouvelle version le LIT (compté à part, non vérifié), le
                # refuse toujours, ou — ce qui n'est pas permis — plante là où l'ancienne refusait proprement.
                issue, valeur = self._rejouer(candidat, appel["args"])
                if issue == "result":
                    rendu.extended += 1
                    if len(rendu.extended_cases) < 20:
                        rendu.extended_cases.append({"args": appel["args"], "got": valeur})
                elif issue in _REFUS_NON_JUGEABLES:
                    rendu.still_refused += 1
                    encore_refuses.add(_cle_du_refus(appel))
                else:
                    rendu.broken_refusals.append({
                        "args": appel["args"], "error": str(valeur)[:300],
                        "why": "the old version refused this input cleanly; the new one crashes on it"})
            if rendu.still_failing or rendu.regressions or rendu.broken_refusals:
                return fin("rejected", f"{len(rendu.still_failing)} failing case(s) not fixed, {len(rendu.regressions)} "
                                       "passing case(s) changed"
                                       + (f", {len(rendu.broken_refusals)} refusal(s) broken" if rendu.broken_refusals else ""),
                           noter=True)
            # Ce que le modèle a VU et laissé refusé : examiné, donc plus reproposé (borné).
            examines = sorted(deja | ({_cle_du_refus(appel) for appel in refus_montres} & encore_refuses) - {""})
            examines = examines[-_MAX_REFUS_EXAMINES:]
            if not rendu.fixed and not rendu.extended:
                raison = "the new version fixes no failing case and still refuses every refused input"
                self._noter_l_essai(name, "nothing_gained", raison, examines)
                return fin("nothing_gained", raison)
            verified = {"kind": "replay", "fixed": rendu.fixed, "checked": rendu.checked, "kept": rendu.kept,
                        "from": ancienne[:12]}
            if rendu.extended:
                verified["extended"] = rendu.extended      # lues, PAS vérifiées : dit dans la fiche
            rendu.tool = self._installer(name, entree, octets, candidat, verified, examines)
        rendu.new_version = hashlib.sha256(chemin.read_bytes()).hexdigest()[:12]
        return fin("improved", "every failing case is fixed and every passing case still gives the same result"
                               + (f"; {rendu.extended} refused input(s) are now accepted, unverified" if rendu.extended else ""))

    def _installer(self, name: str, ancienne_entree: dict[str, Any], anciens_octets: bytes,
                   candidat: GeneratedPythonTool, verified: dict[str, Any],
                   examines: list[str] | None = None) -> GeneratedPythonTool:
        """La candidate a prouvé ce qu'il fallait : l'ancienne version est ARCHIVÉE (`.versions/<nom>/<sha>.py`),
        la nouvelle prend sa place, et le catalogue repart pour elle — avec son histoire, et avec les entrées
        refusées déjà examinées (`examines`) : la version suivante ne les repropose pas non plus."""
        maintenant = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ancienne = str(ancienne_entree.get("sha256") or "")
        nouveau_texte = candidat.file_path.read_bytes().decode("utf-8")
        chemin = self.tools_dir / f"{name}.py"
        with self._verrou:
            archive = self.tools_dir / ".versions" / name / f"{ancienne[:12]}.py"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(anciens_octets)
            atomic_write_text(chemin, nouveau_texte)
            outils = self._lire_catalogue()
            suite = max([int((e.get("provenance") or {}).get("seq") or 0) for e in outils.values()] + [0]) + 1
            permissions = list(candidat.spec.permissions or [])
            outils[name] = {
                "sha256": hashlib.sha256(chemin.read_bytes()).hexdigest(),
                "created": maintenant,
                "description": candidat.spec.description,
                "permissions": permissions,
                "calls": 0, "errors": 0, "consecutive_errors": 0, "unattributed_failures": 0,
                "rejected_calls": 0, "refused_calls": 0, "retired": False,
                "verified_by": dict(verified),
                "provenance": {"tainted": False, "context": "improve", "seq": suite,
                               "visible_tools": sorted(self._visibles - {name}), "improved_from": ancienne},
                "history": [*list(ancienne_entree.get("history") or []),
                            {"sha256": ancienne, "created": ancienne_entree.get("created"),
                             "verified_by": ancienne_entree.get("verified_by") or {"kind": "none"},
                             "replaced": maintenant}],
                "improvement_attempts": int(ancienne_entree.get("improvement_attempts") or 0) + 1,
                "last_improvement": {"status": "improved", "ts": maintenant, "reason": ""},
                **({"refusals_examined": list(examines)} if examines else {}),
            }
            self._ecrire_catalogue(outils)
            self._visibles.add(name)
        outil = load_generated_tool(chemin, sandbox=self._bac_pour(permissions))
        outil.recorder = self._noter_appel
        outil.admission = self._admission
        outil.host_functions = self.host_functions
        outil.verified_by = dict(verified)
        return outil

    def rollback(self, name: str) -> bool:
        """Revient à la version d'AVANT la dernière amélioration (`improve`). ``True`` si c'est fait.

        L'archive doit avoir l'empreinte notée au catalogue : une archive modifiée n'est jamais remise. Les
        compteurs repartent de zéro ; la version quittée reste dans `.versions/<nom>/`, pour inspection."""
        with self._verrou:
            outils = self._lire_catalogue()
            entree = outils.get(name)
            histoire = list((entree or {}).get("history") or [])
            if entree is None or not histoire:
                return False
            passee = histoire[-1]
            empreinte = str(passee.get("sha256") or "")
            dossier = self.tools_dir / ".versions" / name
            chemin = self.tools_dir / f"{name}.py"
            try:
                octets = (dossier / f"{empreinte[:12]}.py").read_bytes()
                if hashlib.sha256(octets).hexdigest() != empreinte:
                    _log.warning("archived version of %r does not match its recorded sha256; NOT rolled back", name)
                    return False
                (dossier / f"{str(entree.get('sha256') or '')[:12]}.py").write_bytes(chemin.read_bytes())
                atomic_write_text(chemin, octets.decode("utf-8"))
            except (OSError, ValueError):
                return False
            entree.update({
                "sha256": empreinte, "created": passee.get("created") or entree.get("created"),
                "verified_by": passee.get("verified_by") or {"kind": "none"},
                "calls": 0, "errors": 0, "consecutive_errors": 0, "unattributed_failures": 0,
                "rejected_calls": 0, "refused_calls": 0, "retired": False, "history": histoire[:-1],
                "last_improvement": {"status": "rolled_back", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "reason": ""},
            })
            self._ecrire_catalogue(outils)
            return True

    def _noter_verification(self, outil: GeneratedPythonTool, verified_by: dict[str, Any]) -> None:
        """L'origine des tests d'un outil, quand elle est connue APRÈS sa construction (`synthesize_tool`)."""
        outil.verified_by = dict(verified_by)
        if not self.persist:
            return
        with self._verrou:
            outils = self._lire_catalogue()
            entree = outils.get(outil.spec.name)
            if entree is not None:
                entree["verified_by"] = dict(verified_by)
                self._ecrire_catalogue(outils)

    def report(self, name: str, call_id: str, *, correct: bool, reason: str | None = None,
               expected: Any = _ABSENT, expected_refusal: bool = False) -> bool:
        """Le VERDICT DE L'HÔTE sur un appel du carnet : l'outil a-t-il bien fait ?

        C'est l'hôte qui juge, pas le modèle. ``correct=False`` : un résultat faux qui n'avait pas planté, un
        échec sur une entrée valide, un refus du schéma à tort — ``expected`` peut donner ce qu'il fallait
        rendre. ``correct=True`` : le résultat est juste, ou l'échec était LÉGITIME (une entrée invalide, refusée
        à raison). Le verdict est écrit sur l'appel (``verdict``), prime sur l'issue dans la fiche (`health`),
        et un nouveau verdict remplace l'ancien. Il ne change ni les compteurs ni le statut de l'outil : voir
        `reinstate` et `retire`. ``reason`` et ``expected`` passent par ``redact`` et sont bornés comme le reste
        du carnet. ``False`` si l'outil ou l'appel est inconnu (déjà sorti du carnet).

        ``expected_refusal=True`` (avec ``correct=False``) : l'outil a RÉPONDU à une entrée invalide — il fallait
        la refuser. Dire seulement « résultat faux » réclamerait un autre résultat ; ici `improve` exige que la
        nouvelle version refuse l'entrée (son schéma, ou `ValueError("invalid input: …")`), et c'est vérifié."""
        if not isinstance(correct, bool):
            raise TypeError(f"correct must be True or False, got {correct!r}")
        if not isinstance(expected_refusal, bool):
            raise TypeError(f"expected_refusal must be True or False, got {expected_refusal!r}")
        if expected_refusal and (correct or expected is not _ABSENT):
            raise ValueError("expected_refusal=True says the tool should have REFUSED the input: it goes with "
                             "correct=False and without expected=")
        verdict: dict[str, Any] = {"correct": correct, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if expected_refusal:
            verdict["expected_refusal"] = True
        try:
            details: dict[str, Any] = {}
            if reason is not None:
                details["reason"] = str(reason if self.redact is None else self.redact(str(reason)))[:300]
            if expected is not _ABSENT:
                texte = _json_canonique(expected if self.redact is None else self.redact(expected))
                details["expected_sha256"] = hashlib.sha256(texte.encode("utf-8")).hexdigest()
                if len(texte) <= self.max_record_chars:
                    details["expected"] = json.loads(texte)
            verdict.update(details)
        except Exception as exc:                       # un `redact` qui plante : le verdict seul, aucune donnée
            _log.warning("redact raised %s; the verdict is recorded without its data", type(exc).__name__)
        with self._verrou:
            if name not in self._lire_catalogue():
                return False
            chemin = self._chemin_du_carnet(name)
            try:
                lignes = chemin.read_text(encoding="utf-8").splitlines()
            except (OSError, ValueError):
                return False
            for index, ligne in enumerate(lignes):
                try:
                    entree = json.loads(ligne)
                except ValueError:
                    continue
                if isinstance(entree, dict) and entree.get("id") == call_id:
                    entree["verdict"] = verdict
                    lignes[index] = json.dumps(entree, ensure_ascii=False)
                    try:
                        atomic_write_text(chemin, "\n".join(lignes) + "\n")
                    except OSError:
                        return False
                    self._lignes_du_carnet[name] = len(lignes)
                    return True
            return False

    def health(self, name: str) -> ToolHealth:
        """La FICHE DE QUALITÉ d'un outil de la bibliothèque (`ToolHealth`, `.summary()`, `.to_dict()`).

        Avec un carnet (`record_calls`), elle est calculée sur ses derniers appels : issues, taux de défaut JUGÉ
        — le verdict de l'hôte (`report`) prime ; sinon `exception` et `timeout` — avec son intervalle de Wilson,
        durée médiane, erreurs regroupées, chiffres par version. Sans carnet, sur les compteurs du catalogue.
        Elle dit aussi d'où viennent les tests de l'outil (`verified_by`). ``KeyError`` si l'outil est inconnu."""
        from .compare import wilson_interval  # import paresseux : `compare` importe l'agent

        with self._verrou:
            entree = self._lire_catalogue().get(name)
        if entree is None:
            raise KeyError(name)
        appels = self.calls(name)
        courante = str(entree.get("sha256") or "")[:12]
        verified = entree.get("verified_by")
        executions = int(entree.get("calls") or 0)
        non_attribues = int(entree.get("unattributed_failures") or 0)
        refuses = int(entree.get("refused_calls") or 0)
        outcomes: dict[str, int] = {}
        verdicts = {"correct": 0, "wrong": 0}
        versions: dict[str, dict[str, int]] = {}
        vues: dict[str, dict[str, Any]] = {}
        durees: list[float] = []
        anciens = 0
        if appels:
            juges = defauts = 0
            for appel in appels:
                version = str(appel.get("version") or "")
                par_version = versions.setdefault(version, {"judged": 0, "defects": 0})
                jugement = _jugement(appel)
                if jugement is not None:
                    par_version["judged"] += 1
                    par_version["defects"] += int(jugement)
                if version and version != courante:
                    # Un appel d'une version PRÉCÉDENTE (remplacée par `improve`, ou reconstruite) : il reste lisible
                    # dans `versions`, mais la fiche parle du code qui tourne — pas des fautes qu'on vient de corriger.
                    anciens += 1
                    continue
                issue = str(appel.get("kind"))
                outcomes[issue] = outcomes.get(issue, 0) + 1
                verdict = appel.get("verdict")
                if isinstance(verdict, dict) and isinstance(verdict.get("correct"), bool):
                    verdicts["correct" if verdict["correct"] else "wrong"] += 1
                if issue in ("ok", *_ISSUES_DE_L_OUTIL) and isinstance(appel.get("seconds"), (int, float)):
                    durees.append(float(appel["seconds"]))
                if jugement is None:
                    continue
                juges += 1
                if not jugement:
                    continue
                defauts += 1
                raison = verdict.get("reason") if isinstance(verdict, dict) else None
                # Quand l'hôte a dit pourquoi, c'est SA raison qu'on montre : elle dit ce qui était attendu.
                texte = str(raison or appel.get("error") or "wrong result (reported by the host)")
                vue = vues.setdefault(_signature(texte), {"count": 0})
                vue.update(error=texte, count=vue["count"] + 1, example_args=appel.get("args"), last_id=appel.get("id"))
        else:
            juges = max(0, executions - non_attribues - refuses)
            defauts = min(juges, int(entree.get("errors") or 0))
            if defauts and entree.get("last_error"):
                vues["catalogue"] = {"error": str(entree["last_error"]), "count": defauts, "example_args": None,
                                     "last_id": None}
        return ToolHealth(
            name=name,
            status="quarantined" if entree.get("quarantined") else "retired" if entree.get("retired") else "active",
            version=str(entree.get("sha256") or "")[:12],
            created=str(entree.get("created") or ""),
            verified_by=dict(verified) if isinstance(verified, dict) else {"kind": "unknown"},
            calls=executions,
            errors=int(entree.get("errors") or 0),
            unattributed_failures=non_attribues,
            rejected_calls=int(entree.get("rejected_calls") or 0),
            refused_calls=refuses,
            basis="call log" if appels else "catalogue",
            window=len(appels) - anciens,
            previous_calls=anciens,
            outcomes=outcomes,
            judged=juges,
            defects=defauts,
            defect_rate=(defauts / juges) if juges else None,
            defect_interval=wilson_interval(defauts, juges) if juges else None,
            median_seconds=round(statistics.median(durees), 3) if durees else None,
            errors_seen=sorted(vues.values(), key=lambda vue: -int(vue["count"])),
            verdicts=verdicts,
            versions=versions,
        )

    def health_all(self) -> dict[str, ToolHealth]:
        """La fiche de chaque outil du catalogue, par nom."""
        with self._verrou:
            noms = sorted(self._lire_catalogue())
        return {nom: self.health(nom) for nom in noms}

    def quarantined(self) -> list[str]:
        """Noms des outils en quarantaine, à faire relire par un humain."""
        with self._verrou:
            return sorted(n for n, e in self._lire_catalogue().items() if e.get("quarantined"))

    def release(self, name: str, reviewed_sha256: str) -> bool:
        """Validation HUMAINE d'un outil en quarantaine. Il faut donner l'empreinte sha256 du
        fichier TEL QUE RELU : si le fichier a changé depuis la construction, ou si l'empreinte
        ne correspond pas, rien n'est libéré (fail-closed). ``True`` si libéré ; l'outil sera
        rechargé au prochain `load_library`."""
        with self._verrou:
            outils = self._lire_catalogue()
            entree = outils.get(name)
            if entree is None or not entree.get("quarantined"):
                return False
            try:
                actuel = hashlib.sha256((self.tools_dir / f"{name}.py").read_bytes()).hexdigest()
            except OSError:
                return False
            if not reviewed_sha256 or actuel != reviewed_sha256 or actuel != entree.get("sha256"):
                return False
            entree["quarantined"] = False
            entree["released"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            entree.pop("quarantine_reason", None)
            self._ecrire_catalogue(outils)
            self._en_quarantaine.discard(name)
            self._run_de_construction.pop(name, None)
            return True

    def _admission(self, nom: str) -> str | None:
        """Contrôle avant CHAQUE appel d'un outil de la bibliothèque (nuit du 11 nov. 2026).

        Un outil en quarantaine restait dans le registre de l'agent qui l'avait construit, et
        appelable à ses runs suivants. Il ne sert plus qu'au run qui l'a construit (le modèle
        l'a fait écrire pour sa tâche en cours) ; partout ailleurs — run suivant, autre agent,
        appel de l'hôte hors run, quarantaine posée par l'hôte — il est refusé jusqu'à `release`.
        """
        with self._verrou:
            entree = self._lire_catalogue().get(nom)
            en_quarantaine = nom in self._en_quarantaine or bool(entree and entree.get("quarantined"))
            if not en_quarantaine:
                return None
            ref = self._run_de_construction.get(nom)
        run = _run_en_cours()
        if ref is not None and run is not None and ref() is run:
            return None
        return (f"tool {nom!r} is in quarantine (built in a run that had ingested untrusted content, "
                f"or quarantined by the host); it is not callable until a human releases it "
                f"(DynamicToolBuilder.release)")

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
            if entree.get("quarantined"):
                _log.warning("library tool %r is quarantined (%s); NOT loaded", nom, entree.get("quarantine_reason", ""))
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
                outil.sandbox = self._bac_pour(outil.spec.permissions)
            except (ToolError, OSError, SyntaxError, ValueError) as exc:     # ToolError : pas de frontière (0.28.0)
                _log.warning("library tool %r not loaded: %s", nom, exc)
                continue
            if outil.spec.name != nom:
                _log.warning("library tool %r declares another name (%r); skipped", nom, outil.spec.name)
                continue
            outil.recorder = self._noter_appel
            outil.verified_by = entree.get("verified_by") if isinstance(entree.get("verified_by"), dict) else None
            outil.admission = self._admission
            outil.host_functions = self.host_functions
            self._visibles.add(nom)
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
            "If an input is invalid for the tool, raise ValueError(\"invalid input: <why>\") — never return an "
            "error value in its place. "
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
