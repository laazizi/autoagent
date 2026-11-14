"""Le banc « outils qui cassent » (main, non publié) — ce que vaut `improve`, mesuré au lieu d'être affirmé.

Un outil créé par le modèle marche sur ses premières entrées ; puis d'autres écritures arrivent, et il échoue.
Que faire ? Le banc compare cinq réactions, À PARTIR DU MÊME premier outil :

* ``none``             — rien : l'outil reste comme il est ;
* ``rebuild``          — le refaire à l'aveugle, depuis la même demande (un retrait suivi d'une recréation) ;
* ``improve``          — `DynamicToolBuilder.improve`, quand l'hôte a dit ce qui était attendu (`report`) ;
* ``improve_unjudged`` — `improve`, quand l'hôte n'a RIEN dit : seuls les plantages sont connus ;
* ``improve_refusals`` — `improve(learn_from_refusals=True)`, l'hôte n'ayant rien dit : le modèle voit aussi ce
  que l'outil a REFUSÉ, et décide ce qui était une demande valide écrite autrement.

Le juge est du CODE : chaque entrée d'une tâche porte son résultat attendu. Après la réaction, l'outil est essayé
sur des entrées que rien n'a jamais vues (`ToolTask.later`) — ni le modèle, ni le carnet. Une UNITÉ (une tâche, un
essai) est résolue quand toutes passent. Une entrée que le premier outil lisait et que la réaction a perdue est
comptée à part (`regressions`). Chaque tâche porte aussi des entrées INVALIDES (`"???"`, puis `"abc"`, `"--"`
jamais vues) : un outil qui se met à leur répondre est compté (`accepted_invalid`) — c'est le risque d'une
réaction qui laisse le modèle décider de ce qui est valide.

Deux niveaux, comme les autres bancs. Niveau 1 : `ScriptedToolProvider`, un « constructeur » scripté — il mesure le
CODE du banc et de `improve`, hors ligne, sans jeton ; `positive_control_tools()` doit y voir `improve` résoudre ce
que `none` ne résout pas, ou le banc est aveugle. Niveau 2 : un vrai fournisseur.

Ce que le banc ne dit pas. Huit tâches de conversion de texte, à un paramètre : pas les outils en général. Les
essais d'un même modèle à température 0 se ressemblent : c'est le nombre de TÂCHES qui porte la preuve, pas le
nombre d'essais. « Résolu » est strict (toutes les entrées suivantes) ; la part d'entrées justes est donnée à côté.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .compare import _decider
from .dynamic import DynamicToolBuilder, ToolBuildRequest
from .errors import ToolError, ToolValidationError
from .providers.base import LLMProvider
from .sandbox import DockerSandbox, GeneratedPythonTool, SubprocessSandbox
from .schema import LLMRequest, LLMResponse, ModelConfig, TokenUsage

__all__ = [
    "ARMS",
    "TOOL_TASKS",
    "ScriptedToolProvider",
    "ToolBenchReport",
    "ToolBenchRow",
    "ToolTask",
    "positive_control_tools",
    "run_tool_bench",
    "same_result",
]

ARMS = ("none", "rebuild", "improve", "improve_unjudged", "improve_refusals")
_NOMS = {
    "none": "rien",
    "rebuild": "refait à l'aveugle",
    "improve": "improve (l'hôte dit l'attendu)",
    "improve_unjudged": "improve (l'hôte ne dit rien)",
    "improve_refusals": "improve (rien dit, les refus servent)",
}
Case = tuple[str, Any]      # (l'entrée, le résultat attendu)


@dataclass(frozen=True)
class ToolTask:
    """Une tâche du banc : une demande décrite par son cas facile, et les entrées qui arrivent ensuite.

    Attributes:
        name: le nom de l'outil à créer.
        capability: ce que l'hôte demande — décrit par son cas FACILE, comme on le fait naturellement.
        easy: les premières entrées, celles que la demande décrit — `(entrée, attendu)`.
        hard: d'autres écritures, VALIDES, qui arrivent ensuite et que le premier outil ne lit pas forcément.
        later: ce qui arrive APRÈS la réaction — jamais montré au modèle, jamais écrit au carnet.
        invalid: des entrées qui ne sont PAS des demandes valides, essayées avec les premières (elles entrent
            au carnet : l'outil doit les refuser, et continuer de les refuser).
        later_invalid: d'autres entrées invalides, jamais vues : l'outil final ne doit PAS leur répondre.
        naive_code, full_code: pour le niveau 1 seulement — la première version (elle ne lit que le cas
            facile) et la version qui lit tout ; `ScriptedToolProvider` les rend.
        param: le nom de l'unique paramètre (une chaîne).
    """

    name: str
    capability: str
    easy: tuple[Case, ...]
    hard: tuple[Case, ...]
    later: tuple[Case, ...]
    naive_code: str = ""
    full_code: str = ""
    param: str = "texte"
    invalid: tuple[str, ...] = ()
    later_invalid: tuple[str, ...] = ()


def same_result(expected: Any, got: Any, *, rel_tol: float = 1e-9) -> bool:
    """LE JUGE : ce que l'outil a rendu est-il le résultat attendu ?

    Un objet à UNE clé vaut sa valeur (`{"minutes": 90}` est jugé comme `90`) : la forme de l'emballage n'est pas
    ce que le banc mesure. Les nombres sont comparés à `rel_tol` près ; un booléen n'est pas un nombre ; une
    chaîne `"90"` n'est pas le nombre 90."""
    if isinstance(got, dict) and len(got) == 1:
        got = next(iter(got.values()))
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected is got
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(expected, got, rel_tol=rel_tol, abs_tol=1e-9)
    return bool(expected == got)


# ── Les tâches fournies ──────────────────────────────────────────────────────
# Chaque entrée n'a qu'une lecture défendable. Les versions naïve et complète sont vérifiées par les tests :
# la naïve lit toutes les entrées faciles et AUCUNE des dures ; la complète les lit toutes. Les naïves cassent de
# trois façons : la plupart PLANTENT, deux rendent une valeur FAUSSE (montant, minutes_depuis_minuit), et deux
# REFUSENT proprement ce qu'elles ne lisent pas (metres, celsius) — ce que fait le premier outil d'un vrai modèle.

_MINUTES_NAIF = r"""
import re

def run(args, context):
    m = re.fullmatch(r'(\d+)h(\d*)', args['texte'])
    return int(m.group(1)) * 60 + int(m.group(2) or 0)
"""
_MINUTES = r"""
import re

def run(args, context):
    t = args['texte'].strip().lower()
    m = re.fullmatch(r'(\d+)\s*[h:]\s*(\d*)', t)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2) or 0)
    m = re.fullmatch(r'(\d+)\s*min', t)
    if m:
        return int(m.group(1))
    raise ValueError('invalid input: duree illisible')
"""

_MONTANT_NAIF = r"""
def run(args, context):
    return float(args['texte'].replace(',', ''))
"""
_MONTANT = r"""
def run(args, context):
    t = args['texte'].replace('€', '').replace(' ', '')
    if ',' in t:
        t = t.replace('.', '').replace(',', '.')
    try:
        return float(t)
    except ValueError:
        raise ValueError('invalid input: montant illisible')
"""

_POURCENT_NAIF = r"""
def run(args, context):
    t = args['texte']
    if not t.endswith('%'):
        raise ValueError('invalid input: pas un pourcentage')
    return int(t[:-1]) / 100
"""
_POURCENT = r"""
def run(args, context):
    t = args['texte'].lower().replace('pour cent', '').replace('%', '').replace(',', '.').strip()
    try:
        return float(t) / 100
    except ValueError:
        raise ValueError('invalid input: pourcentage illisible')
"""

_OCTETS_NAIF = r"""
def run(args, context):
    t = args['texte']
    return int(t[:-2]) * {'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3}[t[-2:]]
"""
_OCTETS = r"""
def run(args, context):
    t = args['texte'].strip().upper().replace(' ', '').replace(',', '.')
    unites = {'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3}
    if len(t) < 3 or t[-1] not in 'BO' or t[-2] not in unites:
        raise ValueError('invalid input: taille illisible')
    try:
        return int(float(t[:-2]) * unites[t[-2]])
    except ValueError:
        raise ValueError('invalid input: taille illisible')
"""

_HEURE_NAIF = r"""
def run(args, context):
    h, m = args['texte'].split(':')
    return int(h) * 60 + int(m[:2])
"""
_HEURE = r"""
import re

def run(args, context):
    m = re.fullmatch(r'(\d{1,2})\s*[:h]\s*(\d{2})?\s*(am|pm)?', args['texte'].strip().lower())
    if not m:
        raise ValueError('invalid input: heure illisible')
    h, minutes = int(m.group(1)), int(m.group(2) or 0)
    if m.group(3):
        h = h % 12 + (12 if m.group(3) == 'pm' else 0)
    if h > 23 or minutes > 59:
        raise ValueError('invalid input: heure impossible')
    return h * 60 + minutes
"""

_METRES_NAIF = r"""
def run(args, context):
    t = args['texte']
    if not t.endswith('km') or not t[:-2].isdigit():
        raise ValueError('invalid input: kilometres attendus, comme 5km')
    return int(t[:-2]) * 1000
"""
_METRES = r"""
def run(args, context):
    t = args['texte'].strip().lower().replace(' ', '').replace(',', '.')
    try:
        if t.endswith('km'):
            return float(t[:-2]) * 1000
        if t.endswith('m'):
            return float(t[:-1])
    except ValueError:
        pass
    raise ValueError('invalid input: distance illisible')
"""

_MOIS_NAIF = r"""
def run(args, context):
    return int(args['texte'].split('-')[1])
"""
_MOIS = r"""
MOIS = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre',
        'novembre', 'décembre']

def run(args, context):
    t = args['texte'].strip().lower()
    try:
        if '-' in t:
            return int(t.split('-')[1])
        if '/' in t:
            return int(t.split('/')[1])
    except (ValueError, IndexError):
        raise ValueError('invalid input: date illisible')
    for numero, nom in enumerate(MOIS, 1):
        if nom in t.split():
            return numero
    raise ValueError('invalid input: date illisible')
"""

_CELSIUS_NAIF = r"""
def run(args, context):
    t = args['texte']
    try:
        if t.endswith('C'):
            return float(t[:-1])
    except ValueError:
        pass
    raise ValueError('invalid input: degres Celsius attendus, comme 20C')
"""
_CELSIUS = r"""
def run(args, context):
    t = args['texte'].strip().upper().replace('°', '').replace(' ', '').replace(',', '.')
    if not t or t[-1] not in 'CFK':
        raise ValueError('invalid input: unite inconnue')
    try:
        v = float(t[:-1])
    except ValueError:
        raise ValueError('invalid input: temperature illisible')
    return v if t[-1] == 'C' else (v - 32) * 5 / 9 if t[-1] == 'F' else v - 273.15
"""

_TACHES: tuple[ToolTask, ...] = (
    ToolTask(
        "minutes_de", "Convertit une durée écrite comme 1h30 en nombre de minutes. Rend le nombre.",
        easy=(("1h30", 90), ("2h", 120), ("0h45", 45)),
        hard=(("90 min", 90), ("1 h 30", 90), ("1:30", 90)),
        later=(("3h15", 195), ("4h", 240), ("45 min", 45), ("2 h 05", 125), ("0:20", 20)),
        naive_code=_MINUTES_NAIF, full_code=_MINUTES),
    ToolTask(
        "montant", "Convertit un montant écrit comme 1234.50 en nombre. Rend le nombre.",
        easy=(("1234.50", 1234.5), ("12", 12), ("0.99", 0.99)),
        hard=(("1 234,50", 1234.5), ("12,5", 12.5), ("1.234,50 €", 1234.5)),
        later=(("99.90", 99.9), ("7", 7), ("2 000,00", 2000), ("3,75 €", 3.75), ("10 000", 10000)),
        naive_code=_MONTANT_NAIF, full_code=_MONTANT),
    ToolTask(
        "pourcentage", "Convertit un pourcentage écrit comme 12% en fraction (12% donne 0.12). Rend le nombre.",
        easy=(("12%", 0.12), ("50%", 0.5), ("100%", 1.0)),
        hard=(("12,5 %", 0.125), ("0.5 pour cent", 0.005), ("7,5%", 0.075)),
        later=(("25%", 0.25), ("3%", 0.03), ("2,5 %", 0.025), ("80 pour cent", 0.8), ("0,5%", 0.005)),
        naive_code=_POURCENT_NAIF, full_code=_POURCENT),
    ToolTask(
        "octets", "Convertit une taille écrite comme 10MB en octets (1 KB = 1024 octets). Rend le nombre entier.",
        easy=(("10MB", 10485760), ("1KB", 1024), ("2GB", 2147483648)),
        hard=(("1.5 GB", 1610612736), ("512 kb", 524288), ("10 Mo", 10485760)),
        later=(("4MB", 4194304), ("3KB", 3072), ("0.5 MB", 524288), ("64 ko", 65536), ("2 Go", 2147483648)),
        naive_code=_OCTETS_NAIF, full_code=_OCTETS),
    ToolTask(
        "minutes_depuis_minuit",
        "Convertit une heure écrite comme 14:30 en minutes depuis minuit. Rend le nombre entier.",
        easy=(("14:30", 870), ("08:05", 485), ("00:00", 0)),
        hard=(("14h30", 870), ("2:30 PM", 870), ("9h", 540)),
        later=(("23:59", 1439), ("07:15", 435), ("7h45", 465), ("11:15 AM", 675), ("12:30 AM", 30)),
        naive_code=_HEURE_NAIF, full_code=_HEURE),
    ToolTask(
        "metres", "Convertit une distance écrite comme 5km en mètres. Rend le nombre.",
        easy=(("5km", 5000), ("12km", 12000), ("1km", 1000)),
        hard=(("2,5 km", 2500), ("800 m", 800), ("3 KM", 3000)),
        later=(("7km", 7000), ("20km", 20000), ("1,2 km", 1200), ("350 m", 350), ("0.5 Km", 500)),
        naive_code=_METRES_NAIF, full_code=_METRES),
    ToolTask(
        "mois_de", "Rend le numéro du mois (1 à 12) d'une date écrite comme 2026-03-15.",
        easy=(("2026-03-15", 3), ("2025-12-01", 12), ("2026-07-31", 7)),
        hard=(("15/03/2026", 3), ("15 mars 2026", 3), ("mars 2026", 3)),
        later=(("2026-11-02", 11), ("2025-01-31", 1), ("25/12/2026", 12), ("1er août 2026", 8), ("février 2027", 2)),
        naive_code=_MOIS_NAIF, full_code=_MOIS),
    ToolTask(
        "celsius", "Convertit une température écrite comme 20C en degrés Celsius. Rend le nombre.",
        easy=(("20C", 20), ("-5C", -5), ("36.6C", 36.6)),
        hard=(("68F", 20), ("20 °C", 20), ("273.15K", 0)),
        later=(("35C", 35), ("15C", 15), ("32 °F", 0), ("100 °C", 100), ("212F", 100)),
        naive_code=_CELSIUS_NAIF, full_code=_CELSIUS),
)
# Les mêmes entrées invalides pour toutes les tâches : du texte qui n'est une demande pour aucune d'elles.
TOOL_TASKS: tuple[ToolTask, ...] = tuple(
    dataclasses.replace(tache, invalid=("???",), later_invalid=("abc", "--")) for tache in _TACHES)


# ── Le niveau 1 : un constructeur scripté ────────────────────────────────────

class ScriptedToolProvider(LLMProvider):
    """Un « modèle constructeur » scripté — le niveau 1 du banc. Il ne mesure jamais un modèle : il mesure le
    CODE (le banc, `improve`, le juge par rejeu), hors ligne et sans jeton.

    À une demande ordinaire, il rend la version NAÏVE de la tâche (`ToolTask.naive_code`). À une demande
    d'amélioration (`improve`), selon ``fix`` : ``"full"`` la version complète ; ``"regress"`` la version complète
    qui casse la première entrée facile (une mauvaise correction : le rejeu doit la refuser) ; ``"same"`` la
    version naïve, inchangée ; ``"guess"`` la version complète qui, au lieu de refuser ce qu'elle ne lit pas, rend 0
    (ce que le rejeu ne PEUT pas voir : c'est le banc qui le compte, `accepted_invalid`). Sans état : sa réponse ne dépend que de la demande. Il déclare 150 jetons par appel,
    pour que le compte des jetons se vérifie."""

    def __init__(self, tasks: Sequence[ToolTask] = TOOL_TASKS, *, fix: str = "full") -> None:
        if fix not in ("full", "regress", "same", "guess"):
            raise ValueError(f"fix must be 'full', 'regress', 'same' or 'guess', got {fix!r}")
        super().__init__(ModelConfig(provider="scripted", model="scripted-1"))
        self.tasks = {task.name: task for task in tasks}
        self.fix = fix
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        demande = json.loads(next(m.content or "" for m in request.messages if m.role == "user"))
        tache = self.tasks[str(demande["preferred_tool_name"])]
        amelioration = str(demande["capability"]).startswith("IMPROVE AN EXISTING TOOL")
        if not amelioration or self.fix == "same":
            code = tache.naive_code
        elif self.fix == "full":
            code = tache.full_code
        elif self.fix == "guess":
            code = (tache.full_code.replace("def run(", "def _complet(")
                    + "\ndef run(args, context):\n    try:\n        return _complet(args, context)\n"
                      "    except ValueError:\n        return 0\n")
        else:
            code = (tache.full_code.replace("def run(", "def _complet(")
                    + f"\ndef run(args, context):\n    if args[{tache.param!r}] == {tache.easy[0][0]!r}:\n"
                      "        return -1\n    return _complet(args, context)\n")
        contenu = json.dumps({
            "tool": {"name": tache.name, "description": tache.capability, "permissions": [],
                     "input_schema": _schema(tache)},
            "code": code, "self_tests": [],
        }, ensure_ascii=False)
        return LLMResponse(content=contenu, model="scripted-1",
                           usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150))


# ── Le banc ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolBenchRow:
    """Une réaction, sur une unité (une tâche, un essai).

    Attributes:
        task, trial, arm: la tâche, le numéro de l'essai (à partir de 1), la réaction.
        status: `"kept"` (none) · `"rebuilt"` / `"build_failed"` (rebuild) · le statut d'`improve`
            (`"improved"`, `"rejected"`, `"nothing_to_fix"`…) · `"first_build_failed"` (pas de premier outil).
        right, total: entrées suivantes (`later`) jugées justes, sur combien.
        regressions: entrées suivantes que le PREMIER outil lisait et que cette réaction a perdues.
        tokens: jetons de la réaction (0 pour `none`) — la première construction est dans `first_tokens`.
        first_hard_right, hard_total: ce que le premier outil savait des entrées dures (le même pour tous les
            bras d'une unité) : s'il les lisait toutes, il n'a pas cassé et l'unité ne dit rien.
        first_tokens: jetons de la première construction, commune aux bras de l'unité.
        detail: pourquoi la réaction n'a rien donné — la raison d'`improve` quand il n'a pas installé de version,
            l'erreur d'une construction refusée ; vide sinon.
        accepted_invalid, invalid_total: entrées INVALIDES jamais vues (`later_invalid`) auxquelles l'outil final
            a répondu par une valeur, sur combien. Il devait les refuser.
    """

    task: str
    trial: int
    arm: str
    status: str
    right: int
    total: int
    regressions: int
    tokens: int
    first_hard_right: int
    hard_total: int
    first_tokens: int
    detail: str = ""
    accepted_invalid: int = 0
    invalid_total: int = 0

    @property
    def solved(self) -> bool:
        """Toutes les entrées suivantes sont justes."""
        return self.total > 0 and self.right == self.total

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "solved": self.solved}


def _pts(x: float) -> str:
    return f"{x * 100:+.0f} pt" + ("" if abs(round(x * 100)) <= 1 else "s")


@dataclass
class ToolBenchReport:
    """Le résultat du banc (`run_tool_bench`) : `rows`, `by_arm()`, `compare(a, b)`, `summary()`, `to_dict()`."""

    rows: list[ToolBenchRow]
    arms: tuple[str, ...]
    trials: int
    model: str = ""
    tasks: tuple[str, ...] = field(default_factory=tuple)

    @property
    def units(self) -> int:
        return len(self.tasks) * self.trials

    @property
    def broke(self) -> int:
        """Unités où le premier outil a VRAIMENT cassé : au moins une entrée dure mal lue. Les autres ne disent
        rien de la réaction (il n'y avait rien à apprendre)."""
        vues = {(r.task, r.trial): r for r in self.rows}
        return sum(1 for r in vues.values() if r.status != "first_build_failed" and r.first_hard_right < r.hard_total)

    @property
    def tokens(self) -> int:
        """Tous les jetons du banc : les premières constructions (une par unité) et les réactions."""
        premiers = {(r.task, r.trial): r.first_tokens for r in self.rows}
        return sum(premiers.values()) + sum(r.tokens for r in self.rows)

    def by_arm(self) -> dict[str, dict[str, Any]]:
        """Par réaction : `units`, `solved`, `right`, `total`, `regressions`, `accepted_invalid`, `invalid_total`,
        `tokens`, `statuses` (par statut)."""
        rendu: dict[str, dict[str, Any]] = {}
        for arm in self.arms:
            lignes = [r for r in self.rows if r.arm == arm]
            statuts: dict[str, int] = {}
            for r in lignes:
                statuts[r.status] = statuts.get(r.status, 0) + 1
            rendu[arm] = {
                "units": len(lignes), "solved": sum(r.solved for r in lignes),
                "right": sum(r.right for r in lignes), "total": sum(r.total for r in lignes),
                "regressions": sum(r.regressions for r in lignes), "tokens": sum(r.tokens for r in lignes),
                "accepted_invalid": sum(r.accepted_invalid for r in lignes),
                "invalid_total": sum(r.invalid_total for r in lignes),
                "statuses": statuts,
            }
        return rendu

    def compare(self, a: str, b: str, *, confidence: float = 0.95) -> dict[str, Any]:
        """Deux réactions comparées sur les unités RÉSOLUES, appariées par tâche — la règle de verdict de
        `compare_configs` (intervalle de Newcombe ET test exact par permutation ; les deux doivent s'accorder).
        Rend `{"a", "b", "delta", "low", "high", "p", "verdict"}` ; `delta` = b - a ; `verdict` :
        `"a_better"`, `"b_better"` ou `"indistinguishable"`."""
        for arm in (a, b):
            if arm not in self.arms:
                raise ValueError(f"unknown arm {arm!r}; this report holds {list(self.arms)}")
        paires = []
        for tache in self.tasks:
            sa = sum(r.solved for r in self.rows if r.task == tache and r.arm == a)
            sb = sum(r.solved for r in self.rows if r.task == tache and r.arm == b)
            paires.append((sa, self.trials, sb, self.trials))
        delta, bas, haut, p, verdict = _decider(paires, confidence)
        return {"a": a, "b": b, "delta": delta, "low": bas, "high": haut, "p": p, "verdict": verdict}

    def summary(self) -> str:
        essais = "essai" if self.trials == 1 else "essais"
        taches = "tâche" if len(self.tasks) == 1 else "tâches"
        unites = "unité" if self.units == 1 else "unités"
        lignes = [f"Banc « outils qui cassent » — {self.model or 'modèle inconnu'} · {len(self.tasks)} {taches} x "
                  f"{self.trials} {essais} = {self.units} {unites} · le premier outil a cassé sur {self.broke} · "
                  f"{self.tokens} jetons"]
        largeur = max(len(_NOMS[arm]) for arm in self.arms)
        for arm, c in self.by_arm().items():
            statuts = ", ".join(f"{n} {s}" for s, n in sorted(c["statuses"].items(), key=lambda p: (-p[1], p[0])))
            invalides = (f" · {c['accepted_invalid']}/{c['invalid_total']} invalide(s) acceptée(s)"
                         if c["invalid_total"] else "")
            lignes.append(f"  {_NOMS[arm]:<{largeur}} : {c['solved']}/{c['units']} unités résolues · {c['right']}/"
                          f"{c['total']} entrées suivantes justes · {c['regressions']} perdue(s){invalides} · "
                          f"{c['tokens']} jetons · {statuts}")
        for a, b, role in (("none", "rebuild", "contrôle — deux constructions à l'aveugle de la même demande"),
                           ("rebuild", "improve", "ce que l'erreur apporte, l'attendu dit"),
                           ("rebuild", "improve_unjudged", "ce que l'erreur apporte, rien dit"),
                           ("rebuild", "improve_refusals", "ce que les refus apportent, sans l'hôte")):
            if a in self.arms and b in self.arms:
                c = self.compare(a, b)
                verdict = {"a_better": f"« {_NOMS[a]} » fait mieux", "b_better": f"« {_NOMS[b]} » fait mieux",
                           "indistinguishable": "indistinguables"}[c["verdict"]]
                lignes.append(f"  {_NOMS[a]} → {_NOMS[b]} : {_pts(c['delta'])} [{_pts(c['low'])} ; {_pts(c['high'])}], "
                              f"p = {c['p']:.3f} → {verdict}   ({role})")
        lignes.append("  Unité = une tâche, un essai ; résolue = toutes les entrées suivantes justes ; « perdue » = une "
                      "entrée que le premier outil lisait ; « invalide acceptée » = une réponse à une entrée qui "
                      "devait être refusée. Le juge est du code.")
        return "\n".join(lignes)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "tasks": list(self.tasks), "arms": list(self.arms), "trials": self.trials,
                "units": self.units, "broke": self.broke, "tokens": self.tokens, "by_arm": self.by_arm(),
                "rows": [r.to_dict() for r in self.rows]}


def _schema(task: ToolTask) -> dict[str, Any]:
    return {"type": "object", "properties": {task.param: {"type": "string"}}, "required": [task.param]}


def _jetons(usage: TokenUsage | None) -> int:
    return int(getattr(usage, "total_tokens", None) or 0)


def _pourquoi(erreur: Exception) -> str:
    return f"{type(erreur).__name__}: {erreur}"[:300]


def _juste(outil: GeneratedPythonTool, args: dict[str, Any], attendu: Any) -> bool:
    """UN appel, jugé : le schéma d'entrée d'abord (comme le ferait le registre), puis le résultat."""
    try:
        issue, valeur = DynamicToolBuilder._rejouer(outil, args)
    except Exception:           # un bac en panne, une erreur imprévue : l'entrée n'est pas lue
        return False
    return issue == "result" and same_result(attendu, valeur)


def _repond(outil: GeneratedPythonTool, args: dict[str, Any]) -> bool:
    """L'outil rend-il une VALEUR pour cette entrée (au lieu de la refuser) ?"""
    try:
        return DynamicToolBuilder._rejouer(outil, args)[0] == "result"
    except Exception:
        return False


def _unite(provider: LLMProvider, task: ToolTask, essai: int, arms: tuple[str, ...],
           sandbox: SubprocessSandbox | DockerSandbox | None, max_repairs: int,
           max_build_tokens: int) -> list[ToolBenchRow]:
    demande = ToolBuildRequest(capability=task.capability, tool_name=task.name, input_schema=_schema(task))
    premiers = (*task.easy, *task.hard)

    def constructeur(dossier: Path, *, carnet: bool = True) -> DynamicToolBuilder:
        # `retire_after_errors=0` : c'est le banc qui décide de la réaction, pas le retrait automatique.
        # `allowed_permissions=set()` : des outils de calcul pur — le modèle ne peut rien s'accorder.
        return DynamicToolBuilder(provider, tools_dir=dossier, sandbox=sandbox, persist=True,
                                  record_calls=4 * (len(premiers) + len(task.invalid)) if carnet else 0,
                                  retire_after_errors=0,
                                  allowed_permissions=set(), max_repairs=max_repairs,
                                  max_build_tokens=max_build_tokens)

    def noter(outil: GeneratedPythonTool) -> tuple[list[bool], int]:
        """L'outil final sur ce qui arrive ensuite : les entrées valides (justes ?) et les invalides (acceptées ?)."""
        # Ces entrées ne doivent JAMAIS entrer au carnet : `improve` les verrait. L'outil ne sert plus qu'à être jugé.
        outil.recorder = None
        justes = [_juste(outil, {task.param: texte}, attendu) for texte, attendu in task.later]
        return justes, sum(_repond(outil, {task.param: texte}) for texte in task.later_invalid)

    with tempfile.TemporaryDirectory(prefix="autoagent_toolbench_", ignore_cleanup_errors=True) as tmp:
        racine = Path(tmp)
        atelier = constructeur(racine / "a")
        try:
            premier = atelier.build(demande)
        except (ToolValidationError, ToolError) as exc:
            # Pas de premier outil : aucune réaction n'a rien à sauver. L'unité compte, ratée pour tous les bras.
            return [ToolBenchRow(task.name, essai, arm, "first_build_failed", 0, len(task.later), 0, 0, 0,
                                 len(task.hard), _jetons(atelier.last_build_usage), _pourquoi(exc), 0,
                                 len(task.later_invalid)) for arm in arms]
        jetons_du_premier = _jetons(atelier.last_build_usage)

        # 1. Les premiers usages — faciles, durs, puis invalides — notés au carnet comme n'importe quel appel.
        for texte in (*(texte for texte, _attendu in premiers), *task.invalid):
            _repond(premier, {task.param: texte})
        appels = {a["args"].get(task.param): a for a in atelier.calls(task.name) if isinstance(a.get("args"), dict)}

        def bien_lu(texte: str, attendu: Any) -> bool:
            appel = appels.get(texte)
            return appel is not None and appel.get("kind") == "ok" and same_result(attendu, appel.get("result"))

        durs_lus = sum(bien_lu(texte, attendu) for texte, attendu in task.hard)
        reference = noter(premier)          # ce que le premier outil lit des entrées suivantes : le bras `none`

        def ligne(arm: str, statut: str, note: tuple[list[bool], int], jetons: int, detail: str = "") -> ToolBenchRow:
            justes, acceptees = note
            perdues = sum(1 for avant, apres in zip(reference[0], justes, strict=True) if avant and not apres)
            return ToolBenchRow(task.name, essai, arm, statut, sum(justes), len(justes), perdues, jetons, durs_lus,
                                len(task.hard), jetons_du_premier, detail, acceptees, len(task.later_invalid))

        def ameliorer(b: DynamicToolBuilder, arm: str, *, refus: bool = False) -> ToolBenchRow:
            resultat = b.improve(task.name, learn_from_refusals=refus)
            # Refusée, ou rien à corriger : l'ancienne version reste — elle lit ce qu'elle lisait.
            note = noter(resultat.tool) if resultat.tool is not None else reference
            return ligne(arm, resultat.status, note, _jetons(resultat.usage),
                         "" if resultat.status == "improved" else resultat.reason[:300])

        lignes: dict[str, ToolBenchRow] = {}
        if "none" in arms:
            lignes["none"] = ligne("none", "kept", reference, 0)
        # La même bibliothèque, copiée AVANT tout verdict, pour chaque réaction où l'hôte n'a rien dit.
        for arm, dossier in (("improve_unjudged", "c"), ("improve_refusals", "d")):
            if arm in arms:
                shutil.copytree(racine / "a", racine / dossier)
        if "improve" in arms:
            # L'hôte (ici : le juge) signale ce qui n'allait pas, avec l'attendu DANS LA FORME que l'outil rend.
            forme = next((appels[t]["result"] for t, attendu in premiers if bien_lu(t, attendu)), None)
            for texte, attendu in premiers:
                appel = appels.get(texte)
                if appel is not None and not bien_lu(texte, attendu):
                    emballe = {next(iter(forme)): attendu} if isinstance(forme, dict) and len(forme) == 1 else attendu
                    atelier.report(task.name, str(appel["id"]), correct=False, expected=emballe)
            for texte in task.invalid:
                appel = appels.get(texte)
                if appel is None:
                    continue
                if appel.get("kind") == "ok":       # il a RÉPONDU à une entrée invalide : il fallait la refuser
                    atelier.report(task.name, str(appel["id"]), correct=False, expected_refusal=True)
                else:                               # son échec était la bonne réponse : elle doit le rester
                    atelier.report(task.name, str(appel["id"]), correct=True, reason="this input is invalid")
            lignes["improve"] = ameliorer(atelier, "improve")
        if "improve_unjudged" in arms:
            lignes["improve_unjudged"] = ameliorer(constructeur(racine / "c"), "improve_unjudged")
        if "improve_refusals" in arms:
            lignes["improve_refusals"] = ameliorer(constructeur(racine / "d"), "improve_refusals", refus=True)
        if "rebuild" in arms:
            autre = constructeur(racine / "b", carnet=False)
            try:
                refait = autre.build(demande)
            except (ToolValidationError, ToolError) as exc:
                lignes["rebuild"] = ligne("rebuild", "build_failed", ([False] * len(task.later), 0),
                                          _jetons(autre.last_build_usage), _pourquoi(exc))
            else:
                lignes["rebuild"] = ligne("rebuild", "rebuilt", noter(refait), _jetons(autre.last_build_usage))
        return [lignes[arm] for arm in arms]


def run_tool_bench(
    provider: LLMProvider,
    *,
    tasks: Sequence[ToolTask] = TOOL_TASKS,
    arms: Sequence[str] = ARMS,
    trials: int = 1,
    sandbox: SubprocessSandbox | DockerSandbox | None = None,
    max_repairs: int = 0,
    max_build_tokens: int = 8192,
) -> ToolBenchReport:
    """Le banc « outils qui cassent » : pour chaque tâche et chaque essai, UN premier outil construit par
    ``provider``, ses premiers usages, puis chaque réaction de ``arms`` à partir de ce même outil, jugée sur les
    entrées suivantes (voir le module).

    ``sandbox`` : le bac où tourne TOUT le code écrit par le modèle (celui de `DynamicToolBuilder` par défaut).
    Avec un vrai modèle, passe une vraie frontière : ``make_sandbox(require_boundary=True)``. ``max_repairs``,
    ``max_build_tokens`` : les reprises et le plafond de sortie du constructeur (`DynamicToolBuilder`) — un modèle
    qui réfléchit a besoin d'un plafond plus haut que 8192. Coût : un appel au modèle par unité, plus un par réaction autre que
    ``none``. Rien n'est gardé sur le disque. ``ValueError`` sur une réaction inconnue, des tâches en double, ou
    ``trials < 1``."""
    taches = list(tasks)
    bras = tuple(arms)
    if not taches:
        raise ValueError("need at least one task")
    if len({t.name for t in taches}) != len(taches):
        raise ValueError("task names must be unique")
    if not bras or len(set(bras)) != len(bras) or any(arm not in ARMS for arm in bras):
        raise ValueError(f"arms must be distinct values among {list(ARMS)}, got {list(bras)}")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError(f"trials must be an integer >= 1, got {trials!r}")
    lignes: list[ToolBenchRow] = []
    for tache in taches:
        for essai in range(1, trials + 1):
            lignes.extend(_unite(provider, tache, essai, bras, sandbox, max_repairs, max_build_tokens))
    modele = str(getattr(getattr(provider, "config", None), "model", "") or "")
    return ToolBenchReport(rows=lignes, arms=bras, trials=trials, model=modele, tasks=tuple(t.name for t in taches))


def positive_control_tools(*, tasks: Sequence[ToolTask] = TOOL_TASKS,
                           sandbox: SubprocessSandbox | DockerSandbox | None = None) -> ToolBenchReport:
    """Le contrôle positif (niveau 1, hors ligne) : avec un constructeur scripté qui rend d'abord la version
    naïve puis, à `improve`, la version complète, ``improve`` DOIT résoudre des unités que ``none`` et ``rebuild``
    ne résolvent pas — et ``improve_refusals`` des unités que ``improve_unjudged`` ne résout pas (celles où la
    version naïve refuse au lieu de planter). Sinon le banc est aveugle, et ses chiffres sur un vrai modèle ne
    valent rien."""
    return run_tool_bench(ScriptedToolProvider(tasks), tasks=tasks, sandbox=sandbox)
