"""Comparer deux configurations d'agent sans se raconter d'histoires.

Le problème. On change un prompt, un modèle, un outil ; un premier échantillon
« montre un gain » ; on publie. Or RELANCER LA MÊME configuration donne déjà des
résultats différents : sur plus de 18 000 trajectoires, environ 54 % de la
variance des résultats vient du simple re-lancement et non du changement de
configuration (Wiedmann et al., arXiv:2610.01618, preprint d'octobre 2026). Un
écart sous ce bruit n'est pas une amélioration, c'est un tirage.

`compare_configs` ne rend un verdict que si la preuve existe :

* **mêmes tâches, mêmes juges** déterministes que `run_k` (jamais un LLM-juge),
  pour les deux configurations ;
* **les bras ALTERNENT** à chaque répétition (ordre tourné, point de départ tiré
  avec `seed`) : une dérive du fournisseur ou un cache chaud touche les deux ;
* **écart apparié par tâche** : la difficulté propre d'une tâche s'annule ;
* **le verdict exige l'accord de DEUX calculs**. D'abord un test exact par
  permutation, stratifié par tâche : si A et B sont identiques sur chaque tâche,
  il déclare une différence avec une probabilité ≤ alpha, par construction (aucune
  approximation, aucun tirage). Ensuite l'intervalle de Wilson-Newcombe combiné
  par tâche, qui dit la TAILLE de l'écart. Pourquoi pas l'intervalle seul : mesuré
  (voir `tests/test_compare_calibration.py`), sur des configurations identiques il
  déclarait un gagnant jusqu'à 17 % du temps au lieu de 5 % ;
* **par défaut : « indistinguable »**. Il faut une preuve pour en sortir ;
* `control=True` rejoue A une seconde fois (contrôle **A/A**) : deux configurations
  identiques doivent sortir « indistinguables ». Sinon la mesure se trompe
  elle-même — biais lié à l'ordre ou à l'identité d'un bras (état partagé que l'un
  abîme ou nettoie, cache, nom de variante utilisé dans un chemin), tentatives non
  indépendantes — et aucun verdict A/B ne vaut. Un contrôle réussi ne prouve pas que
  tout va bien (au plus 5 % des contrôles échouent à tort) : il attrape les gros
  défauts. Une dérive lente du fournisseur, elle, touche les bras de la même façon
  grâce à l'alternance ;
* **« indistinguable » n'est pas « équivalent »** : le rapport dit la plus petite
  différence que CETTE mesure aurait détectée (`detectable_difference`) ;
* le **coût** est comparé avec son intervalle (jetons par tentative, par succès).

Ce que ça ne fait pas, et il vaut mieux le savoir avant de s'en servir :

* l'intervalle et le test valent pour CES tâches. Dire que B est meilleure sur
  des tâches que vous n'avez pas mises dans la suite est une autre affirmation,
  qu'aucun calcul ne remplace ;
* l'intervalle est APPROCHÉ : sa couverture, mesurée par simulation, va d'environ
  83 % à 99 % selon les taux de réussite réels. C'est pourquoi il ne décide pas
  seul ;
* aucune correction pour comparaisons multiples : comparer cinq variantes à une
  référence, c'est cinq verdicts, pas un ;
* aucune équivalence déclarée : prouver « pas pire de plus de X points » demande
  son propre contrôle d'erreur, non fourni ici.

Exemple::

    from autoagent.compare import EvalTask, Variant, compare_configs

    rapport = compare_configs(
        Variant("v1", lambda: construire_agent(prompt_v1)),
        Variant("v2", lambda: construire_agent(prompt_v2)),
        [EvalTask("solde", "Solde du client C-102 ?", lambda r: "4821" in r.output),
         EvalTask("stock", "Stock de la référence R-7 ?", lambda r: "35" in r.output)],
        repeats=8, control=True,
    )
    print(rapport.summary())
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from statistics import NormalDist
from typing import Any

from .eval import Attempt, ReliabilityReport, _duree, _tentative
from .logging import get_logger
from .schema import TokenUsage

__all__ = [
    "ComparisonReport",
    "ControlResult",
    "CostComparison",
    "EvalTask",
    "LatencyComparison",
    "TaskComparison",
    "Variant",
    "compare_configs",
    "detectable_difference",
    "paired_interval",
    "paired_p_value",
    "wilson_interval",
]

_log = get_logger("compare")

# Une tâche vue par les deux bras : (succès A, essais A, succès B, essais B).
Pair = tuple[int, int, int, int]

A_BETTER = "a_better"
B_BETTER = "b_better"
INDISTINGUISHABLE = "indistinguishable"


# ── Calcul : trois fonctions pures, sans état, sans dépendance ───────────────

def _z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def _check_pair(pair: Pair) -> None:
    sa, na, sb, nb = pair
    if na < 1 or nb < 1:
        raise ValueError("each arm needs at least one attempt per task")
    if not (0 <= sa <= na and 0 <= sb <= nb):
        raise ValueError(f"successes out of range in {pair!r}")


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Intervalle de Wilson d'une proportion.

    Contrairement à l'intervalle « normal » (p ± z·√(p(1-p)/n)), il reste sensé aux
    bords — le cas ordinaire avec peu de répétitions : 5 réussites sur 5 ne veut
    pas dire « 100 % ± 0 » mais « au moins environ 57 % ».
    """
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("need n >= 1 and 0 <= successes <= n")
    z = _z(confidence)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    demi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if successes == 0 else max(0.0, centre - demi)
    high = 1.0 if successes == n else min(1.0, centre + demi)
    return low, high


def paired_interval(
    pairs: Sequence[Pair], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Écart moyen par tâche (B - A) et son intervalle approché : `(écart, bas, haut)`.

    L'écart est la MOYENNE des écarts de chaque tâche : la difficulté d'une tâche
    s'annule (c'est l'appariement). L'intervalle est la construction de Newcombe
    — somme des écarts au carré des intervalles de Wilson de chaque bras de
    chaque tâche — étendue d'un écart à une moyenne de T écarts. Elle ne suppose
    rien d'une loi normale aux bords, mais elle est APPROCHÉE : voir l'avertissement
    en tête du module. Sur un seul couple 56/70 contre 48/80 (l'exemple de Newcombe,
    Statistics in Medicine 17:873-890, 1998) elle donne écart 0,2000, intervalle
    [0,0524 ; 0,3339] — identique à l'implémentation indépendante de statsmodels 0.15.0
    (`confint_proportions_2indep`, `method="newcomb"`).
    """
    if not pairs:
        raise ValueError("need at least one task")
    t = len(pairs)
    delta = 0.0
    bas2 = haut2 = 0.0
    for pair in pairs:
        _check_pair(pair)
        sa, na, sb, nb = pair
        pa, pb = sa / na, sb / nb
        la, ua = wilson_interval(sa, na, confidence)
        lb, ub = wilson_interval(sb, nb, confidence)
        delta += pb - pa
        bas2 += (pb - lb) ** 2 + (ua - pa) ** 2
        haut2 += (ub - pb) ** 2 + (pa - la) ** 2
    delta /= t
    return delta, max(-1.0, delta - math.sqrt(bas2) / t), min(1.0, delta + math.sqrt(haut2) / t)


@lru_cache(maxsize=4096)
def _loi_somme(n: int, totaux: tuple[int, ...]) -> tuple[float, ...]:
    """Loi du nombre total de succès attribués à A sous « A et B sont échangeables ».

    Sur une tâche où `k` succès ont été observés en tout (A et B confondus, 2n
    essais), les n essais « de A » sont un tirage sans remise : le nombre de
    succès de A y est hypergéométrique. Les tâches sont indépendantes : la loi de
    la somme est la convolution exacte.
    """
    loi: list[float] = [1.0]
    total = math.comb(2 * n, n)
    for k in totaux:
        bas, haut = max(0, k - n), min(n, k)
        pmf = [math.comb(k, x) * math.comb(2 * n - k, n - x) / total for x in range(bas, haut + 1)]
        suite = [0.0] * (len(loi) + haut)
        for s, ps in enumerate(loi):
            if ps:
                for j, px in enumerate(pmf):
                    suite[s + bas + j] += ps * px
        loi = suite
    return tuple(loi)


def paired_p_value(pairs: Sequence[Pair]) -> float:
    """p-value EXACTE (bilatérale) du test par permutation, stratifié par tâche.

    Hypothèse nulle : sur chaque tâche, A et B ont le même taux de réussite — les
    étiquettes « A » et « B » des essais sont échangeables. Conditionnellement au
    nombre de succès de chaque tâche, la statistique (somme des succès de A) suit
    une loi calculable exactement : aucun tirage, aucune graine, aucune
    approximation. Sous l'hypothèse nulle, la probabilité que cette p-value soit
    ≤ alpha est ≤ alpha (test exact : conservateur quand les données sont discrètes).

    Exige le même nombre d'essais n pour A et B sur TOUTES les tâches — c'est ce que
    `compare_configs` produit.
    """
    if not pairs:
        raise ValueError("need at least one task")
    n = pairs[0][1]
    for pair in pairs:
        _check_pair(pair)
        if pair[1] != n or pair[3] != n:
            raise ValueError("the exact test needs the same number of attempts per arm and per task")
    totaux = tuple(sorted(sa + sb for sa, _na, sb, _nb in pairs))
    k = sum(totaux)
    observe = abs(k - 2 * sum(sa for sa, _na, _sb, _nb in pairs))
    loi = _loi_somme(n, totaux)
    p = sum(prob for s, prob in enumerate(loi) if abs(k - 2 * s) >= observe)
    return min(1.0, p)


def _decider(pairs: Sequence[Pair], confidence: float) -> tuple[float, float, float, float, str]:
    """LA règle de verdict, une seule fois : `(écart, bas, haut, p, verdict)`.

    Il faut l'accord des deux calculs. Quand ils ne s'accordent pas, on ne tranche
    pas : la sortie reste « indistinguable ».
    """
    delta, bas, haut = paired_interval(pairs, confidence)
    p = paired_p_value(pairs)
    alpha = 1.0 - confidence
    if p < alpha and delta > 0 and bas > 0:
        verdict = B_BETTER
    elif p < alpha and delta < 0 and haut < 0:
        verdict = A_BETTER
    else:
        verdict = INDISTINGUISHABLE
    return delta, bas, haut, p, verdict


def detectable_difference(
    n_tasks: int,
    repeats: int,
    *,
    base_rate: float = 0.5,
    power: float = 0.8,
    confidence: float = 0.95,
    simulations: int = 200,
    seed: int = 0,
) -> float | None:
    """Plus petit écart VRAI que ce plan (`n_tasks` x `repeats`) détecterait.

    C'est le garde-fou contre la lecture « indistinguable = équivalentes » : un plan
    trop petit ne voit rien, et le dit ici. Hypothèse de planification : toutes les
    tâches ont le taux `base_rate` pour A et `base_rate ± écart` pour B (0,5 par
    défaut : le cas où le hasard fait le plus de bruit). On cherche, sur une grille
    de 5 points, le plus petit écart que la règle de verdict de `compare_configs`
    détecte avec la probabilité `power`. Résultat arrondi à la grille ; `None` si
    même l'écart maximal ne serait pas détecté. Déterministe (`seed`).

    Coût : quelques centaines de verdicts simulés — instantané pour quelques
    tâches, de l'ordre de la seconde vers 100 essais par bras.
    """
    if n_tasks < 1 or repeats < 1:
        raise ValueError("n_tasks and repeats must be >= 1")
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError("base_rate must be in [0, 1]")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be strictly between 0 and 1")
    sens = 1.0 if base_rate <= 0.5 else -1.0
    cible = B_BETTER if sens > 0 else A_BETTER
    place = 1.0 - base_rate if sens > 0 else base_rate
    grille = [round(0.05 * i, 2) for i in range(1, 21) if 0.05 * i <= place + 1e-9]
    if not grille:
        return None

    def puissance(ecart: float) -> float:
        rng = random.Random(seed)  # mêmes tirages pour chaque écart : puissance monotone
        gagne = 0
        for _ in range(simulations):
            pairs: list[Pair] = []
            for _t in range(n_tasks):
                sa = sum(rng.random() < base_rate for _ in range(repeats))
                tirages = [rng.random() for _ in range(repeats)]
                sb = sum(u < base_rate + sens * ecart for u in tirages)
                pairs.append((sa, repeats, sb, repeats))
            gagne += _decider(pairs, confidence)[4] == cible
        return gagne / simulations

    if puissance(grille[-1]) < power:
        return None
    bas, haut = 0, len(grille) - 1
    while bas < haut:
        milieu = (bas + haut) // 2
        if puissance(grille[milieu]) >= power:
            haut = milieu
        else:
            bas = milieu + 1
    return grille[bas]


# ── Ce qu'on compare ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Variant:
    """Une configuration à comparer : un nom, et un agent — ou, mieux, une FABRIQUE.

    `agent` est un `Agent` ou un callable SANS argument qui en rend un NEUF : chaque
    tentative doit partir d'un état propre (même règle que `run_k`). `params` décrit
    ce qui change (« prompt »: « v2 », « modèle »: …) ; il entre dans l'empreinte.
    """

    name: str
    agent: Any
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalTask:
    """Une tâche de la suite : une consigne, un juge DÉTERMINISTE, un contexte."""

    name: str
    prompt: str
    check: Callable[[Any], bool]
    context: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskComparison:
    """Une tâche, vue par les deux bras. L'intervalle ne vaut que pour cette tâche."""

    name: str
    successes_a: int
    successes_b: int
    n: int
    rate_a: float
    rate_b: float
    delta: float
    low: float
    high: float

    @property
    def direction(self) -> str:
        """`"b"`, `"a"` ou `"tie"` — au sens du taux observé, sans prétendre à la preuve."""
        return "b" if self.delta > 0 else "a" if self.delta < 0 else "tie"


@dataclass(frozen=True)
class CostComparison:
    """Le coût des deux bras (toutes les tentatives, ratées comprises)."""

    tokens_per_attempt_a: float
    tokens_per_attempt_b: float
    relative_change: float | None
    interval: tuple[float, float] | None
    tokens_per_success_a: float | None
    tokens_per_success_b: float | None
    cost_per_success_a: float | None = None
    cost_per_success_b: float | None = None


@dataclass(frozen=True)
class LatencyComparison:
    """La durée des deux bras (0.24.0) : médiane d'UNE tentative, écart relatif, intervalle bootstrap.

    Médiane, pas moyenne : une durée a une queue lourde (un appel réseau lent), et la moyenne d'un
    petit échantillon suit cette queue. L'intervalle est un bootstrap (tentatives rééchantillonnées
    tâche par tâche) : sur peu d'essais il est LARGE, et c'est ce qu'il faut lire."""

    median_seconds_a: float
    median_seconds_b: float
    relative_change: float | None
    interval: tuple[float, float] | None
    n_a: int
    n_b: int


@dataclass(frozen=True)
class ControlResult:
    """Le contrôle A/A : A contre sa propre copie. Il DOIT sortir « indistinguable »."""

    name: str
    delta: float
    low: float
    high: float
    p_value: float
    verdict: str

    @property
    def ok(self) -> bool:
        return self.verdict == INDISTINGUISHABLE


def _pts(x: float) -> str:
    return f"{x * 100:+.0f}"


def _p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


@dataclass
class ComparisonReport:
    """Résultat de `compare_configs`. Les données brutes y sont toutes."""

    a: str
    b: str
    repeats: int
    confidence: float
    tasks: list[TaskComparison]
    rate_a: float
    rate_b: float
    delta: float
    low: float
    high: float
    p_value: float
    p_min: float
    verdict: str
    reports: dict[str, dict[str, ReliabilityReport]]
    fingerprints: dict[str, str]
    cost: CostComparison | None = None
    control: ControlResult | None = None
    detectable: float | None = None
    notes: list[str] = field(default_factory=list)
    latency: LatencyComparison | None = None

    @property
    def trustworthy(self) -> bool:
        """Faux si le contrôle A/A a échoué : aucun verdict A/B ne vaut alors."""
        return self.control is None or self.control.ok

    def summary(self) -> str:
        t = len(self.tasks)
        lignes = [
            f"A « {self.a} » contre B « {self.b} » · {t} tâche(s) x {self.repeats} "
            f"répétition(s) par bras",
            f"Succès : A {self.rate_a:.0%} → B {self.rate_b:.0%} · écart {_pts(self.delta)} points, "
            f"intervalle approché à {self.confidence:.0%} [{_pts(self.low)} ; {_pts(self.high)}] "
            f"· test exact {_p(self.p_value)}",
        ]
        if self.verdict == B_BETTER:
            conclusion = "→ B MEILLEURE : le test exact et l'intervalle s'accordent."
        elif self.verdict == A_BETTER:
            conclusion = "→ A MEILLEURE : le test exact et l'intervalle s'accordent."
        else:
            conclusion = "→ INDISTINGUABLES : aucune preuve d'écart."
            if self.detectable is not None:
                conclusion += (f" Ce n'est PAS « équivalentes » : cette mesure ne détecterait pas "
                               f"moins de ≈{self.detectable * 100:.0f} points (80 % de chances).")
        if not self.trustworthy:
            conclusion += " VERDICT NON FIABLE : le contrôle A/A a échoué."
        lignes.append(conclusion)
        mieux = sum(1 for x in self.tasks if x.direction == "b")
        moins = sum(1 for x in self.tasks if x.direction == "a")
        lignes.append(f"Par tâche : B > A sur {mieux}, B < A sur {moins}, égalité sur {t - mieux - moins}.")
        if self.cost is not None:
            c = self.cost
            ligne = f"Coût : {c.tokens_per_attempt_a:.0f} → {c.tokens_per_attempt_b:.0f} jetons par tentative"
            if c.relative_change is not None and c.interval is not None:
                ligne += (f" ({c.relative_change:+.0%}, intervalle bootstrap "
                          f"[{c.interval[0]:+.0%} ; {c.interval[1]:+.0%}])")
            if c.tokens_per_success_a is not None and c.tokens_per_success_b is not None:
                ligne += (f" · par succès {c.tokens_per_success_a:.0f} → "
                          f"{c.tokens_per_success_b:.0f}")
            lignes.append(ligne)
        # Sous 10 ms des deux côtés (un modèle scripté, un cache), la durée ne dit rien : on ne l'affiche pas
        # (elle reste dans `latency` et dans `to_dict()`).
        if self.latency is not None and max(self.latency.median_seconds_a, self.latency.median_seconds_b) >= 0.01:
            d = self.latency
            ligne = f"Durée : médiane {_duree(d.median_seconds_a)} → {_duree(d.median_seconds_b)} par tentative"
            if d.relative_change is not None and d.interval is not None:
                ligne += (f" ({d.relative_change:+.0%}, intervalle bootstrap "
                          f"[{d.interval[0]:+.0%} ; {d.interval[1]:+.0%}])")
            lignes.append(ligne)
        if self.control is not None:
            c2 = self.control
            etat = "indistinguable, comme attendu" if c2.ok else "A ET SA COPIE DIFFÈRENT"
            lignes.append(f"Contrôle A/A : {etat} (écart {_pts(c2.delta)} points "
                          f"[{_pts(c2.low)} ; {_pts(c2.high)}], {_p(c2.p_value)}).")
        lignes.extend(f"⚠ {n}" for n in self.notes)
        empreintes = " · ".join(f"{k} {v}" for k, v in self.fingerprints.items())
        # Ce que l'empreinte voit — et ne voit pas : elle lit la STRUCTURE de l'agent, jamais
        # le code d'un outil. Deux bras dont l'outil se comporte différemment ont la même
        # empreinte tant que l'écart n'est pas déclaré dans `Variant.params` (mesuré en réel).
        lignes.append(f"Empreintes (structure déclarée, pas le code des outils) : {empreintes}")
        return "\n".join(lignes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe : à archiver à côté des fichiers de la suite."""
        return {
            "a": self.a, "b": self.b, "repeats": self.repeats, "confidence": self.confidence,
            "verdict": self.verdict, "trustworthy": self.trustworthy,
            "rate_a": self.rate_a, "rate_b": self.rate_b, "delta": self.delta,
            "interval": [self.low, self.high], "p_value": self.p_value, "p_min": self.p_min,
            "detectable_difference": self.detectable,
            "tasks": [asdict(x) for x in self.tasks],
            "cost": None if self.cost is None else asdict(self.cost),
            "latency": None if self.latency is None else asdict(self.latency),
            "control": None if self.control is None else {**asdict(self.control), "ok": self.control.ok},
            "fingerprints": dict(self.fingerprints),
            "notes": list(self.notes),
            "reports": {bras: {tache: r.to_dict() for tache, r in taches.items()}
                        for bras, taches in self.reports.items()},
        }


# ── Empreintes : de QUOI parle ce rapport ────────────────────────────────────

def _hash(obj: Any) -> str:
    try:
        blob = json.dumps(
            obj, sort_keys=True, ensure_ascii=False,
            default=lambda o: f"<{type(o).__qualname__}>",   # jamais d'adresse mémoire : stable d'un run à l'autre
        )
    except (TypeError, ValueError):                           # clés de types mélangés, circularité…
        blob = f"<{type(obj).__qualname__}>"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _identite_fonction(fn: Callable[..., Any]) -> list[Any]:
    """Ce qui fait qu'un juge est CE juge : son source, mais aussi les valeurs qu'il
    capture. `lambda r: attendu in r.output` a le même source pour `attendu="42"` et
    `attendu="41"` : sans les valeurs de fermeture, l'empreinte ne verrait pas qu'on
    a changé ce qui compte comme une réussite."""
    if isinstance(fn, functools.partial):
        return ["partial", _identite_fonction(fn.func), list(fn.args), fn.keywords]
    try:
        source: Any = inspect.getsource(fn)
    except (OSError, TypeError):
        source = getattr(fn, "__qualname__", type(fn).__qualname__)
    fermeture = []
    for cellule in getattr(fn, "__closure__", None) or ():
        try:
            fermeture.append(cellule.cell_contents)
        except ValueError:          # cellule pas encore remplie
            fermeture.append("<empty>")
    return [source, fermeture, getattr(fn, "__defaults__", None),
            getattr(fn, "__kwdefaults__", None)]


def _decrire_agent(agent: Any) -> dict[str, Any]:
    """Ce qu'on peut lire d'un agent SANS rien exécuter. Best-effort : fail-open."""
    desc: dict[str, Any] = {"class": type(agent).__qualname__}

    def lire(cle: str, fn: Callable[[], Any]) -> None:
        try:
            valeur = fn()
        except Exception:
            return
        if valeur is not None:
            desc[cle] = valeur

    provider = getattr(agent, "provider", None)
    lire("provider", lambda: None if provider is None else type(provider).__qualname__)
    lire("model", lambda: getattr(getattr(provider, "config", None), "model", None))
    prompt = getattr(agent, "system_prompt", None)
    lire("system_prompt", lambda: _hash(prompt) if isinstance(prompt, str)
         else ("<callable>" if callable(prompt) else None))
    lire("tools", lambda: sorted((s.name, _hash(asdict(s))) for s in agent.registry.specs()))
    lire("bounds", lambda: agent.bounds.to_dict())
    for nom in ("temperature", "max_tokens", "parallel_tool_calls",
                "max_corrections_per_run", "max_dynamic_tools_per_run"):
        lire(nom, functools.partial(getattr, agent, nom, None))
    for nom in ("memory", "tool_policy", "post_turn_hook"):
        lire(nom, functools.partial(_nom_de_classe, getattr(agent, nom, None)))
    return desc


def _nom_de_classe(obj: Any) -> str | None:
    return None if obj is None else type(obj).__qualname__


def _empreinte_suite(tasks: Sequence[EvalTask]) -> str:
    return _hash([[t.name, t.prompt, t.context, _identite_fonction(t.check)] for t in tasks])


# ── Coût : bootstrap sur les tentatives, par tâche ───────────────────────────

def _quantile(valeurs: list[float], q: float) -> float:
    return valeurs[min(len(valeurs) - 1, max(0, round(q * (len(valeurs) - 1))))]


def _comparer_couts(
    att_a: dict[str, list[Attempt]],
    att_b: dict[str, list[Attempt]],
    *,
    confidence: float,
    bootstrap: int,
    seed: int,
    cost_fn: Callable[[TokenUsage], float] | None,
) -> CostComparison | None:
    """None dès qu'une tentative n'a pas rapporté d'usage : on n'invente pas un zéro."""
    for tentatives in (*att_a.values(), *att_b.values()):
        if any(a.total_tokens is None for a in tentatives):
            return None
    listes_a = [[float(a.total_tokens or 0) for a in v] for v in att_a.values()]
    listes_b = [[float(a.total_tokens or 0) for a in v] for v in att_b.values()]
    n_a = sum(len(v) for v in listes_a)
    n_b = sum(len(v) for v in listes_b)
    tot_a = sum(sum(v) for v in listes_a)
    tot_b = sum(sum(v) for v in listes_b)
    relatif: float | None = None
    intervalle: tuple[float, float] | None = None
    if tot_a > 0 and n_a and n_b:
        relatif = (tot_b / n_b) / (tot_a / n_a) - 1.0
        rng = random.Random(seed)
        tirages: list[float] = []
        for _ in range(bootstrap):
            sa = sum(sum(rng.choices(v, k=len(v))) for v in listes_a)
            sb = sum(sum(rng.choices(v, k=len(v))) for v in listes_b)
            if sa > 0:
                tirages.append((sb / n_b) / (sa / n_a) - 1.0)
        if tirages:
            tirages.sort()
            alpha = 1.0 - confidence
            intervalle = (_quantile(tirages, alpha / 2), _quantile(tirages, 1 - alpha / 2))
    toutes_a = [a for v in att_a.values() for a in v]
    toutes_b = [a for v in att_b.values() for a in v]
    rep_a = ReliabilityReport(k=max(1, n_a), attempts=toutes_a)
    rep_b = ReliabilityReport(k=max(1, n_b), attempts=toutes_b)
    return CostComparison(
        tokens_per_attempt_a=tot_a / n_a if n_a else 0.0,
        tokens_per_attempt_b=tot_b / n_b if n_b else 0.0,
        relative_change=relatif,
        interval=intervalle,
        tokens_per_success_a=rep_a.tokens_per_success,
        tokens_per_success_b=rep_b.tokens_per_success,
        cost_per_success_a=None if cost_fn is None else rep_a.cost_per_success(cost_fn),
        cost_per_success_b=None if cost_fn is None else rep_b.cost_per_success(cost_fn),
    )


# ── Durée : bootstrap de la médiane, par tâche (0.24.0) ──────────────────────

def _comparer_durees(
    att_a: dict[str, list[Attempt]],
    att_b: dict[str, list[Attempt]],
    *,
    confidence: float,
    bootstrap: int,
    seed: int,
) -> LatencyComparison | None:
    """None dès qu'une tentative n'a pas de durée mesurée : on n'invente pas un zéro.

    Un flux aléatoire À PART (`seed` décalé) : le bootstrap du coût, lui, ne bouge pas d'un bit."""
    for tentatives in (*att_a.values(), *att_b.values()):
        if any(a.seconds is None for a in tentatives):
            return None
    listes_a = [[float(a.seconds or 0.0) for a in v] for v in att_a.values()]
    listes_b = [[float(a.seconds or 0.0) for a in v] for v in att_b.values()]
    pool_a = [x for v in listes_a for x in v]
    pool_b = [x for v in listes_b for x in v]
    if not pool_a or not pool_b:
        return None
    med_a, med_b = statistics.median(pool_a), statistics.median(pool_b)
    relatif: float | None = None
    intervalle: tuple[float, float] | None = None
    if med_a > 0:
        relatif = med_b / med_a - 1.0
        rng = random.Random(f"{seed!r}:latence")      # un flux à PART, sans arithmétique : `seed` peut être str, bytes, None…
        tirages: list[float] = []
        for _ in range(bootstrap):
            ma = statistics.median([x for v in listes_a for x in rng.choices(v, k=len(v))])
            mb = statistics.median([x for v in listes_b for x in rng.choices(v, k=len(v))])
            if ma > 0:
                tirages.append(mb / ma - 1.0)
        if tirages:
            tirages.sort()
            alpha = 1.0 - confidence
            intervalle = (_quantile(tirages, alpha / 2), _quantile(tirages, 1 - alpha / 2))
    return LatencyComparison(med_a, med_b, relatif, intervalle, len(pool_a), len(pool_b))


# ── Le banc ──────────────────────────────────────────────────────────────────

def _variante(x: Any, defaut: str) -> Variant:
    return x if isinstance(x, Variant) else Variant(name=defaut, agent=x)


def _paires(
    att_a: dict[str, list[Attempt]], att_b: dict[str, list[Attempt]], tasks: Sequence[EvalTask]
) -> list[Pair]:
    return [
        (sum(a.ok for a in att_a[t.name]), len(att_a[t.name]),
         sum(a.ok for a in att_b[t.name]), len(att_b[t.name]))
        for t in tasks
    ]


def compare_configs(
    a: Any,
    b: Any,
    tasks: Sequence[EvalTask],
    *,
    repeats: int = 5,
    confidence: float = 0.95,
    control: bool = False,
    seed: int = 0,
    bootstrap: int = 2000,
    cost_fn: Callable[[TokenUsage], float] | None = None,
    on_attempt: Callable[[str, str, Attempt], None] | None = None,
) -> ComparisonReport:
    """Compare deux configurations sur les mêmes tâches. Voir l'en-tête du module.

    Args:
        a, b: un `Variant` — ou directement un `Agent` / une fabrique d'agents neufs
            (les bras s'appellent alors « A » et « B »). Préférer la fabrique.
        tasks: la suite. Les noms doivent être uniques.
        repeats: essais par bras ET par tâche (5 par défaut, c'est peu : voir
            `detectable_difference` pour savoir ce que votre plan détecte).
        confidence: niveau de l'intervalle ; le test exact utilise alpha = 1 - confidence.
        control: ajoute un bras A' (copie de A) et compare A' à A. Coûte 50 % d'appels
            en plus. À faire au moins une fois par suite, et après tout changement
            de juge ou d'infrastructure.
        seed: ordre des bras, bootstrap du coût et plan de puissance sont déterministes.
        cost_fn: `TokenUsage -> montant` (tarif de l'HÔTE) pour rapporter un coût
            par succès en monnaie. Sans lui, le coût est en jetons.
        on_attempt: `(bras, tâche, tentative)` après chaque tentative (progression,
            archivage). Un rappel qui plante est ignoré : l'observabilité ne casse
            jamais la mesure.

    Nombre d'appels : `len(tasks) x repeats x 2` (x 3 avec `control`). Aucune
    parallélisation : l'ordre doit rester reproductible.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    _z(confidence)
    taches = list(tasks)
    if not taches:
        raise ValueError("need at least one task")
    noms = [t.name for t in taches]
    if len(set(noms)) != len(noms):
        raise ValueError("task names must be unique")
    for t in taches:
        if not callable(t.check):
            raise TypeError(f"check of task {t.name!r} must be a callable (AgentResult) -> bool")
    va, vb = _variante(a, "A"), _variante(b, "B")
    bras: list[Variant] = [va, vb]
    if control:
        bras.append(Variant(name=f"{va.name} (contrôle A/A)", agent=va.agent, params=va.params))
    if len({v.name for v in bras}) != len(bras):
        raise ValueError("the two configurations need distinct names")

    rng = random.Random(seed)
    tentatives: dict[str, dict[str, list[Attempt]]] = {v.name: {t.name: [] for t in taches} for v in bras}
    empreintes_vues: dict[str, set[str]] = {v.name: set() for v in bras}
    for tache in taches:
        decalage = rng.randrange(len(bras))
        for repetition in range(repeats):
            # Ordre tourné : chaque bras passe aussi souvent premier que les autres.
            for pas in range(len(bras)):
                variante = bras[(decalage + repetition + pas) % len(bras)]
                agent = variante.agent() if callable(variante.agent) else variante.agent
                empreintes_vues[variante.name].add(_hash({"agent": _decrire_agent(agent),
                                                          "params": variante.params}))
                tentative = _tentative(agent, tache.prompt, tache.check, tache.context, repetition + 1)
                tentatives[variante.name][tache.name].append(tentative)
                if on_attempt is not None:
                    try:
                        on_attempt(variante.name, tache.name, tentative)
                    except Exception:
                        _log.exception("on_attempt callback failed")  # fail-open

    reports = {
        nom: {t: ReliabilityReport(k=repeats, attempts=liste) for t, liste in par_tache.items()}
        for nom, par_tache in tentatives.items()
    }
    att_a, att_b = tentatives[va.name], tentatives[vb.name]
    pairs = _paires(att_a, att_b, taches)
    delta, bas, haut, p, verdict = _decider(pairs, confidence)
    par_tache: list[TaskComparison] = []
    for t, (sa, na, sb, nb) in zip(taches, pairs, strict=True):
        d, lo, hi = paired_interval([(sa, na, sb, nb)], confidence)
        par_tache.append(TaskComparison(t.name, sa, sb, na, sa / na, sb / nb, d, lo, hi))
    p_min = paired_p_value([(repeats, repeats, 0, repeats)] * len(taches))
    alpha = 1.0 - confidence

    notes: list[str] = []
    for variante in bras:
        erreurs = [x.error for lst in tentatives[variante.name].values() for x in lst if x.error]
        if erreurs:
            notes.append(f"{len(erreurs)} tentative(s) du bras « {variante.name} » ont levé une "
                         f"exception (comptées comme échecs) : vérifier l'infrastructure avant de "
                         f"conclure. Première : {erreurs[0][:140]}")
        if len(empreintes_vues[variante.name]) > 1:
            notes.append(f"La fabrique du bras « {variante.name} » a produit des agents DIFFÉRENTS "
                         f"d'une tentative à l'autre (outils, prompt ou bornes) : ce qui est "
                         f"comparé n'est pas une configuration.")
        if not callable(variante.agent):
            notes.append(f"Le bras « {variante.name} » est une instance réutilisée : son état "
                         f"s'accumule d'une tentative à l'autre. Passer une fabrique.")
    if p_min >= alpha:
        notes.append(f"Plan trop petit : même une séparation parfaite (A toujours, B jamais) ne "
                     f"donnerait que p={p_min:.3f} ≥ {alpha:.2f}. Aucune différence ne pourrait être "
                     f"déclarée : augmenter les répétitions ou les tâches.")
    informatives = sum(1 for sa, na, sb, nb in pairs if 0 < sa + sb < na + nb)
    if informatives == 0:
        notes.append("Plafond ou plancher : sur chaque tâche, les deux bras réussissent TOUJOURS ou "
                     "échouent TOUJOURS. Le juge ne sépare pas ces configurations ; il faut des "
                     "tâches plus difficiles (ou plus faciles).")
    elif informatives < len(taches):
        notes.append(f"{len(taches) - informatives} tâche(s) sur {len(taches)} ne portent aucune "
                     f"information (tous les essais des deux bras ont le même résultat).")
    if len(taches) == 1:
        notes.append("Une seule tâche : le résultat ne vaut que pour elle.")
    if verdict == INDISTINGUISHABLE and p < alpha:
        notes.append(f"Cas limite : le test exact ({_p(p)}) rejette l'égalité mais l'intervalle "
                     f"approché contient 0 (ou l'écart est de signe contraire) : non tranché.")

    resultat_controle: ControlResult | None = None
    if control:
        nom_controle = bras[2].name
        d2, lo2, hi2, p2, v2 = _decider(_paires(att_a, tentatives[nom_controle], taches), confidence)
        resultat_controle = ControlResult(nom_controle, d2, lo2, hi2, p2, v2)
        if not resultat_controle.ok:
            notes.append("Contrôle A/A en échec : A et sa propre copie « diffèrent ». La mesure n'est "
                         "pas fiable (état partagé entre tentatives, tentatives non indépendantes, "
                         "biais lié à l'ordre ou au nom d'un bras). Aucun verdict A/B ne vaut tant que "
                         "ce n'est pas élucidé.")

    detectable: float | None = None
    if verdict == INDISTINGUISHABLE and len(taches) * repeats <= 120:
        try:
            detectable = detectable_difference(len(taches), repeats, confidence=confidence, seed=seed)
        except Exception:
            _log.exception("detectable_difference failed")  # fail-open : une aide, pas la mesure

    latence: LatencyComparison | None = None
    try:
        latence = _comparer_durees(att_a, att_b, confidence=confidence, bootstrap=bootstrap, seed=seed)
    except Exception:
        _log.exception("latency comparison failed")  # fail-open : une aide, pas la mesure — jamais perdre des runs payés

    fingerprints = {"suite": _empreinte_suite(taches)}
    for variante in bras:
        fingerprints[variante.name] = sorted(empreintes_vues[variante.name])[0]
    return ComparisonReport(
        a=va.name, b=vb.name, repeats=repeats, confidence=confidence, tasks=par_tache,
        rate_a=sum(x.rate_a for x in par_tache) / len(par_tache),
        rate_b=sum(x.rate_b for x in par_tache) / len(par_tache),
        delta=delta, low=bas, high=haut, p_value=p, p_min=p_min, verdict=verdict,
        reports=reports, fingerprints=fingerprints,
        cost=_comparer_couts(att_a, att_b, confidence=confidence, bootstrap=bootstrap,
                             seed=seed, cost_fn=cost_fn),
        control=resultat_controle, detectable=detectable, notes=notes,
        latency=latence,
    )
