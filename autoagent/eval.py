"""Mesure de FIABILITÉ d'un agent — `pass^k` (0.18.0).

Un agent noté « 90 % » ne dit presque rien à un exploitant : `pass@1` mesure
*une* tentative. Ce qui compte en production, c'est `pass^k` — la probabilité que
**k** tentatives réussissent TOUTES. Comme `pass^k ≈ p^k`, la courbe s'effondre :
90 % de `pass@1` donne **43 % à k=8** (0,9^8). Un agent qui « marche » à la démo
échoue donc PLUS d'une fois sur deux sur une journée de huit tâches.

Ce module ne fait pas de magie : il exécute la même tâche k fois, demande à un
prédicat de l'HÔTE si le résultat est bon, et rapporte la dispersion. Deux choix
délibérés :

* **Le juge est déterministe, fourni par l'hôte.** Pas de LLM-as-judge : sur les
  échecs d'agent, les juges LLM plafonnent sous 55 % de justesse (accord au niveau
  du hasard sur l'évaluation par sous-chaîne). Un vérificateur *sound* — un schéma,
  un diff de fichiers, une commande qui passe — vaut mieux qu'un avis probabiliste.
* **Zéro dépendance, synchrone.** Combinable avec `ReplaySession` : rejouer k fois
  un fixture enregistré donne une non-régression de fiabilité gratuite, sans clé
  API et sans réseau.

Exemple::

    from autoagent.eval import run_k

    rapport = run_k(
        lambda: construire_mon_agent(),           # un agent NEUF par tentative
        "Combien de lignes ERROR dans app.log ?",
        k=8,
        check=lambda res: "42" in res.output,     # prédicat déterministe
    )
    print(rapport.summary())
    # k=8 · pass@1=0.75 (6/8) · pass^8=0.00 (toutes réussies : non)
    # · estimation pass^8=0.10 · étapes 2-5 (méd. 3)
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .logging import get_logger

__all__ = ["Attempt", "ReliabilityReport", "run_k"]

_log = get_logger("eval")


@dataclass
class Attempt:
    """Une tentative : a-t-elle réussi, en combien d'étapes, à quel coût ?"""

    index: int
    ok: bool
    steps: int = 0
    total_tokens: int | None = None
    output: str = ""
    error: str = ""


@dataclass
class ReliabilityReport:
    """Résultat de `run_k` — brut, non lissé, avec les tentatives détaillées."""

    k: int
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def successes(self) -> int:
        return sum(1 for a in self.attempts if a.ok)

    @property
    def pass_at_1(self) -> float:
        """Taux de réussite d'UNE tentative (l'estimateur usuel de p)."""
        return self.successes / len(self.attempts) if self.attempts else 0.0

    @property
    def pass_hat_k(self) -> float:
        """1.0 si les k tentatives ont TOUTES réussi, 0.0 sinon (observé)."""
        return 1.0 if self.attempts and self.successes == len(self.attempts) else 0.0

    @property
    def estimated_pass_hat_k(self) -> float:
        """`p^k` — ce que deviendrait `pass^k` sur un long horizon.

        Estimation, pas une mesure : elle suppose des tentatives indépendantes.
        Elle sert à montrer l'effondrement exponentiel que `pass@1` masque.
        """
        return self.pass_at_1 ** self.k

    @property
    def steps_range(self) -> tuple[int, int]:
        steps = [a.steps for a in self.attempts if a.steps]
        return (min(steps), max(steps)) if steps else (0, 0)

    @property
    def median_steps(self) -> float:
        steps = [a.steps for a in self.attempts if a.steps]
        return statistics.median(steps) if steps else 0.0

    @property
    def errors(self) -> list[str]:
        return [a.error for a in self.attempts if a.error]

    def summary(self) -> str:
        low, high = self.steps_range
        return (
            f"k={self.k} · pass@1={self.pass_at_1:.2f} "
            f"({self.successes}/{len(self.attempts)}) · "
            f"pass^{self.k}={self.pass_hat_k:.2f} "
            f"(toutes réussies : {'oui' if self.pass_hat_k else 'non'}) · "
            f"estimation pass^{self.k}={self.estimated_pass_hat_k:.2f} · "
            f"étapes {low}-{high} (méd. {self.median_steps:g})"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe — pour archiver un rapport à côté d'un fixture de replay."""
        return {
            "k": self.k,
            "successes": self.successes,
            "pass_at_1": self.pass_at_1,
            "pass_hat_k": self.pass_hat_k,
            "estimated_pass_hat_k": self.estimated_pass_hat_k,
            "median_steps": self.median_steps,
            "attempts": [
                {"index": a.index, "ok": a.ok, "steps": a.steps,
                 "total_tokens": a.total_tokens, "error": a.error}
                for a in self.attempts
            ],
        }


def run_k(
    agent_or_factory: Any,
    prompt: str,
    *,
    k: int = 8,
    check: Callable[[Any], bool],
    context: dict[str, Any] | None = None,
    on_attempt: Callable[[Attempt], None] | None = None,
) -> ReliabilityReport:
    """Exécute `prompt` k fois et mesure la fiabilité.

    Args:
        agent_or_factory: un `Agent`, ou un callable SANS argument qui en rend un
            neuf. Préférer la fabrique dès que les outils ont des effets de bord :
            chaque tentative doit partir d'un état propre, sinon on mesure
            l'accumulation, pas la fiabilité.
        prompt: la tâche, identique à chaque tentative.
        k: nombre de tentatives (τ-bench en utilise 8).
        check: prédicat DÉTERMINISTE `(AgentResult) -> bool`. Un `check` qui lève
            compte comme un échec (on n'avale pas silencieusement un juge cassé :
            son message atterrit dans `Attempt.error`).
        context: passé tel quel à `agent.run(context=...)`.
        on_attempt: rappel après chaque tentative (progression, journalisation).

    Aucune parallélisation : un agent peut avoir des effets de bord et l'ordre
    doit rester reproductible. L'hôte qui veut du parallèle le fait lui-même.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not callable(check):
        raise TypeError("check must be a callable (AgentResult) -> bool")

    report = ReliabilityReport(k=k)
    for index in range(1, k + 1):
        agent = agent_or_factory() if callable(agent_or_factory) else agent_or_factory
        attempt = Attempt(index=index, ok=False)
        try:
            result = agent.run(prompt, context=context)
            attempt.steps = getattr(result, "steps", 0)
            usage = getattr(result, "usage", None)
            attempt.total_tokens = getattr(usage, "total_tokens", None)
            attempt.output = getattr(result, "output", "") or ""
            try:
                attempt.ok = bool(check(result))
            except Exception as exc:
                attempt.error = f"check raised: {type(exc).__name__}: {exc}"
        except Exception as exc:
            # Un run qui plante EST un échec de fiabilité, pas une erreur du banc.
            attempt.error = f"{type(exc).__name__}: {exc}"
        report.attempts.append(attempt)
        if on_attempt is not None:
            try:
                on_attempt(attempt)
            except Exception:
                _log.exception("on_attempt callback failed")  # fail-open
    return report
