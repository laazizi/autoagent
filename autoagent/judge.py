"""Auditer le JUGE avant de croire un banc (0.24.0).

`run_k` et `compare_configs` rendent des chiffres précis, SUIVANT un juge que l'hôte a écrit. Si le juge est
faux — un `"42" in sortie` qui accepte « 420 lignes » —, tous les chiffres le sont, avec la même assurance, et
rien ne le dit. METR l'a mesuré sur des correctifs de code : le correcteur automatique est ≈ 24 points plus
indulgent que la décision des mainteneurs (note du 10 mars 2026 : 4 mainteneurs, 296 PR ; limites des auteurs :
3 dépôts sur 12, « a single benchmark with one agent harness »).

`audit_check` passe le juge sur des exemples dont l'HÔTE connaît la bonne réponse — des `good` qu'il doit
accepter, des `bad` qu'il doit refuser — et rapporte :

* les FAUX POSITIFS (un mauvais exemple accepté : l'indulgence) et les FAUX NÉGATIFS (un bon exemple refusé) ;
* les PLANTAGES (un juge qui lève REFUSE, comme dans `run_k` — mais le message est gardé) ;
* l'INSTABILITÉ (le même exemple, deux verdicts : un juge doit être déterministe) ;
* des NÉGATIFS TRIVIAUX (sortie vide, refus, erreur, réponse tronquée) : un juge qui les accepte est suspect —
  pas condamné : un juge qui vérifie un EFFET peut ignorer la sortie.

Ce que l'audit ne fait pas, et dit : il ne peut que TROUVER des défauts. « Aucun défaut trouvé » sur 21
négatifs borne le taux de faux positifs à ≈ 15 % (borne de Wilson à 95 %), pas à 0. L'audit n'est aussi bon
que les exemples ; il est de l'HÔTE de les écrire — c'est la vérité de terrain, pas celle de la lib.

Exemple::

    from autoagent.judge import audit_check

    audit = audit_check(
        lambda res: "42" in res.output,                  # le juge à auditer
        good=["Il y a 42 lignes ERROR.", "42"],
        bad=["Il y a 420 lignes ERROR.", "Je n'ai pas pu lire le fichier."],
    )
    print(audit.summary())
    # Audit du juge « <lambda> » : 2 exemple(s) à accepter, 2 à refuser, 7 négatif(s) trivial(aux).
    # Faux positifs (mauvais accepté) : 1/2 (50%) · intervalle à 95% [9% ; 91%] — exemples [0]
    # Faux négatifs (bon refusé) : 0/2 (0%) · intervalle à 95% [0% ; 66%]
    # → DÉFAUT TROUVÉ.
    # L'audit ne peut que trouver des défauts, pas prouver que le juge est bon. Avec 1 faux positif sur 2 négatifs, …
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .agent import AgentResult
from .compare import wilson_interval
from .logging import get_logger
from .schema import Message, ToolCall

__all__ = ["JudgeAudit", "ProbeResult", "audit_check", "result_from", "trivial_negatives"]

_log = get_logger("judge")

DEFECT_FOUND = "defect_found"
SUSPICIOUS = "suspicious"
NO_DEFECT_FOUND = "no_defect_found"


def result_from(
    output: str,
    *,
    tool_calls: Iterable[ToolCall] = (),
    finish_reason: str | None = "stop",
    steps: int = 1,
) -> AgentResult:
    """Un `AgentResult` minimal pour écrire un exemple à la main : la sortie, et — si le juge les lit — les
    appels d'outils (dans `messages`, comme dans un vrai run)."""
    return AgentResult(
        output=output,
        messages=[Message(role="assistant", content=output, tool_calls=list(tool_calls))],
        steps=steps,
        finish_reason=finish_reason,
    )


def trivial_negatives() -> list[tuple[str, AgentResult]]:
    """Des réponses qu'aucune tâche ordinaire ne devrait voir acceptées : sortie vide, espaces, refus, « je ne
    sais pas », message d'erreur, réponse coupée (`finish_reason="length"`)."""
    return [
        ("sortie_vide", result_from("")),
        ("espaces", result_from("  \n\t ")),
        ("refus_fr", result_from("Je ne peux pas répondre à cette demande.")),
        ("refus_en", result_from("I'm sorry, I can't help with that.")),
        ("ne_sait_pas", result_from("Je ne sais pas.")),
        ("erreur_d_outil", result_from("Error: the tool failed and returned nothing.")),
        ("tronque", result_from("Voici le résultat que vous avez demandé :", finish_reason="length")),
    ]


def _en_resultat(exemple: Any) -> Any:
    return result_from(exemple) if isinstance(exemple, str) else exemple


@dataclass(frozen=True)
class ProbeResult:
    """Un négatif trivial passé au juge : l'a-t-il accepté ?"""

    label: str
    accepted: bool


@dataclass
class JudgeAudit:
    """Ce que l'audit a trouvé. Les listes d'indices renvoient aux exemples dans l'ordre où l'hôte les a donnés."""

    name: str
    n_good: int
    n_bad: int
    false_negatives: list[int]
    false_positives: list[int]
    errors: list[tuple[str, int, str]]        # (« good » | « bad » | « probe », indice, message)
    unstable: list[tuple[str, int]]
    probes: list[ProbeResult]
    confidence: float = 0.95
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def fp_rate(self) -> float | None:
        return len(self.false_positives) / self.n_bad if self.n_bad else None

    @property
    def fn_rate(self) -> float | None:
        return len(self.false_negatives) / self.n_good if self.n_good else None

    @property
    def fp_interval(self) -> tuple[float, float] | None:
        """Ce que l'échantillon permet de dire du taux de faux positifs — large sur peu d'exemples, et c'est le but."""
        return wilson_interval(len(self.false_positives), self.n_bad, self.confidence) if self.n_bad else None

    @property
    def fn_interval(self) -> tuple[float, float] | None:
        return wilson_interval(len(self.false_negatives), self.n_good, self.confidence) if self.n_good else None

    @property
    def verdict(self) -> str:
        """`defect_found`, `suspicious` ou `no_defect_found` — jamais « sain » : l'audit ne prouve rien."""
        if self.false_positives or self.false_negatives or self.unstable:
            return DEFECT_FOUND
        if self.errors or any(p.accepted for p in self.probes):
            return SUSPICIOUS
        return NO_DEFECT_FOUND

    def summary(self) -> str:
        lignes = [f"Audit du juge « {self.name} » : {self.n_good} exemple(s) à accepter, {self.n_bad} à refuser, "
                  f"{len(self.probes)} négatif(s) trivial(aux)."]

        def taux(n: int, total: int, ivl: tuple[float, float] | None) -> str:
            if not total or ivl is None:
                return "non mesuré (aucun exemple)"
            return f"{n}/{total} ({n / total:.0%}) · intervalle à {self.confidence:.0%} [{ivl[0]:.0%} ; {ivl[1]:.0%}]"

        lignes.append("Faux positifs (mauvais accepté) : " + taux(len(self.false_positives), self.n_bad, self.fp_interval)
                      + (f" — exemples {self.false_positives}" if self.false_positives else ""))
        lignes.append("Faux négatifs (bon refusé) : " + taux(len(self.false_negatives), self.n_good, self.fn_interval)
                      + (f" — exemples {self.false_negatives}" if self.false_negatives else ""))
        acceptes = [p.label for p in self.probes if p.accepted]
        if acceptes:
            lignes.append(f"Négatifs triviaux ACCEPTÉS : {', '.join(acceptes)}.")
        if self.errors:
            cote, i, message = self.errors[0]
            lignes.append(f"Plantages : {len(self.errors)} (compté(s) comme refus). Premier : {cote} #{i} — {message[:120]}")
        if self.unstable:
            lignes.append(f"INSTABLE : {len(self.unstable)} exemple(s) ont reçu deux verdicts différents "
                          "(un juge doit être déterministe).")
        titre = {DEFECT_FOUND: "→ DÉFAUT TROUVÉ", SUSPICIOUS: "→ SUSPECT",
                 NO_DEFECT_FOUND: "→ AUCUN DÉFAUT TROUVÉ"}[self.verdict]
        lignes.append(titre + ".")
        borne = ""
        if self.fp_interval is not None:
            fp, n = len(self.false_positives), self.n_bad
            borne = (f" Avec {fp} faux positif{'s' if fp > 1 else ''} sur {n} négatif{'s' if n > 1 else ''}, le taux réel "
                     f"peut aller jusqu'à {self.fp_interval[1]:.0%}.")
        lignes.append("L'audit ne peut que trouver des défauts, pas prouver que le juge est bon." + borne)
        return "\n".join(lignes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "verdict": self.verdict, "n_good": self.n_good, "n_bad": self.n_bad,
            "false_positives": list(self.false_positives), "false_negatives": list(self.false_negatives),
            "fp_rate": self.fp_rate, "fn_rate": self.fn_rate,
            "fp_interval": None if self.fp_interval is None else list(self.fp_interval),
            "fn_interval": None if self.fn_interval is None else list(self.fn_interval),
            "errors": [list(e) for e in self.errors], "unstable": [list(u) for u in self.unstable],
            "probes": [{"label": p.label, "accepted": p.accepted} for p in self.probes],
            "confidence": self.confidence,
        }


def _verdict_du_juge(check: Callable[[Any], Any], exemple: Any, cote: str, indice: int,
                     stabilite: int, erreurs: list[tuple[str, int, str]],
                     instables: list[tuple[str, int]]) -> bool:
    """Le verdict du juge sur UN exemple : le premier ; un plantage REFUSE (comme `run_k`) et est noté ; si les
    évaluations suivantes divergent du premier verdict, l'exemple est instable."""
    verdicts: list[bool] = []
    for _ in range(max(1, stabilite)):
        try:
            verdicts.append(bool(check(exemple)))
        except Exception as exc:
            verdicts.append(False)
            if not any(c == cote and i == indice for c, i, _m in erreurs):
                erreurs.append((cote, indice, f"{type(exc).__name__}: {exc}"))
    if len(set(verdicts)) > 1:
        instables.append((cote, indice))
    return verdicts[0]


def audit_check(
    check: Callable[[Any], Any],
    *,
    good: Sequence[Any],
    bad: Sequence[Any] = (),
    probes: bool = True,
    stability: int = 2,
    confidence: float = 0.95,
    name: str | None = None,
) -> JudgeAudit:
    """Audite `check` — le prédicat `(AgentResult) -> bool` que tu donnerais à `run_k` / `compare_configs`.

    Args:
        check: le juge à auditer.
        good: des exemples que le juge DOIT accepter. Chacun est un `AgentResult` (ou tout objet que le juge
            sait lire) ou une simple chaîne — la sortie, enveloppée par `result_from`.
        bad: des exemples que le juge DOIT refuser. Les plus utiles sont les PRESQUE-bons (« 420 » quand on
            attend « 42 ») : c'est là que les juges indulgents se trahissent.
        probes: ajoute les `trivial_negatives()` (sortie vide, refus, erreur, réponse tronquée) — un juge qui les
            accepte devient SUSPECT. Sans exemples `bad`, elles sont les seuls négatifs : à ne pas couper.
        stability: nombre d'évaluations par exemple (2 par défaut) ; deux verdicts différents sur le même exemple
            = juge instable. Un juge qui appelle un LLM coûte `stability` fois plus : le mettre à 1 le désactive.
        confidence: niveau des intervalles de Wilson sur les taux de faux positifs / faux négatifs.
    """
    if not callable(check):
        raise TypeError("check must be a callable (AgentResult) -> bool")
    if isinstance(good, (str, bytes)) or isinstance(bad, (str, bytes)):
        raise TypeError("good et bad sont des LISTES d'exemples : une chaîne seule serait lue caractère par caractère "
                        "(écris good=[\"42\"], pas good=\"42\")")
    if isinstance(stability, bool) or not isinstance(stability, int) or stability < 1:
        raise ValueError("stability doit être un entier >= 1 (1 désactive la mesure de stabilité)")
    if not 0 < confidence < 1:
        raise ValueError("confidence doit être dans ]0 ; 1[ (0,95 par défaut)")
    bons, mauvais = list(good), list(bad)
    if not bons:
        raise ValueError("il faut au moins un exemple positif (good) : un juge n'est audité que contre ce qu'il doit accepter ET refuser")
    if not mauvais and not probes:
        raise ValueError("il faut au moins un exemple négatif (bad=…) ou probes=True : un juge audité sur des "
                         "positifs seuls ne peut pas se tromper de la façon qui compte (l'indulgence)")
    nom = name or getattr(check, "__name__", None) or "juge"
    erreurs: list[tuple[str, int, str]] = []
    instables: list[tuple[str, int]] = []

    faux_negatifs = [i for i, ex in enumerate(bons)
                     if not _verdict_du_juge(check, _en_resultat(ex), "good", i, stability, erreurs, instables)]
    faux_positifs = [i for i, ex in enumerate(mauvais)
                     if _verdict_du_juge(check, _en_resultat(ex), "bad", i, stability, erreurs, instables)]
    sondes: list[ProbeResult] = []
    if probes:
        for i, (label, res) in enumerate(trivial_negatives()):
            sondes.append(ProbeResult(label, _verdict_du_juge(check, res, "probe", i, stability, erreurs, instables)))
    return JudgeAudit(
        name=nom, n_good=len(bons), n_bad=len(mauvais), false_negatives=faux_negatifs,
        false_positives=faux_positifs, errors=erreurs, unstable=instables, probes=sondes,
        confidence=confidence,
        examples={"good": [str(getattr(e, "output", e))[:80] for e in bons],
                  "bad": [str(getattr(e, "output", e))[:80] for e in mauvais]},
    )
