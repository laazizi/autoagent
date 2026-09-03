"""Synthèse d'outil par l'exemple : le modèle propose, TES cas tranchent (0.21.0).

    entrée connue + sortie connue  →  l'agent écrit un outil  →  tes cas décident
                                       ↑                              │
                                       └── 2-3 cas ratés en retour ───┘

`DynamicToolBuilder` sait déjà faire écrire un outil par le modèle, le passer à
l'analyse AST, l'exécuter en bac à sable et lancer les auto-tests que le modèle
a fournis. Ce qui manquait tenait en une phrase : **le modèle écrivait les tests
qui le jugeaient.** Il ne triche pas — il se trompe deux fois de la même façon.

Ce module inverse qui juge. L'hôte apporte des exemples (arguments → résultat
attendu). Le modèle écrit un outil. Le code exécute les exemples dans le bac à
sable et compare. Un outil qui rate est jeté et le modèle reçoit quelques cas
ratés pour corriger ; un outil qui passe est enregistré — et entre dans le même
circuit de promotion par empreinte que tout outil dynamique.

LA RÈGLE QUI FAIT TOUT : les exemples sont COUPÉS EN DEUX.

  * les cas MONTRÉS servent au modèle pour écrire et corriger ;
  * les cas CACHÉS (`holdout`) ne lui sont JAMAIS transmis — ni dans la
    demande, ni dans les retours d'échec, même pas leur contenu quand ils
    ratent. On lui dit « N cas que tu n'as pas vus échouent », rien de plus.

Sans cette coupure, un modèle à qui l'on demande de faire passer des cas écrit
un outil qui les traite UN PAR UN (`if entree == …`) : 100 % de réussite, 0 %
d'utilité. Ce n'est pas de la malhonnêteté, c'est ce qu'on lui a demandé. La
coupure transforme « fais passer ces cas » en « trouve la règle ».

Ce que la boucle NE FAIT PAS, et pourquoi :

  * elle ne rend pas le modèle plus intelligent — elle convertit des essais en
    justesse, ce qui n'est possible que quand la vérité est déjà connue ;
  * elle ne vaut rien sans exemples de qualité : un jeu d'exemples faux produit
    un outil faux qui passe ;
  * elle ne remplace pas la promotion humaine : l'outil accepté reste un outil
    dynamique, en bac à sable, tant qu'une personne ne l'a pas promu.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .dynamic import DynamicToolBuilder, ToolBuildRequest
from .errors import ToolError, ToolValidationError
from .sandbox import GeneratedPythonTool
from .schema import LLMRequest, LLMResponse, TokenUsage

if TYPE_CHECKING:
    from .agent import Agent

__all__ = ["Attempt", "Example", "SynthesisResult", "synthesize_tool"]


@dataclass(frozen=True)
class Example:
    """Un cas de vérité : ces arguments doivent produire exactement ce résultat."""

    args: dict[str, Any]
    expected: Any


@dataclass
class Attempt:
    """Ce qui s'est passé à UN essai — le journal que l'hôte relit pour décider."""

    index: int
    tool_name: str | None = None
    shown_passed: int = 0
    shown_total: int = 0
    holdout_passed: int | None = None      # None = pas évalué (les montrés ont raté)
    holdout_total: int = 0
    error: str | None = None               # échec de construction (JSON, AST, auto-test…)
    accepted: bool = False


@dataclass
class SynthesisResult:
    tool: GeneratedPythonTool | None
    attempts: list[Attempt] = field(default_factory=list)
    usage: TokenUsage | None = None
    shown: int = 0
    holdout: int = 0

    @property
    def accepted(self) -> bool:
        return self.tool is not None

    def summary(self) -> str:
        """Une ligne honnête, pour un journal ou un rapport."""
        if self.accepted:
            return (f"accepté à l'essai {len(self.attempts)} : {self.tool.spec.name} "  # type: ignore[union-attr]
                    f"({self.shown} montrés + {self.holdout} cachés, tous passés)")
        return f"refusé après {len(self.attempts)} essais ({self.shown} montrés, {self.holdout} cachés)"


class _UsageTap:
    """Un fournisseur qui compte : `DynamicToolBuilder.build` jette l'usage de
    ses appels, et on veut que la synthèse rapporte ce qu'elle a coûté."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.config = getattr(inner, "config", None)
        self.entree = 0
        self.sortie = 0
        self.vu = False

    def complete(self, request: LLMRequest) -> LLMResponse:
        reponse: LLMResponse = self._inner.complete(request)
        if reponse.usage is not None:
            self.vu = True
            self.entree += reponse.usage.input_tokens or 0
            self.sortie += reponse.usage.output_tokens or 0
        return reponse

    def __getattr__(self, name: str) -> Any:            # stream(), etc.
        return getattr(self._inner, name)


def _json_equal(a: Any, b: Any) -> bool:
    """Égalité au sens JSON : `{"a":1,"b":2}` == `{"b":2,"a":1}`, `1.0 == 1`,
    mais `True != 1`. Le résultat revient du bac à sable après un aller-retour
    JSON, donc c'est la seule comparaison qui a un sens."""
    try:
        return bool(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))
    except (TypeError, ValueError):
        return bool(a == b)


def _run_cases(tool: GeneratedPythonTool, cases: list[Example]) -> list[tuple[Example, Any]]:
    """Exécute les cas dans le bac à sable ; rend ceux qui RATENT avec ce qu'on a
    obtenu (ou l'erreur), dans l'ordre."""
    rates: list[tuple[Example, Any]] = []
    for ex in cases:
        try:
            obtenu = tool(**ex.args)
        except ToolError as exc:
            rates.append((ex, f"error: {exc}"))
            continue
        if not _json_equal(obtenu, ex.expected):
            rates.append((ex, obtenu))
    return rates


def _split(examples: list[Example], holdout: float, seed: int) -> tuple[list[Example], list[Example]]:
    """Coupure DÉTERMINISTE (graine) : rejouable, et le même jeu donne la même
    coupure d'un run à l'autre — sinon un essai « réussi » ne serait pas
    comparable au suivant."""
    if len(examples) < 2:
        return list(examples), []
    melange = list(examples)
    random.Random(seed).shuffle(melange)
    n_cache = max(1, round(len(melange) * holdout)) if holdout > 0 else 0
    n_cache = min(n_cache, len(melange) - 1)         # au moins UN cas montré
    return melange[n_cache:], melange[:n_cache]


def _capability(goal: str, shown: list[Example], feedback: str | None) -> str:
    """La demande faite au modèle. Les cas montrés y sont ; les cachés JAMAIS."""
    lignes = [
        goal.strip(),
        "",
        "The tool MUST return exactly the expected value for each example below. "
        "Find the general rule — the tool will also be checked on cases you are NOT shown, "
        "so hard-coding these examples will fail.",
        "",
        "Examples (args -> expected):",
    ]
    for ex in shown:
        lignes.append(f"  {json.dumps(ex.args, ensure_ascii=False)} -> "
                      f"{json.dumps(ex.expected, ensure_ascii=False)}")
    if feedback:
        lignes += ["", "Your previous attempt was rejected:", feedback]
    return "\n".join(lignes)


def _feedback_from_shown(rates: list[tuple[Example, Any]], k: int) -> str:
    lignes = [f"{len(rates)} of the shown examples failed. First failures:"]
    for ex, obtenu in rates[:k]:
        lignes.append(f"  args={json.dumps(ex.args, ensure_ascii=False)}  "
                      f"expected={json.dumps(ex.expected, ensure_ascii=False)}  "
                      f"got={json.dumps(obtenu, ensure_ascii=False, default=str)}")
    return "\n".join(lignes)


def synthesize_tool(
    builder: DynamicToolBuilder,
    goal: str,
    examples: list[Example] | list[tuple[dict[str, Any], Any]],
    *,
    tool_name: str | None = None,
    input_schema: dict[str, Any] | None = None,
    holdout: float = 0.4,
    max_attempts: int = 5,
    feedback_cases: int = 3,
    seed: int = 0,
    register_on: Agent | None = None,
) -> SynthesisResult:
    """Fait écrire un outil au modèle jusqu'à ce qu'il passe TES exemples.

    Args:
        builder: le `DynamicToolBuilder` (bac à sable, dossier d'outils, AST).
        goal: ce que l'outil doit faire, en clair.
        examples: la vérité — `Example(args, expected)` ou `(args, expected)`.
        holdout: part des exemples CACHÉE au modèle (0.4 = 40 %). Au moins un
            cas reste montré ; avec un seul exemple, rien n'est caché — et
            `SynthesisResult.holdout == 0` te le dit.
        max_attempts: borne dure du nombre d'essais.
        feedback_cases: combien de cas MONTRÉS ratés on renvoie au modèle.
        seed: graine de la coupure — même jeu, même coupure.
        register_on: si fourni, l'outil accepté est enregistré sur cet agent,
            par le même chemin qu'un outil dynamique ordinaire.

    Un essai est ACCEPTÉ si tous les cas montrés ET tous les cas cachés passent.
    Quand seuls des cas cachés ratent, le modèle apprend combien, jamais
    lesquels : c'est ce qui l'empêche d'apprendre le jeu par cœur.

    Un outil refusé est supprimé du disque : rien de non validé ne reste
    chargeable dans `tools_dir`.
    """
    exs = [e if isinstance(e, Example) else Example(dict(e[0]), e[1]) for e in examples]
    if not exs:
        raise ValueError("synthesize_tool attend au moins un exemple")
    shown, hidden = _split(exs, holdout, seed)

    tap = _UsageTap(builder.provider)
    builder.provider = tap  # type: ignore[assignment]  # proxy, remis en place à la fin
    resultat = SynthesisResult(tool=None, shown=len(shown), holdout=len(hidden))
    feedback: str | None = None
    try:
        for index in range(1, max_attempts + 1):
            essai = Attempt(index=index, shown_total=len(shown), holdout_total=len(hidden))
            resultat.attempts.append(essai)
            try:
                tool = builder.build(ToolBuildRequest(
                    capability=_capability(goal, shown, feedback),
                    tool_name=tool_name, input_schema=input_schema))
            except ToolValidationError as exc:
                essai.error = str(exc)
                feedback = f"The tool could not be built: {exc}"
                continue
            essai.tool_name = tool.spec.name

            rates_montres = _run_cases(tool, shown)
            essai.shown_passed = len(shown) - len(rates_montres)
            if rates_montres:
                feedback = _feedback_from_shown(rates_montres, feedback_cases)
                _discard(tool)
                continue

            rates_caches = _run_cases(tool, hidden)
            essai.holdout_passed = len(hidden) - len(rates_caches)
            if rates_caches:
                # JAMAIS le contenu : seulement le compte.
                feedback = (f"All shown examples pass, but {len(rates_caches)} of "
                            f"{len(hidden)} UNSEEN cases fail. Your rule is too specific "
                            f"to the examples — generalise it.")
                _discard(tool)
                continue

            essai.accepted = True
            resultat.tool = tool
            if register_on is not None:
                register_on.registry.replace(tool.spec, tool)
            break
    finally:
        builder.provider = tap._inner
        if tap.vu:
            resultat.usage = TokenUsage(input_tokens=tap.entree, output_tokens=tap.sortie)
    return resultat


def _discard(tool: GeneratedPythonTool) -> None:
    """Un outil refusé ne doit pas rester chargeable : on le retire du disque —
    son bytecode aussi, sinon un essai suivant de même nom, même taille et même
    seconde exécuterait le code REFUSÉ (vu en CI)."""
    try:
        tool.file_path.unlink(missing_ok=True)
        cache = tool.file_path.parent / "__pycache__"
        if cache.is_dir():
            for pyc in cache.glob(f"{tool.file_path.stem}.*.pyc"):
                pyc.unlink(missing_ok=True)
    except OSError:
        pass
