"""30 — Le mode témoin : la garde photographie au lieu de verbaliser.

Un radar qui verbalise dès la première seconde, on ne l'installe pas : on ne
sait pas s'il est bien réglé, et on l'apprend quand les plaintes arrivent.

C'est le problème de toute borne. `max_repeated_tool_calls=2` en production,
c'est un pari : et si un agent légitime avait vraiment besoin d'appeler trois
fois ? Résultat courant — on ne l'active jamais, ou on l'active une fois, ça
casse, on l'éteint pour toujours.

    Agent(..., max_repeated_tool_calls=2, shadow_guards=True)

Le verdict est calculé et TRACÉ (`loop_guard_would_block`), mais pas appliqué.
Rien ne casse. À la fin, `run_end` porte le compte :

    « cette borne aurait refusé N appels »

Une semaine de trafic réel plus tard, tu lis le rapport, tu regardes les cas, et
tu actives en SACHANT. Et comme la lib sait rejouer un run à l'identique
(démo 21), tu peux poser la même question au mois DERNIER — hors ligne, sans
payer un seul appel.

⚠️ Le mode témoin ne PROTÈGE pas. C'est un mode de mesure, pas un défaut sûr.
Et il ne touche jamais à `tool_policy` : la politique de l'hôte est la frontière
de l'hôte, un drapeau de bibliothèque n'a pas à pouvoir l'éteindre.

Aucune clé API : le fournisseur est scripté dans le fichier. Forcer une vraie
boucle avec un vrai modèle serait du théâtre peu fiable — et le sujet ici, c'est
le rapport de la garde, pas le modèle.

    python examples_autoagent/30_mode_temoin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:                       # exécution directe
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Console Windows en cp1252 : sans ça les filets et les accents plantent la
# sortie. Les autres démos héritent ce réglage de `_common`, dont celle-ci se
# passe volontairement (aucune clé, aucun fournisseur réel).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from autoagent import Agent, TraceEmitter                      # noqa: E402
from autoagent.providers.base import LLMProvider               # noqa: E402
from autoagent.schema import (                                 # noqa: E402
    LLMResponse,
    ModelConfig,
    TokenUsage,
    ToolCall,
)

TOURS = 6          # le modèle redemande 6 fois la même chose
BORNE = 2          # on n'en tolère que 2


class ModeleQuiBoucle(LLMProvider):
    """Un modèle scripté qui redemande TOUJOURS le même appel — le défaut que
    la garde anti-boucle existe pour attraper."""

    def __init__(self) -> None:
        super().__init__(ModelConfig(provider="scripte", model="boucleur", api_key="x"))
        self.restants = TOURS

    def complete(self, request):  # type: ignore[no-untyped-def]
        if self.restants <= 0:
            return LLMResponse(content="j'abandonne")
        self.restants -= 1
        return LLMResponse(
            tool_calls=[ToolCall(id=f"c{self.restants}", name="statut_capteur",
                                 arguments={"capteur": "cmp-lyo-07"})],
            usage=TokenUsage(input_tokens=120, output_tokens=8),
        )


def essai(temoin: bool) -> tuple[int, int, list[dict]]:
    """Rend (exécutions réelles, refus observés, détail des cas)."""
    executions = {"n": 0}
    cas: list[dict] = []
    bilan: dict = {}

    def on_event(ev) -> None:
        if ev.type in ("loop_guard_block", "loop_guard_would_block"):
            cas.append({"type": ev.type, **ev.payload})
        elif ev.type == "run_end":
            bilan.update(ev.payload)

    with TraceEmitter(on_event=on_event) as trace:
        agent = Agent(ModeleQuiBoucle(), max_steps=TOURS + 2, trace=trace,
                      max_repeated_tool_calls=BORNE, shadow_guards=temoin)

        @agent.tool
        def statut_capteur(capteur: str) -> dict:
            """Interroge un capteur (ici : l'effet de bord qu'on veut compter)."""
            executions["n"] += 1
            return {"capteur": capteur, "etat": "ok"}

        agent.run("Donne-moi le statut du capteur cmp-lyo-07.")

    return executions["n"], bilan.get("would_block", 0), cas


def main() -> None:
    print(f"Un modèle qui redemande {TOURS} fois le MÊME appel d'outil.")
    print(f"Borne posée : max_repeated_tool_calls={BORNE}.")
    print("─" * 76)

    exe_t, observes, cas_t = essai(temoin=True)
    print("\n  MODE TÉMOIN — shadow_guards=True (le radar photographie)")
    print(f"    outil réellement exécuté : {exe_t} fois")
    print(f"    refus OBSERVÉS           : {observes}")
    print(f"    événements tracés        : {sorted({c['type'] for c in cas_t})}")

    exe_n, _, cas_n = essai(temoin=False)
    print("\n  MODE NORMAL — la borne s'applique (le radar verbalise)")
    print(f"    outil réellement exécuté : {exe_n} fois")
    print(f"    événements tracés        : {sorted({c['type'] for c in cas_n})}")

    print("\n" + "─" * 76)
    print("LE RAPPORT — ce que tu lirais après une semaine de trafic réel\n")
    print(f"  La borne max_repeated_tool_calls={BORNE} aurait refusé "
          f"{observes} appels.")
    for cas in cas_t:
        print(f"    · {cas['name']}(…) — {cas['repeats']}ᵉ fois identique, "
              f"étape {cas['step']}")
    print()
    print("  Tu regardes ces cas. Étaient-ce de vraies boucles, ou un agent")
    print("  légitime qui avait besoin d'insister ? Tu le sais AVANT d'activer,")
    print(f"  et sans avoir rien cassé : l'outil a bien tourné {exe_t} fois.")
    print()
    print("  Une fois la borne activée pour de bon, le même scénario s'arrête")
    print(f"  à {exe_n} exécutions au lieu de {exe_t} — c'est le second run ci-dessus.")

    print("\n" + "─" * 76)
    print("DEUX LIMITES, DITES CLAIREMENT\n")
    print("  1. Le mode témoin ne PROTÈGE pas. Pendant l'observation, la boucle")
    print("     tourne vraiment : elle consomme des jetons et rejoue ses effets")
    print("     de bord. C'est un mode de mesure, jamais un défaut.")
    print()
    print("  2. Il ne touche PAS à `tool_policy`. La politique de l'hôte est la")
    print("     frontière de l'hôte : un drapeau de bibliothèque n'a pas à pouvoir")
    print("     éteindre le code que tu as écrit pour dire non. Seules les gardes")
    print("     INTÉGRÉES (anti-boucle, trifecta) sont observables.")


if __name__ == "__main__":
    main()
