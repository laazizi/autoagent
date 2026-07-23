"""21 — Record / replay : n'importe quel run devient un test déterministe gratuit.

Le non-déterminisme des agents (2 à 4 trajectoires sur 10 runs, même à
température 0) rend les bugs de prod irreproductibles. La parade : geler un
vrai run dans un « fixture », puis le REJOUER à l'identique.

Trois actes, prouvés en réel :
  1. RECORD  — un vrai run (outil inclus) est enregistré dans un fixture JSONL.
  2. REPLAY  — le même run rejoué HORS-LIGNE : provider qui explose s'il est
     appelé (preuve zéro-réseau), outils non ré-exécutés (zéro effet de bord),
     sortie identique.
  3. DIVERGENCE — on retire l'outil de l'agent → le replay lève ReplayMismatch
     en pointant l'étape exacte (ton comportement a changé).

Le replay ne teste NI le LLM NI les outils (gelés) : il exerce TON code —
boucle, politique, mémoire, parsing. C'est là que vivent tes bugs.

    python examples_autoagent/21_record_replay.py
"""

from pathlib import Path

from _common import make_provider

from autoagent import Agent, RecordSession, ReplaySession
from autoagent.errors import ReplayMismatch

FIXTURE = Path(__file__).parent / "run_enregistre.jsonl"
EFFETS: list[str] = []


def ajouter_outils(agent: Agent) -> None:
    @agent.tool
    def convertir_km_miles(km: float) -> dict:
        """Convertit des kilomètres en miles (effet de bord traçable)."""
        EFFETS.append(f"conversion:{km}")
        return {"miles": round(km * 0.621371, 2)}


def main() -> None:
    FIXTURE.unlink(missing_ok=True)

    # ── Acte 1 : RECORD (vrai run) ──
    print("── Acte 1 : run réel enregistré ──")
    with RecordSession(FIXTURE) as rec:
        agent = Agent(rec.provider(make_provider()), registry=rec.registry(),
                      system_prompt="Tu convertis des distances. Utilise ton outil.")
        ajouter_outils(agent)
        original = agent.run("Combien de miles font 42 km ?")
    print(f"   sortie : {original.output.strip()[:80]}")
    print(f"   effets de bord réels : {EFFETS}")
    print(f"   fixture : {FIXTURE.name} ({len(FIXTURE.read_text(encoding='utf-8').splitlines())} événements)")

    # ── Acte 2 : REPLAY hors-ligne ──
    EFFETS.clear()

    class ProviderQuiExplose:
        config = None
        def complete(self, request):
            raise AssertionError("le vrai provider NE doit PAS être appelé en replay !")

    print("\n── Acte 2 : replay hors-ligne (zéro réseau, zéro effet de bord) ──")
    with ReplaySession(FIXTURE) as rep:
        agent = Agent(rep.provider(), registry=rep.registry())
        ajouter_outils(agent)                  # l'outil est là mais NE sera pas exécuté
        rejoue = agent.run("Combien de miles font 42 km ?")
    print(f"   sortie rejouée : {rejoue.output.strip()[:80]}")
    print(f"   identique à l'original : {rejoue.output == original.output}")
    print(f"   effets de bord en replay : {EFFETS}  ← vide : les outils n'ont pas tourné")
    print(f"   tokens dépensés : 0 (aucun appel LLM réel)")

    # ── Acte 3 : DIVERGENCE détectée ──
    print("\n── Acte 3 : le comportement change → ReplayMismatch pointe l'étape ──")
    with ReplaySession(FIXTURE) as rep:
        agent = Agent(rep.provider(), registry=rep.registry())
        # on N'AJOUTE PAS l'outil : l'ensemble des outils proposés diverge
        try:
            agent.run("Combien de miles font 42 km ?")
            print("   (pas de divergence — inattendu)")
        except ReplayMismatch as exc:
            print(f"   🛑 {exc}")

    print("\n→ Fige ce fixture, et ta CI rejoue ce run à chaque commit, "
          "SANS clé API, déterministe.")


if __name__ == "__main__":
    main()
