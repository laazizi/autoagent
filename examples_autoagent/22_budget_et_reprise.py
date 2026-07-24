"""22 — Maîtriser la dépense : budget dur, arrêt net, reprise sans rien perdre.

Deux niveaux de contrôle du coût, prouvés en réel :

  1. `Agent(token_budget=N)` — plafond par run. Vérifié AVANT chaque appel
     LLM : dès que le cumulé ATTEINT N, l'appel suivant n'est pas émis et la
     boucle lève `TokenBudgetExceeded`. Honnête : l'appel qui franchit la
     ligne se termine, donc tu peux dépasser N du coût du DERNIER appel —
     jamais plus (pas d'emballement possible).

  2. `exc.state` — l'exception porte un snapshot REPRENABLE. « Budget épuisé »
     n'est donc pas un crash qui jette le travail : tu vois ce qui a été
     dépensé, et tu DÉCIDES — arrêter, ou relever le budget et reprendre
     exactement où ça s'était arrêté (`agent.resume`). Le coût reste sous
     contrôle, sans perdre les étapes déjà payées.

Bonus (niveau session) : un plafond GLOBAL en euros, en simple code hôte —
on cumule `result.usage` et on arrête de lancer des runs quand le budget est
atteint. La lib borne le run ; l'hôte borne la campagne.

    python examples_autoagent/22_budget_et_reprise.py
"""

from _common import make_provider

from autoagent import Agent, TokenBudgetExceeded


def _agent(budget: int) -> Agent:
    agent = Agent(
        make_provider(),
        max_steps=10,
        token_budget=budget,
        system_prompt="Tu calcules pas à pas avec ton outil, un nombre à la fois.",
    )

    @agent.tool
    def carre(n: int) -> dict:
        """Renvoie le carré d'un entier."""
        return {"carre": n * n}

    return agent


def main() -> None:
    # ── Niveau 1 : plafond dur → arrêt net ──
    print("── Budget serré : l'agent s'arrête dès le plafond atteint ──")
    agent = _agent(budget=600)           # volontairement bas : quelques étapes puis stop
    tache = "Donne le carré de 2, puis 3, puis 4, puis 5, puis 6 — un appel d'outil chacun."

    try:
        resultat = agent.run(tache)
        print(f"terminé dans le budget : {resultat.output.strip()[:80]}")
        print(f"dépensé : {resultat.usage.total_tokens} tokens")
        return
    except TokenBudgetExceeded as exc:
        print(f"🛑 stoppé à {exc.spent} tokens (plafond {agent.token_budget} — "
              "le dernier appel a fini, aucun de plus n'est émis).")
        print("   Le travail déjà fait est dans exc.state (reprenable).")

        # ── Niveau 2 : décider — ici on relève le budget et on REPREND ──
        print("\n── On relève le budget et on reprend où ça s'était arrêté ──")
        agent.token_budget = 8000        # nouveau plafond
        resultat = agent.resume(exc.state)   # NE recommence PAS de zéro
        print(f"repris et terminé : {resultat.output.strip()[:120]}")
        print(f"dépense totale (avant + après reprise) : {resultat.usage.total_tokens} tokens")

    # ── Bonus : plafond GLOBAL en euros sur toute une session ──
    print("\n── Plafond de session en € (code hôte) : on arrête de lancer des runs ──")
    PRIX_PAR_1M = 0.30                    # $/1M tokens du modèle « pas cher » (exemple)
    PLAFOND_EUR = 0.002
    depense_eur = 0.0
    for i in range(1, 6):
        if depense_eur >= PLAFOND_EUR:
            print(f"   plafond de {PLAFOND_EUR}€ atteint → on ne lance PAS le run {i}.")
            break
        a = _agent(budget=5000)
        r = a.run(f"Donne le carré de {i}.")
        cout = (r.usage.total_tokens or 0) / 1_000_000 * PRIX_PAR_1M
        depense_eur += cout
        print(f"   run {i} : {r.usage.total_tokens} tokens (~{cout*100:.4f} cents) "
              f"| cumul session : {depense_eur*100:.3f} cents")


if __name__ == "__main__":
    main()
