"""28 — Élaguer les vieux résultats d'outils : la même tâche, deux fois moins de contexte.

La démo 26 borne la LARGEUR d'un résultat d'outil (`max_tool_result_chars`) :
ce qui entre une fois dans le transcript. Elle ne dit rien de sa DURÉE.

Or un agent n'a pas de mémoire côté fournisseur : à chaque étape, TOUT le
transcript repart. Le résultat de 3 000 caractères lu à l'étape 1 est donc
renvoyé à l'étape 2, puis 3, puis 4 — alors que le modèle en a déjà tiré ce
qu'il voulait. Et il n'est jamais dans le préfixe mis en cache (démo 27),
puisque l'historique change à chaque tour. C'est du plein tarif, à chaque fois.

    Agent(prune_tool_results_after=1)

Au-delà des N plus récents, un résultat d'outil garde son rôle et son
`tool_call_id` — la conversation reste bien formée — mais perd sa charge, dans
la VUE envoyée au fournisseur uniquement. Le transcript rendu à l'hôte, la
trace et les snapshots gardent tout : on économise des jetons, pas des preuves.

Trois points que la démo rend visibles :

  1. LE MARQUEUR DIT QUE LE RÉSULTAT ÉTAIT VALIDE. « Supprimé » tout court fait
     replanifier un modèle autour d'un échec qui n'a pas eu lieu. Il nomme
     l'outil et la taille, et invite à rappeler l'outil si besoin.
  2. LA TEINTE SURVIT. Un résultat untrusted élagué reste encadré : sans ça,
     élaguer « laverait » la teinte et désarmerait la garde trifecta.
  3. ÇA NE GROSSIT JAMAIS. Un résultat plus court que son propre marqueur est
     laissé tel quel — une borne qui coûte du contexte n'est pas une borne.

    python examples_autoagent/28_elagage_contexte.py
"""

from __future__ import annotations

from _common import make_provider

from autoagent import Agent, TraceEmitter

JOURS = (1, 2, 3, 4)

TACHE = (
    "Lis le journal des jours 1, 2, 3 puis 4 — un appel d'outil à la fois. "
    "Réponds ensuite par une seule phrase : le total d'erreurs sur les 4 jours."
)


def journal(jour: int) -> str:
    """Un journal réaliste : beaucoup de lignes, une seule qui compte."""
    lignes = [
        f"2026-08-{jour:02d} 0{h}:{m:02d}:00 INFO  capteur={c} trames=1240 ok"
        for h in range(1, 6) for m in (0, 15, 30, 45) for c in ("A12", "B07")
    ]
    lignes.append(f"2026-08-{jour:02d} 23:59:00 SUMMARY erreurs={jour * 2}")
    return "\n".join(lignes)


def construire(elagage: int | None, trace: TraceEmitter) -> Agent:
    agent = Agent(
        make_provider(),
        max_steps=10,
        trace=trace,
        prune_tool_results_after=elagage,
        system_prompt="Tu utilises tes outils, un appel à la fois, et tu es bref.",
    )

    @agent.tool
    def lire_journal(jour: int) -> str:
        """Renvoie le journal brut d'une journée."""
        return journal(jour)

    return agent


def mesurer(titre: str, elagage: int | None) -> tuple[int, str]:
    economies: list[int] = []

    def on_event(ev) -> None:
        if ev.type == "context_pruned":
            economies.append(ev.payload["chars_saved"])

    with TraceEmitter(on_event=on_event) as trace:
        resultat = construire(elagage, trace).run(TACHE)

    entree = resultat.usage.input_tokens or 0
    print(f"\n  {titre}")
    print(f"    entrée cumulée : {entree} jetons")
    if economies:
        print(f"    élagages       : {len(economies)} tours, "
              f"{sum(economies)} caractères retirés de la vue")
    print(f"    réponse        : {resultat.output.strip()[:90]}")
    return entree, resultat.output.strip()


def main() -> None:
    provider = make_provider()
    taille = len(journal(1))
    print(f"Modèle : {provider.config.provider} / {provider.config.model}")
    print(f"Chaque journal fait {taille} caractères ; l'agent en lit {len(JOURS)}.")
    print("Un seul paramètre change entre les deux runs.")
    print("─" * 76)

    sans, rep_sans = mesurer("SANS élagage — chaque journal repart à chaque étape", None)
    avec, rep_avec = mesurer("AVEC prune_tool_results_after=1", 1)

    print("\n" + "─" * 76)
    print("CE QU'IL FAUT RETENIR\n")
    if sans and avec:
        ecart = sans - avec
        sens = "économisés" if ecart > 0 else "de PLUS"
        print(f"  Entrée cumulée : {sans} jetons sans élagage, {avec} avec.")
        print(f"  Soit {abs(ecart)} jetons {sens} — {abs(ecart) / sans:.0%} de l'entrée.")
        print("  L'économie porte sur l'ENTRÉE, la part qu'on repaie à CHAQUE étape :")
        print("  elle grandit donc avec le nombre d'étapes, pas avec la taille du run.")
    print()
    print("  Les deux réponses doivent dire la même chose ; c'est tout l'enjeu :")
    print(f"    sans : {rep_sans[:60]}")
    print(f"    avec : {rep_avec[:60]}")
    print()
    print("  Si elles divergent, le seuil est trop agressif pour cette tâche —")
    print("  monte-le. Ce que le modèle a encore besoin de RELIRE doit rester ;")
    print("  ce dont il a déjà tiré sa conclusion peut partir. Aucun réglage")
    print("  universel : ça dépend de la tâche, et ça se mesure comme ici.")
    print()
    print("  À noter : le transcript RENDU garde tout. `resultat.messages` porte")
    print("  les journaux complets, la trace aussi. On élague ce qu'on ENVOIE,")
    print("  jamais ce qu'on GARDE.")


if __name__ == "__main__":
    main()
