"""29 — Déléguer à plusieurs spécialistes EN MÊME TEMPS.

`as_tool` (démo 08) expose UN spécialiste. Un superviseur qui en consulte trois
les fait donc passer l'un après l'autre : il attend la SOMME des latences.

    delegate_to({"comptage": expert_a, "juridique": expert_b, ...})

Un seul outil, plusieurs demandes en un appel, exécutées ensemble. La démo
mesure les deux formes sur les MÊMES trois questions, avec de vrais appels
réseau — c'est le seul endroit où la différence est visible.

LE CONTRAT QUI REND ÇA SÛR, et qui est tout l'intérêt : l'appel ne rend la main
que lorsque TOUS les spécialistes ont fini. Paralléliser, ce n'est pas rendre le
système asynchrone. `token_budget` est vérifié avant chaque appel LLM sur la
dépense DÉJÀ CONNUE : avec des sous-agents encore en vol, le plafond ne bornerait
plus que ce qui a atterri, et le chiffre manquant n'existerait pas encore — donc
aucune comptabilité ne pourrait le rattraper. On gagne le temps, on ne perd pas
la borne. La démo vérifie que la dépense des trois est bien dans `result.usage`.

Deux pièges traités par la lib, et que la démo rappelle :

  * un même spécialiste reçoit ses demandes EN SÉRIE (un `Agent` ne sert qu'un
    appelant à la fois) ; seuls des spécialistes différents partent ensemble ;
  * l'ordre des réponses suit l'ordre des DEMANDES, pas l'ordre d'arrivée —
    sinon le transcript cesse d'être déterministe et le rejeu casse.

    python examples_autoagent/29_delegation_parallele.py
"""

from __future__ import annotations

import time

from _common import make_provider

from autoagent import Agent, delegate_to

QUESTIONS = {
    "comptage": "En une phrase : à quoi sert un comptage routier directionnel ?",
    "enquete": "En une phrase : qu'est-ce qu'une enquête déplacements ménages ?",
    "capteur": "En une phrase : quelle différence entre boucle magnétique et radar ?",
}


def _specialiste(domaine: str) -> Agent:
    return Agent(
        make_provider(),
        max_steps=2,
        system_prompt=f"Tu es un expert {domaine} en mobilité. Tu réponds en UNE phrase.",
    )


def _mesurer(titre: str, agent: Agent, tache: str) -> tuple[float, int, str]:
    debut = time.monotonic()
    resultat = agent.run(tache)
    duree = time.monotonic() - debut
    appels = sum(len(m.tool_calls or []) for m in resultat.messages)
    jetons = (resultat.usage.total_tokens if resultat.usage else 0) or 0
    print(f"\n  {titre}")
    print(f"    durée         : {duree:5.1f} s")
    print(f"    appels d'outil: {appels}")
    print(f"    jetons du run : {jetons} (spécialistes compris)")
    return duree, jetons, resultat.output.strip()


def main() -> None:
    provider = make_provider()
    print(f"Modèle : {provider.config.provider} / {provider.config.model}")
    print("Trois spécialistes, les mêmes trois questions, deux façons de les poser.")
    print("─" * 76)

    tache = ("Pose à chaque spécialiste la question de son domaine, puis résume "
             "leurs trois réponses en trois lignes.\n"
             + "\n".join(f"- {d} : {q}" for d, q in QUESTIONS.items()))

    # ── Forme A : un outil par spécialiste (l'agent les appelle l'un après l'autre)
    sequentiel = Agent(make_provider(), max_steps=8,
                       system_prompt="Tu interroges tes spécialistes puis tu résumes.")
    for domaine in QUESTIONS:
        sequentiel.add_tool(_specialiste(domaine).as_tool(
            name=f"demander_{domaine}",
            description=f"Pose une question au spécialiste {domaine}."))
    duree_a, jetons_a, _ = _mesurer("A · un outil par spécialiste (as_tool)",
                                    sequentiel, tache)

    # ── Forme B : un seul outil, les demandes partent ensemble
    parallele = Agent(make_provider(), max_steps=8,
                      system_prompt="Tu interroges tes spécialistes puis tu résumes.")
    parallele.add_tool(delegate_to({d: _specialiste(d) for d in QUESTIONS}))
    duree_b, jetons_b, sortie_b = _mesurer("B · un seul outil (delegate_to)",
                                           parallele, tache)

    print("\n" + "─" * 76)
    print("CE QU'IL FAUT RETENIR\n")
    if duree_b < duree_a:
        print(f"  {duree_a:.1f} s puis {duree_b:.1f} s : "
              f"{(1 - duree_b / duree_a):.0%} de temps en moins.")
        print("  Le gain vient des latences qui se RECOUVRENT, pas d'un travail")
        print("  économisé — le nombre de jetons, lui, reste du même ordre :")
        print(f"  {jetons_a} puis {jetons_b}.")
    else:
        print(f"  Pas de gain ici ({duree_a:.1f} s puis {duree_b:.1f} s).")
        print("  Cause la plus probable : le modèle n'a pas groupé ses demandes en")
        print("  UN seul appel — il reste libre de les envoyer une par une. Le gain")
        print("  dépend donc de son choix, pas seulement de l'outil. Relance :")
        print("  la démo mesure, elle ne triche pas.")

    print()
    print("  Et la dépense des spécialistes est DANS le total ci-dessus : elle")
    print("  compte dans `token_budget`. Un plafond qu'on pourrait contourner en")
    print("  déléguant ne serait pas un plafond.")
    print()
    print("  Ce qui n'est PAS fait ici, volontairement : les spécialistes ne se")
    print("  parlent pas entre eux, et rien ne survit à un redémarrage. Ça, c'est")
    print("  un serveur à faire tourner — pas une bibliothèque qu'on lit en entier.")
    print()
    print(f"  Synthèse du superviseur B :\n    {sortie_b[:300]}")


if __name__ == "__main__":
    main()
