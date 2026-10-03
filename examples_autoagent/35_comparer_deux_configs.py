"""35 - Comparer deux configurations sans se raconter d'histoires.

Tu changes un outil, un prompt, un modele ; un premier essai « montre un gain » ; tu
publies. Or relancer la MEME configuration donne deja des resultats differents : environ
54 % de la variance des resultats d'agents vient du simple re-lancement, pas du
changement de configuration (Wiedmann et al., arXiv:2610.01618, preprint d'octobre
2026). Un ecart sous ce bruit n'est pas une amelioration, c'est un tirage.

    compare_configs(a, b, taches, repeats=5, control=True)

Ce que la demo montre, sur un vrai modele :

  1. quatre taches dont la reponse n'est connue que d'un OUTIL (le modele ne peut pas la
     deviner), et deux configurations : l'outil sain, et le meme outil qui se trompe
     60 % du temps. Les bras ALTERNENT a chaque repetition : une derive du fournisseur
     ou un cache chaud touche les deux ;
  2. le verdict : un test exact par permutation ET l'intervalle de Wilson-Newcombe
     doivent s'accorder. Sinon : « indistinguables ». L'intervalle seul, mesure sur des
     configurations IDENTIQUES, declarait un gagnant jusqu'a 14,6 % du temps (au lieu
     de 5 %) : voir tests/test_compare_calibration.py ;
  3. le controle A/A : A rejoue contre sa propre copie doit sortir « indistinguable » ;
  4. ce que TON plan est capable de voir. « Indistinguable » n'est pas « equivalent » :
     avec peu de taches et peu de repetitions, la mesure est aveugle aux ecarts moyens.

Le juge est du CODE (la valeur exacte est-elle dans la reponse ?), jamais un LLM.

Cout : 4 taches x 5 repetitions x 3 bras = 60 runs, environ 120 appels au modele.

    python examples_autoagent/35_comparer_deux_configs.py
"""

from __future__ import annotations

import random
import re

from _common import make_provider

from autoagent import Agent, EvalTask, Variant, compare_configs
from autoagent.compare import detectable_difference

SYSTEME = "Tu reponds uniquement par le resultat demande, apres avoir utilise l'outil."

# Valeurs qu'aucun modele ne peut deviner : seul l'outil les connait.
VALEURS = {"C-102": 4821, "R-7": 3517, "billing": 7342, "S-12": 2906}
CONSIGNES = {
    "solde": ("C-102", "Quel est le solde du client C-102 ? Utilise ton outil, reponds par le nombre."),
    "stock": ("R-7", "Combien d'unites en stock pour la reference R-7 ? Utilise ton outil, reponds par le nombre."),
    "quota": ("billing", "Quel est le quota mensuel du service billing ? Utilise ton outil, reponds par le nombre."),
    "seuil": ("S-12", "Quel est le seuil d'alerte de la sonde S-12 ? Utilise ton outil, reponds par le nombre."),
}


def juge(attendu: int):
    cible = str(attendu)
    return lambda res: cible in re.sub(r"\D", "", res.output or "")


TACHES = [EvalTask(nom, consigne, juge(VALEURS[cle])) for nom, (cle, consigne) in CONSIGNES.items()]


def main() -> None:
    provider = make_provider()

    def fabrique(p_erreur: float, graine: int):
        """Un agent NEUF par tentative ; l'outil se trompe avec la probabilite p_erreur."""
        rng = random.Random(graine)

        def construire() -> Agent:
            agent = Agent(provider, system_prompt=SYSTEME, max_steps=4)

            def lire(identifiant: str) -> dict:
                """Lit, dans le systeme d'information, la valeur d'un identifiant
                (client, reference, service ou sonde)."""
                valeur = VALEURS.get(identifiant)
                if valeur is None:
                    return {"erreur": "identifiant inconnu"}
                return {"valeur": valeur + 111 if rng.random() < p_erreur else valeur}

            agent.add_tool(lire)
            return agent

        return construire

    print("Quatre taches, un juge en code, trois bras : l'outil sain (A), le meme outil")
    print("rejoue pour le controle A/A, et l'outil qui se trompe 60 % du temps (B).")
    print("Compte une a deux minutes par bras sur un modele rapide.\n")

    def progression(bras: str, tache: str, tentative) -> None:  # type: ignore[no-untyped-def]
        print(f"  {bras:<26} {tache:<6} essai {tentative.index} : "
              f"{'ok' if tentative.ok else 'rate'}", flush=True)

    # `params` DECLARE ce qui change : l'empreinte lit la structure de l'agent (prompt,
    # schemas d'outils, bornes), pas le code d'un outil. Sans cette ligne, les deux bras
    # auraient la meme empreinte alors que leur outil se comporte differemment.
    rapport = compare_configs(
        Variant("outil sain", fabrique(0.0, 1), params={"p_erreur": 0.0}),
        Variant("outil faux 60 %", fabrique(0.6, 2), params={"p_erreur": 0.6}),
        TACHES, repeats=5, control=True, seed=1, on_attempt=progression,
    )

    print("\n" + "-" * 78)
    print(rapport.summary())
    print("\nPar tache (A = sain, B = faux 60 %) :")
    for x in rapport.tasks:
        print(f"  {x.name:<6} A {x.successes_a}/{x.n}   B {x.successes_b}/{x.n}   "
              f"ecart {x.delta * 100:+.0f} points")

    print("\n" + "-" * 78)
    print("CE QUE TON PLAN EST CAPABLE DE VOIR (puissance 80 %, taux de base 50 %)\n")
    print("  taches x repetitions   plus petit ecart detecte")
    for taches, rep in ((2, 5), (4, 5), (4, 10), (8, 10), (12, 10)):
        d = detectable_difference(taches, rep)
        lu = "aucun, meme enorme" if d is None else f"{d * 100:.0f} points"
        print(f"  {taches:>3} x {rep:<3}               {lu}")

    print("\n" + "-" * 78)
    print("CE QU'IL FAUT RETENIR\n")
    print("  * Un verdict n'est rendu que si DEUX calculs s'accordent ; par defaut, la")
    print("    reponse est « indistinguables ». Il faut une preuve pour en sortir.")
    print("  * « Indistinguables » ne veut PAS dire « equivalentes » : la ligne du plan")
    print("    ci-dessus dit l'ecart minimal que cette mesure aurait vu. Avec 4 taches x 5")
    print("    repetitions, un ecart de 30 points passe a travers plus d'une fois sur deux.")
    print("  * Le controle A/A (control=True) coute 50 % d'appels de plus. Il attrape les gros")
    print("    defauts du banc (biais d'ordre ou de bras, etat partage) ; il ne prouve pas que")
    print("    tout va bien. Une derive du fournisseur, elle, touche les bras de la meme facon.")
    print("  * Le resultat vaut pour CES taches. Dire que B est meilleure sur d'autres taches")
    print("    est une autre affirmation, qu'aucun calcul ne remplace.")
    print("  * L'empreinte de la suite et des bras (derniere ligne du rapport) dit DE QUOI")
    print("    parle ce rapport : change le juge, le prompt ou un schema d'outil, elle change.")
    print("    Elle ne lit PAS le code d'un outil : ce qui differe la-dedans, on le DECLARE")
    print("    dans Variant(params=...).")


if __name__ == "__main__":
    main()
