"""40 - Auditer le JUGE avant de croire un banc.

`run_k` et `compare_configs` rendent des chiffres precis, SUIVANT un juge que tu as ecrit.
Si le juge est faux, tous les chiffres le sont - avec la meme assurance, et rien ne le dit.
METR a mesure le meme phenomene sur des correctifs de code : le correcteur automatique est
environ 24 points plus indulgent que la decision des mainteneurs (note du 10 mars 2026,
4 mainteneurs, 296 PR ; limites des auteurs : 3 depots sur 12, « a single benchmark with
one agent harness »).

    audit_check(juge, good=[...], bad=[...])

Ce que la demo montre, hors ligne et sans cle :

  1. un agent FAUX (il repond « 420 lignes » quand la bonne reponse est 42) note par deux
     juges de la meme tache. Le juge par sous-chaine (`"42" in sortie`) dit 100 % de
     reussite ; le juge exact dit 0 %. Les deux ont l'air serieux ;
  2. `audit_check` attrape le juge indulgent AVANT de lui confier le moindre run : deux
     faux positifs sur six negatifs (les presque-bons « 420 » et « 142 »), intervalle
     [10 % ; 70 %] - alors qu'aucun des negatifs triviaux (sortie vide, refus, erreur)
     n'est accepte : ce sont les presque-bons qui le trahissent ;
  3. le juge de la demo 35 (`cible in chiffres(sortie)`) passe l'audit sur des negatifs
     ordinaires et se trahit sur les presque-bons (« 420 », « 142 », « 4 et 2 ») : ce
     sont ces exemples-la qu'un juge indulgent oublie, et qu'il faut lui montrer.

Ce que l'audit ne fait pas : il ne peut que TROUVER des defauts. « Aucun defaut trouve »
sur 21 negatifs borne le taux de faux positifs a environ 15 %, pas a 0 - et il n'est aussi
bon que tes exemples : la verite de terrain est de l'HOTE.

    python examples_autoagent/40_auditer_le_juge.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if __package__ is None:                       # execution directe
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from autoagent import Agent  # noqa: E402
from autoagent.eval import run_k  # noqa: E402
from autoagent.judge import audit_check  # noqa: E402
from autoagent.providers.base import LLMProvider  # noqa: E402
from autoagent.schema import LLMResponse, ModelConfig  # noqa: E402

BONNE_REPONSE = 42
CONSIGNE = "Combien de lignes ERROR dans app.log ? Reponds par le nombre."


class AgentFaux(LLMProvider):
    """Un modele scripte qui se trompe toujours de la meme facon : « 420 » au lieu de 42."""

    def __init__(self) -> None:
        super().__init__(ModelConfig(provider="faux", model="faux-1"))

    def complete(self, request):
        return LLMResponse(model="faux-1", content="Il y a 420 lignes ERROR dans app.log.")


# Le juge INDULGENT : « 42 » est bien une sous-chaine de « 420 ».
def juge_indulgent(res) -> bool:
    return str(BONNE_REPONSE) in res.output


# Le juge EXACT : « 42 » en mot entier (pas colle a d'autres chiffres).
MOT_ENTIER = re.compile(rf"(?<!\d){BONNE_REPONSE}(?!\d)")


def juge_exact(res) -> bool:
    return MOT_ENTIER.search(res.output) is not None


# Le juge de la demo 35 : les chiffres de la sortie, concatenes.
def juge_chiffres(res) -> bool:
    return str(BONNE_REPONSE) in re.sub(r"\D", "", res.output or "")


# La verite de terrain, ECRITE PAR L'HOTE : ce que le juge doit accepter, ce qu'il doit refuser.
BONS = [
    "Il y a 42 lignes ERROR dans app.log.",
    "42",
    "Le fichier contient 42 erreurs.",
]
ORDINAIRES = [
    "Il y a 24 lignes ERROR.",
    "Je n'ai pas pu lire le fichier.",
    "Il y a 17 lignes ERROR.",
]
PIEGES = [
    "Il y a 420 lignes ERROR dans app.log.",      # presque bon : « 42 » est une sous-chaine de « 420 »
    "Il y a 142 lignes ERROR.",                   # « 42 » est une sous-chaine de « 142 »
    "Entre 4 et 2 lignes.",                       # « 4 » puis « 2 » : les chiffres collent en « 42 »
]
MAUVAIS = ORDINAIRES + PIEGES


def main() -> None:
    print("=" * 74)
    print("1) Un agent FAUX, note par deux juges de la meme tache")
    print("=" * 74)
    for nom, juge in (("juge indulgent  (\"42\" in sortie)", juge_indulgent), ("juge exact      (42 en mot entier)", juge_exact)):
        rapport = run_k(lambda: Agent(AgentFaux(), system_prompt="Tu comptes les lignes."), CONSIGNE, k=5, check=juge)
        print(f"  {nom} : {rapport.successes}/5 reussites  ->  pass@1 = {rapport.pass_at_1:.0%}")
    print("  L'agent se trompe a CHAQUE essai. Le juge indulgent le declare parfait.")

    print()
    print("=" * 74)
    print("2) L'audit attrape le juge indulgent AVANT le premier run")
    print("=" * 74)
    print(audit_check(juge_indulgent, good=BONS, bad=MAUVAIS, name="juge indulgent").summary())
    print()
    print(audit_check(juge_exact, good=BONS, bad=MAUVAIS, name="juge exact").summary())

    print()
    print("=" * 74)
    print("3) Le juge de la demo 35, sur des exemples ordinaires puis avec les pieges")
    print("=" * 74)
    ordinaire = audit_check(juge_chiffres, good=BONS, bad=ORDINAIRES, name="juge de la demo 35")
    print(f"  sur des negatifs ordinaires -> {ordinaire.verdict} (faux positifs : {ordinaire.false_positives})")
    piege = audit_check(juge_chiffres, good=BONS, bad=ORDINAIRES + PIEGES, name="juge de la demo 35 + pieges")
    noms = [PIEGES[i - len(ORDINAIRES)] for i in piege.false_positives]
    print(f"  avec les presque-bons ajoutes -> {piege.verdict}")
    for nom in noms:
        print(f"      accepte a tort : « {nom} »")
    print("  Un juge qui lit « les chiffres de la sortie » accepte tout ce qui contient la bonne suite")
    print("  de chiffres, collee ou eclatee. Il passe l'audit tant qu'on ne lui montre que des cas faciles.")
    print("\nLecon : un juge est du code, donc il a des bugs - on l'audite comme le reste.")


if __name__ == "__main__":
    main()
