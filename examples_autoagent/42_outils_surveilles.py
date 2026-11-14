"""42 - Surveiller, mesurer, corriger les outils que le modele a crees : le carnet, la fiche, puis improve.

L'agent ecrit l'outil qui lui manque (demo 06) et la bibliotheque le garde (`persist=True`). Mais que
vaut cet outil a l'usage ? Jusqu'ici on n'avait que des compteurs. Le carnet (`record_calls=N`, 0.28.0)
note chaque appel : ce que l'outil a recu, l'issue, la cause, la duree, la version du code.

Six choses a voir, sans cle (le « modele » qui ecrit les outils est scripte) :

  1. un outil a defaut plante sur une entree qu'il n'attendait pas : le carnet dit SUR QUOI ;
  2. une fonction de l'hote refusee par la politique n'est pas un defaut de l'outil : elle est comptee
     a part et ne le fait pas retirer (avant, deux refus retiraient un outil sain) ;
  3. le carnet garde ce que l'outil recoit : `redact=` masque un nom AVANT toute ecriture ;
  4. la FICHE DE QUALITE (`builder.health(nom)`) : d'ou viennent les tests de l'outil, son taux de defaut
     avec son intervalle, ses erreurs regroupees. Le carnet ne voit ni un resultat faux qui ne plante pas,
     ni un refus legitime : c'est l'HOTE qui juge (`builder.report(nom, id, correct=...)`), pas le modele ;
  5. APPRENDRE de l'erreur (`builder.improve(nom)`) : le modele recoit le vrai cas rate et ecrit une nouvelle
     version ; du CODE la juge - le cas rate doit etre resolu, et les cas qui marchaient, que le modele n'a
     jamais vus, doivent donner le meme resultat. Sinon l'ancienne version reste ;
  6. APPRENDRE D'UN REFUS, sans rien demander a l'hote (`improve(nom, learn_from_refusals=True)`) : un outil
     ecrit par un vrai modele refuse proprement ce qu'il ne sait pas lire, au lieu de planter - et un refus
     que personne n'a juge n'est pas un defaut. Le modele decide si l'entree refusee etait valide ; le code
     prouve que rien de ce qui marchait ne change. Ce qu'il lit en plus est dit « non verifie ».

    python examples_autoagent/42_outils_surveilles.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import _common  # noqa: F401  (chemin vers la lib du depot, console UTF-8)

from autoagent import Agent, DynamicToolBuilder, ToolBuildRequest
from autoagent.errors import ToolError
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, ToolCall

# Ce qu'un modele aurait pu ecrire : il convertit « 1h30 » en minutes, et n'a pas prevu « 90 min ».
CODE_DUREE = (
    "def run(args, context):\n"
    "    heures, minutes = args['duree'].split('h')\n"
    "    return {'minutes': int(heures) * 60 + int(minutes or 0)}\n"
)
# Un outil qui passe par une fonction de l'hote pour lire une fiche.
CODE_FICHE = (
    "def run(args, context):\n"
    "    fiche = context['call_host']('lire_fiche', {'client': args['client']})\n"
    "    return {'client': args['client'], 'ville': fiche['ville']}\n"
)


def _outil(nom: str, description: str, code: str, parametre: str) -> str:
    return json.dumps({
        "tool": {"name": nom, "description": description, "permissions": [],
                 "input_schema": {"type": "object", "properties": {parametre: {"type": "string"}}, "required": [parametre]}},
        "code": code, "self_tests": [],
    })


class Scripte(LLMProvider):
    """Joue le modele : rend les reponses prevues, dans l'ordre. Aucun appel reseau."""

    def __init__(self, reponses: list[Any]) -> None:
        super().__init__(ModelConfig(provider="scripte", model="scripte", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        suivante = self.reponses.pop(0) if self.reponses else "fini"
        return suivante if isinstance(suivante, LLMResponse) else LLMResponse(content=suivante)


def _appel(numero: int, nom: str, **arguments: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"c{numero}", name=nom, arguments=arguments)])


def _afficher(constructeur: DynamicToolBuilder, nom: str) -> None:
    for appel in constructeur.calls(nom):
        vu = appel.get("result") if appel["ok"] else appel.get("error")
        print(f"    {appel['kind']:9s} {appel['seconds']:5.2f} s  {json.dumps(appel.get('args'), ensure_ascii=False):28s} -> "
              f"{str(vu)[:70]}")
    e = constructeur.catalogue()[nom]
    print(f"    compteurs : {e['calls']} appels, {e['errors']} defauts de l'outil, "
          f"{e['unattributed_failures']} echecs non attribues, retire = {e['retired']}")


def partie_1(dossier: Path) -> None:
    print("=" * 78)
    print("1) Un outil a defaut : le carnet dit SUR QUOI il a plante")
    print("=" * 78)
    constructeur = DynamicToolBuilder(Scripte([_outil("convertir_duree", "Convertit 1h30 en minutes.", CODE_DUREE, "duree")]),
                                      tools_dir=dossier, persist=True, record_calls=20)
    outil = constructeur.build(ToolBuildRequest(capability="convertir une duree en minutes"))
    for duree in ("1h30", "2h", "90 min", "0h45"):
        try:
            outil(duree=duree)
        except ToolError:
            pass
    _afficher(constructeur, "convertir_duree")
    rates = constructeur.calls("convertir_duree", failed=True)
    print(f"\n  Les entrees a reprendre : {[a['args']['duree'] for a in rates]}")
    print("  Sans le carnet on savait « 1 erreur » ; avec, on sait laquelle - de quoi corriger l'outil.")


def partie_2(dossier: Path) -> None:
    print()
    print("=" * 78)
    print("2) Une fonction de l'hote refusee par la politique n'est pas un defaut de l'outil")
    print("=" * 78)

    def lire_fiche(client: str) -> dict:
        """La fiche d'un client."""
        return {"ville": "Lyon"}

    DynamicToolBuilder(Scripte([_outil("ville_du_client", "La ville d'un client.", CODE_FICHE, "client")]),
                       tools_dir=dossier, persist=True, host_functions={"lire_fiche": lire_fiche}
                       ).build(ToolBuildRequest(capability="la ville d'un client"))

    constructeur = DynamicToolBuilder(Scripte([]), tools_dir=dossier, persist=True, record_calls=20, retire_after_errors=2)
    agent = Agent(Scripte([_appel(1, "ville_du_client", client="C-17"), _appel(2, "ville_du_client", client="C-18"),
                           _appel(3, "ville_du_client", client="C-19")]),
                  max_steps=5,
                  # La politique de l'hote : les fiches ne se lisent pas depuis du code ecrit par le modele.
                  tool_policy=lambda ctx: "fiches interdites aux outils generes" if ctx.call.name == "lire_fiche" else None)
    agent.enable_dynamic_tools(constructeur, host_functions={"lire_fiche": lire_fiche})
    agent.run("Dans quelle ville sont les clients C-17, C-18 et C-19 ?")
    _afficher(constructeur, "ville_du_client")
    print("\n  Trois refus, retire_after_errors=2 : l'outil n'est PAS retire. Le refus vient de la politique de")
    print("  l'hote (issue « host »), pas du code de l'outil. Avant, il aurait ete retire au deuxieme appel.")


def partie_3(dossier: Path) -> None:
    print()
    print("=" * 78)
    print("3) Le carnet garde ce que l'outil recoit : redact= passe avant toute ecriture")
    print("=" * 78)
    code = "def run(args, context):\n    return {'bonjour': args['nom'].upper()}\n"

    def masquer(valeur: Any) -> Any:
        """La pseudonymisation de l'hote (ici : un seul nom, pour l'exemple)."""
        return json.loads(json.dumps(valeur, ensure_ascii=False).replace("Dupont", "<NOM>").replace("DUPONT", "<NOM>"))

    constructeur = DynamicToolBuilder(Scripte([_outil("saluer", "Salue quelqu'un.", code, "nom")]),
                                      tools_dir=dossier, persist=True, record_calls=20, redact=masquer)
    outil = constructeur.build(ToolBuildRequest(capability="saluer"))
    print("  l'outil, lui, voit tout :", outil(nom="Dupont"))
    (appel,) = constructeur.calls("saluer")
    print("  le carnet garde         :", {"args": appel["args"], "result": appel["result"], "redacted": appel["redacted"]})
    sur_disque = (dossier / "saluer.calls.jsonl").read_text(encoding="utf-8")
    print("  « Dupont » sur le disque :", "oui" if "upont" in sur_disque.lower().replace("<nom>", "") else "non")


# Une moyenne qui divise toujours par 2 : juste pour deux valeurs, FAUSSE au-dela - et elle ne plante pas.
CODE_MOYENNE = (
    "def run(args, context):\n"
    "    valeurs = args['valeurs']\n"
    "    if not valeurs:\n"
    "        raise ValueError('liste vide')\n"
    "    return {'moyenne': sum(valeurs) / 2}\n"
)


def partie_4(dossier: Path) -> DynamicToolBuilder:
    print()
    print("=" * 78)
    print("4) La fiche de qualite, et le verdict de l'hote")
    print("=" * 78)
    reponse = json.dumps({
        "tool": {"name": "moyenne", "description": "La moyenne d'une liste de nombres.", "permissions": [],
                 "input_schema": {"type": "object", "required": ["valeurs"],
                                  "properties": {"valeurs": {"type": "array", "items": {"type": "number"}}}}},
        "code": CODE_MOYENNE, "self_tests": [],
    })
    constructeur = DynamicToolBuilder(Scripte([reponse]), tools_dir=dossier, persist=True, record_calls=20)
    outil = constructeur.build(ToolBuildRequest(capability="la moyenne d'une liste"))
    for valeurs in ([10, 30], [10, 20, 30], []):
        try:
            print(f"  moyenne({valeurs}) -> {outil(valeurs=valeurs)}")
        except ToolError as echec:
            print(f"  moyenne({valeurs}) -> ECHEC {str(echec)[:60]}")

    print("\n  La fiche, telle que le code la voit :\n")
    print("  " + constructeur.health("moyenne").summary().replace("\n", "\n  "))
    print("\n  Elle se trompe deux fois : la liste vide est un refus LEGITIME, et moyenne([10, 20, 30]) = 30")
    print("  est FAUX sans avoir plante. Seul l'hote peut le dire :\n")

    juste, faux, refus = constructeur.calls("moyenne")
    constructeur.report("moyenne", refus["id"], correct=True, reason="liste vide : refus attendu")
    constructeur.report("moyenne", faux["id"], correct=False, reason="la moyenne de 10, 20, 30 est 20",
                        expected={"moyenne": 20})
    print("  " + constructeur.health("moyenne").summary().replace("\n", "\n  "))
    print("\n  Le taux est le meme (1 sur 3), mais ce n'est plus le meme appel : le vrai defaut est trouve,")
    print("  avec l'entree qui le montre et le resultat attendu - ce qu'il faut pour corriger l'outil.")
    return constructeur


# Deux « corrections » que le modele pourrait rendre. La premiere repare le cas signale EN DUR et casse le reste.
CODE_MOYENNE_HATIVE = (
    "def run(args, context):\n"
    "    valeurs = args['valeurs']\n"
    "    if len(valeurs) == 3:\n"
    "        return {'moyenne': sum(valeurs) / 3}\n"
    "    return {'moyenne': sum(valeurs)}\n"
)
CODE_MOYENNE_JUSTE = (
    "def run(args, context):\n"
    "    valeurs = args['valeurs']\n"
    "    if not valeurs:\n"
    "        raise ValueError('invalid input: liste vide')\n"
    "    return {'moyenne': sum(valeurs) / len(valeurs)}\n"
)


def partie_5(constructeur: DynamicToolBuilder) -> None:
    print()
    print("=" * 78)
    print("5) Apprendre de l'erreur : une nouvelle version, gardee seulement si du code le prouve")
    print("=" * 78)

    def proposition(code: str) -> str:
        return json.dumps({
            "tool": {"name": "moyenne", "description": "La moyenne d'une liste de nombres.", "permissions": [],
                     "input_schema": {"type": "object", "required": ["valeurs"],
                                      "properties": {"valeurs": {"type": "array", "items": {"type": "number"}}}}},
            "code": code, "self_tests": [],
        })

    constructeur.provider.reponses.extend([proposition(CODE_MOYENNE_HATIVE), proposition(CODE_MOYENNE_JUSTE)])
    print("  Le modele recoit le code actuel et le cas que l'hote a juge faux. Premier essai : il repare")
    print("  ce cas-la en dur, et change un resultat qui etait juste.\n")
    premier = constructeur.improve("moyenne")
    print("  " + premier.summary())
    for cas in premier.regressions:
        print(f"      change : moyenne({cas['args']['valeurs']}) rendait {cas['expected']}, rendrait {cas['got']}")
    print("\n  Second essai :\n")
    second = constructeur.improve("moyenne")
    print("  " + second.summary())
    outil = second.tool
    assert outil is not None
    print(f"\n  moyenne([10, 20, 30]) -> {outil(valeurs=[10, 20, 30])}")
    try:
        outil(valeurs=[])
    except ToolError as refus:
        print(f"  moyenne([]) -> {str(refus)[:50]}")
    print(f"  ... note « {constructeur.calls('moyenne')[-1]['kind']} » au carnet : un refus DECLARE par l'outil")
    print("  (ValueError('invalid input: ...')), plus un plantage. Il ne compte pas contre lui.")
    print("\n  " + constructeur.health("moyenne").summary().replace("\n", "\n  "))
    print("\n  Le modele n'a vu que le cas rate ; le cas qui marchait a ete rejoue sans qu'il le voie.")
    print("  L'ancienne version est gardee : builder.rollback('moyenne') la remet.")




# Un outil comme un vrai modele l'ecrit : il REFUSE proprement ce qu'il ne sait pas lire.
CODE_POURCENT = (
    "def run(args, context):\n"
    "    t = args['texte']\n"
    "    if not t.endswith('%'):\n"
    "        raise ValueError('invalid input: pas un pourcentage')\n"
    "    return float(t[:-1]) / 100\n"
)
# La version que le modele rend quand on lui montre ce que l'outil a refuse.
CODE_POURCENT_ETENDU = (
    "def run(args, context):\n"
    "    t = args['texte'].lower().replace('pour cent', '%').replace(' ', '')\n"
    "    if not t.endswith('%'):\n"
    "        raise ValueError('invalid input: pas un pourcentage')\n"
    "    try:\n"
    "        return float(t[:-1].replace(',', '.')) / 100\n"
    "    except ValueError:\n"
    "        raise ValueError('invalid input: pas un pourcentage')\n"
)


def partie_6(dossier: Path) -> None:
    print()
    print("=" * 78)
    print("6) Apprendre d'un REFUS, sans rien demander a l'hote")
    print("=" * 78)
    demande = "Convertit un pourcentage ecrit comme 12% en fraction."
    modele = Scripte([_outil("pourcentage", demande, CODE_POURCENT, "texte"),
                      _outil("pourcentage", demande, CODE_POURCENT_ETENDU, "texte")])
    constructeur = DynamicToolBuilder(modele, tools_dir=dossier, persist=True, record_calls=20)
    outil = constructeur.build(ToolBuildRequest(capability=demande))
    print(f"  pourcentage('12%') -> {outil(texte='12%')}")
    for texte in ("80 pour cent", "???"):
        try:
            outil(texte=texte)
        except ToolError as refus:
            print(f"  pourcentage({texte!r}) -> REFUSE ({str(refus)[:49]})")
    print("\n  L'outil ne plante pas : il refuse. Personne ne dit si c'etait a raison. Pour `improve`, il n'y a")
    print("  donc rien a corriger :\n")
    print("  " + constructeur.improve("pourcentage").summary())
    print("\n  Avec `learn_from_refusals=True`, le modele voit les deux entrees refusees, et decide :\n")
    resultat = constructeur.improve("pourcentage", learn_from_refusals=True)
    print("  " + resultat.summary())
    for cas in resultat.extended_cases:
        print(f"      lue maintenant : {cas['args']['texte']!r} -> {cas['got']}   (a relire : personne ne l'a verifiee)")
    print(f"      encore refusee : {resultat.still_refused} entree (« ??? »)")
    print("\n  Ce que le code a prouve : le cas reussi rend le meme resultat, et aucun refus n'est devenu un")
    print("  plantage. Ce qu'il n'a PAS prouve : que 0.8 est la bonne lecture de « 80 pour cent ». La fiche le dit :\n")
    print("  " + constructeur.health("pourcentage").summary().splitlines()[0])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        partie_1(Path(tmp) / "a")
        partie_2(Path(tmp) / "b")
        partie_3(Path(tmp) / "c")
        partie_5(partie_4(Path(tmp) / "d"))
        partie_6(Path(tmp) / "e")
    print("\n  Limites : sans verdict de l'hote, un resultat faux reste « ok » - et il compte meme comme un cas")
    print("  REUSSI a ne pas changer. improve ne s'applique pas a un outil qui appelle des fonctions de l'hote")
    print("  (le rejouer referait des effets de bord). « Meme resultat qu'avant » prouve qu'on n'a rien casse,")
    print("  pas que l'outil est juste partout. Et une entree lue grace a `learn_from_refusals` n'est PAS verifiee :")
    print("  c'est le modele qui l'a jugee valide - le niveau de preuve d'une premiere construction, pas plus.")
