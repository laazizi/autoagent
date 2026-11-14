"""43 - Le banc « outils qui cassent » : ce que vaut `improve`, mesure au lieu d'etre affirme.

Un outil cree par le modele marche sur ses premieres entrees ; puis d'autres ecritures arrivent,
et il echoue. Que faire ? Le banc (`autoagent/toolbench.py`) compare quatre reactions, A PARTIR DU
MEME premier outil :

    rien · refait a l'aveugle · improve (l'hote dit l'attendu) · improve (l'hote ne dit rien)

Le juge est du CODE : chaque entree d'une tache porte son resultat attendu. Apres la reaction,
l'outil est essaye sur des entrees que rien n'a jamais vues - ni le modele, ni le carnet. Une unite
(une tache, un essai) est RESOLUE quand toutes passent.

Sans cle (niveau 1) : un constructeur SCRIPTE rend d'abord une version naive puis, a `improve`, la
version complete. Il mesure le code du banc et de `improve` - jamais un modele. C'est le CONTROLE
POSITIF : `improve` doit y resoudre ce que « rien » ne resout pas, sinon le banc est aveugle.

    python examples_autoagent/43_banc_outils_qui_cassent.py           # 3 taches, une par sorte de casse
    python examples_autoagent/43_banc_outils_qui_cassent.py --tout    # les 8 taches (plus long)

Avec un vrai modele (niveau 2) : il ecrit vraiment les outils, derriere une vraie frontiere
(le noyau sous Linux, sinon Docker) - un appel au modele par unite, plus un par reaction :

    python examples_autoagent/43_banc_outils_qui_cassent.py --provider deepseek [--essais 2]
    python examples_autoagent/43_banc_outils_qui_cassent.py --provider gemini --plafond 24000

`--plafond` : le plafond de sortie du constructeur (`max_build_tokens`). Un modele qui reflechit le depense
avant d'ecrire l'outil - mesure sur Gemini : 7 861 jetons de reflexion sur 8 192, reponse coupee.

Ce que ca ne dit pas : huit taches de conversion de texte, pas les outils en general. A
temperature 0 les essais d'un meme modele se ressemblent : c'est le nombre de TACHES qui porte
la preuve, pas le nombre d'essais.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ is None:                       # execution directe
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from autoagent.sandbox import make_sandbox  # noqa: E402
from autoagent.toolbench import (  # noqa: E402
    TOOL_TASKS,
    ToolBenchReport,
    positive_control_tools,
    run_tool_bench,
)

# Une tache par sorte de casse : le premier outil PLANTE, rend une valeur FAUSSE, REFUSE une entree valide.
TROIS = ("minutes_de", "montant", "pourcentage")


def ligne(titre: str) -> None:
    print("\n" + "=" * 78 + f"\n{titre}\n" + "=" * 78)


def detail(rapport: ToolBenchReport) -> None:
    print(f"\n  {'tache':<22} " + " ".join(f"{arm:<17}" for arm in rapport.arms))
    for tache in rapport.tasks:
        cellules = []
        for arm in rapport.arms:
            r = next(x for x in rapport.rows if x.task == tache and x.arm == arm and x.trial == 1)
            cellules.append(f"{r.right}/{r.total} {r.status}"[:17])
        print(f"  {tache:<22} " + " ".join(f"{c:<17}" for c in cellules))
    print("  (premier essai de chaque tache : entrees suivantes justes, puis ce que la reaction a fait)")


def niveau_1(tout: bool) -> None:
    taches = [t for t in TOOL_TASKS if tout or t.name in TROIS]
    ligne(f"Niveau 1 - le controle positif : un constructeur scripte, {len(taches)} taches, hors ligne")
    exemple = taches[0]
    print(f"  Exemple de tache : « {exemple.capability} »")
    print(f"    premieres entrees  : {', '.join(repr(t) for t, _ in exemple.easy)}")
    print(f"    puis arrivent      : {', '.join(repr(t) for t, _ in exemple.hard)}")
    print(f"    et ensuite (jamais vues par le modele ni par le carnet) : {', '.join(repr(t) for t, _ in exemple.later)}\n")
    # Le code execute ici est celui du banc (les versions naive et complete des taches), pas celui d'un
    # modele : l'avertissement « bac sans frontiere » de la bibliotheque n'apprendrait rien.
    logging.getLogger("autoagent").setLevel(logging.ERROR)
    rapport = positive_control_tools(tasks=taches)
    print(rapport.summary())
    detail(rapport)
    c = rapport.by_arm()
    voit = c["improve"]["solved"] > max(c["none"]["solved"], c["rebuild"]["solved"])
    print("\n  Controle positif :", f"le banc VOIT ce qu'improve apporte ({c['improve']['solved']} unites resolues "
          f"contre {c['rebuild']['solved']})." if voit else "LE BANC EST AVEUGLE.")
    comparaison = rapport.compare("rebuild", "improve")
    if comparaison["verdict"] == "indistinguishable":
        print(f"  Mais {len(taches)} taches ne suffisent pas a TRANCHER (p = {comparaison['p']:.3f} : « indistinguables ») :")
        print("  c'est le nombre de taches qui porte la preuve. Lance --tout pour les huit.")
    restes = [r.task for r in rapport.rows if r.arm == "improve_unjudged" and r.status == "rejected"]
    if restes:
        print(f"  Sans verdict de l'hote, seul le PLANTAGE s'apprend. Ici : {', '.join(restes)}. Le premier outil y")
        print("  rendait une valeur FAUSSE sans planter ; personne ne l'a signalee, elle compte donc comme un cas")
        print("  reussi, et la bonne correction, qui la change, est refusee (« rejected »). C'est la limite, mesuree.")


def niveau_2(essais: int, plafond: int) -> None:
    from _common import make_provider

    provider = make_provider(sys.argv[1:])
    bac = make_sandbox(require_boundary=True, timeout=60.0)
    ligne(f"Niveau 2 - un vrai modele ecrit les outils · bac : {bac.isolation()['kind']} · {essais} essai(s) par tache")
    rapport = run_tool_bench(provider, trials=essais, sandbox=bac, max_build_tokens=plafond)
    print(rapport.summary())
    detail(rapport)
    for r in rapport.rows:
        if r.detail and r.status in ("build_failed", "first_build_failed", "rejected"):
            print(f"  {r.task} / {r.arm} : {r.status} - {r.detail[:150]}")
    print("\n  A lire avec ses limites : huit taches de conversion de texte, un seul modele, et des essais")
    print("  qui se ressemblent a temperature 0. « Indistinguables » veut dire « pas tranche », pas « egaux ».")


if __name__ == "__main__":
    lecteur = argparse.ArgumentParser()
    lecteur.add_argument("--provider", default=None)
    lecteur.add_argument("--tout", action="store_true")
    lecteur.add_argument("--essais", type=int, default=1)
    lecteur.add_argument("--plafond", type=int, default=8192)
    options, _reste = lecteur.parse_known_args()
    if options.provider:
        niveau_2(options.essais, options.plafond)
    else:
        niveau_1(options.tout)
