"""41 - Lire le web sans ouvrir le bac, et une frontiere obligatoire pour le code du modele.

Deux reglages qui vont ensemble (main, non publie) :

  1. `Agent(require_boundary=True)` : le code ecrit par le modele (code comme action, run_python,
     outils generes) ne tourne QUE derriere une vraie frontiere - le confinement par le noyau sous
     Linux, sinon Docker s'il tourne, sinon l'agent refuse de demarrer. Ses sous-agents en heritent
     le temps d'une delegation.
  2. `http_fetch_tool([...])` : un outil qui lit une page web PAR L'HOTE, sur une liste de sites.
     Le bac reste coupe du reseau ; le programme du modele l'appelle par `call_host`, donc par la
     porte unique. GET seul, https, adresses internes et metadonnees refusees, taille et temps bornes.

Sans cle et sans reseau, la partie 1 montre ce que la machine fournit comme frontiere et ce que
l'outil refuse AVANT toute connexion. Avec `--reel`, un vrai modele lit une vraie page (pypi.org)
en ecrivant un programme, derriere la frontiere.

    python examples_autoagent/41_lecture_web_bornee.py
    python examples_autoagent/41_lecture_web_bornee.py --reel [--provider deepseek]
"""

from __future__ import annotations

import sys

import _common  # noqa: F401  (chemin vers la lib du depot, console UTF-8)

from autoagent import Agent
from autoagent.errors import ToolError
from autoagent.webfetch import http_fetch_tool

SITES = ["pypi.org"]


def partie_1() -> None:
    print("=" * 78)
    print("1) La frontiere que cette machine fournit au code du modele")
    print("=" * 78)
    agent = Agent(None, require_boundary=True)  # type: ignore[arg-type]   # aucun appel modele ici
    try:
        agent.enable_code_action()
        print("  require_boundary=True ->", agent.python_runner.sandbox.isolation()["kind"],
              "(frontiere de l'OS :", agent.python_runner.sandbox.isolation()["os_boundary"], ")")
    except ToolError as refus:
        print("  require_boundary=True -> REFUS a la configuration :", str(refus)[:160])
    sans = Agent(None)  # type: ignore[arg-type]
    sans.enable_code_action()
    print("  sans le reglage        ->", sans.python_runner.sandbox.isolation()["kind"],
          "(frontiere de l'OS :", sans.python_runner.sandbox.isolation()["os_boundary"], ")")

    print()
    print("=" * 78)
    print("2) Ce que http_fetch refuse AVANT toute connexion (sites permis :", ", ".join(SITES) + ")")
    print("=" * 78)
    lire = http_fetch_tool(SITES)
    for url in ("http://pypi.org/",                       # http sans l'option
                "https://evil.example/",                  # hors liste
                "https://pypi.org.evil.example/",         # un nom qui COMMENCE par le site permis
                "https://moi@pypi.org/",                  # identifiants dans l'URL
                "https://pypi.org:8443/",                 # autre port
                "https://169.254.169.254/latest/"):       # les metadonnees d'un nuage, ecrites en adresse
        try:
            lire(url)
            print(f"  ACCEPTEE {url}")
        except ToolError as refus:
            print(f"  refusee  {url:34s} -> {str(refus)[:90]}")
    interne = http_fetch_tool(["localhost"])
    try:
        interne("https://localhost/")
    except ToolError as refus:
        print(f"  refusee  {'https://localhost/ (site permis !)':34s} -> {str(refus)[:90]}")
    print("\n  Une adresse interne est refusee meme quand le nom est dans la liste : la regle porte")
    print("  sur l'adresse RESOLUE, et la connexion se fait sur cette adresse-la.")


def partie_2(argv: list[str]) -> None:
    from _common import make_provider

    print()
    print("=" * 78)
    print("3) Un vrai modele lit une vraie page, en ecrivant un programme, derriere la frontiere")
    print("=" * 78)
    agent = Agent(make_provider(argv), max_steps=8, require_boundary=True, url_guard="deny",
                  system_prompt="Tu reponds a partir de pages web officielles. Quand une page est grosse, "
                                "ecris un programme qui en extrait l'information.")
    agent.add_tool(http_fetch_tool(SITES))
    agent.enable_code_action(compact=True, timeout=90.0)
    print("  bac du code :", agent.python_runner.sandbox.isolation()["kind"])
    res = agent.run("Quelle est la derniere version publiee d'autoagent-core sur PyPI "
                    "(https://pypi.org/pypi/autoagent-core/json), et a quelle date (AAAA-MM-JJ) ?")
    print("  reponse :", res.output)
    if res.usage is not None:
        print("  jetons  :", res.usage.total_tokens)


if __name__ == "__main__":
    partie_1()
    if "--reel" in sys.argv:
        partie_2([a for a in sys.argv[1:] if a != "--reel"])
    else:
        print("\n  (--reel : un vrai modele lit une vraie page ; il faut une cle dans .env et le reseau)")
