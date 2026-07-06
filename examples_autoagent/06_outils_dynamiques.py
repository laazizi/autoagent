"""06 — Outils dynamiques : l'agent ÉCRIT l'outil qui lui manque.

La signature d'autoagent. L'agent démarre SANS outil de calcul ; face à la
question, il appelle `create_python_tool` : un LLM écrit le module Python,
la lib le valide (AST : pas d'os/eval/réseau non autorisé), l'exécute en
SANDBOX (subprocess isolé, ou Docker si dispo), puis l'agent l'UTILISE.
Le fichier généré est posé sur disque — inspecte-le !

    python examples_autoagent/06_outils_dynamiques.py
"""

from _common import ROOT, make_provider

from autoagent import Agent, DynamicToolBuilder

TOOLS_DIR = ROOT / "examples_autoagent" / "outils_generes"


def main() -> None:
    provider = make_provider()
    agent = Agent(provider, max_steps=8, max_dynamic_tools_per_run=2)
    agent.enable_dynamic_tools(DynamicToolBuilder(provider, tools_dir=TOOLS_DIR))

    result = agent.run(
        "Tu n'as aucun outil de maths : crée un outil qui calcule le PGCD (plus grand "
        "commun diviseur) de deux entiers par l'algorithme d'Euclide, puis donne-moi "
        "le PGCD de 462 et 1071."
    )

    print(result.output)
    generes = sorted(TOOLS_DIR.glob("*.py"))
    print(f"\n[outil(s) écrit(s) par l'agent dans {TOOLS_DIR.name}/ : "
          f"{', '.join(p.name for p in generes) or 'aucun'}]")
    print("[ouvre le fichier : TOOL = {...} + def run(args, context) — validé AST, exécuté sandbox]")


if __name__ == "__main__":
    main()
