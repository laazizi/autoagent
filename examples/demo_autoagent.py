"""Démo autoagent : un ORCHESTRATEUR qui valide le travail de ses agents.

Deux fichiers de données, trois agents :

    orchestrateur ──► analyste_serveur    (lit app.log      — erreurs serveur)
          │      └──► analyste_paiements  (lit payments.log — paiements échoués)
          │
          ├── RECOUPE leurs résultats (chaque erreur « gateway 502 » doit
          │   correspondre à un paiement FAILED), contre-vérifie lui-même
          │   avec ses propres outils en cas de doute,
          └── puis ÉCRIT son rapport dans un espace CLÔTURÉ (_out/, .md
              uniquement — traversée de chemin et autres extensions refusées
              PAR DU CODE, pas par une prière dans le prompt).

Tout est de l'autoagent standard :
  * @agent.tool           — le schéma JSON est généré depuis les annotations
  * agent.as_tool()       — un agent devient l'outil d'un autre (multi-agent en 2 lignes)
  * ProjectWorkspace      — écritures bornées : allowlist d'extensions, anti-traversée
  * TraceEmitter partagé  — TOUTE la hiérarchie dans un seul arbre de spans (JSONL)

Lancer :  GEMINI_API_KEY=...  python demo_autoagent.py
"""

import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))               # racine du repo → `import autoagent`

from autoagent import Agent, ProjectWorkspace, TraceEmitter

trace = TraceEmitter(file=str(ICI / "trace_orchestre.jsonl"))   # un arbre pour tout l'essaim

# Le SEUL endroit où l'orchestrateur peut écrire : _out/, markdown uniquement.
rapports = ProjectWorkspace(ICI / "_out", allowed_write_extensions={".md"})


def lire_fichier(nom: str) -> dict:
    """Lit un fichier texte du dossier de démo."""
    return {"contenu": (ICI / nom).read_text(encoding="utf-8")[:10000]}


def compter(texte: str, mot: str) -> int:
    """Compte les occurrences d'un mot dans un texte (un LLM ne sait pas compter)."""
    return texte.lower().count(mot.lower())


def creer_analyste(nom: str, mission: str) -> Agent:
    analyste = Agent.from_model("gemini", "gemini-3.5-flash",
                                system_prompt=f"Tu es {nom}. {mission} "
                                "Tu ne rapportes que des chiffres vérifiés par tes outils.",
                                temperature=0.0, trace=trace)
    analyste.tool(lire_fichier)
    analyste.tool(compter)
    return analyste


analyste_serveur = creer_analyste(
    "analyste_serveur", "Tu analyses app.log (journal serveur) : erreurs et messages.")
analyste_paiements = creer_analyste(
    "analyste_paiements", "Tu analyses payments.log : paiements échoués, raisons, montants.")

orchestrateur = Agent.from_model(
    "gemini", "gemini-3.5-flash",
    system_prompt=(
        "Tu es l'orchestrateur. Délègue l'analyse de chaque fichier au bon spécialiste. "
        "Puis VALIDE leur travail : chaque erreur serveur « payment gateway returned 502 » "
        "doit correspondre à un paiement FAILED avec la raison gateway_502. Si les chiffres "
        "ne collent pas, contre-vérifie toi-même avec tes propres outils avant de conclure. "
        "Termine en ENREGISTRANT un court rapport validé dans « rapport.md » (ecrire_rapport) : "
        "constats, cohérence des deux analyses, montant total non encaissé — puis réponds "
        "par une conclusion en une seule ligne."),
    temperature=0.0, max_steps=12, trace=trace)

orchestrateur.add_tool(analyste_serveur.as_tool(
    name="interroger_analyste_serveur",
    description="Délègue une question sur app.log (erreurs serveur) au spécialiste."))
orchestrateur.add_tool(analyste_paiements.as_tool(
    name="interroger_analyste_paiements",
    description="Délègue une question sur payments.log (paiements échoués) au spécialiste."))
orchestrateur.tool(lire_fichier)                   # ses propres outils d'audit :
orchestrateur.tool(compter)                        # faire confiance, mais vérifier


@orchestrateur.tool
def ecrire_rapport(chemin: str, contenu: str) -> dict:
    """Enregistre le rapport final (markdown, dans la zone rapports uniquement)."""
    return rapports.write_file(chemin, contenu, reason="rapport d'incident")


resultat = orchestrateur.run(
    "Analyse l'incident du jour : combien d'erreurs serveur, combien de paiements "
    "échoués, les deux analyses sont-elles cohérentes, et quel montant n'a pas pu "
    "être encaissé ?")
print(resultat.output)
rapport = ICI / "_out" / "rapport.md"
print(f"\n📄 rapport enregistré : {rapport.exists()} ({rapport})")
print(f"({resultat.steps} tours d'orchestrateur — arbre complet dans trace_orchestre.jsonl)")

# ── La clôture, démontrée de façon DÉTERMINISTE (aucun LLM impliqué) ─────────
print("\nCe que le workspace bloque, PAR DU CODE — les mêmes gardes que traverse l'agent :")
for tentative in ('rapports.write_file("../demo_autoagent.py", "piraté")',
                  'rapports.write_file("virus.exe", "piraté")',
                  'rapports.write_file("C:/Windows/x.md", "piraté")'):
    try:
        eval(tentative)
        print(f"  {tentative}  →  ⚠️ AUTORISÉ (ne devrait pas arriver)")
    except Exception as exc:
        print(f"  {tentative}  →  ❌ {exc}")
