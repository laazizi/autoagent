"""19 — Boucle autonome fermée : plan → build → vérifie → corrige → apprend.

Le pattern « Loop » complet, assemblé à partir des briques existantes de la
lib — AUCUN framework requis, aucune primitive nouvelle :

    rôle de la vidéo        brique autoagent
    ────────────────        ─────────────────────────────────────────────
    Déclencheur             une ligne de cron (externe, comme il se doit)
    Planificateur/Builder   Agent + outils
    Vérificateur            post_turn_hook (critères = du CODE, pas un avis)
    Auto-correction         le hook renvoie l'écart → l'agent se corrige
    Mémoire des leçons      FactMemory (persistée entre les exécutions)
    État                    fichier JSON + RunState si interruption
    Cadre (loop FERMÉE)     max_steps + token_budget + workspace borné

Checklist « faut-il une loop ? » (cochée) : tâche RÉCURRENTE (hebdo) ; succès
IDENTIFIABLE automatiquement (critères objectifs vérifiés par code) ;
AUTONOMIE complète (zéro saisie humaine) ; AUCUN jugement subjectif requis.

Chaque exécution = un battement de la boucle. Pour la rendre autonome :

    # Linux/macOS (cron, lundi 9h) :
    0 9 * * 1  cd /chemin && python examples_autoagent/19_boucle_autonome.py
    # Windows (Planificateur de tâches) :
    schtasks /Create /SC WEEKLY /D MON /ST 09:00 /TN rapport-capteurs ^
      /TR "python C:\\chemin\\examples_autoagent\\19_boucle_autonome.py"

    python examples_autoagent/19_boucle_autonome.py    # un battement, maintenant
"""

import json
import re
from pathlib import Path

from _common import make_provider

from autoagent import Agent, FactMemory, Message, ProjectWorkspace

ICI = Path(__file__).parent
ETAT = ICI / "boucle_etat.json"            # état ENTRE les battements
RAPPORTS = ICI / "boucle_rapports"         # zone d'écriture bornée

CRITERES = """Le rapport DOIT contenir, exactement :
- une section « ## Synthèse » (2 phrases max) ;
- une section « ## Chiffres » avec le total de passages ET le taux de
  disponibilité en pourcentage (avec le signe %) ;
- une section « ## Anomalies » citant CHAQUE capteur en panne par son id ;
- moins de 220 mots au total."""


def donnees_de_la_semaine(semaine: int) -> dict:
    """Simule la télémétrie de la semaine (en prod : ta vraie base)."""
    capteurs = {f"C{i:02d}": 4000 + (i * 137 + semaine * 911) % 3000 for i in range(1, 9)}
    en_panne = [f"C{(semaine * 3 + k) % 8 + 1:02d}" for k in range(semaine % 2 + 1)]
    for cid in en_panne:
        capteurs[cid] = 0
    return {"semaine": semaine, "passages": capteurs, "en_panne": sorted(set(en_panne))}


def verifier_conformite(texte: str, donnees: dict) -> list[str]:
    """Le VÉRIFICATEUR : des critères objectifs, en code — pas un avis LLM."""
    ecarts = []
    for section in ("## Synthèse", "## Chiffres", "## Anomalies"):
        if section not in texte:
            ecarts.append(f"section manquante : « {section} »")
    if "%" not in texte:
        ecarts.append("le taux de disponibilité en % est absent")
    total = sum(donnees["passages"].values())
    if f"{total}" not in texte.replace(" ", "").replace("\u202f", ""):
        ecarts.append(f"le total de passages ({total}) n'apparaît pas")
    for cid in donnees["en_panne"]:
        if cid not in texte:
            ecarts.append(f"capteur en panne non cité : {cid}")
    if len(texte.split()) > 220:
        ecarts.append(f"trop long : {len(texte.split())} mots (max 220)")
    return ecarts


def main() -> None:
    provider = make_provider()
    etat = json.loads(ETAT.read_text(encoding="utf-8")) if ETAT.exists() else {"semaine": 0, "historique": []}
    semaine = etat["semaine"] + 1
    donnees = donnees_de_la_semaine(semaine)
    zone = ProjectWorkspace(RAPPORTS, allowed_write_extensions={".md"})
    lecons = FactMemory(provider, path=ICI / "boucle_lecons.json")   # mémoire ENTRE battements

    corrections = {"n": 0}

    def controleur(ctx) -> Message | None:
        """Vérifie le rapport écrit ; renvoie l'ÉCART précis → auto-correction."""
        fichier = RAPPORTS / f"semaine_{semaine}.md"
        texte = fichier.read_text(encoding="utf-8") if fichier.exists() else ""
        ecarts = verifier_conformite(texte, donnees)
        if not ecarts:
            return None                                   # conforme → la boucle se clôt
        corrections["n"] += 1
        print(f"  🔁 auto-correction {corrections['n']} : {'; '.join(ecarts)}")
        return Message(role="user", content=(
            "Le rapport n'est PAS conforme. Corrige puis réécris-le "
            f"(ecrire_rapport). Écarts exacts :\n- " + "\n- ".join(ecarts)
        ))

    agent = Agent(
        provider,
        max_steps=10,
        max_corrections_per_run=3,        # le cadre de la loop FERMÉE
        token_budget=60_000,
        memory=lecons,                    # les leçons des battements passés
        post_turn_hook=controleur,
        system_prompt=(
            "Tu produis le rapport hebdomadaire du parc de capteurs. Utilise "
            "donnees_semaine puis ÉCRIS le rapport avec ecrire_rapport. "
            f"Critères de conformité STRICTS :\n{CRITERES}"
        ),
    )
    agent.register_recall_tool()
    agent.register_remember_tool()

    @agent.tool
    def donnees_semaine() -> dict:
        """Télémétrie de la semaine courante : passages par capteur et pannes."""
        return donnees

    @agent.tool
    def ecrire_rapport(contenu: str) -> dict:
        """Écrit/remplace le rapport de la semaine (markdown, zone bornée)."""
        return zone.write_file(f"semaine_{semaine}.md", contenu, reason="rapport hebdo")

    print(f"═══ Battement de boucle : semaine {semaine} ═══")
    resultat = agent.run(
        f"Produis le rapport de la semaine {semaine}. Consulte d'abord recall "
        "pour d'éventuelles leçons des semaines passées."
    )

    # ── clôture du battement : leçon + état pour le prochain ──
    if corrections["n"]:
        lecon = re.sub(r"\s+", " ", (
            f"semaine {semaine} : {corrections['n']} correction(s) — penser à "
            "vérifier sections/total/%/capteurs cités AVANT d'écrire"
        ))
        lecons.remember(lecon, subject="rapport-hebdo")
    etat = {"semaine": semaine,
            "historique": etat["historique"] + [{"semaine": semaine, "corrections": corrections["n"],
                                                  "tokens": resultat.usage.total_tokens if resultat.usage else None}]}
    ETAT.write_text(json.dumps(etat, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ conforme après {corrections['n']} auto-correction(s), "
          f"{resultat.steps} tours, {resultat.usage.total_tokens if resultat.usage else '?'} tokens")
    print(f"📄 {RAPPORTS / f'semaine_{semaine}.md'}")
    print(f"🧠 leçons mémorisées : {len(lecons.facts())} — prochain battement : semaine {semaine + 1}")


if __name__ == "__main__":
    main()
