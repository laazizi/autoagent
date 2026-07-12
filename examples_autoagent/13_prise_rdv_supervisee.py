"""13 — Prise de RDV supervisée : les 3 cerveaux de cati_service, sans la voix.

Le cas le plus riche : un répondeur de prise de rendez-vous. On enlève la couche
vocale (Gemini Live / Asterisk, qui n'est PAS de l'autoagent) et il reste
l'architecture à TROIS cerveaux séparés, chacun une brique de la lib :

  1. FLUX DÉTERMINISTE  (Orchestrator)  — le moteur possède la machine à états ;
     le LLM ne fait qu'interpréter/reformuler. Validation Python par slot :
     un jour hors planning ou une heure hors fenêtre sont REFUSÉS par le code.
  2. SUPERVISEUR        (Agent + tools) — un agent autoagent qui a la VUE GLOBALE
     (état + transcript + mémoire) et TRANCHE via UN outil : valider() ou
     demander_correction(). post_turn_hook = on le force à trancher.
  3. MÉMOIRE PAR APPELANT (Memory Protocol) — le numéro = l'identité. Chaque
     appel est résumé et rangé sur disque ; au rappel, l'agent RETROUVE le passé
     (register_recall_tool → outil rappeler_memoire).

Exécutable tel quel (client simulé) :
    python examples_autoagent/13_prise_rdv_supervisee.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from _common import make_provider

from autoagent import (Agent, AgentTurnContext, Message, Orchestrator, Step,
                       TraceEmitter)

# ── Config du RDV (comme config_rdv.json de cati_service) ────────────────────
JOURS = ["mardi", "mercredi", "jeudi", "vendredi", "samedi"]
FENETRES = {"semaine": (17, 21), "samedi": (10, 15)}   # heures d'ouverture
DUREE_MIN = 35
CHAMPS = [
    ("nom",     "le nom et prénom de la personne"),
    ("jour",    "le jour du rendez-vous (du mardi au samedi)"),
    ("heure",   "l'heure du rendez-vous"),
    ("tel",     "le numéro de rappel (format français)"),
    ("confirm", "la confirmation du récapitulatif (oui / non)"),
]
MEM_DIR = Path(__file__).resolve().parent / "memoire_demo"
MEM_DIR.mkdir(exist_ok=True)
TRACE = Path(__file__).resolve().parent / "trace_rdv.jsonl"


# ── CERVEAU 3 : mémoire par appelant (Memory Protocol : compact + recall) ────
class CallerMemory:
    """Mémoire d'UN numéro. Branchée sur l'agent superviseur (`memory=` +
    register_recall_tool) : il fouille lui-même les appels passés."""

    def __init__(self, numero: str) -> None:
        self.numero = re.sub(r"\D", "", numero) or "inconnu"

    def _file(self) -> Path:
        return MEM_DIR / f"{self.numero}.json"

    def compact(self, messages: list[Message]) -> list[Message]:
        return messages                                   # runs courts → inchangé

    def recall(self, query: str, k: int = 3) -> list[Message]:
        try:
            appels = json.loads(self._file().read_text(encoding="utf-8"))["appels"]
        except Exception:
            return []
        mots = [m for m in re.split(r"\W+", (query or "").lower()) if len(m) > 2]
        scored = []
        for ap in appels:
            texte = f"[appel du {ap['date']}] {ap['resume']} — RDV={json.dumps(ap['reponses'], ensure_ascii=False)}"
            score = sum(texte.lower().count(m) for m in mots) if mots else 1
            if score:
                scored.append((score, texte))
        scored.sort(key=lambda s: -s[0])
        return [Message(role="user", content=t) for _, t in scored[:k]]

    def save(self, resume: str, reponses: dict) -> None:
        data = {"appels": []}
        if self._file().is_file():
            data = json.loads(self._file().read_text(encoding="utf-8"))
        data["appels"].append({"date": time.strftime("%Y-%m-%d %H:%M"),
                               "resume": resume, "reponses": reponses})
        data["appels"] = data["appels"][-10:]
        self._file().write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ── CERVEAU 1 : validation déterministe des slots (comme flow.py) ────────────
def _fenetre(jour: str) -> tuple[int, int, str]:
    if "samedi" in jour.lower():
        return (*FENETRES["samedi"], "le samedi")
    return (*FENETRES["semaine"], "en semaine")

def valider_slot(champ: str, valeur, etat: dict) -> str | None:
    """Renvoie un message d'erreur (→ rejet, reformulé) ou None (→ accepté)."""
    v = str(valeur).lower().strip()
    if champ == "jour":
        if not any(j in v for j in JOURS):
            return f"les enquêteurs ne rappellent que : {', '.join(JOURS)}"
    elif champ == "heure":
        m = re.search(r"(\d{1,2})\s*[h:]\s*(\d{0,2})", v)
        if not m:
            return "je n'ai pas compris l'heure (ex. 18h, 14h30)"
        h = int(m.group(1))
        lo, hi, libelle = _fenetre(etat.get("jour", ""))
        if not (lo <= h < hi or (h == hi)):
            return (f"{libelle} c'est entre {lo}h et {hi}h (l'entretien dure ~{DUREE_MIN} min) — "
                    "proposez un autre créneau")
    elif champ == "tel":
        if not re.match(r"^(0|\+?33)\d{9}$", re.sub(r"[ .\-]", "", v)):
            return "il me faut un numéro français valide (ex. 06 12 34 56 78)"
    return None


# ── Le flux (Orchestrator) piloté par un client SIMULÉ ───────────────────────
def prise_de_rdv(provider, scenario: list[str]) -> tuple[dict, str]:
    answers: dict[str, object] = {}
    transcript: list[str] = []

    def current_steps():
        restants = [(c, lbl) for c, lbl in CHAMPS if c not in answers]
        return [Step(id=c, payload={"demande": lbl}) for c, lbl in restants[:2]]

    def record(step_id: str, value) -> str | None:
        err = valider_slot(step_id, value, answers)
        if err:
            return err                                    # rejet : l'étape reste
        answers[step_id] = int(re.search(r"\d{1,2}", str(value)).group()) if step_id == "heure" \
            else value
        return None

    orch = Orchestrator(provider, current_steps=current_steps, record=record,
                        describe=lambda sid, val: f"{sid} = {val}",
                        closing_text="Parfait, votre rendez-vous est confirmé. Merci et à bientôt !")

    print("Léa : Bonjour, service rendez-vous de l'institut Alyce, je suis Léa !")
    for reponse in scenario:
        if not current_steps():
            break
        print(f"\nClient : {reponse}")
        transcript.append(f"CLIENT : {reponse}")
        rep_lea = ""
        for ev in orch.turn(reponse):
            if ev.type == "text":
                rep_lea += ev.text
        print(f"Léa : {rep_lea.strip()}")
        transcript.append(f"LÉA : {rep_lea.strip()}")
    return answers, "\n".join(transcript)


# ── CERVEAU 2 : le superviseur (Agent autoagent qui TRANCHE) ─────────────────
_SYS_SUP = (
    "Tu es l'ORCHESTRATEUR silencieux d'un répondeur de prise de RDV (l'agent "
    "vocal s'appelle Léa). Tu as la vue globale : état du questionnaire, "
    "transcript, et la mémoire des appels passés de ce numéro (outil "
    "rappeler_memoire — utilise-le si le passé éclaire le présent). Vérifie que "
    "le dernier échange est cohérent, puis TRANCHE en appelant EXACTEMENT UN "
    "outil : valider() si tout va bien (cas NORMAL, sois conservateur) ; "
    "demander_correction(probleme, consigne) s'il y a un VRAI problème (mauvaise "
    "valeur notée, réponse ambiguë validée à tort, info déjà donnée lors d'un "
    "appel passé). Dans le doute → valider()."
)

def superviser(provider, trace, numero: str, etat: str, transcript: str) -> dict:
    verdict: dict = {"decide": False}

    def force_verdict(ctx: AgentTurnContext) -> Message | None:
        if not verdict["decide"]:
            return Message(role="user", content="Tu n'as pas tranché : appelle valider() "
                           "ou demander_correction(probleme, consigne) MAINTENANT.")
        return None

    agent = Agent(provider, system_prompt=_SYS_SUP, max_steps=4, temperature=0.0,
                  memory=CallerMemory(numero), trace=trace,
                  post_turn_hook=force_verdict, max_corrections_per_run=1)
    agent.register_recall_tool(
        name="rappeler_memoire", default_k=3,
        description="Fouille les appels PASSÉS de ce numéro (résumés + RDV pris).")

    @agent.tool
    def valider(raison: str = "") -> dict:
        """Le dernier échange est correct : l'appel continue normalement."""
        verdict.update(decide=True, ok=True, raison=raison)
        return {"enregistre": True}

    @agent.tool
    def demander_correction(probleme: str, consigne: str) -> dict:
        """Signale un vrai problème + une consigne corrective courte pour Léa."""
        verdict.update(decide=True, ok=False, probleme=probleme, consigne=consigne)
        return {"enregistre": True}

    agent.run(f"ÉTAT DU QUESTIONNAIRE :\n{etat}\n\nTRANSCRIPT :\n{transcript}\n\nTranche.")
    return verdict


def main() -> None:
    provider = make_provider()
    trace = TraceEmitter(file=str(TRACE))
    numero = "0612345678"

    print("═" * 70 + "\nCERVEAU 1 — FLUX DÉTERMINISTE (Orchestrator + validation Python)\n" + "═" * 70)
    scenario = [
        "Bonjour, Marie Dupont à l'appareil.",
        "Est-ce que lundi serait possible ?",     # ← REFUSÉ par le moteur (hors planning)
        "d'accord, alors jeudi",                   # ← accepté
        "vers 18h ce serait parfait",              # ← accepté (fenêtre semaine 17-21h)
        "vous me rappelez au 06 12 34 56 78",      # ← accepté
        "oui, c'est parfait pour moi",             # ← confirmation
    ]
    answers, transcript = prise_de_rdv(provider, scenario)
    etat = "\n".join(f"  {c} = {answers.get(c, '—')}" for c, _ in CHAMPS)
    print(f"\n→ RDV capturé par le MOTEUR (pas par le LLM) : {answers}")

    print("\n" + "═" * 70 + "\nCERVEAU 2 — SUPERVISEUR (Agent qui tranche : valider / corriger)\n" + "═" * 70)
    print("\n• Sur le transcript RÉEL (propre) :")
    v1 = superviser(provider, trace, numero, etat, transcript)
    print(f"  → verdict : {'✅ VALIDÉ' if v1.get('ok') else '⚠️ CORRECTION'} "
          f"— {v1.get('raison') or v1.get('consigne', '')}")

    print("\n• Sur un transcript DOUTEUX (Léa a noté 18h mais le client a dit 18h30) :")
    faux = transcript.replace("vers 18h ce serait parfait",
                              "plutôt 18h30 si possible") + "\nLÉA : C'est noté, jeudi à 18h."
    v2 = superviser(provider, trace, numero, etat + "\n  (heure notée : 18h)", faux)
    print(f"  → verdict : {'✅ VALIDÉ' if v2.get('ok') else '⚠️ CORRECTION'} "
          f"— {v2.get('probleme','')} / consigne : {v2.get('consigne','')}")

    print("\n" + "═" * 70 + "\nCERVEAU 3 — MÉMOIRE PAR APPELANT (le numéro rappelle)\n" + "═" * 70)
    mem = CallerMemory(numero)
    mem.save(resume=f"Marie Dupont a pris RDV le {answers.get('jour')} à {answers.get('heure')}h, "
             f"rappel au {answers.get('tel')}.", reponses=answers)
    print(f"\n• Appel mémorisé pour le {numero}.")
    print("• Le MÊME numéro rappelle → l'agent retrouve le passé (recall) :")
    for m in mem.recall("rendez-vous jour heure"):
        print(f"    ↳ {m.content}")

    print(f"\n[trace complète de l'essaim : {TRACE.name}]")


if __name__ == "__main__":
    main()
