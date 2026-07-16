"""Éval comportementale de la mémoire : que retient VRAIMENT l'agent ?

Douze scénarios multi-sessions (style LoCoMo, en français) : des faits sont
établis lors d'appels passés — parfois contredits ou rendus caducs — puis une
question est posée dans une session NEUVE. Trois configurations comparées :

  * sans_memoire      — agent neuf, aucune mémoire (plancher attendu)
  * summarizing       — SummarizingMemory (résumé roulant)
  * fact_memory       — FactMemory (+ outil recall)

Le score : la réponse contient une des formulations attendues ET aucune des
formulations interdites (ex. la valeur PÉRIMÉE après contradiction).

    python evals/eval_memoire.py            # provider choisi comme les démos (.env)
    python evals/eval_memoire.py --limit 3  # essai rapide

Résultats écrits dans evals/resultats.json. Coût : ~60-90 appels du modèle
« pas cher » configuré (2-3 min).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples_autoagent"))

from _common import make_provider  # noqa: E402

from autoagent import Agent, BufferMemory, FactMemory, Message, SummarizingMemory  # noqa: E402

# ── Scénarios ────────────────────────────────────────────────────────────────
# Chaque session = liste de tours (user, assistant). Le runner ajoute du
# REMPLISSAGE (tours anodins) pour dépasser le seuil de compaction — comme
# dans un vrai appel. `attendu` = formulations acceptées (une suffit),
# `interdit` = formulations qui invalident (valeur périmée).

SCENARIOS = [
    {
        "id": "retention-simple",
        "sessions": [[
            ("Bonjour, c'est Mme Diallo. Mon code d'accès immeuble est le 4271B.",
             "C'est noté Mme Diallo : code 4271B."),
        ]],
        "question": "Quel est le code d'accès de l'immeuble de Mme Diallo ?",
        "attendu": ["4271B"], "interdit": [],
    },
    {
        "id": "contradiction-horaire",
        "sessions": [
            [("Je préfère être rappelé le soir après 18h.", "Noté : le soir après 18h.")],
            [("Finalement rappelez-moi plutôt le matin, avant 9h.", "C'est modifié : le matin avant 9h.")],
        ],
        "question": "À quel moment de la journée faut-il rappeler cette personne ?",
        "attendu": ["matin"], "interdit": ["soir"],
    },
    {
        "id": "fait-caduc",
        "sessions": [
            [("Nous avons deux voitures et un scooter au foyer.", "Deux voitures et un scooter, noté.")],
            [("On a vendu le scooter le mois dernier.", "Je note : plus de scooter.")],
        ],
        "question": "Quels véhicules possède ce foyer aujourd'hui ?",
        "attendu": ["deux voitures"], "interdit": ["scooter"],
    },
    {
        "id": "valeur-numerique",
        "sessions": [[
            ("Mon loyer est de 743 euros par mois.", "743 euros, c'est noté."),
        ]],
        "question": "Quel est le montant du loyer mensuel de cette personne ?",
        "attendu": ["743"], "interdit": [],
    },
    {
        "id": "multi-faits",
        "sessions": [[
            ("Je m'appelle M. Ferreira, j'habite à Villeurbanne et je travaille à Part-Dieu.",
             "Noté : Villeurbanne, travail à Part-Dieu."),
            ("J'y vais en métro, ligne A puis B.", "Métro A puis B, très bien."),
        ]],
        "question": "Comment M. Ferreira se rend-il à son travail ?",
        "attendu": ["métro", "metro"], "interdit": [],
    },
    {
        "id": "contradiction-adresse",
        "sessions": [
            [("J'habite 12 rue des Lilas à Bron.", "12 rue des Lilas à Bron, noté.")],
            [("J'ai déménagé : maintenant c'est 3 avenue Berthelot à Lyon.",
              "Je mets à jour : 3 avenue Berthelot, Lyon.")],
        ],
        "question": "Quelle est l'adresse actuelle de cette personne ?",
        "attendu": ["Berthelot"], "interdit": ["Lilas", "Bron"],
    },
    {
        "id": "engagement-date",
        "sessions": [[
            ("Notez que je serai absent tout le mois d'août, je pars au Portugal.",
             "C'est noté : absent tout le mois d'août."),
        ]],
        "question": "Quand cette personne sera-t-elle absente ?",
        "attendu": ["août", "aout"], "interdit": [],
    },
    {
        "id": "distracteurs",
        "sessions": [[
            ("Ma fille prend le bus C3 pour le lycée.", "Bus C3, noté."),
            ("Mon fils va au collège à vélo.", "À vélo, très bien."),
            ("Et moi je covoiture avec un collègue le jeudi uniquement.",
             "Covoiturage le jeudi, c'est noté."),
        ]],
        "question": "Quel jour cette personne covoiture-t-elle ?",
        "attendu": ["jeudi"], "interdit": [],
    },
    {
        "id": "double-contradiction",
        "sessions": [
            [("Rappelez-moi sur le fixe : 04 72 11 22 33.", "Sur le fixe, noté.")],
            [("Plutôt sur mon portable en fait : 06 45 67 89 10.", "Le portable, c'est noté.")],
            [("Non finalement le fixe c'est mieux, le 04 72 11 22 33.", "Retour au fixe, très bien.")],
        ],
        "question": "Sur quel numéro faut-il rappeler cette personne ?",
        "attendu": ["04 72 11 22 33", "fixe"], "interdit": [],
    },
    {
        "id": "preference-negative",
        "sessions": [[
            ("Surtout ne m'envoyez JAMAIS de SMS, uniquement des appels.",
             "C'est noté : jamais de SMS, uniquement des appels."),
        ]],
        "question": "Peut-on envoyer un SMS à cette personne ?",
        "attendu": ["non", "jamais", "pas de sms", "uniquement des appels"], "interdit": [],
    },
    {
        "id": "profession-employeur",
        "sessions": [[
            ("Je suis infirmière de nuit à l'hôpital Édouard-Herriot.",
             "Infirmière de nuit à Édouard-Herriot, noté."),
        ]],
        "question": "Où travaille cette personne et à quel rythme ?",
        "attendu": ["Herriot"], "interdit": [],
    },
    {
        "id": "correction-orthographe",
        "sessions": [
            [("Mon nom c'est Kowalski. K-O-W-A-L-S-K-I.", "Kowalski, noté.")],
            [("Vous l'aviez mal écrit la dernière fois : c'est bien Kowalski avec un K au début.",
              "Bien noté, Kowalski avec un K.")],
        ],
        "question": "Quel est le nom de famille de cette personne ?",
        "attendu": ["Kowalski"], "interdit": [],
    },
]

REMPLISSAGE = [
    ("D'accord, continuons le questionnaire.", "Très bien, question suivante."),
    ("Oui je suis toujours là.", "Parfait, poursuivons."),
    ("Hmm, laissez-moi réfléchir une seconde.", "Prenez votre temps."),
]

SYSTEM = (
    "Tu es l'assistant d'un centre d'enquêtes. Tu réponds à des questions sur "
    "un enquêté à partir de ta mémoire. Si un outil recall est disponible, "
    "utilise-le AVANT de répondre. Réponds en une phrase précise ; si tu ne "
    "sais pas, dis « je ne sais pas »."
)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return re.sub(r"[̀-ͯ]", "", text)


def _session_messages(tours) -> list[Message]:
    msgs = [Message(role="system", content=SYSTEM)]
    for user, assistant in list(tours) + REMPLISSAGE:
        msgs.append(Message(role="user", content=user))
        msgs.append(Message(role="assistant", content=assistant))
    return msgs


def _score(reponse: str, attendu: list[str], interdit: list[str]) -> bool:
    reponse_n = _norm(reponse)
    ok = any(_norm(a) in reponse_n for a in attendu)
    ko = any(_norm(i) in reponse_n for i in interdit)
    return ok and not ko


def _make_memory(config: str, provider):
    if config == "fact_memory":
        return FactMemory(provider, max_messages=6, keep_recent=2)
    if config == "summarizing":
        return SummarizingMemory(provider, max_messages=6, keep_recent=2)
    if config == "buffer":
        return BufferMemory(max_messages=6)
    return None  # sans_memoire


def run_eval(configs: list[str], limit: int | None) -> dict:
    provider = make_provider()
    scenarios = SCENARIOS[:limit] if limit else SCENARIOS
    resultats: dict = {c: {"scores": {}, "reponses": {}} for c in configs}

    for scenario in scenarios:
        for config in configs:
            memoire = _make_memory(config, provider)
            if memoire is not None:
                for session in scenario["sessions"]:
                    memoire.compact(_session_messages(session))
                if hasattr(memoire, "flush"):
                    memoire.flush(timeout=60)
            agent = Agent(provider, system_prompt=SYSTEM, max_steps=4, memory=memoire)
            if memoire is not None:
                agent.register_recall_tool()
            try:
                reponse = agent.run(scenario["question"]).output or ""
            except Exception as exc:  # noqa: BLE001
                reponse = f"[erreur: {exc}]"
            bon = _score(reponse, scenario["attendu"], scenario["interdit"])
            resultats[config]["scores"][scenario["id"]] = bon
            resultats[config]["reponses"][scenario["id"]] = reponse.strip()[:200]
            print(f"  {scenario['id']:<24} {config:<14} {'✅' if bon else '❌'}")

    print("\n══ Bilan ══")
    for config in configs:
        scores = resultats[config]["scores"]
        taux = 100 * sum(scores.values()) / len(scores)
        resultats[config]["accuracy"] = round(taux, 1)
        print(f"  {config:<14} {sum(scores.values())}/{len(scores)}  ({taux:.0f} %)")
    return resultats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--configs", default="sans_memoire,summarizing,fact_memory")
    args = parser.parse_args()
    resultats = run_eval(args.configs.split(","), args.limit)
    out = Path(__file__).parent / "resultats.json"
    out.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndétail → {out}")


if __name__ == "__main__":
    main()
