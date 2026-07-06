"""11 — Flux déterministe : l'`Orchestrator` (le LLM ne pilote PAS).

Quand un PROCESSUS doit être garanti (questionnaire, formulaire réglementé),
tu ne veux pas d'un agent autonome. L'`Orchestrator` inverse le contrôle :
TON code possède la machine à états (`current_steps` + `record`), le LLM ne
fait que deux micro-tâches — interpréter la réponse et reformuler la
question. Il ne peut ni sauter, ni inventer, ni réordonner une étape.

    python examples_autoagent/11_flux_deterministe.py
    python examples_autoagent/11_flux_deterministe.py --reponse "moi c'est Ana, 30 ans, Lyon"
"""

import argparse

from _common import make_provider

from autoagent import Orchestrator, Step

CHAMPS = ["prenom", "age", "ville"]


def main() -> None:
    provider = make_provider()
    # make_provider ignore les args inconnus -> on lit le nôtre séparément.
    ap = argparse.ArgumentParser()
    ap.add_argument("--reponse", default="Bonjour, moi c'est Ana, j'ai 30 ans et j'habite à Lyon.")
    reponse_utilisateur = ap.parse_known_args()[0].reponse

    answers: dict[str, object] = {}

    def current_steps():
        # L'état appartient au host : on n'expose QUE les champs pas encore remplis.
        restants = [c for c in CHAMPS if c not in answers]
        return [Step(id=c, payload={"demande": c}) for c in restants[:2]]

    def record(step_id: str, value) -> str | None:
        if step_id == "age":
            try:
                value = int(value)
            except (TypeError, ValueError):
                return "L'âge doit être un nombre."   # rejet -> l'étape reste, l'erreur est reformulée
        answers[step_id] = value
        return None  # accepté

    orch = Orchestrator(provider, current_steps=current_steps, record=record)

    print(f"Utilisateur : {reponse_utilisateur}\n")
    print("Léa (reformulation streamée) : ", end="", flush=True)
    for ev in orch.turn(reponse_utilisateur):
        if ev.type == "text":
            print(ev.text, end="", flush=True)
        elif ev.type == "recorded":
            pass  # valeur validée + stockée (silencieux ici)
    print(f"\n\n[champs capturés par le HOST (pas par le LLM) : {answers}]")


if __name__ == "__main__":
    main()
