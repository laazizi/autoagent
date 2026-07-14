"""16 — Clarification : l'agent POSE des questions quand la demande est vague.

Le pattern « clarification » (ou elicitation) en deux garde-fous complémentaires :

  1. L'outil `demander_a_l_humain` — la clarification est un APPEL D'OUTIL :
     le LLM décide de demander au lieu de deviner, l'échange est visible dans
     la trace, et l'hôte contrôle le canal (ici la console ; en prod : une
     modale web, un SMS, un ticket…).
  2. Le `post_turn_hook` en filet — les modèles ont un biais fort à répondre
     quand même : si l'agent conclut SANS avoir clarifié une demande vague,
     l'hôte le renvoie poser sa question au lieu de laisser passer une
     réponse devinée.

Consigne volontairement floue : « Organise la réunion. » — l'agent doit
demander (par ex.) avec qui, quand, sur quel sujet, avant de produire
l'invitation.

Variante DURABLE (bot vocal, worker — impossible de bloquer sur input()) :
même outil + `tool_policy` qui lève `ApprovalRequired` sur cet outil → le run
se met en pause avec un `RunState` sérialisable, la question part vers
l'humain par le canal que tu veux, et `agent.resume(state)` reprend avec la
réponse. Voir §19–20 du dev-doc.

    python examples_autoagent/16_questions_clarification.py
"""

from _common import make_provider

from autoagent import Agent, Message

# La démo tourne aussi sans humain au clavier (CI, lecteur pressé) : des
# réponses pré-remplies sont servies si tu appuies juste sur Entrée.
REPONSES_EXEMPLE = iter([
    "avec l'équipe capteurs, on doit choisir le fournisseur des Jetson",
    "jeudi à 14h, une heure en visio",
])


def main() -> None:
    agent = Agent(
        make_provider(),
        max_steps=8,
        system_prompt=(
            "Tu organises des tâches pour l'utilisateur. RÈGLE : si la demande "
            "est vague ou qu'il te manque une information indispensable "
            "(quoi, qui, quand…), utilise demander_a_l_humain — UNE question "
            "précise à la fois, deux questions maximum. Ne devine JAMAIS un "
            "détail important. Quand tu as ce qu'il faut, produis le résultat "
            "final (ici : le texte d'invitation à la réunion)."
        ),
        post_turn_hook=exiger_clarification,
    )

    @agent.tool
    def demander_a_l_humain(question: str) -> dict:
        """Pose UNE question de clarification à l'utilisateur et renvoie sa réponse.

        À utiliser dès que la demande est ambiguë ou incomplète, AVANT d'agir."""
        print(f"\n  ❓ L'agent demande : {question}")
        saisie = input("  → ta réponse (Entrée = réponse d'exemple) : ").strip()
        if not saisie:
            saisie = next(REPONSES_EXEMPLE, "comme tu préfères")
            print(f"    (réponse d'exemple : {saisie})")
        return {"reponse": saisie}

    # Consigne VOLONTAIREMENT vague — c'est le sujet de la démo.
    resultat = agent.run("Organise la réunion.")
    print(f"\n=== Résultat final ({resultat.steps} tours) ===\n{resultat.output}")


def exiger_clarification(ctx) -> Message | None:
    """Filet de sécurité : pas de réponse finale devinée sur une demande vague.

    Si l'agent veut conclure alors qu'il n'a POSÉ AUCUNE question de tout le
    run, on le renvoie clarifier (une seule fois — max_corrections_per_run).
    En prod, remplace ce critère naïf par le tien : champs obligatoires
    manquants, montant absent, date ambiguë…"""
    a_clarifie = any(tc.name == "demander_a_l_humain" for tc in ctx.tool_calls)
    if not a_clarifie and ctx.correction_count == 0:
        return Message(
            role="user",
            content=(
                "Ta réponse repose sur des suppositions : tu n'as rien clarifié. "
                "Pose d'abord ta question la plus importante avec demander_a_l_humain."
            ),
        )
    return None


if __name__ == "__main__":
    main()
