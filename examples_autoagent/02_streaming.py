"""02 — Streaming : la réponse arrive token par token, les outils en direct.

`run_stream` émet des `StreamEvent` typés : `text` (delta à afficher),
`tool_start`/`tool_end` (activité outils), puis `done` (résultat complet à
persister). Tous les providers streament nativement (SSE).

    python examples_autoagent/02_streaming.py
"""

from _common import make_provider

from autoagent import Agent


def main() -> None:
    agent = Agent(make_provider(), max_steps=6)

    @agent.tool
    def compter_lettres(mot: str, lettre: str) -> dict:
        """Compte les occurrences d'une lettre dans un mot."""
        return {"occurrences": mot.lower().count(lettre.lower())}

    print("réponse en direct : ", end="", flush=True)
    for event in agent.run_stream(
        "Combien de 'r' dans 'anticonstitutionnellement' ? Puis explique en une phrase."
    ):
        if event.type == "text":
            print(event.text, end="", flush=True)      # delta token par token
        elif event.type == "tool_start":
            print(f"\n  🔧 {event.tool_name}…", end="", flush=True)
        elif event.type == "tool_end":
            print(f" {event.tool_status}\n", end="", flush=True)
        elif event.type == "done":
            print(f"\n\n[terminé en {event.steps} tours | tokens: "
                  f"{event.usage.total_tokens if event.usage else '?'}]")
        elif event.type == "error":
            print(f"\n[erreur: {event.error}]")


if __name__ == "__main__":
    main()
