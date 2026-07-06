"""07 — Multi-agents : superviseur → spécialistes via `as_tool`.

`Agent.as_tool()` expose un agent comme OUTIL d'un autre. Ici un superviseur
délègue à un chercheur (qui a un outil) puis à un rédacteur (style). Chaque
spécialiste a son prompt, ses outils, son budget. Le résultat de chaque
délégation porte son coût — et un TraceEmitter partagé montre tout l'essaim.

    python examples_autoagent/07_multi_agents.py
"""

from _common import make_provider

from autoagent import Agent, TraceEmitter

_STOCK = {"capteurs_actifs": 128, "en_panne": 3, "couverture_km": 540}


def main() -> None:
    provider = make_provider()
    trace = TraceEmitter(on_event=lambda ev: (
        print(f"   [trace] {ev.type:16} {ev.payload.get('name', ev.payload.get('model',''))}")
        if ev.type in ("run_start", "tool_call_start") else None
    ))

    chercheur = Agent(provider, system_prompt="Tu fournis des chiffres du parc de capteurs.",
                      max_steps=5, token_budget=15_000, trace=trace)

    @chercheur.tool
    def etat_parc() -> dict:
        """Renvoie l'état courant du parc de capteurs."""
        return _STOCK

    redacteur = Agent(provider, max_steps=3, token_budget=8_000, trace=trace,
                      system_prompt="Tu écris un mini-bulletin : 2 phrases, ton factuel.")

    superviseur = Agent(
        provider, max_steps=6, trace=trace,
        system_prompt=("Pour répondre : demande les chiffres à `chercheur`, puis fais "
                       "rédiger le bulletin par `redacteur`, et rends son texte."),
    )
    superviseur.add_tool(chercheur.as_tool(
        name="chercheur", description="Donne les chiffres réels du parc de capteurs."))
    superviseur.add_tool(redacteur.as_tool(
        name="redacteur", description="Rédige un bulletin court à partir de faits."))

    result = superviseur.run("Fais-moi le bulletin hebdo de l'état du parc de capteurs.")
    print("\n=== Bulletin ===\n" + result.output)


if __name__ == "__main__":
    main()
