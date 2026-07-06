"""04 — Observabilité + budget : traces typées, coût par run, plafond dur.

Trois pouvoirs :
  * `TraceEmitter` : chaque étape (run_start, llm_request, tool_call_*,
    run_end) devient un événement typé — vers un fichier JSONL ET/OU un
    callback (WebSocket, Langfuse…). Les secrets sont redactés.
  * `AgentResult.usage` : le coût tokens agrégé de chaque run.
  * `Agent(token_budget=N)` : plafond dur — au-delà, `TokenBudgetExceeded`
    au lieu d'un appel LLM de plus.

    python examples_autoagent/04_observabilite_budget.py
"""

from _common import ROOT, make_provider

from autoagent import Agent, TokenBudgetExceeded, TraceEmitter


def main() -> None:
    provider = make_provider()
    trace_file = ROOT / "examples_autoagent" / "trace_demo.jsonl"

    def on_event(ev) -> None:  # callback synchrone — ici un simple print
        print(f"   [trace] {ev.type:20} {ev.payload.get('name', '')}")

    with TraceEmitter(file=trace_file, on_event=on_event) as trace:
        agent = Agent(provider, max_steps=6, trace=trace)

        @agent.tool
        def carre(n: float) -> dict:
            """Renvoie le carré d'un nombre."""
            return {"carre": n * n}

        print("— Run normal (trace + coût) —")
        result = agent.run("Quel est le carré de 12.5 ?")
        print(f"\n{result.output}")
        if result.usage:
            print(f"[coût du run : {result.usage.input_tokens} in + "
                  f"{result.usage.output_tokens} out = {result.usage.total_tokens} tokens]")
        print(f"[trace JSONL : {trace_file.name}]\n")

        print("— Run avec token_budget=1 (plafond volontairement intenable) —")
        radin = Agent(provider, max_steps=6, trace=trace, token_budget=1)

        @radin.tool
        def cube(n: float) -> dict:
            """Renvoie le cube d'un nombre."""
            return {"cube": n ** 3}

        try:
            radin.run("Quel est le cube de 3 ? Utilise l'outil.")
        except TokenBudgetExceeded as exc:
            print(f"→ stoppé net : {exc}")
            print(f"  (dépensé : {getattr(exc, 'spent', '?')} tokens ; "
                  "l'appel suivant n'a jamais été émis)")


if __name__ == "__main__":
    main()
