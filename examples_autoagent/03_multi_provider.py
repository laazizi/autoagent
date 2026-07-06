"""03 — Multi-provider : le MÊME code d'agent sur chaque LLM.

Un `Agent` ne connaît pas son provider — change une ligne et le même agent
tourne sur Gemini, DeepSeek, OpenAI ou Anthropic. Ce script lance la même
question sur CHAQUE provider dont la clé est dans .env et compare
latence / tokens / réponse.

Bonus (voir commentaire) : `RoutingProvider` compose deux providers —
texte → le moins cher, image → un modèle vision — sans toucher à l'agent.

    python examples_autoagent/03_multi_provider.py
"""

import os
import time

from _common import DEFAULTS, KEYS, load_env

from autoagent import Agent, ModelConfig, create_provider

QUESTION = "Donne 3 idées d'indicateurs pour un dashboard de comptage routier. Concis."

# RoutingProvider (bonus) :
#   routeur = RoutingProvider(
#       default=create_provider(ModelConfig(provider="deepseek", model="deepseek-chat")),
#       vision=create_provider(ModelConfig(provider="gemini", model="gemini-2.5-flash")),
#   )
#   agent = Agent(routeur)   # un message AVEC image part vers Gemini, sinon DeepSeek


def main() -> None:
    load_env()
    disponibles = [name for name, key in KEYS.items() if os.getenv(key)]
    if not disponibles:
        raise SystemExit("Aucune clé LLM dans .env.")
    for name in disponibles:
        provider = create_provider(ModelConfig(provider=name, model=DEFAULTS[name], timeout=120.0))
        agent = Agent(provider, max_steps=3)
        t0 = time.monotonic()
        try:
            result = agent.run(QUESTION)
        except Exception as exc:  # noqa: BLE001 — un provider en panne ne bloque pas la comparaison
            print(f"=== {name:10} ÉCHEC : {type(exc).__name__}: {str(exc)[:80]}\n")
            continue
        elapsed = time.monotonic() - t0
        tokens = result.usage.total_tokens if result.usage else "?"
        print(f"=== {name} ({DEFAULTS[name]}) — {elapsed:.1f}s, {tokens} tokens")
        print(result.output.strip()[:400], "\n")


if __name__ == "__main__":
    main()
