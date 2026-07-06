"""01 — La base : un agent + des outils en 15 lignes.

Le cœur d'autoagent : tu décores des fonctions Python, la lib génère leur
schéma JSON depuis la signature (types + docstring), et l'agent boucle
LLM ↔ outils jusqu'à la réponse finale. `result.usage` te donne le coût.

    python examples_autoagent/01_hello_tools.py
"""

from _common import make_provider

from autoagent import Agent


def main() -> None:
    agent = Agent(make_provider(), max_steps=6)

    @agent.tool
    def additionner(a: float, b: float) -> dict:
        """Additionne deux nombres."""
        return {"somme": a + b}

    @agent.tool
    def celsius_vers_fahrenheit(celsius: float) -> dict:
        """Convertit une température de Celsius vers Fahrenheit."""
        return {"fahrenheit": celsius * 9 / 5 + 32}

    result = agent.run("Combien font 21.5 + 20.5 ? Et 30°C en Fahrenheit ?")

    print(result.output)
    print(f"\n[{result.steps} tours LLM | tokens: "
          f"{result.usage.total_tokens if result.usage else 'non rapportés'}]")


if __name__ == "__main__":
    main()
