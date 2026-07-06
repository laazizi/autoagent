"""05 — Mémoire résumante : le contexte reste borné SANS amnésie.

`BufferMemory` tronque (les vieux tours disparaissent). `SummarizingMemory`
les replie dans un résumé LLM incrémental : la conversation ci-dessous fait
30+ messages, la fenêtre n'en garde que 8… et l'agent retrouve quand même
une décision prise « il y a longtemps », car elle vit dans le résumé.

    python examples_autoagent/05_memoire_resumante.py
"""

from _common import make_provider

from autoagent import Agent, Message, SummarizingMemory


def _longue_conversation() -> list[Message]:
    """Simule 15 tours d'un projet (dont UNE décision clé noyée au milieu)."""
    messages = [Message(role="system", content="Tu assistes un chef de projet capteurs.")]
    for i in range(15):
        if i == 4:
            user = ("Décision validée : le budget matériel 2026 est fixé à 48 500 € "
                    "et le fournisseur retenu est Jetson-Distrib Lyon.")
        else:
            user = f"Point d'avancement numéro {i} : rien de bloquant sur le lot {i}."
        messages.append(Message(role="user", content=user))
        messages.append(Message(role="assistant", content=f"Bien noté pour le point {i}."))
    return messages


def main() -> None:
    provider = make_provider()
    memory = SummarizingMemory(provider, max_messages=12, keep_recent=6)
    agent = Agent(provider, memory=memory, max_steps=4)
    agent.register_recall_tool()  # bonus : l'agent peut re-fouiller les tours repliés

    historique = _longue_conversation()
    historique.append(Message(
        role="user",
        content="Rappelle-moi le budget matériel décidé et le fournisseur retenu.",
    ))

    result = agent.run_messages(historique)

    compacted = [m for m in result.messages if m.role != "system"]
    print(result.output)
    print(f"\n[{len(historique)} messages en entrée → {len(compacted)} non-système "
          "envoyés au LLM (le reste vit dans le résumé)]")
    resume = next((m for m in result.messages if m.content.startswith("[Résumé")), None)
    if resume:
        print(f"[extrait du résumé : {resume.content[:160]}…]")


if __name__ == "__main__":
    main()
