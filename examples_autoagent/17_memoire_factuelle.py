"""17 — Mémoire factuelle : des FAITS tenus à jour, pas un résumé qui empile.

`FactMemory` (0.12.0) fait passer les vieux tours par une extraction LLM qui
maintient des faits atomiques via add / update / delete : une contradiction
REMPLACE le fait périmé (« préfère le soir » → « préfère le matin »), un fait
caduc disparaît (« le scooter a été vendu »). Le tout dans un JSON lisible par
identité (`path=`) : auditable à la main, RGPD = supprimer le fichier.

La démo rejoue le scénario d'un enquêté rappelé deux fois :

  appel 1  →  préférences + équipement extraits en faits
  appel 2  →  CONTRADICTION (soir→matin) + fait caduc (scooter vendu)
  agent    →  remember (« notez que je pars en août ») puis recall

    python examples_autoagent/17_memoire_factuelle.py
"""

from pathlib import Path

from _common import make_provider

from autoagent import Agent, FactMemory, Message

STORE = Path(__file__).parent / "faits_martin.json"   # 1 fichier par identité


def conversation(*tours: tuple[str, str]) -> list[Message]:
    msgs = [Message(role="system", content="Tu es l'assistant d'enquête mobilité.")]
    msgs += [Message(role=r, content=t) for r, t in tours]
    return msgs


def afficher(memoire: FactMemory, titre: str) -> None:
    print(f"\n── {titre} ──")
    for f in memoire.facts():
        sujet = f" ({f['subject']})" if f.get("subject") else ""
        print(f"  [#{f['id']}] {f['fact']}{sujet}")


def main() -> None:
    STORE.unlink(missing_ok=True)                      # démo reproductible
    provider = make_provider()
    memoire = FactMemory(provider, path=STORE, max_messages=6, keep_recent=2)

    # ── Appel 1 : l'enquêté se présente (compact() extrait les faits) ──
    memoire.compact(conversation(
        ("user", "Bonjour, c'est M. Martin. Je préfère être rappelé le soir, après 18h."),
        ("assistant", "C'est noté, nous vous rappellerons le soir après 18h."),
        ("user", "Nous avons deux voitures au foyer, et mon fils a un scooter."),
        ("assistant", "Très bien : deux voitures et un scooter."),
        ("user", "Mon fixe c'est le 04 72 11 22 33."),
        ("assistant", "Parfait, c'est noté."),
        ("user", "On peut continuer le questionnaire ?"),
        ("assistant", "Bien sûr."),
    ))
    afficher(memoire, "Faits après l'appel 1")

    # ── Appel 2 : contradiction + fait caduc → update / delete, PAS d'empilement ──
    memoire.compact(conversation(
        ("user", "Rebonjour, c'est M. Martin. Finalement rappelez-moi plutôt LE MATIN."),
        ("assistant", "C'est modifié : le matin désormais."),
        ("user", "Et on a vendu le scooter de mon fils le mois dernier."),
        ("assistant", "Je note : plus de scooter, il reste deux voitures."),
        ("user", "Voilà, on reprend ?"),
        ("assistant", "Reprenons."),
        ("user", "Allez-y."),
        ("assistant", "Très bien."),
    ))
    afficher(memoire, "Faits après l'appel 2 (soir→matin mis à jour, scooter supprimé)")

    # ── L'agent écrit et lit sa mémoire lui-même (remember / recall) ──
    agent = Agent(
        provider,
        memory=memoire,
        system_prompt=(
            "Tu es l'assistant d'enquête. Quand l'appelant te demande de NOTER "
            "une information durable, utilise remember puis confirme. Pour "
            "retrouver une information mémorisée, utilise recall."
        ),
        max_steps=6,
    )
    agent.register_remember_tool()     # écriture volontaire (tracée)
    agent.register_recall_tool()       # lecture à la demande

    r1 = agent.run("Notez bien que je serai absent tout le mois d'août, je pars en vacances.")
    print(f"\nagent (remember) : {r1.output.strip()}")

    r2 = agent.run("Quel est le meilleur moment pour rappeler M. Martin ? Vérifie avec recall.")
    print(f"agent (recall)   : {r2.output.strip()}")

    print(f"\n📄 Base de faits auditable : {STORE.name}")
    print(STORE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
