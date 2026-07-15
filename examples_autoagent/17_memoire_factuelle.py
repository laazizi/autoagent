"""17 — Mémoire factuelle : des FAITS tenus à jour, pas un résumé qui empile.

`FactMemory` fait passer les vieux tours par une extraction LLM qui maintient
des faits atomiques via add / update / delete : une contradiction REMPLACE le
fait périmé (« préfère le soir » → « préfère le matin »), un fait caduc
disparaît (« le scooter a été vendu »). JSON lisible par identité (`path=`).

Et les deux atouts v2 (sleep-time + sens) :

  * ``background=True`` — l'appel LLM d'extraction part dans un THREAD :
    ``compact()`` rend la main en <1 ms (chronométré ci-dessous), la
    conversation n'attend jamais la mémoire, et le transcript n'est replié
    qu'APRÈS la sauvegarde des faits (échec = rien perdu).
  * ``embed_fn=`` — recherche par le SENS : « véhicule » retrouve « deux
    voitures » sans un seul mot commun (embeddings Gemini si la clé est là,
    sinon repli lexical automatique).

    python examples_autoagent/17_memoire_factuelle.py
"""

import json
import os
import time
import urllib.request
from pathlib import Path

from _common import make_provider

from autoagent import Agent, FactMemory, Message

STORE = Path(__file__).parent / "faits_martin.json"   # 1 fichier par identité


def embed_gemini(texts: list[str]) -> list[list[float]]:
    """Embeddings réels via l'API Gemini (batch, urllib pur, zéro dépendance)."""
    corps = json.dumps({"requests": [
        {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]}}
        for t in texts
    ]}).encode()
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-embedding-001:batchEmbedContents",
        data=corps,
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    with urllib.request.urlopen(req, timeout=30) as reponse:  # noqa: S310
        return [e["values"] for e in json.loads(reponse.read())["embeddings"]]


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
    semantique = "GEMINI_API_KEY" in os.environ
    memoire = FactMemory(
        provider, path=STORE, max_messages=6, keep_recent=2,
        background=True,                               # sleep-time : jamais bloquant
        embed_fn=embed_gemini if semantique else None, # recherche par le sens
    )

    # ── Appel 1 : l'enquêté se présente — et compact() ne bloque PAS ──
    appel1 = conversation(
        ("user", "Bonjour, c'est M. Martin. Je préfère être rappelé le soir, après 18h."),
        ("assistant", "C'est noté, nous vous rappellerons le soir après 18h."),
        ("user", "Nous avons deux voitures au foyer, et mon fils a un scooter."),
        ("assistant", "Très bien : deux voitures et un scooter."),
        ("user", "Mon fixe c'est le 04 72 11 22 33."),
        ("assistant", "Parfait, c'est noté."),
        ("user", "On peut continuer le questionnaire ?"),
        ("assistant", "Bien sûr."),
    )
    debut = time.perf_counter()
    memoire.compact(appel1)
    print(f"⚡ compact() a rendu la main en {(time.perf_counter() - debut) * 1000:.1f} ms "
          "— l'extraction LLM tourne en coulisses (sleep-time)")
    memoire.flush(timeout=60)                          # on attend juste pour l'affichage
    afficher(memoire, "Faits après l'appel 1 (extraits en arrière-plan)")

    # ── Appel 2 : contradiction + fait caduc → update / delete ──
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
    memoire.flush(timeout=60)
    afficher(memoire, "Faits après l'appel 2 (soir→matin mis à jour, scooter supprimé)")

    # ── Recherche par le SENS : aucun mot commun avec le fait ──
    if semantique:
        matches = memoire.recall("véhicule")
        print("\n🔍 recall('véhicule') — mot absent des faits — trouve par le sens :")
        for m in matches[:1]:
            print(f"  {m.content}")
    else:
        print("\n(pas de GEMINI_API_KEY : recall lexical — fournis embed_fn pour le sens)")

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
    agent.register_remember_tool()
    agent.register_recall_tool()

    r1 = agent.run("Notez bien que je serai absent tout le mois d'août, je pars en vacances.")
    print(f"\nagent (remember) : {r1.output.strip()}")
    r2 = agent.run("Quel est le meilleur moment pour rappeler M. Martin ? Vérifie avec recall.")
    print(f"agent (recall)   : {r2.output.strip()}")

    print(f"\n📄 Base de faits auditable : {STORE.name}")
    print(STORE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
