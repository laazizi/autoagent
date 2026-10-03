"""Scénario de coupure BRUTALE, lancé en SOUS-PROCESSUS par `tests/test_journal_kill.py`.

    python _journal_scenario.py run    <journal> <effets> <kill_at>
    python _journal_scenario.py resume <journal> <effets> [pause|retry]

`kill_at` = « frontière[@outil][#n] » (ou « - » : ne pas tuer). À la frontière visée le processus
se tue par `os._exit(137)` : aucun `finally`, aucun `atexit`, aucune fermeture de fichier — comme
un `kill -9` ou une coupure de courant. Les EFFETS (un mail, un compteur) s'écrivent dans un
fichier, `fsync` : c'est ce que le test relit pour compter ce qui est VRAIMENT parti.

Sorties : 0 = terminé ; 3 = `OutcomeUnknown` (les appels à l'issue inconnue sont imprimés en JSON) ;
4 = journal verrouillé par un autre processus ; 137 = tué à la frontière demandée.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoagent import Agent, Journal, JournalLocked, OutcomeUnknown, idempotency_key
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMResponse, ModelConfig, ToolCall


class Scripte(LLMProvider):
    def __init__(self) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))

    def complete(self, request):  # type: ignore[no-untyped-def]
        if any(m.role == "tool" for m in request.messages):
            return LLMResponse(content="fini")
        return LLMResponse(tool_calls=[
            ToolCall(id="c1", name="envoyer_mail", arguments={"destinataire": "moi@example.com"}),
            ToolCall(id="c2", name="lire_compteur", arguments={}),
        ])


def construire(journal: Journal, effets: Path) -> Agent:
    def effet(nom: str) -> None:
        with open(effets, "a", encoding="utf-8") as f:
            f.write(f"{nom} {idempotency_key()}\n")
            f.flush()
            os.fsync(f.fileno())

    agent = Agent(Scripte(), max_steps=4, journal=journal)

    @agent.tool
    def envoyer_mail(destinataire: str) -> dict:
        """Envoie un mail."""
        effet("envoyer_mail")
        if os.environ.get("LENT"):
            time.sleep(float(os.environ["LENT"]))        # pour tenir le verrou le temps d'une course
        return {"envoye": True}

    @agent.tool(idempotent=True)
    def lire_compteur() -> dict:
        """Lit un compteur."""
        effet("lire_compteur")
        return {"n": 7}

    return agent


def tueur(spec: str):
    """`frontière[@outil][#n]` : tue à la n-ième occurrence de la frontière (pour cet outil)."""
    m = re.fullmatch(r"(?P<nom>[a-z_:]+)(?:@(?P<outil>\w+))?(?:#(?P<n>\d+))?", spec)
    if m is None:
        raise SystemExit(f"kill_at illisible : {spec!r}")
    nom, outil, cible_n = m["nom"], m["outil"] or "", int(m["n"] or 1)
    vu = {"n": 0}

    def crochet(frontiere: str, infos: dict) -> None:
        if frontiere == nom and (not outil or infos.get("name") == outil):
            vu["n"] += 1
            if vu["n"] == cible_n:
                os._exit(137)

    return crochet


def main() -> int:
    mode, chemin, effets = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    try:
        if mode == "run":
            kill = sys.argv[4]
            journal = Journal(chemin, on_boundary=None if kill == "-" else tueur(kill))
            construire(journal, effets).run("go")
            journal.close()
            return 0
        journal = Journal(chemin)
        if os.environ.get("MARQUEUR"):
            Path(os.environ["MARQUEUR"]).write_text("verrou pris", encoding="utf-8")
        try:
            construire(journal, effets).resume_from_journal(
                on_unknown=sys.argv[4] if len(sys.argv) > 4 else "pause")
        except OutcomeUnknown as exc:
            print(json.dumps({"unknown": [{"id": c.id, "name": c.name} for c in exc.calls]}))
            journal.close()
            return 3
        journal.close()
        return 0
    except JournalLocked:
        return 4


if __name__ == "__main__":
    sys.exit(main())
