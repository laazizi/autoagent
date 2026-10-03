"""37 - La coupure brutale : le mail est parti, le processus meurt. Que fait la reprise ?

Un run reprend d'un instantane pris a la fin de chaque etape. Si le processus meurt PENDANT une
etape - le premier outil a envoye son mail, rien n'est encore ecrit - la reprise repart de
l'instantane precedent, ou de zero, et REFAIT l'etape : le mail part deux fois. La litterature
releve le meme defaut chez les frameworks d'agents : a la reprise, les effets sont refaits
(Khan, arXiv:2608.03836, preprint a auteur unique, aout 2026).

    Agent(provider, journal=Journal("run.jsonl"))
    agent.resume_from_journal()

Le journal ecrit l'INTENTION d'un appel (fsync) AVANT l'effet, son resultat APRES. A la reprise :
un appel dont le resultat est ecrit n'est jamais relance ; un appel dont l'issue est INCONNUE
(l'intention est ecrite, pas le resultat : le mail est peut-etre parti) n'est PAS relance en
silence - `OutcomeUnknown`, et c'est a l'hote de dire ce qui s'est passe.

La demo tue VRAIMENT un processus enfant (`os._exit`, aucun nettoyage) APRES l'effet, deux fois :
avec la reprise classique (checkpoint seul), puis avec le journal. Elle compte les mails dans un
fichier ecrit par l'outil lui-meme, fsync : c'est la verite terrain.

Aucune cle API : le modele est scripte dans le fichier.

    python examples_autoagent/37_coupure_brutale.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ is None:                       # execution directe
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from autoagent import Agent, Journal, OutcomeUnknown, idempotency_key  # noqa: E402
from autoagent.providers.base import LLMProvider  # noqa: E402
from autoagent.schema import LLMResponse, ModelConfig, ToolCall  # noqa: E402


class Scripte(LLMProvider):
    """Un modele scripte, sans etat : il demande le mail tant qu'aucun resultat d'outil n'est la."""

    def __init__(self) -> None:
        super().__init__(ModelConfig(provider="scripte", model="scripte", api_key="x"))

    def complete(self, request):  # type: ignore[no-untyped-def]
        if any(m.role == "tool" for m in request.messages):
            return LLMResponse(content="Mail envoye.")
        return LLMResponse(tool_calls=[ToolCall(id="c1", name="envoyer_mail",
                                                arguments={"destinataire": "moi@example.com"})])


def agent(effets: Path, *, journal: Journal | None, mourir: bool) -> Agent:
    a = Agent(Scripte(), max_steps=4, journal=journal)

    @a.tool
    def envoyer_mail(destinataire: str) -> dict:
        """Envoie un e-mail : effet irreversible."""
        with open(effets, "a", encoding="utf-8") as f:                  # LE mail part : verite terrain
            f.write(f"mail a {destinataire} cle={idempotency_key()}\n")
            f.flush()
            os.fsync(f.fileno())
        if mourir:
            os._exit(137)                                                # coupure brutale APRES l'effet
        return {"envoye": True}

    return a


def enfant(variante: str, mode: str, dossier: Path) -> int:
    """Le code du processus enfant (celui qui meurt)."""
    effets, instantane = dossier / "effets.log", dossier / "instantane.json"
    if variante == "checkpoint":                                         # la reprise « classique »
        a = agent(effets, journal=None, mourir=(mode == "run"))
        if mode == "run":
            a.run("Envoie un mail.", checkpoint=lambda s: instantane.write_text(json.dumps(s.to_dict())))
        else:                                                            # pas d'instantane : on relance
            a.run("Envoie un mail.")
        return 0
    journal = Journal(dossier / "run.jsonl")
    try:
        a = agent(effets, journal=journal, mourir=(mode == "run"))
        if mode == "run":
            a.run("Envoie un mail.")
        else:
            a.resume_from_journal()
        return 0
    except OutcomeUnknown as exc:
        print(json.dumps({"inconnus": [c.name for c in exc.calls]}))
        return 3
    finally:
        journal.close()


def lancer(variante: str, mode: str, dossier: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, __file__, "--enfant", variante, mode, str(dossier)],
                          capture_output=True, text=True, encoding="utf-8", timeout=120, check=False)


def mails(dossier: Path) -> int:
    f = dossier / "effets.log"
    return len(f.read_text(encoding="utf-8").splitlines()) if f.exists() else 0


def main() -> None:
    print("Le modele demande UN mail. Le processus est tue juste APRES son envoi.\n")
    with tempfile.TemporaryDirectory() as t:
        d1 = Path(t) / "checkpoint"
        d1.mkdir()
        r = lancer("checkpoint", "run", d1)
        print(f"  [reprise classique] le processus est mort (code {r.returncode}) ; mails partis : {mails(d1)}")
        lancer("checkpoint", "resume", d1)
        print(f"  [reprise classique] apres la reprise : {mails(d1)} mails")

        d2 = Path(t) / "journal"
        d2.mkdir()
        r = lancer("journal", "run", d2)
        print(f"\n  [journal] le processus est mort (code {r.returncode}) ; mails partis : {mails(d2)}")
        r = lancer("journal", "resume", d2)
        print(f"  [journal] la reprise s'arrete : code {r.returncode}, {r.stdout.strip()} - rien n'est relance ; "
              f"mails : {mails(d2)}")
        with Journal(d2 / "run.jsonl") as journal:                       # l'hote a verifie : le mail est bien parti
            appel = journal.run_view(journal.run_ids()[0]).in_doubt()[0]
            journal.resolve(appel.id, ok=True, result={"envoye": True, "constate": "present dans effets.log"})
        r = lancer("journal", "resume", d2)
        print(f"  [journal] apres que l'hote a tranche : reprise terminee (code {r.returncode}) ; "
              f"mails : {mails(d2)}")
        classique, avec_journal = mails(d1), mails(d2)

    print("\n" + "-" * 78)
    print(f"{'':<44}{'mails partis':>16}")
    print(f"{'reprise classique (checkpoint seul)':<44}{classique:>16}")
    print(f"{'journal durable':<44}{avec_journal:>16}")
    print("\nCE QU'IL FAUT RETENIR\n")
    print("  * Une reprise qui repart de l'instantane precedent refait l'etape interrompue. Un mail,")
    print("    un paiement, un appel d'API : l'effet part deux fois. Le journal ecrit l'INTENTION avant")
    print("    l'effet et le RESULTAT apres ; l'ecart entre les deux est la fenetre ou une coupure laisse")
    print("    une issue INCONNUE.")
    print("  * Une issue inconnue n'est jamais relancee en silence : `OutcomeUnknown`, avant qu'aucun outil")
    print("    de l'etape ne tourne. L'hote dit ce qui s'est passe (journal.resolve), puis reprend. Un outil")
    print("    declare `idempotent=True` est relance tout seul : c'est ce que le drapeau promet.")
    print("  * `idempotency_key()` donne a l'outil une cle STABLE d'avant a apres la coupure : a passer au")
    print("    service externe, qui dedoublonne lui-meme.")
    print("  * Fail-CLOSED : si l'intention ne peut pas etre ecrite, l'effet ne part pas. C'est le contrat")
    print("    inverse de la trace (observabilite, fail-open) : deux mecanismes, deux contrats.")
    print("  * Teste par de vraies morts de processus. Une panne de courant depend en plus de l'honnetete")
    print("    du disque face a fsync : non teste ici.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--enfant":
        sys.exit(enfant(sys.argv[2], sys.argv[3], Path(sys.argv[4])))
    main()
