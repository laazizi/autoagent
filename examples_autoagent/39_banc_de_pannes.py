"""39 - Un banc de pannes fournisseur : de vrais sockets, de vrais fournisseurs.

Un agent vocal vit ou meurt sur ce que fait la bibliotheque quand le fournisseur flanche :
un 529 « surcharge », une erreur annoncee EN COURS de flux, une coupure en pleine phrase, un
silence. La 0.23.1 avait corrige des cas ou un flux coupe finissait en « succes » tronque -
mais en les prouvant sur des reponses SIMULEES injectees a la place du reseau.

Ici : `FaultServer` (autoagent/faults.py) est un VRAI serveur HTTP local (stdlib, port libre)
qui parle le format de fil d'OpenAI et des compatibles (DeepSeek, OpenRouter...), d'Anthropic
et de Gemini, et dont on script les pannes. Les VRAIS fournisseurs de la bibliotheque s'y
branchent, et un juge EN CODE classe ce qui en sort :

    complete · typed_error · truncated_success · untyped_error · hang

Le contrat : face a une panne, une reponse COMPLETE ou une erreur TYPEE (`ProviderError`, dont
`retryable` dit si cela vaut la peine de reessayer) - jamais un texte coupe rendu comme la
reponse, jamais une exception brute que l'hote ne sait pas attraper, jamais un appel qui ne
revient pas.

Mesure sur la 0.23.1 : 15 defauts sur 45 cas (quinze pannes x trois formats de fil). Un flux
OpenAI-compatible ou Gemini coupe PROPREMENT rendait le texte tronque comme une reponse ; une
connexion coupee net ou figee en plein flux levait un `IncompleteRead` / `TimeoutError` bruts ;
une erreur en HTTP 200 (Anthropic, Gemini) etait prise pour une reponse vide ; un corps illisible
levait `UnicodeDecodeError`. Tous corriges en 0.24.0 - et le banc reste la pour que ca ne revienne pas.

Ce que ca ne dit pas : un serveur LOCAL n'est pas un fournisseur (pas de TLS, pas de latence
reseau), et les formats de fil sont reproduits d'apres les adaptateurs et les documentations.
Hors ligne, sans cle : le « fournisseur » est le serveur de ce fichier.

    python examples_autoagent/39_banc_de_pannes.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ is None:                       # execution directe
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from autoagent import ProviderError  # noqa: E402
from autoagent.faults import FAULT_CASES, FaultServer, default_provider, run_fault_bench  # noqa: E402
from autoagent.schema import LLMRequest, Message  # noqa: E402

REQUETE = LLMRequest(messages=[Message(role="user", content="Dis bonjour.")])

# Les relances de la bibliotheque journalisent des WARNING : on les montre, dans l'ordre, avec le reste.
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="  [journal] %(message)s")


def ligne(titre: str) -> None:
    print("\n" + "=" * 78 + f"\n{titre}\n" + "=" * 78)


def histoires() -> None:
    ligne("1) Un 529 « surcharge » puis un succes : les relances absorbent la panne")
    with FaultServer(["http_529", "ok"]) as serveur:
        reponse = default_provider("anthropic", serveur.url, 5.0).complete(REQUETE)
        print(f"  requetes recues par le serveur : {[r.fault for r in serveur.requests]}")
        print(f"  reponse : {reponse.content!r}  (l'hote n'a rien vu)")

    ligne("2) Un flux coupe PROPREMENT apres quelques morceaux (format OpenAI-compatible)")
    with FaultServer(["cut_clean"]) as serveur:
        provider = default_provider("openai", serveur.url, 5.0)
        recu = ""
        try:
            for morceau in provider.stream(REQUETE):
                if morceau.type == "text":
                    recu += morceau.text or ""
                elif morceau.type == "final":
                    print(f"  le fournisseur a rendu une reponse « complete » : {morceau.response.content!r}")
        except ProviderError as exc:
            print(f"  texte recu avant la coupure : {recu!r}")
            print(f"  erreur TYPEE : ProviderError(retryable={exc.retryable}) - {str(exc)[:90]}")
            print("  -> l'hote sait que la phrase est incomplete, et que reessayer vaut la peine.")

    ligne("3) Une connexion coupee NET en plein morceau (le serveur ferme en plein envoi)")
    with FaultServer(["cut_reset"]) as serveur:
        try:
            list(default_provider("gemini", serveur.url, 5.0).stream(REQUETE))
        except ProviderError as exc:
            print(f"  ProviderError(retryable={exc.retryable}) - cause d'origine : {type(exc.__cause__).__name__}")
        except Exception as exc:
            print(f"  EXCEPTION BRUTE : {type(exc).__name__} - l'hote ne sait pas l'attraper proprement")


def matrice() -> None:
    ligne("Le banc complet : 15 pannes x 3 formats de fil, juge en code (une dizaine de secondes)")
    rapport = run_fault_bench(timeout=0.4)
    cases = {c.name: c for c in FAULT_CASES}
    par = {}
    for r in rapport.rows:
        par.setdefault(r.case, {})[r.wire] = r
    print(f"  {'panne':<34} {'openai':<20} {'anthropic':<20} {'gemini':<20}")
    for nom in cases:
        cellules = []
        for wire in ("openai", "anthropic", "gemini"):
            r = par[nom][wire]
            cellules.append(f"{r.outcome}{'' if r.accepted else ' !!'}")
        print(f"  {nom:<34} {cellules[0]:<20} {cellules[1]:<20} {cellules[2]:<20}")
    print(f"\n  {len(rapport.rows)} cas, {len(rapport.failures)} defaut(s).")
    if rapport.failures:
        print("  « !! » = un defaut : texte tronque rendu comme une reponse, exception brute, ou appel qui ne revient pas.")
    else:
        print("  Aucun defaut : chaque panne finit en reponse complete ou en erreur typee.")


if __name__ == "__main__":
    histoires()
    matrice()
