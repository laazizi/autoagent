"""12 — Pseudonymisation : le LLM ne voit JAMAIS les vraies données (RGPD).

Pattern « coffre-fort à la frontière ». Avant qu'un texte n'atteigne le LLM,
le HOST remplace les données personnelles (nom, e-mail, téléphone) par des
JETONS opaques ([PERSONNE_1], [EMAIL_1]…) et garde la correspondance CÔTÉ
HOST. Le modèle raisonne sur les jetons ; sa sortie est « dé-tokenisée »
localement — les vraies valeurs réapparaissent uniquement dans le résultat
final, sans avoir transité par le fournisseur LLM.

Le script PROUVE qu'aucune vraie donnée n'est dans le payload envoyé au LLM.

    python examples_autoagent/12_pseudonymisation_pii.py
"""

import re

from _common import make_provider

from autoagent import LLMRequest, Message

# --- Dossier client RÉEL (fictif, mais c'est le genre de données à protéger) -
DOSSIER = {
    "nom": "Camille Berthier",
    "email": "camille.berthier@example.com",
    "tel": "06 12 34 56 78",
}
DEMANDE = (
    f"Rédige un SMS de relance court et poli pour {DOSSIER['nom']} : son rendez-vous "
    f"d'installation du capteur est demain 9h. Termine par « répondez à {DOSSIER['email']} "
    f"ou au {DOSSIER['tel']} ». Garde les mentions entre crochets EXACTEMENT telles quelles."
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_TEL = re.compile(r"0[1-9](?:[ .\-]?\d{2}){4}")


class Vault:
    """Remplace les PII par des jetons ; seul le HOST connaît la correspondance."""

    def __init__(self) -> None:
        self._to_real: dict[str, str] = {}
        self._to_token: dict[str, str] = {}
        self._n: dict[str, int] = {}

    def _tok(self, kind: str, real: str) -> str:
        if real not in self._to_token:
            self._n[kind] = self._n.get(kind, 0) + 1
            token = f"[{kind}_{self._n[kind]}]"
            self._to_token[real] = token
            self._to_real[token] = real
        return self._to_token[real]

    def mask(self, text: str, noms: tuple[str, ...] = ()) -> str:
        for nom in noms:  # noms connus d'abord (regex ne les attrape pas)
            text = text.replace(nom, self._tok("PERSONNE", nom))
        text = _EMAIL.sub(lambda m: self._tok("EMAIL", m.group()), text)
        text = _TEL.sub(lambda m: self._tok("TEL", m.group()), text)
        return text

    def unmask(self, text: str) -> str:
        for token, real in self._to_real.items():
            text = text.replace(token, real)
        return text


def main() -> None:
    provider = make_provider()
    vault = Vault()

    masque = vault.mask(DEMANDE, noms=(DOSSIER["nom"],))
    print("1) Ce que le LLM reçoit RÉELLEMENT (données masquées) :\n")
    print("   " + masque.replace("\n", "\n   "), "\n")

    # Garde-fou : AUCUNE vraie PII ne doit être dans le payload sortant.
    fuites = [v for v in DOSSIER.values() if v in masque]
    assert not fuites, f"FUITE de PII vers le LLM : {fuites}"
    print(f"2) Vérification : 0 donnée réelle dans le payload (jetons: "
          f"{list(vault._to_real)}).\n")

    response = provider.complete(LLMRequest(
        messages=[
            Message(role="system", content="Tu rédiges des SMS pros. Garde les [JETONS] intacts."),
            Message(role="user", content=masque),
        ],
        temperature=0,
        max_tokens=300,
    ))
    print("3) Sortie du LLM (encore masquée) :\n")
    print("   " + response.content.strip().replace("\n", "\n   "), "\n")

    final = vault.unmask(response.content)
    print("4) Après dé-tokenisation CÔTÉ HOST (vraies valeurs restaurées) :\n")
    print("   " + final.strip().replace("\n", "\n   "))


if __name__ == "__main__":
    main()
