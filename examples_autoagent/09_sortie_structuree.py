"""08 — Sortie structurée : du JSON fiable, pas du texte à re-parser.

`response_format={"type": "json_object"}` active le JSON mode natif
(OpenAI/DeepSeek/Gemini ; consigne stricte pour Anthropic). On appelle le
provider DIRECTEMENT (sans boucle d'agent) pour extraire une fiche
structurée d'un texte libre — le pattern « extraction » classique.

    python examples_autoagent/08_sortie_structuree.py
"""

import json

from _common import make_provider

from autoagent import LLMRequest, Message

TEXTE = (
    "Bonjour, ici l'exploitant A7. Le capteur CMP-VAL-SN à Valence (GPS 44.95, 4.87) "
    "ne remonte plus rien depuis 8h ce matin, sens sud. Il faisait très chaud hier. "
    "Merci de regarder en priorité haute."
)


def main() -> None:
    provider = make_provider()

    request = LLMRequest(
        messages=[
            Message(role="system", content=(
                "Extrais une fiche d'incident. Réponds en JSON avec EXACTEMENT ces clés : "
                "capteur_id (str), lieu (str), gps (objet lat/lon), heure (str), "
                "direction (str), priorite (str parmi basse/normale/haute).")),
            Message(role="user", content=TEXTE),
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=400,
    )
    response = provider.complete(request)

    fiche = json.loads(response.content)  # parse direct : le JSON mode garantit la forme
    print("Fiche extraite (dict Python) :\n")
    print(json.dumps(fiche, indent=2, ensure_ascii=False))
    print(f"\ncapteur = {fiche.get('capteur_id')} | priorité = {fiche.get('priorite')}")


if __name__ == "__main__":
    main()
