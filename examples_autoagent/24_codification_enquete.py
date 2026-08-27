"""24 — Codifier des réponses libres d'enquête : le travail, pas la démo.

Un dépouillement d'enquête déplacements. Les enquêtés répondent en langage
libre ; l'exploitation a besoin de codes d'une nomenclature fermée. Ce rangement
se fait à la main, ligne par ligne.

Ce que fait le modèle : il propose un code pour chaque verbatim.
Ce que fait le code : il **refuse tout ce qui n'est pas dans la nomenclature** et
escalade vers un humain. Donc la sortie ne peut PAS contenir un code inventé —
ce n'est pas une performance du modèle, c'est une propriété du programme.

Les trois nombres à la fin sont ceux d'un exploitant :
    codifié automatiquement · escaladé à un humain · hors nomenclature (toujours 0)

    python examples_autoagent/24_codification_enquete.py
"""

import json
import re

from _common import make_provider

from autoagent import LLMRequest, Message

# ── La nomenclature. Fermée, décidée par le métier, jamais par le modèle. ─────
MODES = ["voiture", "transport en commun", "vélo", "marche", "deux-roues motorisé"]
MOTIFS = ["travail", "études", "achats", "loisirs", "santé", "démarches",
          "accompagnement", "autre"]
SEUIL_CONFIANCE = 0.75          # sous ce seuil : relecture humaine. Le host décide.

# ── Les réponses telles qu'elles arrivent du terrain. ────────────────────────
VERBATIMS = [
    "je vais au bureau en métro tous les matins",
    "j'ai déposé les enfants à l'école puis je suis allé bosser, en voiture",
    "à pied jusqu'à la boulangerie du coin",
    "en vélo au boulot, une vingtaine de minutes",
    "j'ai pris le tram puis fini à pied jusqu'au cabinet du médecin",
    "en trottinette électrique",                       # hors nomenclature
    "scooter, pour aller à la fac",
    "j'sais plus comment j'y suis allé",               # inexploitable
    "covoiturage avec un collègue jusqu'à l'usine",
    "en TER jusqu'à Lyon Part-Dieu, rendez-vous à la préfecture",
    "j'ai fait du télétravail",                        # ce n'est pas un déplacement
    "on est parti en famille au bord du lac, deux voitures",
]

SYSTEME = f"""Tu codifies des réponses libres d'une enquête déplacements.
Pour CHAQUE verbatim, propose un mode et un motif, avec une confiance de 0 à 1.

Modes autorisés : {", ".join(MODES)}
Motifs autorisés : {", ".join(MOTIFS)}

Règles :
- En cas de trajet à plusieurs modes, retiens le mode PRINCIPAL (le plus long).
- Si le verbatim ne permet pas de trancher, ou ne décrit pas un déplacement,
  mets une confiance basse (< 0.5) plutôt que de deviner.
- N'invente jamais un code absent des listes ci-dessus.

Réponds en JSON strict : {{"lignes": [{{"i": <index>, "mode": "<mode>",
"motif": "<motif>", "confiance": <nombre>}}, ...]}} — un objet par verbatim."""


def _lignes_utilisables(brut: str) -> dict[int, dict]:
    """Extrait les objets COMPLETS, et rien d'autre.

    Un modèle peut tronquer sa réponse en plein JSON (vu en réel avec Gemini).
    On ne tente pas de réparer la queue : on garde les objets entiers et on
    ignore le reste. Conséquence : une réponse tronquée ne corrompt pas le
    livrable, elle envoie simplement plus de lignes à un humain. C'est le bon
    sens de l'échec.
    """
    try:
        entier = json.loads(brut)
        lignes = entier.get("lignes", []) if isinstance(entier, dict) else []
    except json.JSONDecodeError:
        lignes = []
        for bloc in re.finditer(r"\{[^{}]*\}", brut):     # objets sans imbrication
            try:
                lignes.append(json.loads(bloc.group()))
            except json.JSONDecodeError:
                continue
    return {int(l["i"]): l for l in lignes
            if isinstance(l, dict) and str(l.get("i", "")).lstrip("-").isdigit()}


def main() -> None:
    provider = make_provider()

    demande = "\n".join(f"{i}. {v}" for i, v in enumerate(VERBATIMS))
    reponse = provider.complete(LLMRequest(
        messages=[Message(role="system", content=SYSTEME),
                  Message(role="user", content=demande)],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=3000,
    ))
    propositions = _lignes_utilisables(reponse.content)

    # ── Le contrôle. C'est ici que la garantie se joue, pas dans le prompt. ──
    codifies, humains, hors_nomenclature = [], [], 0
    for i, verbatim in enumerate(VERBATIMS):
        prop = propositions.get(i)
        if prop is None:
            humains.append((i, "aucune proposition du modèle"))
            continue
        mode, motif = str(prop.get("mode", "")).strip(), str(prop.get("motif", "")).strip()
        try:
            confiance = float(prop.get("confiance", 0))
        except (TypeError, ValueError):
            confiance = 0.0

        if mode not in MODES or motif not in MOTIFS:
            hors_nomenclature += 1        # le modèle a inventé : on jette
            humains.append((i, f"code hors nomenclature ({mode or '?'} / {motif or '?'})"))
        elif confiance < SEUIL_CONFIANCE:
            humains.append((i, f"confiance {confiance:.2f} < {SEUIL_CONFIANCE}"))
        else:
            codifies.append((i, mode, motif, confiance))

    # ── Le livrable : c'est ÇA que voit un exploitant. ───────────────────────
    print("=" * 92)
    print("DÉPOUILLEMENT — réponses libres rangées dans la nomenclature de l'enquête")
    print("=" * 92)
    print(f"{'':3} {'réponse de l’enquêté':<46} {'mode':<20} {'motif':<14}")
    print("-" * 92)
    for i, verbatim in enumerate(VERBATIMS):
        ligne = next((c for c in codifies if c[0] == i), None)
        court = verbatim if len(verbatim) <= 45 else verbatim[:44] + "…"
        if ligne:
            _, mode, motif, conf = ligne
            print(f"{i:>3} {court:<46} {mode:<20} {motif:<14} {conf:.2f}")
        else:
            raison = next(r for j, r in humains if j == i)
            print(f"{i:>3} {court:<46} {'→ RELECTURE HUMAINE':<35} ({raison})")

    total = len(VERBATIMS)
    print("-" * 92)
    print(f"  {len(codifies)}/{total} codifiés automatiquement "
          f"({100 * len(codifies) // total} %)")
    print(f"  {len(humains)}/{total} renvoyés à un humain — avec le motif, pas en silence")
    print(f"  {hors_nomenclature} code hors nomenclature dans le livrable")
    print()
    print("Le dernier chiffre est le seul qui ne dépende pas du modèle : un code")
    print("absent de la nomenclature est rejeté par une comparaison Python. Il ne")
    print("PEUT PAS entrer dans le fichier de sortie, quel que soit le modèle,")
    print("quelle que soit sa version, même s'il se trompe.")
    print()
    # Le modèle n'a rien inventé ce run — la garantie n'aurait donc rien prouvé.
    # On l'exerce explicitement, par le même chemin de code que ci-dessus.
    print("Preuve, plutôt que promesse — on soumet des codes inventés au même contrôle :")
    for mode, motif in [("trottinette électrique", "travail"),
                        ("voiture", "visite familiale"),
                        ("téléportation", "curiosité")]:
        accepte = mode in MODES and motif in MOTIFS
        verdict = "accepté" if accepte else "REJETÉ"
        print(f"  {mode!r} / {motif!r} → {verdict}")
    print("  Aucun `if` à ajouter : c'est le contrôle qui a servi aux 12 lignes.")
    print()
    print("Ce que ça change : le rangement se fait tout seul sur la majorité des")
    print("lignes, et le temps humain va sur celles qui le méritent vraiment.")


if __name__ == "__main__":
    main()
