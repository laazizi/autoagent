"""23 — Questionnaire de mobilité (CATI) : le host possède le questionnaire.

Le cas métier le plus simple à montrer, et le plus convaincant : une enquête
déplacements menée au téléphone. L'enquêté répond en langage libre, en désordre,
parfois hors barème — et le processus reste garanti.

Ce que fait le LLM : deux micro-tâches. Il *interprète* une phrase libre
(« hier, de Villeurbanne à Lyon, pour le travail » → trois champs) et il
*reformule* la question suivante poliment.

Ce que fait le LLM : rien d'autre. Il ne choisit pas l'ordre des questions, il
ne valide aucune réponse, il ne peut ni sauter ni inventer une étape. Toutes les
règles ci-dessous sont des `if` en Python — y compris celle qu'aucune consigne
ne tiendrait de façon fiable : *une origine identique à la destination n'est pas
un déplacement*.

Le point que la démo rend visible : le modèle **propose**, le code **décide**.
Chaque tour affiche les deux colonnes.

À noter dans l'échange scénarisé : au 3ᵉ tour l'enquêté dit « 25 minutes » alors
que la fiche garde les 20 du 2ᵉ tour. Ce n'est pas un bug — un créneau rempli
n'est plus exposé, donc plus modifiable. Autoriser les corrections en cours
d'entretien est un choix explicite du host (`accept_extra`), pas un défaut.

    python examples_autoagent/23_questionnaire_mobilite.py
    python examples_autoagent/23_questionnaire_mobilite.py --reponse "hier, Bron vers Lyon, en bus, 40 min, pour le travail"
"""

import argparse
from datetime import date, timedelta

from _common import make_provider

from autoagent import Orchestrator, Step

# ── Le barème. C'est une donnée du host, pas une consigne au modèle. ──────────
MOTIFS = ["travail", "études", "achats", "loisirs", "santé", "démarches", "autre"]
MODES = ["voiture", "transport en commun", "vélo", "marche", "deux-roues motorisé"]
DUREE_MIN, DUREE_MAX = 1, 300          # minutes ; au-delà on redemande

QUESTIONS = [
    ("date_dep", "Quel jour avez-vous fait ce déplacement ?",
     {"attendu": "une date, jamais dans le futur"}),
    ("origine", "De quelle commune partiez-vous ?", {"attendu": "un nom de commune"}),
    ("destination", "Et vous alliez dans quelle commune ?", {"attendu": "un nom de commune"}),
    ("motif", "Pour quel motif ?", {"attendu": "un motif du barème", "barème": MOTIFS}),
    ("mode", "Avec quel mode de transport principal ?",
     {"attendu": "un mode du barème", "barème": MODES}),
    ("duree_min", "Combien de temps a duré le trajet, en minutes ?",
     {"attendu": "un entier en minutes", "entre": [DUREE_MIN, DUREE_MAX]}),
]
ORDRE = [q[0] for q in QUESTIONS]

# Prompt d'EXTRACTION seulement. Il ne valide rien : tout ce qu'il produit passe
# ensuite par `valider()`, qui peut le refuser. Le renforcer améliore la capture,
# ça n'affaiblit aucune garantie — c'est là toute la différence avec un agent
# dont la consigne EST la règle.
EXTRACTION_FR = """Tu es un module d'extraction pour un questionnaire d'enquête.
Tu reçois l'étape en cours (`current_step`), les étapes à venir (`upcoming_steps`)
et la réponse brute de l'enquêté. Réponds en JSON STRICT, sans rien d'autre :

{"status": "answered", "values": [{"id": "<id d'étape>", "value": <valeur>}, ...]}
ou {"status": "unclear"}
ou {"status": "offtopic", "note": "<ce que l'enquêté a dit>"}
ou {"status": "refused"}

Règles :
- Une réponse d'enquêté couvre SOUVENT plusieurs étapes d'un coup. Extrais-les
  TOUTES : l'étape en cours d'abord, puis chaque étape à venir que la phrase
  renseigne, même partiellement.
- « de X à Y » renseigne l'origine ET la destination. « pour le travail »
  renseigne le motif. « en bus, 40 minutes » renseigne le mode ET la durée.
- Recopie ce que dit l'enquêté sans le corriger : si le mode annoncé n'est pas
  au barème, propose-le quand même tel quel. Ce n'est pas à toi de trancher.
- "refused" seulement si l'enquêté refuse EXPLICITEMENT de répondre.
JSON STRICT uniquement."""


def main() -> None:
    provider = make_provider()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reponse", action="append",
                    help="une réponse d'enquêté ; répétable. Sans l'option, un échange scénarisé.")
    fourni = ap.parse_known_args()[0].reponse
    scenarise = fourni is None
    reponses = fourni or [
        "Bonjour, hier je suis allé de Villeurbanne à Lyon, c'était pour aller travailler.",
        "En trottinette électrique, une vingtaine de minutes.",   # hors barème → refusé
        "Alors mettez vélo, et disons 25 minutes.",
    ]

    fiche: dict[str, object] = {}
    proposes: list[tuple[str, object]] = []      # ce que le modèle a proposé
    refuses: list[tuple[str, str]] = []          # ce que le code a rejeté, et pourquoi
    abandons: list[str] = []                     # l'enquêté a refusé de répondre

    def current_steps():
        """L'état appartient au host : ce qui reste, dans NOTRE ordre.

        On expose tout l'horizon restant, pas seulement la question en cours :
        un enquêté répond en désordre et on veut tout encaisser. La lib
        n'accepte QUE des créneaux exposés ici — un champ absent de cette liste
        est jeté, même si le modèle le propose.
        """
        return [Step(id=cid, payload={"question": texte, **contraintes})
                for cid, texte, contraintes in QUESTIONS if cid not in fiche]

    def controler(step_id: str, value) -> str | None:
        """Le seul endroit où une valeur peut entrer dans la fiche.

        Retourner une chaîne = REFUS : l'étape reste ouverte et le modèle
        reformule le motif. Retourner None = accepté.
        """
        texte = str(value).strip()

        if step_id == "date_dep":
            jour = _lire_jour(texte)
            if jour is None:
                return "Je n'ai pas saisi la date. Quel jour était-ce ?"
            if jour > date.today():                      # règle CATI classique
                return "Cette date est dans le futur ; il me faut un déplacement déjà effectué."
            value = jour.isoformat()

        elif step_id in ("origine", "destination"):
            if len(texte) < 2:
                return "Il me faut un nom de commune."
            # Règle CROISÉE : impossible à garantir par une consigne, triviale ici.
            autre = "destination" if step_id == "origine" else "origine"
            if fiche.get(autre) and texte.lower() == str(fiche[autre]).lower():
                return (f"L'origine et la destination sont identiques ({texte}) : "
                        "ce n'est pas un déplacement. Où alliez-vous exactement ?")
            value = texte.title()

        elif step_id == "motif":
            trouve = _dans_bareme(texte, MOTIFS)
            if trouve is None:
                return f"Ce motif n'est pas au barème. Choisissez parmi : {', '.join(MOTIFS)}."
            value = trouve

        elif step_id == "mode":
            trouve = _dans_bareme(texte, MODES)
            if trouve is None:
                return (f"« {texte} » n'est pas dans la nomenclature de l'enquête. "
                        f"Le mode principal était plutôt : {', '.join(MODES)} ?")
            value = trouve

        elif step_id == "duree_min":
            try:
                minutes = int(round(float(texte.replace(",", "."))))
            except ValueError:
                return "Il me faut une durée en minutes, en chiffres."
            if not DUREE_MIN <= minutes <= DUREE_MAX:
                return (f"{minutes} minutes sort du plausible pour un déplacement local "
                        f"(on attend entre {DUREE_MIN} et {DUREE_MAX}).")
            value = minutes

        fiche[step_id] = value
        return None

    def valider(step_id: str, value) -> str | None:
        """Enveloppe : garde trace de ce qui a été proposé, et des refus."""
        proposes.append((step_id, value))
        motif = controler(step_id, value)
        if motif:
            refuses.append((step_id, motif))
        return motif

    def abandonner(step_id: str) -> None:
        """L'enquêté refuse de répondre. C'est un droit : on marque et on AVANCE."""
        abandons.append(step_id)
        fiche.setdefault(step_id, "refus de répondre")

    orch = Orchestrator(
        provider,
        current_steps=current_steps,
        record=valider,
        interpret_system=EXTRACTION_FR,
        on_refused=abandonner,
        closing_text="C'est noté, le questionnaire est complet. Merci beaucoup !",
    )

    largeur = 74
    print("=" * largeur)
    print("Enquête déplacements — le questionnaire appartient au CODE, pas au modèle")
    print("=" * largeur)

    for tour, reponse in enumerate(reponses, 1):
        avant, n_ref, n_prop = dict(fiche), len(refuses), len(proposes)
        print(f"\n--- tour {tour} " + "-" * (largeur - 12))
        print(f"Enquêté   : {reponse}")
        print("Enquêteur : ", end="", flush=True)
        fini = False
        for ev in orch.turn(reponse):
            if ev.type == "text":
                print(ev.text, end="", flush=True)
            elif ev.type == "done":
                fini = ev.flow_complete
        print()

        for cid, brut in proposes[n_prop:]:
            print(f"  → le MODÈLE propose : {cid} = {brut!r}")
        nouveaux = {k: v for k, v in fiche.items() if k not in avant}
        if nouveaux:
            print(f"  ✓ le CODE retient   : {nouveaux}")
        for cid, motif in refuses[n_ref:]:
            print(f"  ✗ le CODE refuse    : {cid} — {motif}")
        if len(proposes) == n_prop:
            print("  · le modèle n'a rien proposé pour ce tour")
        if fini:
            break
    tours_utilises = tour

    print("\n" + "=" * largeur)
    print("Fiche finale, remplie par le host (le LLM n'y a jamais écrit) :")
    for cid in ORDRE:
        print(f"  {cid:<12} {fiche.get(cid, '— manquant')}")
    manquants = [c for c in ORDRE if c not in fiche]
    print(f"\n{len(proposes)} valeurs proposées par le modèle · "
          f"{len(fiche)}/{len(ORDRE)} retenues · {len(refuses)} refusées par le code")
    if abandons:
        print(f"Refus de répondre (un droit, le flux avance) : {', '.join(abandons)}")
    if manquants:
        print(f"Le questionnaire reste OUVERT sur : {', '.join(manquants)}.")
        print("Aucune valeur inventée, aucune étape sautée : c'est le point de la démo.")
    # ── Ce que le modèle apporte, chiffré. Sans ça, la démo ne prouve que la
    #    contrainte — et un directeur répond « on sait déjà faire un questionnaire ».
    classique = len(ORDRE)          # un script CATI pose UNE question par champ
    hors_bareme = len(refuses)
    print()
    print("Ce que le modèle fait gagner :")
    print(f"  {tours_utilises} tours au lieu de {classique} — un script classique pose une")
    print(f"  question par champ. Soit {100 - round(100 * tours_utilises / classique)} % "
          "d'échanges en moins sur ce déplacement.")
    print(f"  {len(fiche) / tours_utilises:.1f} champ(s) validé(s) par tour.")
    if hors_bareme:
        print(f"  {hors_bareme} réponse(s) hors nomenclature reformulée(s) en langage naturel,")
        print("  au lieu d'un « réponse invalide, choisissez entre 1 et 5 ».")
    print()
    print("Ce que le code garantit, lui, ne dépend d'aucun de ces chiffres :")
    print("  aucune valeur hors barème ne peut entrer dans la fiche, jamais.")

    if scenarise:
        print()
        print("Note : « 25 minutes » au 3ᵉ tour n'a pas écrasé les 20 du 2ᵉ — un créneau")
        print("rempli n'est plus exposé, donc plus modifiable. Autoriser les corrections")
        print("en cours d'entretien est un choix explicite du host (`accept_extra`).")


def _dans_bareme(texte: str, bareme: list[str]) -> str | None:
    """Tolérant sur la forme, strict sur le fond : le résultat est une valeur du barème."""
    t = texte.lower()
    for valeur in bareme:
        if valeur in t or t in valeur:
            return valeur
    return None


def _lire_jour(texte: str) -> date | None:
    t = texte.lower()
    if "avant-hier" in t:
        return date.today() - timedelta(days=2)
    if "hier" in t:
        return date.today() - timedelta(days=1)
    if "aujourd" in t:
        return date.today()
    try:
        return date.fromisoformat(texte.strip())
    except ValueError:
        pass
    for sep in ("/", "-", "."):
        morceaux = [m for m in texte.replace(sep, " ").split() if m.isdigit()]
        if len(morceaux) >= 2:
            jour, mois = int(morceaux[0]), int(morceaux[1])
            annee = int(morceaux[2]) if len(morceaux) > 2 else date.today().year
            if annee < 100:
                annee += 2000
            try:
                return date(annee, mois, jour)
            except ValueError:
                return None
    return None


if __name__ == "__main__":
    main()
