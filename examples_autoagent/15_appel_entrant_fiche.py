"""15 — Appel entrant : identifier l'appelant, ou créer sa fiche.

Un standard téléphonique reçoit un appel avec un NUMÉRO. L'agent applique une
CASCADE de repli, en choisissant lui-même le bon outil à chaque étape :

    1. fiche_locale(numero)        — connu en base locale ? → on salue.
    2. sinon annuaire_externe(...)  — connu dans l'AUTRE système (CRM) ?
    3. sinon on DISCUTE : demander_a_la_personne(...) pour obtenir nom + adresse,
       puis creer_fiche(...) — la fiche existe désormais en local.

C'est l'agent qui décide de l'escalade (pas un `if` en dur) : on lui donne les
outils + la procédure, il enchaîne. Ici le CRM et la personne sont simulés pour
que la démo tourne seule ; en prod on branche un vrai CRM / la voix.

    python examples_autoagent/15_appel_entrant_fiche.py
"""

from __future__ import annotations

from _common import make_provider

from autoagent import Agent

# ── Les "sources" (simulées) ────────────────────────────────────────────────
FICHES_LOCALES: dict[str, dict] = {
    "0611111111": {"nom": "Marie Dupont", "adresse": "12 rue de la Paix, Lyon"},
}
CRM_EXTERNE: dict[str, dict] = {                       # l'AUTRE système
    "0622222222": {"nom": "Paul Martin", "adresse": "5 av. Jean Jaurès, Grenoble"},
}
PERSONNES: dict[str, dict] = {                         # ce que dira l'appelant inconnu
    "0633333333": {"nom": "Sophie Bernard", "adresse": "8 place Bellecour, Lyon"},
}


def traiter_appel(provider, numero: str) -> str:
    etat = {"num": numero}                             # l'appelant courant (pour la voix simulée)

    agent = Agent(provider, max_steps=10, temperature=0.0, system_prompt=(
        "Tu es le standard téléphonique. Un appel arrive avec un numéro. "
        "Procédure STRICTE, dans l'ordre, en t'arrêtant dès que tu as la fiche :\n"
        "1) fiche_locale(numero) ;\n"
        "2) si absente → annuaire_externe(numero) ;\n"
        "3) si toujours absente → demande à la personne son NOM puis son ADRESSE "
        "(demander_a_la_personne), puis creer_fiche(numero, nom, adresse).\n"
        "Termine par une phrase d'accueil personnalisée, en disant d'où vient la "
        "fiche (base locale, CRM, ou fiche créée à l'instant)."))

    @agent.tool
    def fiche_locale(numero: str) -> dict:
        """Cherche l'appelant dans la base LOCALE."""
        f = FICHES_LOCALES.get(numero)
        print(f"   [local]    {numero} → {'trouvé' if f else 'inconnu'}")
        return {"trouve": bool(f), "fiche": f}

    @agent.tool
    def annuaire_externe(numero: str) -> dict:
        """Cherche l'appelant dans l'AUTRE système (CRM externe)."""
        f = CRM_EXTERNE.get(numero)
        print(f"   [CRM]      {numero} → {'trouvé' if f else 'inconnu'}")
        return {"trouve": bool(f), "fiche": f}

    @agent.tool
    def demander_a_la_personne(question: str) -> dict:
        """Pose une question à l'appelant et renvoie sa réponse (voix simulée)."""
        profil = PERSONNES.get(etat["num"], {})
        q = question.lower()
        if any(w in q for w in ("nom", "prénom", "appelez", "identité", "qui")):
            rep = profil.get("nom", "je préfère ne pas dire")
        elif any(w in q for w in ("adresse", "habitez", "domicile", "rue", "ville", "où")):
            rep = profil.get("adresse", "je ne souhaite pas la donner")
        else:
            rep = "Pouvez-vous reformuler ?"
        print(f"   [personne] « {question} » → « {rep} »")
        return {"reponse": rep}

    @agent.tool
    def creer_fiche(numero: str, nom: str, adresse: str) -> dict:
        """Crée la fiche de l'appelant dans la base LOCALE."""
        FICHES_LOCALES[numero] = {"nom": nom, "adresse": adresse}
        print(f"   [fiche]    créée pour {numero} : {nom} — {adresse}")
        return {"cree": True, "fiche": FICHES_LOCALES[numero]}

    return agent.run(f"Appel entrant du numéro {numero}. Identifie l'appelant.").output


def main() -> None:
    provider = make_provider()
    for numero, titre in [
        ("0611111111", "A — numéro CONNU en local"),
        ("0622222222", "B — inconnu en local, TROUVÉ dans le CRM externe"),
        ("0633333333", "C — inconnu partout → on crée la fiche en discutant"),
    ]:
        print(f"\n{'═'*70}\nAPPEL {titre}\n{'═'*70}")
        print(traiter_appel(provider, numero))

    print(f"\n[base locale après les appels : {list(FICHES_LOCALES)}]  "
          f"← le 0633333333 y est maintenant")


if __name__ == "__main__":
    main()
