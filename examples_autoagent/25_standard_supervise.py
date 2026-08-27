"""25 — Standard téléphonique supervisé : l'agent choisit, l'humain autorise.

La démo faite pour être PROJETÉE. Elle tient toute la thèse en un fichier :

  1. L'agent CHOISIT ses outils. On ne code pas la cascade d'identification :
     on donne quatre outils et une procédure. Selon ce qu'il trouve, il fait
     une, deux ou cinq étapes. Ajouter une source, c'est ajouter un outil —
     pas une branche dans un `if`.

  2. Il AGIT. La dernière étape écrit dans la base. C'est à partir de là que
     « qu'a-t-il le droit de faire » devient une vraie question.

  3. L'humain garde l'écriture. `tool_policy` lève `ApprovalRequired` sur
     `creer_fiche` : la boucle s'arrête AVANT que le moindre outil du tour ne
     s'exécute, et rend un instantané reprenable. Tu autorises, ou tu refuses.

  4. Un refus ne casse rien. L'agent voit le verdict, se replanifie, et
     annonce honnêtement qu'il n'a pas pu créer la fiche. La base est intacte.

Le fournisseur est déclaré EXPLICITEMENT ci-dessous, sans passer par
l'utilitaire des démos : c'est la ligne qu'un responsable sécurité veut lire.

    python examples_autoagent/25_standard_supervise.py
"""

from __future__ import annotations

from _common import load_env        # lit .env — seul emprunt à l'utilitaire

from autoagent import (Agent, ApprovalRequired, ModelConfig, ToolPolicyContext,
                       create_provider)

load_env()

# ── UNE ligne à changer, et rien d'autre dans tout le fichier. ───────────────
CONFIG = ModelConfig(provider="deepseek",  model="deepseek-chat")
# CONFIG = ModelConfig(provider="gemini",    model="gemini-3.6-flash")
# CONFIG = ModelConfig(provider="openai",    model="gpt-5-mini")
# CONFIG = ModelConfig(provider="anthropic", model="claude-sonnet-5")
# CONFIG = ModelConfig(provider="openai",    model="mistral-small",
#                      base_url="http://localhost:11434/v1")   # modèle hébergé chez nous

# ── Les « sources » (simulées pour que la démo tourne seule) ─────────────────
FICHES_LOCALES: dict[str, dict] = {
    "0611111111": {"nom": "Marie Dupont", "adresse": "12 rue de la Paix, Lyon"},
}
CRM_EXTERNE: dict[str, dict] = {
    "0622222222": {"nom": "Paul Martin", "adresse": "5 av. Jean Jaurès, Grenoble"},
}
PERSONNES: dict[str, dict] = {          # ce que répondra un appelant inconnu
    "0633333333": {"nom": "Sophie Bernard", "adresse": "8 place Bellecour, Lyon"},
    "0644444444": {"nom": "Karim Haddad", "adresse": "3 rue Garibaldi, Villeurbanne"},
}

PROCEDURE = (
    "Tu es le standard téléphonique. Un appel arrive avec un numéro. "
    "Procédure STRICTE, dans l'ordre, en t'arrêtant dès que tu as la fiche :\n"
    "1) fiche_locale(numero) ;\n"
    "2) si absente → annuaire_externe(numero) ;\n"
    "3) si toujours absente → demande à la personne son NOM puis son ADRESSE "
    "(demander_a_la_personne), puis creer_fiche(numero, nom, adresse).\n"
    "Si la création de fiche est refusée, ne réessaie pas.\n\n"
    "Ta réponse finale est UNIQUEMENT ce que la personne entend au téléphone : "
    "une ou deux phrases, aucun commentaire sur ta procédure ni sur tes outils."
)


def construire(decisions: dict[str, str]) -> Agent:
    """Un agent neuf par appel : le second appelant n'hérite pas du premier."""

    def politique(ctx: ToolPolicyContext):
        """La règle du bureau : écrire dans la base demande un accord humain.

        Au retour de `resume`, les appels en attente REPASSENT par ici. D'où
        les trois branches : autorisé (None), refusé (une chaîne que le modèle
        lira), pas encore tranché (on repause — c'est idempotent).
        """
        if ctx.call.name != "creer_fiche":
            return None                                  # le reste passe
        verdict = decisions.get(ctx.call.name)
        if verdict == "autorise":
            return None
        if verdict == "refuse":
            return "Création de fiche refusée par le superviseur."
        raise ApprovalRequired(
            f"« creer_fiche » veut écrire dans la base — {ctx.call.arguments}")

    agent = Agent(create_provider(CONFIG), max_steps=10, temperature=0.0,
                  system_prompt=PROCEDURE, tool_policy=politique)

    @agent.tool
    def fiche_locale(numero: str) -> dict:
        """Cherche le numéro dans la base locale."""
        trouve = FICHES_LOCALES.get(numero)
        print(f"   [local]    {numero} → {'trouvé' if trouve else 'inconnu'}")
        return trouve or {"trouve": False}

    @agent.tool
    def annuaire_externe(numero: str) -> dict:
        """Cherche le numéro dans le CRM externe."""
        trouve = CRM_EXTERNE.get(numero)
        print(f"   [CRM]      {numero} → {'trouvé' if trouve else 'inconnu'}")
        return trouve or {"trouve": False}

    @agent.tool
    def demander_a_la_personne(numero: str, question: str) -> dict:
        """Pose une question à l'appelant (voix simulée ici)."""
        personne = PERSONNES.get(numero, {})
        reponse = (personne.get("nom") if "nom" in question.lower()
                   else personne.get("adresse", "je ne sais pas"))
        print(f"   [personne] « {question} » → « {reponse} »")
        return {"reponse": reponse}

    @agent.tool
    def creer_fiche(numero: str, nom: str, adresse: str) -> dict:
        """Écrit une nouvelle fiche dans la base locale."""
        FICHES_LOCALES[numero] = {"nom": nom, "adresse": adresse}
        print(f"   [fiche]    ÉCRITE pour {numero} : {nom} — {adresse}")
        return {"cree": True}

    return agent


def traiter(numero: str, decision_humaine: str | None) -> None:
    """Un appel. `decision_humaine` = ce que tu répondras si la boucle pause."""
    decisions: dict[str, str] = {}
    agent = construire(decisions)
    demande = f"Appel entrant du numéro {numero}. Identifie l'appelant."

    try:
        print(f"   {agent.run(demande).output.strip()}")
        return
    except ApprovalRequired as pause:
        print(f"\n   ⏸  LA BOUCLE S'ARRÊTE — aucun outil de ce tour n'a tourné")
        print(f"      {pause}")
        print(f"      outils en attente : {[c.name for c in pause.calls]}")

        if decision_humaine == "autorise":
            print("      → tu AUTORISES\n")
            decisions["creer_fiche"] = "autorise"
        else:
            print("      → tu REFUSES\n")
            decisions["creer_fiche"] = "refuse"

        resultat = agent.resume(pause.state)
        print(f"   {resultat.output.strip()}")


def main() -> None:
    print(f"[fournisseur déclaré dans le fichier : {CONFIG.provider} / {CONFIG.model}]")

    scenarios = [
        ("0611111111", None, "A — connu en local : aucune écriture, aucune pause"),
        ("0633333333", "autorise", "B — inconnu partout : il veut écrire → tu AUTORISES"),
        ("0644444444", "refuse", "C — inconnu partout : il veut écrire → tu REFUSES"),
    ]
    for numero, decision, titre in scenarios:
        print(f"\n{'═' * 74}\nAPPEL {titre}\n{'═' * 74}")
        traiter(numero, decision)

    print(f"\n{'═' * 74}")
    print(f"Base locale à la fin : {sorted(FICHES_LOCALES)}")
    print()
    print("  0611111111  était déjà là")
    print("  0633333333  ajouté — tu l'avais autorisé")
    if "0644444444" in FICHES_LOCALES:
        print("  0644444444  ⚠ PRÉSENT alors que tu avais refusé — ce serait un bug")
    else:
        print("  0644444444  ABSENT — tu avais refusé, et le refus a tenu")
    print()
    print("Le refus n'a rien cassé : l'agent a vu le verdict, s'est replanifié,")
    print("et a terminé son appel proprement.")
    print()
    print("Regarde bien sa dernière phrase à Karim. D'un run à l'autre, parfois")
    print("il annonce le refus, parfois il n'en parle pas — la consigne le lui")
    print("demande, il ne la suit pas toujours. C'est EXACTEMENT ce qu'une")
    print("consigne ne garantit pas.")
    print()
    print("Ce qui est garanti, c'est la ligne du dessus : la fiche n'existe pas.")
    print("Elle n'existe pas parce qu'une fonction Python a refusé — pas parce")
    print("qu'on a demandé gentiment au modèle de ne pas la créer.")


if __name__ == "__main__":
    main()
