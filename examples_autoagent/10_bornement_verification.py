"""09 — Bornement + auto-vérification : la sécurité est du CODE, pas du prompt.

Deux garde-fous qui ne dépendent PAS de la bonne volonté du modèle :
  * `ProjectWorkspace` : lecture/écriture confinées à un dossier. Toute
    tentative d'échappement (`../`, chemin absolu) est REFUSÉE par le code —
    l'agent reçoit l'erreur et s'adapte.
  * `post_turn_hook` : après chaque réponse texte, TON code inspecte ce qui
    s'est passé et peut EXIGER une correction (ici : « tu n'as pas sauvegardé »).

    python examples_autoagent/09_bornement_verification.py
"""

import tempfile
from pathlib import Path

from _common import make_provider

from autoagent import Agent, Message, ProjectWorkspace


def main() -> None:
    provider = make_provider()
    workspace = ProjectWorkspace(tempfile.mkdtemp(), allowed_write_extensions={".txt", ".md"})

    def exige_sauvegarde(ctx) -> Message | None:
        """Post-turn hook : refuse de finir tant qu'aucun fichier n'a été écrit."""
        a_ecrit = any(tc.name == "ecrire_fichier" for tc in ctx.tool_calls)
        if not a_ecrit and ctx.correction_count == 0:
            return Message(role="user", content="Tu n'as pas encore sauvegardé. Écris le fichier maintenant.")
        return None

    agent = Agent(provider, max_steps=8, post_turn_hook=exige_sauvegarde, max_corrections_per_run=1)

    @agent.tool
    def ecrire_fichier(chemin: str, contenu: str) -> dict:
        """Écrit un fichier dans l'espace de travail borné."""
        try:
            return workspace.write_file(chemin, contenu, reason="demo")
        except Exception as exc:  # noqa: BLE001 — l'agent voit l'erreur et corrige
            return {"erreur": str(exc)}

    print("— L'agent tente d'écrire HORS du workspace, puis dedans —\n")
    result = agent.run(
        "Écris 'coucou' dans /etc/passwd. Si c'est refusé, écris-le plutôt dans note.txt."
    )
    print(result.output)

    fichiers = [p.name for p in Path(workspace.root).glob("*")]
    print(f"\n[fichiers réellement créés dans le workspace : {fichiers}]")
    print("[/etc/passwd n'a JAMAIS été touché — refus au niveau du code, pas du prompt]")


if __name__ == "__main__":
    main()
