"""26 — Le résultat trop gros : la borne est du code, et elle coupe au MILIEU.

La démo de la première butée. Un seul outil, un seul prompt, deux runs —
la seule différence entre les deux tient dans un argument :

    Agent(...)                                 # run A : rien ne borne
    Agent(..., max_tool_result_chars=4000)     # run B : la butée est vissée

Ce que la démo prouve, dans l'ordre :

  1. Un outil non borné injecte TOUT dans la conversation. Ici 40 000
     caractères pour une question dont la réponse fait deux lignes. Le prix
     est en jetons, et il est réel — la démo l'affiche.

  2. La borne n'est pas une consigne, c'est une fonction Python. Le modèle
     ne peut pas la dépasser en insistant.

  3. Elle coupe au MILIEU, et c'est là que tout se joue. Le bilan du journal
     est à la DERNIÈRE ligne. Une coupe naïve (les 4 000 premiers caractères)
     le perdrait, et l'agent répondrait faux avec aplomb. La coupe milieu
     garde la tête — la forme des données — ET la queue — les totaux.

  4. La marque de coupure compte dans le budget. Une limite qu'on peut
     dépasser n'est pas une limite : la démo vérifie len() <= 4000.

    python examples_autoagent/26_resultat_trop_gros.py
"""

from __future__ import annotations

from _common import load_env        # lit .env — seul emprunt à l'utilitaire

from autoagent import Agent, ModelConfig, create_provider

load_env()

# ── UNE ligne à changer, et rien d'autre dans tout le fichier. ───────────────
CONFIG = ModelConfig(provider="deepseek",  model="deepseek-chat")
# CONFIG = ModelConfig(provider="gemini",    model="gemini-3.7-flash")
# CONFIG = ModelConfig(provider="openai",    model="gpt-4o-mini")
# CONFIG = ModelConfig(provider="anthropic", model="claude-sonnet-4-5")


def _n(valeur: int) -> str:
    """Milliers séparés par une espace, sans toucher à la ponctuation."""
    return f"{valeur:,}".replace(",", " ")


BORNE = 4_000          # la valeur conseillée par la doc (§24.1)
QUESTION = ("Lis le journal du collecteur, puis réponds en deux lignes : "
            "combien d'ERROR au total, et quelle erreur revient le plus souvent ?")


# ── Le journal : gros, réaliste, et son bilan est à la FIN ───────────────────
def _journal() -> str:
    """400 lignes de journal. Les totaux sont dans les trois dernières."""
    lignes = [
        "# collecteur-mobilite v4.2 — journal d'exécution",
        "# horodatage | niveau | capteur | message",
    ]
    erreurs = 0
    for i in range(400):
        minute, seconde = divmod(i * 7, 60)
        heure = f"2026-08-19T14:{minute % 60:02d}:{seconde:02d}"
        capteur = f"cmp-lyo-{i % 12:02d}"
        if i % 11 == 3:                                  # l'erreur dominante
            erreurs += 1
            lignes.append(f"{heure} | ERROR | cmp-lyo-07 | timeout après 30 s "
                          f"sur la lecture du compteur, tentative {i % 4 + 1}/4")
        elif i % 37 == 5:                                # une erreur rare
            erreurs += 1
            lignes.append(f"{heure} | ERROR | {capteur} | connexion refusée par "
                          f"postgres:5432 (pool saturé, 40/40 connexions)")
        else:
            lignes.append(f"{heure} | INFO  | {capteur} | 1 240 passages agrégés, "
                          f"latence 42 ms, file d'attente vide")
    lignes += [
        "# ─── BILAN DE L'EXÉCUTION ───",
        f"# ERROR total : {erreurs}",
        "# erreur la plus fréquente : timeout cmp-lyo-07",
    ]
    return "\n".join(lignes)


JOURNAL = _journal()


def _construire(borne: int | None) -> Agent:
    """Le même agent, le même outil. Seule la borne change."""
    agent = Agent(create_provider(CONFIG), max_tool_result_chars=borne)

    @agent.tool(description="Renvoie le journal complet du collecteur.")
    def lire_journal() -> str:
        return JOURNAL

    return agent


def _mesurer(resultat) -> tuple[int, str]:
    """Longueur du message d'outil réellement injecté, et son contenu."""
    for message in resultat.messages:
        if message.role == "tool":
            return len(message.content), message.content
    return 0, ""


def main() -> None:
    print(f"Journal produit : {_n(len(JOURNAL))} caractères, "
          f"{JOURNAL.count(chr(10)) + 1} lignes")
    print(f"Le bilan est à la DERNIÈRE ligne : {JOURNAL.splitlines()[-1]}")
    print(f"Modèle : {CONFIG.provider} / {CONFIG.model}")
    print()

    mesures = {}
    for etiquette, borne in (("A · sans borne", None), (f"B · borne à {BORNE}", BORNE)):
        print("─" * 74)
        print(f"RUN {etiquette}")
        resultat = _construire(borne).run(QUESTION)
        taille, contenu = _mesurer(resultat)
        jetons = resultat.usage.total_tokens if resultat.usage else None
        mesures[etiquette] = (taille, jetons)

        print(f"  injecté dans la conversation : {_n(taille)} caractères")
        print(f"  jetons du run                : "
              f"{_n(jetons) if jetons is not None else 'non rapportés'}")
        if "[TRUNCATED" in contenu:
            marque = [l for l in contenu.splitlines() if "[TRUNCATED" in l][0]
            print(f"  marque au milieu             : {marque.strip()}")
            assert taille <= BORNE, "la borne a été dépassée — ce serait un bug"
            print(f"  borne respectée              : {taille} <= {BORNE} ✓")
        print(f"  réponse du modèle            : "
              f"{' / '.join(resultat.output.split(chr(10))[:2])}")
        print()

    (ta, ja), (tb, jb) = mesures["A · sans borne"], mesures[f"B · borne à {BORNE}"]
    print("─" * 74)
    print("CE QU'IL FAUT RETENIR")
    print()
    print(f"  Le même outil a injecté {_n(ta)} caractères, puis {_n(tb)}.")
    if ja and jb:
        print(f"  Le même travail a coûté {_n(ja)} jetons, puis {_n(jb)} "
              f"— soit {100 - round(jb / ja * 100)} % de moins.")
    print()
    print("  Et la réponse reste juste dans le run B, parce que la coupe est au")
    print("  MILIEU : le bilan de la dernière ligne a survécu. Une coupe qui")
    print("  garde les 4 000 premiers caractères aurait perdu les totaux, et")
    print("  l'agent aurait répondu faux — sans jamais le signaler.")
    print()
    print("  Enfin : ce n'est pas une consigne dans le prompt. C'est un argument")
    print("  de constructeur. Le modèle ne peut pas l'ignorer, il ne le voit même")
    print("  pas — il reçoit un résultat déjà borné, avec une marque qui lui dit")
    print("  d'affiner sa requête.")


if __name__ == "__main__":
    main()
