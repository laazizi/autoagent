"""31 — Synthèse par l'exemple : le modèle écrit l'outil, TES cas décident.

La démo 06 fait écrire au modèle l'outil qui lui manque, et le modèle fournit
ses propres auto-tests. Le point faible tient en une phrase : il écrit les tests
qui le jugent. Ici on inverse qui juge.

    synthesize_tool(builder, "…ce que l'outil doit faire…", exemples)

Tu apportes des cas de VÉRITÉ (arguments → résultat attendu). Le modèle écrit un
outil. Le code l'exécute en bac à sable sur tes cas et compare. Un outil qui
rate est jeté — fichier compris — et le modèle reçoit deux ou trois cas ratés
pour corriger. Un outil qui passe est enregistré.

LE PIÈGE, ET LA PARADE. Un modèle à qui l'on demande de « faire passer ces
cas » écrit volontiers un outil qui les traite un par un (`if entree == …`) :
100 % de réussite, 0 % d'utilité. La parade : les exemples sont COUPÉS. Le
modèle en voit 60 % ; **les 40 % restants ne lui sont jamais transmis**, même
quand ils ratent — il apprend seulement COMBIEN ont raté. « Fais passer ces
cas » devient « trouve la règle ».

Le cas : des lignes de journal capteur à transformer en enregistrement propre.
La règle a trois pièges (deux formats de date, un niveau parfois absent, des
espaces variables) — un modèle qui code les exemples par cœur y tombera.

Aucune clé Anthropic requise ; Gemini / OpenAI / DeepSeek via .env.

    python examples_autoagent/31_synthese_par_exemple.py
"""

from __future__ import annotations

import time

from _common import ROOT, make_provider

from autoagent import Example, synthesize_tool
from autoagent.dynamic import DynamicToolBuilder

# ── La vérité : dix lignes réelles et ce qu'on attend de chacune ─────────────
# Trois formats se mélangent, comme dans un vrai export : ISO avec « T », ISO
# avec espace, et « JJ/MM/AAAA hh:mm ». Le niveau manque parfois (→ "INFO").
EXEMPLES = [
    Example({"ligne": "2026-08-19T14:02:11 ERROR cmp-lyo-07 timeout 30s"},
            {"date": "2026-08-19", "heure": "14:02", "niveau": "ERROR", "capteur": "cmp-lyo-07"}),
    Example({"ligne": "2026-08-19 14:03:40  INFO  cmp-lyo-01  1240 passages"},
            {"date": "2026-08-19", "heure": "14:03", "niveau": "INFO", "capteur": "cmp-lyo-01"}),
    Example({"ligne": "19/08/2026 14:05 WARN cmp-lyo-12 latence 210ms"},
            {"date": "2026-08-19", "heure": "14:05", "niveau": "WARN", "capteur": "cmp-lyo-12"}),
    Example({"ligne": "2026-08-20T08:15:00 cmp-gre-03 boot ok"},
            {"date": "2026-08-20", "heure": "08:15", "niveau": "INFO", "capteur": "cmp-gre-03"}),
    Example({"ligne": "20/08/2026 08:16 ERROR cmp-gre-03 connexion refusee"},
            {"date": "2026-08-20", "heure": "08:16", "niveau": "ERROR", "capteur": "cmp-gre-03"}),
    Example({"ligne": "2026-08-20 08:17:59 DEBUG cmp-gre-04 trame 0x1F"},
            {"date": "2026-08-20", "heure": "08:17", "niveau": "DEBUG", "capteur": "cmp-gre-04"}),
    Example({"ligne": "2026-08-21T23:59:59 WARN cmp-nan-09 pile faible"},
            {"date": "2026-08-21", "heure": "23:59", "niveau": "WARN", "capteur": "cmp-nan-09"}),
    Example({"ligne": "21/08/2026 00:01 cmp-nan-09 reprise"},
            {"date": "2026-08-21", "heure": "00:01", "niveau": "INFO", "capteur": "cmp-nan-09"}),
    Example({"ligne": "2026-08-22 12:00:00 ERROR cmp-lyo-07 timeout 30s"},
            {"date": "2026-08-22", "heure": "12:00", "niveau": "ERROR", "capteur": "cmp-lyo-07"}),
    Example({"ligne": "22/08/2026 12:30   INFO   cmp-lyo-02   ok"},
            {"date": "2026-08-22", "heure": "12:30", "niveau": "INFO", "capteur": "cmp-lyo-02"}),
]

# Une ligne que NI le modèle NI la boucle n'ont vue : le vrai test final.
JAMAIS_VUE = "23/08/2026 06:45 cmp-bdx-11 demarrage"
ATTENDU = {"date": "2026-08-23", "heure": "06:45", "niveau": "INFO", "capteur": "cmp-bdx-11"}


def main() -> None:
    provider = make_provider()
    builder = DynamicToolBuilder(provider, tools_dir=ROOT / ".autoagent" / "tools_demo31",
                                 timeout=15)
    print(f"Modèle : {provider.config.provider} / {provider.config.model}")
    print(f"{len(EXEMPLES)} cas de vérité · 40 % cachés au modèle · 5 essais maximum")
    print("─" * 76)

    debut = time.monotonic()
    res = synthesize_tool(
        builder,
        goal=("Parse one sensor log line into a record with keys date (YYYY-MM-DD), "
              "heure (HH:MM), niveau (ERROR/WARN/INFO/DEBUG; INFO when absent) and "
              "capteur (the token that starts with 'cmp-'). Dates may be ISO with 'T' "
              "or space, or DD/MM/YYYY."),
        examples=EXEMPLES,
        tool_name="parser_ligne_capteur",
        holdout=0.4,
        max_attempts=5,
        seed=0,
    )
    duree = time.monotonic() - debut

    print(f"\n  cas montrés au modèle : {res.shown} · cas cachés : {res.holdout}\n")
    print("  essai  montrés  cachés   verdict")
    for a in res.attempts:
        caches = "—" if a.holdout_passed is None else f"{a.holdout_passed}/{a.holdout_total}"
        if a.error:
            verdict = f"construction ratée : {a.error[:50]}"
        elif a.accepted:
            verdict = "ACCEPTÉ"
        elif a.holdout_passed is None:
            verdict = "rate des cas montrés → retour au modèle avec 3 exemples"
        else:
            verdict = "passe les montrés, rate des cachés → « trop spécifique », sans détail"
        print(f"    {a.index}     {a.shown_passed}/{a.shown_total}     {caches:>5}    {verdict}")

    print(f"\n  {res.summary()}")
    print(f"  durée {duree:.1f} s · coût "
          f"{(res.usage.total_tokens if res.usage else 0) or 0} jetons")

    print("\n" + "─" * 76)
    if res.accepted:
        obtenu = res.tool(ligne=JAMAIS_VUE)
        ok = obtenu == ATTENDU
        print("LE VRAI TEST — une ligne que personne n'a vue, ni le modèle ni la boucle :\n")
        print(f"  {JAMAIS_VUE}")
        print(f"  → {obtenu}")
        print(f"  {'✓ juste' if ok else '✗ FAUX — attendu ' + str(ATTENDU)}")
        print()
        print("  L'outil est en bac à sable, dans .autoagent/tools_demo31/. Il n'est PAS")
        print("  promu : ça, c'est ton geste, par le manifeste à empreinte (démo 07).")
    else:
        print("REFUSÉ après", len(res.attempts), "essais — et c'est un résultat honnête.")
        print("  Aucun outil non validé n'est resté sur le disque. Relis les essais :")
        print("  s'il rate toujours les cachés, la règle est peut-être mal décrite ;")
        print("  s'il rate les montrés, les exemples se contredisent peut-être.")

    print()
    print("  Ce que la boucle ne fait pas : rendre le modèle plus intelligent. Elle")
    print("  convertit des essais en justesse — possible seulement parce que la")
    print("  vérité était déjà connue. Là où tu as les données ET la bonne réponse,")
    print("  elle est très rentable. Ailleurs, elle n'a pas de juge.")


if __name__ == "__main__":
    main()
