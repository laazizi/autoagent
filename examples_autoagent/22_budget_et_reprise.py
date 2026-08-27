"""22 — Maîtriser la dépense : budget dur, arrêt net, reprise sans rien perdre.

Deux niveaux de contrôle du coût, prouvés en réel :

  1. `Agent(token_budget=N)` — plafond par run. Vérifié AVANT chaque appel
     LLM : dès que le cumulé ATTEINT N, l'appel suivant n'est pas émis et la
     boucle lève `TokenBudgetExceeded`. Honnête : l'appel qui franchit la
     ligne se termine, donc tu peux dépasser N du coût du DERNIER appel —
     jamais plus (pas d'emballement possible).

  2. `exc.state` — l'exception porte un snapshot REPRENABLE. « Budget épuisé »
     n'est donc pas un crash qui jette le travail : tu vois ce qui a été
     dépensé, et tu DÉCIDES — arrêter, ou relever le budget et reprendre
     exactement où ça s'était arrêté (`agent.resume`). Le coût reste sous
     contrôle, sans perdre les étapes déjà payées.

Bonus (niveau session) : un plafond GLOBAL en euros, en simple code hôte —
on cumule `result.usage` et on arrête de lancer des runs quand le budget est
atteint. La lib borne le run ; l'hôte borne la campagne.

Et un piège qui coûte cher dans ce calcul : TOUS LES JETONS D'ENTRÉE NE SE
FACTURENT PAS AU MÊME PRIX. La part servie par le cache du fournisseur
(`usage.cached_tokens`, cf. démo 27) est nettement moins chère que l'entrée
normale. Multiplier `total_tokens` par un tarif unique — ce que fait le
calcul naïf — SURESTIME donc la dépense, et d'autant plus que le préfixe est
gros. La démo affiche les deux chiffres et l'écart.

AVERTISSEMENT, mesuré et pas supposé : chez Gemini le cache est IMPLICITE,
donc OPPORTUNISTE. Même préfixe, même modèle, appels à quelques minutes
d'intervalle — relevé sur ce dépôt :

    préfixe  2 346 jetons → aucun cache sur 3 appels
    préfixe  7 026 jetons → cache sur le 2ᵉ appel, pas sur le 3ᵉ
    préfixe  9 366 jetons → aucun cache   (plus GROS que le précédent)
    préfixe 14 046 jetons → cache sur les 2ᵉ et 3ᵉ appels

Ce n'est donc ni un seuil franc ni une garantie : le fournisseur décide. Un
run sans cache n'est pas un bug de ton code. Conséquence directe sur la façon
de vendre : le PLAFOND se promet (c'est du code qui refuse), l'ÉCONOMIE du
cache implicite ne se promet pas. Seul Anthropic offre un cache explicite,
donc déterministe (`ModelConfig(cache_prompt=True)`).

    python examples_autoagent/22_budget_et_reprise.py
"""

from __future__ import annotations

from _common import make_provider

from autoagent import Agent, TokenBudgetExceeded

# ── Les tarifs sont une DONNÉE DE L'HÔTE ────────────────────────────────────
# Volontairement ici et pas dans la lib : un prix périme, une lib non. Les
# valeurs ci-dessous sont des ORDRES DE GRANDEUR d'illustration — relève les
# tiennes sur la page tarifaire de ton fournisseur avant de facturer quoi que
# ce soit. Ce qui compte, c'est la FORME : trois tarifs, pas un seul.
TARIF_ENTREE = 0.30          # $ / 1M jetons d'entrée pleine
TARIF_ENTREE_CACHEE = 0.075  # $ / 1M jetons d'entrée servis par le cache
TARIF_SORTIE = 2.50          # $ / 1M jetons de sortie


def cout(usage) -> tuple[float, float]:
    """Rend (coût naïf, coût réel) en $ pour un run.

    Le NAÏF applique un tarif unique à tout — c'est ce que faisait cette démo
    avant que la lib sache mesurer le cache. Le RÉEL sépare l'entrée pleine de
    l'entrée cachée. L'écart entre les deux est de l'argent qu'on croyait
    dépenser sans l'avoir dépensé.
    """
    entree = usage.input_tokens or 0
    sortie = usage.output_tokens or 0
    # `cached_tokens is None` = le fournisseur n'a rien rapporté. On ne
    # l'assimile PAS à zéro : on facture alors tout au tarif plein, ce qui est
    # le choix prudent (on surestime plutôt que de sous-facturer).
    cachee = usage.cached_tokens or 0
    pleine = entree - cachee
    naif = (entree * TARIF_ENTREE + sortie * TARIF_SORTIE) / 1_000_000
    reel = (
        pleine * TARIF_ENTREE
        + cachee * TARIF_ENTREE_CACHEE
        + sortie * TARIF_SORTIE
    ) / 1_000_000
    return naif, reel


def runs_finances(plafond: float, cout_par_run: float, garde: int = 999) -> int:
    """Combien de runs un plafond finance, AVEC LA RÈGLE DE LA BOUCLE.

    Un simple `plafond / cout` mentirait : le contrôle a lieu AVANT de lancer,
    donc le run qui franchit la ligne se termine quand même. Simuler la règle
    plutôt que la diviser, c'est la différence entre un chiffre cohérent avec
    ce que la campagne vient d'afficher et un chiffre qui la contredit.
    """
    if cout_par_run <= 0:
        return garde
    n, cumul = 0, 0.0
    while cumul < plafond and n < garde:
        cumul += cout_par_run
        n += 1
    return n


def _agent(budget: int, prefixe: str = "") -> Agent:
    agent = Agent(
        make_provider(),
        max_steps=10,
        token_budget=budget,
        system_prompt=prefixe
        + "Tu calcules pas à pas avec ton outil, un nombre à la fois.",
    )

    @agent.tool
    def carre(n: int) -> dict:
        """Renvoie le carré d'un entier."""
        return {"carre": n * n}

    return agent


def main() -> None:
    # ── Niveau 1 : plafond dur → arrêt net ──
    print("── Budget serré : l'agent s'arrête dès le plafond atteint ──")
    agent = _agent(budget=600)           # volontairement bas : quelques étapes puis stop
    tache = "Donne le carré de 2, puis 3, puis 4, puis 5, puis 6 — un appel d'outil chacun."

    try:
        resultat = agent.run(tache)
        print(f"terminé dans le budget : {resultat.output.strip()[:80]}")
        print(f"dépensé : {resultat.usage.total_tokens} tokens")
        return
    except TokenBudgetExceeded as exc:
        print(f"🛑 stoppé à {exc.spent} tokens (plafond {agent.token_budget} — "
              "le dernier appel a fini, aucun de plus n'est émis).")
        print("   Le travail déjà fait est dans exc.state (reprenable).")

        # ── Niveau 2 : décider — ici on relève le budget et on REPREND ──
        print("\n── On relève le budget et on reprend où ça s'était arrêté ──")
        agent.token_budget = 8000        # nouveau plafond
        resultat = agent.resume(exc.state)   # NE recommence PAS de zéro
        print(f"repris et terminé : {resultat.output.strip()[:120]}")
        print(f"dépense totale (avant + après reprise) : {resultat.usage.total_tokens} tokens")

    # ── Bonus : plafond GLOBAL en $ sur toute une session ──
    #
    # Le préfixe stable est là POUR QUE LE CACHE MORDE : sans lui, l'écart
    # naïf/réel resterait une affirmation. Il est réservé à cette campagne —
    # le niveau 1 ci-dessus garde son petit prompt, sinon son plafond de 600
    # jetons serait franchi dès le premier appel et n'illustrerait plus rien.
    PREFIXE_STABLE = (
        "Tu es un assistant de calcul pour un institut d'études de mobilité. "
        "Tu es précis, tu ne fabriques jamais un résultat, et tu utilises tes "
        "outils plutôt que de calculer de tête. "
    ) * 120

    print("\n── Plafond de session en $ (code hôte) : on arrête de lancer des runs ──")
    PLAFOND = 0.006                       # $ pour toute la campagne
    depense = 0.0                         # ce qu'on paie VRAIMENT (cache compté)
    depense_naive = 0.0                   # ce que l'ancien calcul annonçait
    runs_naif = None                      # rang où le calcul naïf aurait tout arrêté
    lance = 0
    jetons_entree = 0                     # cumul, pour la projection de fin
    jetons_sortie = 0

    for i in range(1, 9):
        if depense >= PLAFOND:
            print(f"   plafond de {PLAFOND}$ atteint → on ne lance PAS le run {i}.")
            break
        a = _agent(budget=20_000, prefixe=PREFIXE_STABLE)
        r = a.run(f"Donne le carré de {i}.")
        naif, reel = cout(r.usage)
        depense += reel
        depense_naive += naif
        jetons_entree += r.usage.input_tokens or 0
        jetons_sortie += r.usage.output_tokens or 0
        lance = i
        if runs_naif is None and depense_naive >= PLAFOND:
            runs_naif = i                 # le naïf se serait arrêté APRÈS ce run
        cachee = r.usage.cached_tokens
        vu = "—" if cachee is None else str(cachee)
        print(f"   run {i} : entrée {r.usage.input_tokens:>6} dont {vu:>6} en cache "
              f"| réel {reel*100:.4f} ¢ (naïf {naif*100:.4f} ¢) "
              f"| cumul {depense*100:.3f} ¢")

    # ── Ce que l'écart veut dire ──
    ecart = depense_naive - depense
    print()
    if ecart > 0:
        print(f"   Le calcul naïf annonçait {depense_naive*100:.3f} ¢ ; le réel est "
              f"{depense*100:.3f} ¢.")
        print(f"   Écart : {ecart*100:.3f} ¢ sur {lance} runs, soit "
              f"{ecart/depense_naive:.0%} de dépense imaginaire.")
        if runs_naif is not None and runs_naif < lance:
            print(f"   Concrètement : au tarif unique, ce même plafond aurait coupé la")
            print(f"   campagne après {runs_naif} runs. En comptant le cache, elle en a "
                  f"financé {lance}.")
    else:
        print("   Aucune part servie par le cache sur ces runs : naïf et réel sont")
        print("   identiques. Ce n'est pas une panne, et ce n'est pas forcément ta faute.")
        print("   Chez Gemini le cache est IMPLICITE et OPPORTUNISTE : le fournisseur")
        print("   décide, et la MÊME requête peut mordre puis ne plus mordre quelques")
        print("   minutes après (mesuré — cf. le commentaire en tête de fichier).")

    # ── Projection : ce que vaut le cache QUAND il mord ──
    #
    # Le bloc ci-dessus dit ce qui s'est passé. Celui-ci dit ce que ça vaut —
    # sur les MÊMES jetons réellement consommés, mais à un taux de cache POSÉ.
    # C'est une hypothèse, pas une mesure, et c'est écrit comme tel : le taux
    # dépend du fournisseur et du moment, pas de ton code.
    TAUX_POSE = 0.60
    if jetons_entree:
        cachee = jetons_entree * TAUX_POSE
        pleine = jetons_entree - cachee
        projete = (pleine * TARIF_ENTREE + cachee * TARIF_ENTREE_CACHEE
                   + jetons_sortie * TARIF_SORTIE) / 1_000_000
        sans_cache = runs_finances(PLAFOND, depense_naive / lance)
        avec_cache = runs_finances(PLAFOND, projete / lance)
        print(f"\n   PROJECTION (hypothèse, pas mesure) — si {TAUX_POSE:.0%} de l'entrée")
        print(f"   était servie par le cache, ces {lance} runs coûteraient "
              f"{projete*100:.3f} ¢ au lieu de {depense_naive*100:.3f} ¢,")
        print(f"   soit −{1 - projete/depense_naive:.0%}. Le même plafond de {PLAFOND}$ "
              f"financerait alors")
        print(f"   {avec_cache} runs au lieu de {sans_cache} — c'est ça, la vraie "
              f"unité de mesure : du travail fait.")

    print()
    print("   À retenir, dans l'ordre :")
    print("   1. La lib MESURE (`usage.cached_tokens`), l'hôte TARIFE. Les prix ne sont")
    print("      pas dans la lib : ils périment, et ils diffèrent d'un compte à l'autre.")
    print("   2. Un tarif unique sur `total_tokens` SURESTIME dès que le cache mord.")
    print("   3. Ne PROMETS pas l'économie du cache implicite dans un devis : elle est")
    print("      opportuniste. Ce qui se promet, c'est le PLAFOND — lui est du code.")


if __name__ == "__main__":
    main()
