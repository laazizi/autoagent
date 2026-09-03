"""27 — Le cache de prompt : la même question, trois fois, de moins en moins chère.

Un agent n'a pas de mémoire : à CHAQUE tour, on lui renvoie tout — le prompt
système, les schémas de tous ses outils, et l'historique. Sur un run de huit
étapes, le même préfixe part donc huit fois.

Les fournisseurs savent servir ce préfixe depuis un cache. La démo montre les
trois choses qu'il faut avoir comprises :

  1. LE PREMIER RUN NE GAGNE RIEN. Le cache s'écrit ; il n'y a rien à lire.
     Le fournisseur ne rapporte pas de part servie — on affiche « inconnu »,
     pas « zéro ». Les deux ne veulent pas dire la même chose.

  2. LES SUIVANTS PAIENT MOINS. La part servie par le cache apparaît, et le
     taux avec. C'est mesuré, pas estimé.

  3. L'ENTRÉE TOTALE NE BOUGE PAS. Le cache est un SOUS-ENSEMBLE de l'entrée,
     jamais un ajout — sinon on facturerait deux fois les mêmes jetons. La
     démo le vérifie par une assertion.

Le préfixe doit être STABLE et assez GROS : chaque fournisseur a un seuil
minimal en dessous duquel il ne met rien en cache. Un prompt système qui
change à chaque appel (une date, un identifiant) ne sera jamais mis en cache.

MAIS ces deux conditions ne SUFFISENT PAS, et c'est le point le moins connu :
chez Gemini et OpenAI le cache est IMPLICITE — le fournisseur décide seul,
sans garantie. Mesuré sur ce dépôt : le même préfixe de 7 026 jetons a mordu
sur un appel et pas sur le suivant, et un préfixe de 9 366 jetons n'a rien
donné là où un de 14 046 marchait. Cette démo peut donc afficher « inconnu »
partout un jour et 59 % le lendemain, sans qu'une ligne ait changé. Et ça se
mesure FOURNISSEUR PAR FOURNISSEUR : la même démo sur DeepSeek a rapporté un
zéro MESURÉ au 1ᵉʳ appel puis 100 % (7 552 / 7 571) aux suivants — déterministe
en pratique. Anthropic est seulement le seul à exiger un marqueur explicite
(voir la fin de cette démo). Conséquence pratique : le cache est un bonus qu'on
CONSTATE, jamais une économie qu'on PROMET dans un devis — sauf à l'avoir
mesuré chez le fournisseur qu'on facture.

    python examples_autoagent/27_cache_de_prompt.py
"""

from __future__ import annotations

from _common import make_provider

from autoagent import Agent

# ── Le préfixe stable : c'est LUI qu'on veut voir passer en cache ────────────
# Répété pour dépasser le seuil minimal de mise en cache des fournisseurs.
SYSTEME = (
    "Tu es un assistant d'analyse de données de mobilité pour un institut "
    "d'études spécialisé dans les comptages routiers et les enquêtes "
    "déplacements. Tu réponds de façon concise et factuelle, et tu ne "
    "fabriques jamais un chiffre que tu n'as pas lu. "
) * 120

QUESTION = "Réponds simplement : OK"
RUNS = 3


def _ligne(n: int, usage) -> tuple[int, int | None]:
    """Affiche un run et rend (entrée, part servie par le cache)."""
    if usage is None:
        print(f"  run {n} : le fournisseur n'a rapporté aucun usage")
        return 0, None

    taux = usage.cache_hit_ratio
    if taux is None:
        # Distinction essentielle : « rien rapporté » n'est pas « zéro mesuré ».
        lu = "inconnu" if usage.cached_tokens is None else f"{usage.cached_tokens}"
        affichage = "— (le cache s'écrit)" if usage.cached_tokens is None else "0 %"
    else:
        lu = str(usage.cached_tokens)
        affichage = f"{taux:.0%}"

    print(f"  run {n} : entrée {usage.input_tokens:>6} · sortie "
          f"{usage.output_tokens:>4} · servi par le cache {lu:>7} · {affichage}")
    return usage.input_tokens or 0, usage.cached_tokens


def main() -> None:
    provider = make_provider()
    modele = provider.config.model
    print(f"Modèle : {provider.config.provider} / {modele}")
    print(f"Préfixe stable : ~{len(SYSTEME) // 4} jetons estimés\n")
    print("─" * 74)

    entrees: list[int] = []
    caches: list[int | None] = []
    for n in range(1, RUNS + 1):
        # Un agent NEUF à chaque fois : on ne teste pas la mémoire d'un objet,
        # on teste le cache du fournisseur, de l'autre côté du réseau.
        agent = Agent(provider, system_prompt=SYSTEME, max_steps=2)
        entree, cache = _ligne(n, agent.run(QUESTION).usage)
        entrees.append(entree)
        caches.append(cache)

    print("─" * 74)
    print("CE QU'IL FAUT RETENIR\n")

    if len(set(entrees)) == 1 and entrees[0]:
        print(f"  L'entrée est restée à {entrees[0]} jetons aux {RUNS} runs.")
        print("  Le cache est un SOUS-ENSEMBLE de l'entrée, pas un ajout : on ne")
        print("  facture jamais deux fois les mêmes jetons.")
    else:
        print(f"  Entrées observées : {entrees} — elles devraient être identiques ;")
        print("  si elles varient, c'est que le préfixe n'est pas stable.")

    servis = [c for c in caches if c]
    if servis:
        print(f"\n  À partir du 2ᵉ run, {servis[-1]} jetons sur {entrees[-1]} sont")
        print("  servis depuis le cache — à chaque tour, pas une seule fois. Sur un")
        print("  agent qui fait huit étapes, ce sont huit fois ces jetons-là qui")
        print("  changent de tarif. En euros, l'écart dépend du prix du jeton caché")
        print("  chez ton fournisseur : la démo 22 fait ce calcul-là.")
    else:
        print("\n  Aucune part servie par le cache sur ces runs — et c'est un")
        print("  résultat NORMAL, pas une panne. Le cache implicite est décidé par")
        print("  le fournisseur : le même préfixe mord à un moment et plus à un")
        print("  autre (mesuré sur ce dépôt, cf. l'en-tête de la démo 22). Relance")
        print("  la démo : tu n'obtiendras pas forcément le même résultat.")
        print("  Les causes que TU contrôles, s'il ne mord jamais : préfixe trop")
        print("  court pour le seuil, ou pas strictement identique d'un appel à")
        print("  l'autre (une date, un identifiant… suffisent à le casser).")

    print("\n  Rien n'a été demandé dans la requête : Gemini, OpenAI et les")
    print("  endpoints compatibles (DeepSeek…) mettent le préfixe stable en cache")
    print("  TOUT SEULS. Le seul travail de la lib est de le RAPPORTER — sans")
    print("  quoi l'économie serait invisible.")
    print()
    print("  Anthropic est le seul à exiger un marqueur explicite. On l'active")
    print("  ainsi, et c'est la seule ligne à changer :")
    print()
    print('      ModelConfig(provider="anthropic", model="claude-sonnet-4-5",')
    print("                  cache_prompt=True)")
    print()
    print("  Il est éteint par défaut, et c'est voulu : chez Anthropic, ÉCRIRE")
    print("  dans le cache coûte plus cher qu'une entrée normale. Sur un préfixe")
    print("  court, ou utilisé une seule fois, l'activer fait perdre de l'argent.")

    # La garantie qu'on ne double compte pas, vérifiée et pas seulement affirmée.
    for entree, cache in zip(entrees, caches, strict=False):
        if cache is not None and entree:
            assert cache <= entree, "la part cachée dépasse l'entrée — ce serait un bug"


if __name__ == "__main__":
    main()
