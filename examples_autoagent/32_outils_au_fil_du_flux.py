"""32 — Exécuter les outils PENDANT que le modèle parle encore.

En streaming, un appel d'outil est souvent complet bien avant la fin du
message : le modèle a décidé « lire capteur A », puis continue d'émettre
« lire capteur B », puis du texte. Jusqu'ici la boucle attendait le `final`
pour lancer quoi que ce soit — le temps de génération et le temps d'outil
s'ADDITIONNAIENT.

    @agent.tool(idempotent=True)
    def lire_capteur(nom: str) -> dict: ...

Avec `idempotent=True`, la boucle lance l'outil dès que son appel est complet
(chunk `tool_call`), pendant que le modèle émet la suite. PASTE (arXiv
2603.18897) mesure -43 % de temps par tâche avec ce recouvrement.

LA RÈGLE QUI REND ÇA SÛR — et elle est du code, pas une consigne :

  * JAMAIS sans `idempotent=True`. Un flux qui casse jette le résultat anticipé ;
    un outil idempotent n'a rien changé dans le monde. Un outil à effet de bord
    (envoi, écriture, paiement) ne doit pas porter ce drapeau — la lib ne le
    devine pas, l'hôte le déclare (papier compagnon arXiv 2606.07846 : « un
    mauvais résultat spéculatif ne peut pas annuler l'irréversible »).
  * Les GARDES passent AVANT le lancement, exactement comme sur la voie normale :
    anti-boucle, trifecta, politique de l'hôte. Un appel refusé n'est pas lancé.
  * Le chemin normal CONSOMME le résultat, il ne relance pas : un seul appel,
    et le transcript est identique à l'octet à celui d'un run sans anticipation.

Ce que la démo mesure : le même run streamé deux fois, seul `idempotent`
change. Les deux outils dorment 1,2 s chacun pour rendre le recouvrement
visible. Le gain dépend du modèle : s'il n'émet pas ses deux appels dans le
MÊME tour, il n'y a rien à recouvrir — la démo le dit alors, elle ne triche pas.

    python examples_autoagent/32_outils_au_fil_du_flux.py
"""

from __future__ import annotations

import time

from _common import make_provider

from autoagent import Agent, TraceEmitter

SOMMEIL = 1.2      # secondes par capteur — un vrai appel réseau lent
TACHE = ("Lis les capteurs cmp-lyo-07 et cmp-gre-03 (un appel d'outil par capteur, "
         "les deux dans la même réponse), puis donne leur état en une phrase.")


def construire(idempotent: bool, trace: TraceEmitter) -> Agent:
    agent = Agent(make_provider(), max_steps=6, trace=trace,
                  system_prompt="Tu utilises tes outils puis tu réponds en une phrase.")

    @agent.tool(idempotent=idempotent)
    def lire_capteur(nom: str) -> dict:
        """Interroge un capteur (lent : 1,2 s) et rend son état."""
        time.sleep(SOMMEIL)
        return {"capteur": nom, "etat": "ok", "trames": 1240}

    return agent


def mesurer(titre: str, idempotent: bool) -> tuple[float, int, int, str]:
    anticipes = [0]
    appels_par_tour: list[int] = []

    def on_event(ev) -> None:
        if ev.type == "tool_call_early_start":
            anticipes[0] += 1
        elif ev.type == "llm_response":
            appels_par_tour.append(ev.payload.get("tool_call_count", 0))

    debut = time.monotonic()
    sortie = ""
    with TraceEmitter(on_event=on_event) as trace:
        for ev in construire(idempotent, trace).run_stream(TACHE):
            if ev.type == "done":
                sortie = (ev.output or "").strip()
    duree = time.monotonic() - debut

    max_par_tour = max(appels_par_tour or [0])
    print(f"\n  {titre}")
    print(f"    durée totale          : {duree:5.1f} s")
    print(f"    appels dans un même tour : {max_par_tour}")
    print(f"    lancés en avance      : {anticipes[0]}")
    print(f"    réponse               : {sortie[:80]}")
    return duree, anticipes[0], max_par_tour, sortie


def main() -> None:
    provider = make_provider()
    print(f"Modèle : {provider.config.provider} / {provider.config.model}")
    print(f"Deux capteurs, {SOMMEIL} s chacun. Même run streamé deux fois ; seul "
          f"`idempotent` change.")
    print("─" * 76)

    d_sans, _, _, _ = mesurer("SANS idempotent — les outils attendent la fin du message", False)
    d_avec, n_avance, par_tour, _ = mesurer("AVEC idempotent=True — lancés dès leur chunk", True)

    print("\n" + "─" * 76)
    print("CE QU'IL FAUT RETENIR\n")
    if n_avance and d_avec < d_sans:
        print(f"  {d_sans:.1f} s puis {d_avec:.1f} s : {1 - d_avec / d_sans:.0%} de temps en moins,")
        print(f"  {n_avance} outil(s) lancé(s) pendant que le modèle parlait encore.")
        print("  Le travail est le même ; c'est l'ATTENTE qui s'est recouverte.")
    elif par_tour < 2:
        print("  Pas de gain mesurable ici : le modèle n'a pas émis ses deux appels")
        print("  dans le même tour, il n'y avait donc rien à recouvrir. Le gain")
        print("  dépend de SA façon de répondre, pas seulement de l'outil. Relance.")
    else:
        print(f"  {d_sans:.1f} s puis {d_avec:.1f} s — pas de gain net sur ce run.")
        print("  Possible quand la génération finit avant l'outil : on ne peut pas")
        print("  recouvrir plus que la durée de ce qui reste à générer.")
    print()
    print("  Le transcript est IDENTIQUE dans les deux cas — mêmes messages, même")
    print("  ordre, même trace. Seul l'instant de lancement change. Et rien n'est")
    print("  parti en avance sans le drapeau : un effet de bord ne se spécule pas.")


if __name__ == "__main__":
    main()
