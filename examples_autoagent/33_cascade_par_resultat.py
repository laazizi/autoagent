"""33 — La cascade : le petit modèle d'abord, le gros seulement si TON juge dit non.

`RoutingProvider` (démo 03) route AVANT l'appel, sur la forme de la requête.
La cascade route APRÈS, sur le résultat : le petit modèle répond, un juge EN
CODE vérifie, et on ne paie le gros que si le petit a raté.

    cascade([lite, pro], tache, check=juge)

CE QUI DÉCIDE DE MONTER EST DU CODE. Si c'est le petit modèle qui dit « je ne
suis pas sûr », on est revenu à une consigne qu'on espère. Ici le juge est un
prédicat déterministe sur le résultat — le même contrat que `run_k` (démo 04).

La démo enchaîne quatre tâches vérifiables (la réponse doit être un JSON qui
respecte une propriété qu'on peut TESTER : des nombres premiers, une somme, un
tri…). Pour chacune : cascade lite → pro. On compare la facture totale à celle
d'un run « tout sur le gros » — et on montre le nombre d'escalades. Les paliers
ratés SE PAIENT : la dépense affichée est celle de tout ce qui a été essayé.

Deux garanties qui ne se voient pas dans les chiffres, mais dans les tests :
une pause d'approbation (`ApprovalRequired`) remonte au lieu d'être « rattrapée »
par un modèle plus gros — la cascade ne contourne jamais `tool_policy` ; et un
palier qui plante est un palier raté, pas une exception qui casse tout.

Paliers : gemini-3.5-flash-lite → gemini-3.7-flash (liste des modèles relevée
sur l'API). Avec une autre clé, change les deux ModelConfig.

    python examples_autoagent/33_cascade_par_resultat.py
"""

from __future__ import annotations

import json
import re

from _common import load_env

from autoagent import Agent, ModelConfig, cascade, create_provider

load_env()
LITE = ModelConfig(provider="gemini", model="gemini-3.5-flash-lite", timeout=120)
PRO = ModelConfig(provider="gemini", model="gemini-3.7-flash", timeout=120)

SYSTEME = ("Réponds UNIQUEMENT par un objet JSON valide, sans texte autour, "
           "avec la clé demandée.")


def _premier(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _json(res) -> dict | None:  # type: ignore[no-untyped-def]
    texte = res.output.strip()
    m = re.search(r"\{.*\}", texte, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


# (tâche, juge) — chaque juge est du CODE qui vérifie une propriété, pas un avis.
TACHES = [
    ("Donne les cinq premiers nombres premiers strictement supérieurs à 1000, "
     "clé 'premiers' (liste d'entiers croissants).",
     lambda r: (d := _json(r)) is not None and d.get("premiers") == [1009, 1013, 1019, 1021, 1031]),
    ("Trie ces mots par ordre alphabétique français strict : ['zèbre','écureuil','abeille','Éléphant'], "
     "clé 'tries' (liste).",
     lambda r: (d := _json(r)) is not None and isinstance(d.get("tries"), list)
     and [w.lower().replace("é", "e").replace("è", "e") for w in d["tries"]]
     == sorted(["zebre", "ecureuil", "abeille", "elephant"])),
    ("Combien de secondes dans 3 jours, 7 heures et 15 minutes ? clé 'secondes' (entier).",
     lambda r: (d := _json(r)) is not None and d.get("secondes") == 3 * 86400 + 7 * 3600 + 15 * 60),
    ("Le mot 'ressasser' est-il un palindrome (lettres seules, sans accent) ? clé 'palindrome' (booléen).",
     lambda r: (d := _json(r)) is not None and d.get("palindrome") is True),
]


def _agent(config: ModelConfig) -> Agent:
    return Agent(create_provider(config), system_prompt=SYSTEME, max_steps=2)


def main() -> None:
    print(f"Paliers : {LITE.model}  →  {PRO.model}")
    print(f"{len(TACHES)} tâches, chacune avec un juge en code.")
    print("─" * 76)

    total_cascade = 0
    total_pro_seul = 0
    escalades = 0
    jetons_lite = 0        # ce que la cascade a dépensé sur le palier lite
    jetons_pro_casc = 0    # … et sur le palier pro (escalades)
    print("\n  CASCADE (lite d'abord, pro si le juge refuse)")
    for i, (tache, juge) in enumerate(TACHES, 1):
        r = cascade([lambda: _agent(LITE), lambda: _agent(PRO)], tache, check=juge)
        jetons = (r.usage.total_tokens if r.usage else 0) or 0
        total_cascade += jetons
        escalades += r.escalations
        for a in r.attempts:
            j = (a.usage.total_tokens if a.usage else 0) or 0
            if a.model == LITE.model:
                jetons_lite += j
            else:
                jetons_pro_casc += j
        chemin = " → ".join(f"{a.model.replace('gemini-', '')}{'✓' if a.ok else '✗'}" for a in r.attempts)
        print(f"    tâche {i} : {chemin:<38} {jetons:>5} jetons"
              + ("" if r.accepted else "   ← AUCUN palier accepté"))

    print("\n  TOUT SUR LE GROS (référence)")
    for i, (tache, juge) in enumerate(TACHES, 1):
        res = _agent(PRO).run(tache)
        jetons = (res.usage.total_tokens if res.usage else 0) or 0
        total_pro_seul += jetons
        print(f"    tâche {i} : {'✓' if juge(res) else '✗'} {jetons:>5} jetons")

    print("\n" + "─" * 76)
    print("CE QU'IL FAUT RETENIR\n")
    print(f"  Cascade : {total_cascade} jetons ({jetons_lite} sur lite + {jetons_pro_casc} sur pro), "
          f"{escalades} escalade(s) sur {len(TACHES)} tâches.")
    print(f"  Gros seul : {total_pro_seul} jetons.")
    if total_cascade and total_pro_seul:
        ecart = total_cascade / total_pro_seul - 1
        if ecart > 0:
            print(f"\n  EN JETONS, LA CASCADE A COÛTÉ PLUS : {ecart:+.0%}. C'est normal, et la")
            print("  démo le dit plutôt que de l'arrondir : une escalade paie DEUX paliers,")
            print("  et sur des tâches de 60 à 100 jetons l'essai raté ne s'amortit pas.")
        else:
            print(f"\n  En jetons, la cascade a coûté {ecart:+.0%}.")
        # Seuil : la cascade gagne en EUROS si prix_lite/prix_pro < (pro_seul - pro_casc) / lite
        economise_pro = total_pro_seul - jetons_pro_casc
        if jetons_lite > 0 and economise_pro > 0:
            seuil = economise_pro / jetons_lite
            print("\n  LE SEUIL QUI DÉCIDE : la cascade gagne en euros si le jeton lite coûte")
            print(f"  moins de {seuil:.0%} du jeton pro. (Elle a évité {economise_pro} jetons de")
            print(f"  pro, contre {jetons_lite} jetons de lite dépensés.) Le rapport de prix")
            print("  réel est TA donnée — la lib ne présume aucun tarif (démo 22).")
        elif economise_pro <= 0:
            print("\n  Ici la cascade n'a rien évité sur le gros modèle : elle ne peut pas")
            print("  gagner, quel que soit le tarif. Le taux d'acceptation du palier lite")
            print("  est ce qui décide — mesure-le sur TES tâches avant de déployer.")
    print()
    print("  Le juge est du CODE : un JSON qui respecte une propriété vérifiable. Sans")
    print("  juge déterministe, il n'y a pas de cascade — il y a un pari. Et les")
    print("  paliers ratés sont DANS la facture : c'est ce qu'elle a vraiment coûté.")


if __name__ == "__main__":
    main()
