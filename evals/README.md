# evals — éval comportementale de la mémoire

Les tests unitaires vérifient que le code fait ce qu'on lui demande ; ce banc
mesure **ce que l'agent retient vraiment** à travers plusieurs sessions.

## Méthode

12 scénarios multi-sessions en français (style LoCoMo réduit) : des faits sont
établis lors d'appels passés — parfois **contredits** (« le soir » → « le
matin ») ou rendus **caducs** (« on a vendu le scooter ») — puis une question
est posée dans une session **neuve**. Du remplissage force la compaction,
comme dans un vrai appel.

Réponse comptée juste si elle contient une formulation attendue **et aucune
formulation interdite** (la valeur périmée, après contradiction).

## Configurations comparées

| Config | Ce que c'est |
|---|---|
| `sans_memoire` | agent neuf, aucune mémoire — le plancher (les ✅ sont des coups de chance du modèle) |
| `summarizing` | `SummarizingMemory` — mémoire de **conversation** (résumé roulant) ; par design, elle ne prétend PAS survivre à une session neuve |
| `fact_memory` | `FactMemory` + outil `recall` — mémoire d'**identité** (faits tenus à jour) |

## Lancer

```bash
python evals/eval_memoire.py              # provider résolu comme les démos (.env)
python evals/eval_memoire.py --limit 3    # essai rapide
```

Résultats détaillés (réponses incluses) dans `resultats.json`. Coût : ~60-90
appels du modèle configuré. Les scores varient légèrement d'un run à l'autre
(LLM réel) — l'ordre de grandeur est stable.

---

# Banc d'efficacité — combien de tâches, à quel coût, en combien de temps

`eval_efficacite.py` mesure ce que la lib tire d'un modèle sur des tâches qui
ressemblent à un usage réel de données de mobilité : **le taux de réussite, les
jetons dépensés (échecs compris) et la durée**, pour une configuration donnée.
Il sert de point de départ avant de toucher aux réglages : un levier n'est
gardé que s'il améliore ces trois chiffres sur CES tâches.

## Les vingt tâches (cinq familles de quatre)

| Famille | Ce que l'agent doit faire | Outil |
|---|---|---|
| A. extraction | une réponse libre d'enquête → un JSON (commune, heure de départ, modes) | aucun |
| B. codification | une réponse libre → le code d'une nomenclature de motifs | `nomenclature_motifs` |
| C. sql | une question sur une base de comptages (20 capteurs, une semaine, horaire) | `executer_sql` (SELECT, lecture seule) |
| D. gros résultat | une réponse cachée dans un journal (~9 000 caractères) ou un export CSV (~26 000) | `lire_journal`, `export_comptages_csv` |
| E. plusieurs étapes | enchaîner ou combiner des outils (jusqu'à cinq journaux) | tous |

Tous les outils sont donnés à l'agent pour toutes les tâches : il choisit.

**Données simulées** (graine fixe) : rien de réel ne part chez le fournisseur.
**Réponses attendues calculées** depuis ces données — jamais saisies à la main —
et recalculées par un second chemin (SQL, CSV) à la vérification. **Juges en
code** : la valeur de la dernière ligne « RÉPONSE: … » (ou le JSON pour
l'extraction) ; une réponse qui hésite (« 6 ou 7 », « environ 7 ») est fausse.
Chaque juge est audité par `audit_check` (presque-bons et négatifs triviaux)
avant tout appel payant.

## Lancer

```bash
python evals/eval_efficacite.py --verifier                  # hors ligne, gratuit : données + audit des juges
python evals/eval_efficacite.py --provider deepseek --k 3   # 60 runs
python evals/eval_efficacite.py --provider gemini --k 3
python evals/eval_efficacite.py --provider gemini --k 1 --taches D,E1   # un sous-ensemble
python evals/eval_efficacite.py --provider deepseek --k 3 --config code # un réglage de la lib, seul
```

`--config` choisit UN réglage, le reste identique au point de départ :
`telle_quelle` (défauts de la lib), `code` (code comme action câblé à la main :
`enable_run_python(PythonRunner(host_functions=<les outils>))` + une phrase de
consigne), `code_une_ligne` (`agent.enable_code_action()`, rien d'autre),
`coupe` (`max_tool_result_chars=4000`), `elagage`
(`prune_tool_results_after=1`), `parallele` (`parallel_tool_calls=True`).
Les outils du banc sont locaux et instantanés : `parallele` ne peut rien y
gagner en temps — ce banc ne mesure pas ce réglage.

`--comparer <réglage>` confirme un réglage contre `telle_quelle` avec
`compare_configs` : les deux bras ALTERNENT à chaque répétition, et le rapport
donne le verdict de réussite (test exact + intervalle), la variation des jetons
par tentative et de la durée avec leurs intervalles bootstrap :

```bash
python evals/eval_efficacite.py --provider deepseek --k 3 --comparer code_une_ligne   # 120 runs
```

Résultats dans `resultats_efficacite.json`, une entrée par fournisseur/modèle :
le résumé par famille et, pour chaque tentative, réussite, étapes, jetons
(entrée, sortie, cache), durée, erreur et la fin de la réponse. Les jetons
d'un run qui LÈVE (plafond d'étapes…) sont relevés sur l'exception : un échec
se paie aussi.

## Limites

Vingt tâches, courtes, en français, dans UN domaine ; k=3 répétitions par
tâche : un écart de quelques points entre deux configurations n'est pas
détectable (voir `detectable_difference` de `autoagent.compare`). Les données
sont simulées : elles ressemblent à un usage réel, elles n'en sont pas un.
