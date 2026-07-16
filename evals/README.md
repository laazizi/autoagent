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
