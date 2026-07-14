# examples_autoagent — le potentiel d'autoagent en 16 démos

Quinze scripts courts, **exécutables tels quels**, montrant chacun UNE facette
de la lib (la n°13 les combine). Rangés du plus simple au plus avancé.

## Installation

```bash
pip install -r requirements.txt          # jsonschema + le provider voulu
# une clé dans .env à la racine (au moins une) :
#   GEMINI_API_KEY=...   (ou DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)
```

Chaque exemple choisit **automatiquement** le premier provider dont la clé est
présente. Pour forcer : `--provider gemini --model gemini-2.5-flash`.

## Les démos

| # | Fichier | Ce que ça montre | Clé API ? |
|---|---|---|---|
| 01 | `01_hello_tools.py` | Le cœur : agent + outils décorés, schéma auto depuis la signature, boucle LLM↔outils, `result.usage` | oui |
| 02 | `02_streaming.py` | `run_stream` : réponse token-par-token + événements outils en direct | oui |
| 03 | `03_multi_provider.py` | Le **même** agent sur chaque LLM configuré (latence/tokens comparés) ; `RoutingProvider` en bonus | oui |
| 04 | `04_observabilite_budget.py` | `TraceEmitter` (JSONL + callback), coût par run, `token_budget` (plafond dur) | oui |
| 05 | `05_memoire_resumante.py` | `SummarizingMemory` : contexte borné **sans amnésie** (30+ msgs → 8, décision clé retrouvée) | oui |
| 06 | `06_outils_dynamiques.py` | L'agent **écrit** l'outil qui lui manque → validé AST → exécuté sandbox → utilisé | oui |
| 07 | `07_sandbox_securite.py` | **Sécurité = code** : AST refuse le dangereux, sandbox isole, pont host-function whitelisté | **NON** |
| 08 | `08_multi_agents.py` | `Agent.as_tool()` : superviseur → spécialistes (chercheur + rédacteur), trace partagée | oui |
| 09 | `09_sortie_structuree.py` | `response_format` : JSON mode natif → extraction fiable (pas de re-parsing) | oui |
| 10 | `10_bornement_verification.py` | `ProjectWorkspace` (écriture confinée) + `post_turn_hook` (exiger une action) | oui |
| 11 | `11_flux_deterministe.py` | `Orchestrator` : le host possède la machine à états, le LLM interprète/reformule seulement | oui |
| 12 | `12_pseudonymisation_pii.py` | **RGPD** : le host masque les PII (nom/email/tél) en jetons ; le LLM ne voit JAMAIS les vraies données, restaurées côté host | oui |
| 13 | `13_prise_rdv_supervisee.py` | **La démo complète** (inspirée de cati_service) : les 3 cerveaux — `Orchestrator` (flux + validation), `Agent` superviseur (valider/corriger via outils + hook), mémoire par appelant (Memory + recall) | oui |
| 14 | `14_base_sql.py` | **Base SQL comme source** : l'agent inspecte le schéma, écrit un SELECT, la lib l'exécute en LECTURE SEULE (écriture refusée par le code) et répond sur des lignes réelles. SQLite (stdlib) | oui |
| 15 | `15_appel_entrant_fiche.py` | **Standard téléphonique** : cascade de repli pilotée par l'agent — fiche locale → CRM externe → sinon il DISCUTE avec l'appelant pour créer sa fiche. Plusieurs outils, l'agent choisit l'escalade | oui |
| 16 | `16_questions_clarification.py` | **Clarification** : demande vague → l'agent POSE des questions à l'humain (outil `demander_a_l_humain`) avant d'agir ; `post_turn_hook` en filet (interdit de conclure en devinant) | oui |

## Par où commencer

- **Zéro clé, tout de suite** : `python examples_autoagent/07_sandbox_securite.py`
- **La démo « wow »** : `06` (l'agent code son outil) puis `08` (agents qui délèguent).
- **Le cas complet, façon prod** : `13` (prise de RDV supervisée) — combine flux
  déterministe + agent superviseur + mémoire par appelant (l'archi de cati_service).
- **Pour un usage produit** : `04` (coût/observabilité), `05` (mémoire), `10` (bornement), `12` (RGPD/PII).

## Choisir la bonne primitive

- Tâche ouverte, l'IA décide → **`Agent`** (01–10).
- Processus garanti (questionnaire, formulaire) → **`Orchestrator`** (11) : le LLM ne
  peut ni sauter ni inventer d'étape.

## Note

Les artefacts générés (`outils_generes/`, `trace_demo.jsonl`) sont ignorés par
git — c'est normal qu'ils apparaissent après un run.
