# examples_autoagent — le potentiel d'autoagent en 28 démos

Vingt-huit scripts courts, **exécutables tels quels**, montrant chacun UNE facette
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
| 17 | `17_memoire_factuelle.py` | **Mémoire factuelle** (`FactMemory`) : faits atomiques tenus À JOUR (une contradiction REMPLACE, un fait caduc DISPARAÎT), JSON auditable par identité, outils `remember`/`recall` — + v2 : consolidation **sleep-time** (`background=True`, compact <1 ms) et recall **par le sens** (`embed_fn`) | oui |
| 18 | `18_corpus_url.py` | **Gros corpus depuis une URL** (~1M tokens) : l'agent TÉLÉCHARGE, le host INDEXE, l'agent navigue par outils (`chercher`/`lire_passage`) — répond en consommant ~1 % du corpus au lieu de tout injecter. Recherche par le SENS (embeddings + cache) si GEMINI_API_KEY, sinon lexical | oui |
| 19 | `19_boucle_autonome.py` | **Boucle autonome fermée** (« Loops ») : plan → build → VÉRIFICATEUR en code → auto-correction → leçons en `FactMemory`, état persisté entre les battements, relançable par cron — le pattern complet sans framework | oui |
| 20 | `20_injection_dejouee.py` | **Injection indirecte déjouée** : un outil `untrusted=True` lit une page piégée → le run est *teinté* → `tool_policy` met en PAUSE tout outil sensible avant effet de bord. La barrière est du code, prouvée déterministe | oui |
| 21 | `21_record_replay.py` | **Record / replay déterministe** : un vrai run est gelé dans un fixture JSONL puis rejoué HORS-LIGNE (zéro réseau, zéro effet de bord, sortie identique) ; une divergence lève `ReplayMismatch`. N'importe quel run → test CI gratuit | oui |
| 22 | `22_budget_et_reprise.py` | **Maîtriser la dépense** : `token_budget` (plafond dur par run) → arrêt net à l'épuisement ; `exc.state` = reprise sans rien perdre (relève le budget, `resume`) ; + plafond de session en $ en code hôte, avec le **vrai** calcul de coût (entrée pleine / entrée cachée / sortie, tarifs fournis par l'hôte) : un tarif unique sur `total_tokens` surestime. Documente en tête le relevé qui montre que le cache implicite de Gemini est **opportuniste** — même préfixe, mord ou pas | oui |
| 23 | `23_questionnaire_mobilite.py` | **Questionnaire CATI (enquête déplacements)** : le host possède la machine à états et le barème ; le LLM ne fait qu'extraire une phrase libre en désordre et reformuler. Chaque tour montre ce que le modèle *propose* face à ce que le code *retient* — dont une règle croisée (origine = destination → refus) qu'aucune consigne ne tiendrait | oui |
| 24 | `24_codification_enquete.py` | **Codifier des réponses libres** (dépouillement d'enquête) : le modèle propose un code, le host **rejette tout ce qui n'est pas dans la nomenclature** et escalade avec le motif. Trois chiffres d'exploitant, dont un que le modèle ne peut pas dégrader : *0 code hors nomenclature*. Tolère une réponse tronquée (plus de relecture humaine, jamais de livrable corrompu) | oui |
| 25 | `25_standard_supervise.py` | **La démo à projeter** : l'agent choisit sa cascade d'identification (1, 2 ou 5 étapes selon ce qu'il trouve) puis veut ECRIRE en base → `tool_policy` leve `ApprovalRequired`, la boucle s'arrete avant tout outil, tu autorises ou tu refuses, `resume` continue. Fournisseur declare explicitement (pas de `_common`) | oui |
| 26 | `26_resultat_trop_gros.py` | **La première butée, mesurée** : le même outil et le même prompt, deux runs — seul `max_tool_result_chars=4000` change. 41 597 caractères injectés puis 4 000, 17 993 jetons puis 2 453 (−86 %), et la réponse reste juste : la coupe est au MILIEU, donc le bilan de la dernière ligne survit. La marque compte dans le budget (`len() <= 4000` vérifié dans la démo) | oui |
| 27 | `27_cache_de_prompt.py` | **Le cache de prompt, mesuré** : la même question trois fois, un préfixe système stable de ~6 800 jetons. Run 1 : le cache s'écrit, rien n'est rapporté (« inconnu », pas « zéro »). Runs 2 et 3 (relevé du 26/08/2026) : **4 073 jetons sur 6 847 servis par le cache, 59 %** — et l'entrée totale ne bouge pas, le cache est un sous-ensemble. ⚠️ Non reproductible à volonté : le cache implicite est décidé par le fournisseur, la même démo peut n'afficher aucun cache (cf. l'en-tête de la 22). Montre aussi `cache_prompt=True`, le marqueur qu'Anthropic seul exige — le seul cache déterministe | oui |
| 28 | `28_elagage_contexte.py` | **La butée de DURÉE** (la 26 borne la largeur, celle-ci la durée de vie) : quatre journaux lus l'un après l'autre, seul `prune_tool_results_after=1` change. **16 360 jetons d'entrée puis 7 592 (−54 %)**, réponse identique — parce qu'un vieux résultat repart à CHAQUE étape et n'est jamais dans le préfixe caché. Le marqueur dit que le résultat était *valide* (sinon le modèle replanifie autour d'un échec imaginaire), la teinte untrusted est reconduite, et le transcript rendu garde tout : on élague ce qu'on ENVOIE, pas ce qu'on GARDE | oui |

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
