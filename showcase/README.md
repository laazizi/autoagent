# App vivante — vitrine autoagent (back + front)

Une application de démonstration : un **chat en streaming** où l'agent, au fil des
phases, **pilote l'écran**, **fabrique ses propres outils** et **s'auto-étend** —
le tout sous des bornes qui sont du **code** (workspace, sandbox, approbation
humaine), fidèle à la thèse d'autoagent.

> ⚠️ C'est un **consommateur** de la lib : il a des dépendances (FastAPI). La lib
> `autoagent`, elle, reste zéro-dépendance. Ce dossier n'est jamais inclus dans le
> paquet publié.

## Lancer

```bash
# depuis la racine du dépôt (autoagent importé depuis ../)
pip install -r showcase/requirements.txt
GEMINI_API_KEY=...  python -m uvicorn showcase.backend.main:app --reload
# → http://127.0.0.1:8000
```

**Sans clé** : l'app démarre en **mode démo hors-ligne** (provider factice qui
streame une réponse canned). Tout le pipeline — streaming SSE, historique,
écran piloté — reste testable sans réseau ni clé. Forcer ce mode : `AUTOAGENT_FAKE=1`.

## Architecture (désir → brique autoagent)

Essaim multi-agent (pattern `as_tool`) : **Orchestrateur** → **Agent Données** +
**Agent HTML**. L'orchestrateur ne met que de courtes phrases dans le chat ; tout
affichage riche devient du VRAI HTML dans l'écran central.

| Fonction | Brique autoagent |
|---|---|
| Chat streaming | `agent.run_messages_stream()` → `StreamEvent` relayés en **SSE** |
| Historique par session | `Message.to_dict/from_dict` persistés en JSON (`data/sessions/`) |
| L'agent pilote l'écran | Agent HTML génère du vrai HTML → **iframe sandboxé** + `ProjectWorkspace` (.html) |
| Données réelles | outil `meteo` (open-meteo, sans clé) porté par l'Agent Données |
| Outils dynamiques | `enable_dynamic_tools(DynamicToolBuilder(sandbox=…))` — build validé AST + exécuté en sandbox |
| Gate d'approbation | `tool_policy` → `ApprovalRequired` → **pause** (`RunState` sur disque) → `resume_stream` |
| Trace vivante | `TraceEmitter(on_event=…)` partagée par les 3 agents → arbre de spans en direct dans l'UI |
| Galerie de pages | chaque écran généré est persisté et navigable (onglets) |
| **API vivante** | `GET /api/live/{session}/{outil}` exécute un outil en direct (meteo / outil créé, en sandbox) ; un pont `window.appelerAPI(outil, args)` est injecté dans chaque page → le front **fetch** l'API (rechargeable) au lieu de figer les données |
| Suppression | `DELETE /api/sessions/{id}` efface le chat **et** son workspace (pages, outils, trace) |

## Feuille de route

- [x] **P0** — squelette : FastAPI + SSE streaming + historique persisté + front une page.
- [x] **P1** — essaim multi-agent : l'Agent HTML rend de vraies pages dans le canvas central.
- [x] **P2** — outils dynamiques (sandbox) + gate d'approbation humain (`ApprovalRequired`/`resume`).
- [x] **P3** — évolutivité : galerie de pages persistantes + **trace vivante** de l'essaim.

> Note : l'« évolutivité » est réalisée via `DynamicToolBuilder` (l'app gagne des
> capacités/outils) + galerie de pages persistantes (l'app gagne des vues), pas via
> `EvolutionRuntime` (qui fait s'auto-réécrire le code source — capacité plus lourde
> et risquée, hors périmètre de cette démo).

## Structure

```
showcase/
  backend/
    main.py           # FastAPI : /api/chat, /api/approve, SSE (text/tool/render/trace/approval)
    agent_factory.py  # essaim (orchestrateur + données + HTML), tool_policy, dynamic tools, trace
    sessions.py       # persistance : conversations, canvas, galerie, RunState en attente
  frontend/
    index.html        # une page : Chat (gauche) · Écran HTML (centre) · Historique (droite)
  data/               # runtime (gitignoré) : sessions, workspaces, outils générés, traces
```
