# autoagent — documentation développeur

> Référence technique complète pour intégrer, étendre et tester `autoagent` dans un projet Python.
> **Public visé** : devs qui vont écrire des tools, brancher l'agent sur leur app, ou éventuellement contribuer à la lib.

**Auteur** : Mohamed LAAZIZI · **Équipe** : Alyce R&D · **Version** : 2026-07-14 · **Couvre autoagent** : 0.12.0 (publié sur PyPI : [`autoagent-core`](https://pypi.org/project/autoagent-core/))

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Installation et démarrage rapide](#2-installation-et-démarrage-rapide)
3. [Concepts fondamentaux](#3-concepts-fondamentaux)
4. [Référence API](#4-référence-api)
   - 4.1 [`Agent`](#41-agent)
   - 4.2 [`AgentResult`](#42-agentresult)
   - 4.3 [`ToolRegistry`](#43-toolregistry)
   - 4.4 [`ModelConfig` et `create_provider`](#44-modelconfig-et-create_provider)
   - 4.5 [Tracing : `TraceEmitter` / `TraceEvent`](#45-tracing--traceemitter--traceevent) *(0.5.0)*
   - 4.6 [Mémoire : `Memory` / `BufferMemory`](#46-mémoire--memory--buffermemory) *(0.6.0)*
   - 4.7 [`post_turn_hook` — boucle de vérification hôte](#47-post_turn_hook--boucle-de-vérification-hôte) *(0.2.0)*
   - 4.8 [`cancel_token` — annulation coopérative](#48-cancel_token--annulation-coopérative) *(0.2.0)*
   - 4.9 [Messages multimodaux : `ImageAttachment`](#49-messages-multimodaux--imageattachment) *(0.4.0)*
   - 4.10 [`reasoning_content` (DeepSeek / o-series)](#410-reasoning_content-deepseek--o-series) *(0.3.2)*
5. [La boucle interne (run_messages)](#5-la-boucle-interne-run_messages)
6. [Écrire des tools](#6-écrire-des-tools)
7. [Génération automatique du JSON schema](#7-génération-automatique-du-json-schema)
8. [Providers (OpenAI, Anthropic, DeepSeek, Gemini)](#8-providers)
9. [ProjectWorkspace — lectures/écritures bornées](#9-projectworkspace--lecturesécritures-bornées)
10. [EvolutionRuntime — l'agent pilote un projet entier](#10-evolutionruntime--lagent-pilote-un-projet-entier)
11. [Tools dynamiques — l'agent invente ses outils](#11-tools-dynamiques)
12. [Tests](#12-tests)
13. [Extension : ton propre provider, ton propre runtime](#13-extension--ton-propre-provider-ton-propre-runtime)
14. [Pièges fréquents et FAQ](#14-pièges-fréquents-et-faq)
15. [`Orchestrator` — flux déterministe piloté par le host](#15-orchestrator--flux-déterministe-piloté-par-le-host) *(0.9.0)*
16. [Nouveautés 0.8.0 → 0.10.0](#16-nouveautés-080--0100) — streaming, `SummarizingMemory`, `as_tool`, `token_budget`, `RoutingProvider`…
17. [`MCPClient` — outils MCP branchés comme des tools locaux](#17-mcpclient--outils-mcp-branchés-comme-des-tools-locaux) *(0.11.0)*
18. [`OTelTraceExporter` — traces vers OpenTelemetry](#18-oteltraceexporter--traces-vers-opentelemetry) *(0.11.0)*
19. [`RunState` — checkpoint / resume (agents longue durée)](#19-runstate--checkpoint--resume-agents-longue-durée) *(0.11.0)*
20. [`tool_policy` — politique d'exécution des outils & approval gate](#20-tool_policy--politique-dexécution-des-outils--approval-gate) *(0.11.0)*
21. [`FactMemory` — mémoire factuelle tenue à jour](#21-factmemory--mémoire-factuelle-tenue-à-jour) *(0.12.0)*

---

## 1. Vue d'ensemble

### 1.1 Philosophie

`autoagent` est un **noyau d'agent** Python — pas un framework. Sa thèse :

- **L'agent doit être lisible** : tu peux lire toute la lib en une heure (~800 LOC)
- **Le bornement est du code Python, pas du prompt** : `ProjectWorkspace` + permissions + AST + sandbox
- **Zéro dépendance** pour le cœur : Python ≥3.10 + `urllib` + `dataclasses`
- **Multi-provider** sans abstractions inutiles : un `Provider` = une méthode `complete(LLMRequest)`

### 1.2 Ce que ce n'est pas

- ❌ Pas LangChain (chains, prompts templates, memory backends, callbacks…)
- ❌ Pas un framework "agent orchestration" (CrewAI, AutoGen…)
- ❌ Pas async — la boucle est synchrone, simple, déterministe
- ❌ Pas de RAG intégré (à toi de fournir un tool `search_docs` si besoin)

### 1.3 Ce que c'est

- ✅ Une **boucle LLM ↔ tools** propre et auditable
- ✅ Un **système de bornement** (ProjectWorkspace + permissions tags)
- ✅ Un **sandbox de génération de code Python** (DynamicToolBuilder)
- ✅ Un **adaptateur multi-provider** que tu peux étendre en 50 lignes
- ✅ Un **système d'observabilité** structuré (TraceEmitter) — événements typés, redaction de secrets intégrée
- ✅ Une **abstraction mémoire** minimale (Memory protocol + BufferMemory) — vector-backed en option dans `examples/`
- ✅ Du **multimodal** (`ImageAttachment` côté `Message`, sérialisation par provider)
- ✅ Un **post-turn hook** d'hôte pour boucle de vérification, et un **cancel_token** coopératif

### 1.4 Carte des modules

| Module | Rôle | Ajouté en |
|---|---|---|
| `autoagent/agent.py` | Boucle `run_messages` + `post_turn_hook` + `cancel_token` + tracing | — |
| `autoagent/schema.py` | `Message`, `ToolCall`, `ToolSpec`, `ImageAttachment`, `reasoning_content` | — |
| `autoagent/registry.py` | `ToolRegistry`, génération de schema | — |
| `autoagent/workspace.py` | `ProjectWorkspace` (bornement disque) | — |
| `autoagent/evolution.py` | `EvolutionRuntime` (l'agent modifie un projet) | — |
| `autoagent/dynamic.py` | `DynamicToolBuilder`, `ToolBuildRequest` (l'agent génère ses tools) | — |
| `autoagent/sandbox.py` | `SubprocessSandbox` + **`DockerSandbox`** (isolation OS) + pont host-function + denylist AST durcie + `make_sandbox` | — |
| `autoagent/approval.py` | `ToolManifest` (allowlist par hash) + `load_tools` (natif/sandbox) + promotion humaine + CLI | — |
| `autoagent/pipeline.py` | `PipelineManager` (slots `pipeline.json` hot-swap) | — |
| `autoagent/orchestrator.py` | `Orchestrator` — flux déterministe piloté par le host (le LLM interprète + reformule seulement) | 0.9.0 |
| `autoagent/http.py` | `post_json` / `post_sse` (retry + backoff sur erreurs transitoires) | — |
| `autoagent/errors.py` | exceptions : `ToolError`, `ToolValidationError`, `ProviderError`… | — |
| `autoagent/providers/*.py` | OpenAI / Anthropic / DeepSeek / Gemini (+ `stream()` SSE 0.8.0) | — |
| `autoagent/providers/routing.py` | `RoutingProvider` — dispatch par requête (texte→cheap, image→vision), §16.7 | — |
| `autoagent/logging.py` | Logger + `SecretRedactingFilter` | — |
| **`autoagent/trace.py`** | **`TraceEmitter`, `TraceEvent`, `truncate_preview`** | **0.5.0** |
| **`autoagent/memory.py`** | **`Memory` Protocol, `BufferMemory`, `SummarizingMemory` (0.10.0), `FactMemory` (§21)** | **0.6.0** |
| **`autoagent/mcp.py`** | **`MCPClient` — outils d'un serveur MCP montés comme tools locaux (stdio, zéro dép.), §17** | **0.11.0** |
| **`autoagent/otel.py`** | **`OTelTraceExporter` — trace → spans OpenTelemetry (dépendance optionnelle), §18** | **0.11.0** |

---

## 2. Installation et démarrage rapide

### 2.1 Pré-requis

- Python ≥ 3.10
- Une clé API d'un provider LLM (OpenAI, Anthropic, DeepSeek, Gemini)
- **Docker** — *optionnel mais recommandé* : il fournit la vraie isolation OS des tools dynamiques
  (`DockerSandbox`, §11.4). Sans démon Docker, `make_sandbox()` retombe automatiquement sur
  `SubprocessSandbox` (durcissement AST seul, pas d'isolation réseau). Pas besoin de Docker si tu
  n'utilises pas les tools dynamiques.

### 2.2 Installer

```bash
pip install autoagent-core            # s'importe `import autoagent`
pip install autoagent-core[otel]      # + export OpenTelemetry (§18)
```

Ou depuis les sources (dev de la lib) :

```bash
git clone https://github.com/laazizi/autoagent.git
cd autoagent
pip install jsonschema                # seule dépendance du cœur
```

`.env` à la racine :
```
OPENAI_API_KEY=sk-...
# ou
ANTHROPIC_API_KEY=sk-ant-...
# ou
DEEPSEEK_API_KEY=sk-...
# ou
GEMINI_API_KEY=...
```

### 2.3 Hello world

```python
from autoagent import Agent

agent = Agent.from_model("openai", "gpt-4o-mini")

@agent.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

result = agent.run("Combien font 21 + 21 ?")
print(result.output)   # "42"
print(result.steps)    # 2 — un appel à add, puis une réponse texte
print(len(result.messages))  # historique complet
```

### 2.4 Lancer les exemples

```bash
python examples/basic.py           # tool Python classique
python examples/dynamic_tools.py   # agent qui crée ses tools
python examples/software_evolution.py  # agent qui modifie un projet
python examples/web_app_evolution.py   # agent qui édite une page HTML
python examples/qt_panel_evolution.py  # agent qui édite un plugin PyQt5
```

---

## 3. Concepts fondamentaux

### 3.1 Agent = LLM + tools + boucle

À chaque tour :

```
1. Envoie l'historique au LLM avec la liste des tools disponibles
2. Le LLM répond :
   - soit du texte final → on s'arrête
   - soit 1+ appels d'outils
3. On exécute les outils en local (Python)
4. On ajoute les résultats à l'historique (role="tool")
5. Si max_steps atteint → stop. Sinon → retour étape 1
```

La boucle est dans [`autoagent/agent.py`](autoagent/agent.py) — méthode `Agent.run_messages(history)`.

### 3.2 Les 5 classes principales

| Classe | Fichier | Rôle |
|---|---|---|
| `Agent` | `autoagent/agent.py` | Orchestrateur de la boucle |
| `Tool` (concept) | `autoagent/registry.py` | Une fonction Python exposée au LLM avec son schema |
| `Provider` (interface) | `autoagent/providers/` | Adaptateur OpenAI/Anthropic/DeepSeek/Gemini |
| `Message` | `autoagent/schema.py` | Élément d'historique : role, content, tool_calls, tool_call_id, attachments, reasoning_content |
| `ProjectWorkspace` | `autoagent/workspace.py` | Lecture/écriture bornées avec historique pour rollback |
| `TraceEmitter` | `autoagent/trace.py` | Événements lifecycle (run_start, tool_call_*, run_end…) → JSONL + callback |
| `Memory` / `BufferMemory` | `autoagent/memory.py` | Compaction/recall de l'historique avant chaque run |

### 3.3 Le format d'un Message

```python
@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_call_id: str | None = None      # uniquement pour role="tool"
    name: str | None = None              # nom du tool (pour role="tool")
    tool_calls: list[ToolCall] = field(default_factory=list)  # pour role="assistant"
    # 0.4.0 — images attachées (pertinent pour role="user")
    attachments: list[ImageAttachment] = field(default_factory=list)
    # 0.3.2 — trace "thinking" renvoyée par certains modèles (DeepSeek thinking,
    # o-series avec reveal). À ré-injecter dans le tour suivant.
    reasoning_content: str | None = None
```

Et `ToolCall` :
```python
@dataclass
class ToolCall:
    id: str                  # identifiant unique généré par le LLM
    name: str                # nom du tool
    arguments: dict          # JSON-désérialisé
```

### 3.4 Le flux d'un message dans l'historique

```
[
  Message(role="system", content="Tu es..."),
  Message(role="user", content="Combien font 21 + 21 ?"),
  Message(role="assistant", content="", tool_calls=[
      ToolCall(id="call_abc", name="add", arguments={"a": 21, "b": 21})
  ]),
  Message(role="tool", tool_call_id="call_abc", name="add", content="42"),
  Message(role="assistant", content="42 + 42 = 42."),  # texte final
]
```

L'`assistant` peut avoir `content` vide ET `tool_calls` rempli — c'est valide, le LLM appelle juste un tool.

---

## 4. Référence API

### 4.1 `Agent`

```python
class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        registry: ToolRegistry | None = None,
        system_prompt: str | Callable[[], str] = DEFAULT_SYSTEM_PROMPT,  # callable : 0.10.0
        max_steps: int = 8,
        max_dynamic_tools_per_run: int = 3,
        temperature: float | None = None,
        max_tokens: int | None = None,
        post_turn_hook: PostTurnHook | None = None,       # 0.2.0
        max_corrections_per_run: int = 1,                  # 0.2.0
        trace: TraceEmitter | None = None,                 # 0.5.0
        memory: Memory | None = None,                      # 0.6.0
        parallel_tool_calls: bool = False,                 # 0.10.0
        token_budget: int | None = None,                   # 0.10.0
        tool_policy: ToolPolicy | None = None,             # 0.11.0
    ): ...
```

**Paramètres** (tous keyword-only sauf `provider`) :

| Param | Type | Défaut | Rôle |
|---|---|---|---|
| `provider` | `LLMProvider` | — | Fournisseur LLM concret. Utilise `create_provider(ModelConfig(...))` pour aller vite. |
| `registry` | `ToolRegistry \| None` | `None` (registry vide) | Pour partager un registry pré-rempli entre agents. |
| `system_prompt` | `str` | `DEFAULT_SYSTEM_PROMPT` | Instruction système préfixée à chaque run. |
| `max_steps` | `int` | `8` | Cap dur sur le nombre de tours LLM. Lève `MaxStepsExceeded` au-delà. |
| `max_dynamic_tools_per_run` | `int` | `3` | Cap dur sur `create_python_tool` (voir §11). |
| `temperature` | `float \| None` | `None` | Forwardé au provider si défini. |
| `max_tokens` | `int \| None` | `None` | Forwardé au provider si défini. |
| `post_turn_hook` | `PostTurnHook \| None` | `None` | Callback hôte appelé quand le LLM produit une réponse texte-only. Peut injecter une correction. Voir §4.7. |
| `max_corrections_per_run` | `int` | `1` | Cap dur sur le nombre de corrections que le hook peut injecter. |
| `trace` | `TraceEmitter \| None` | `None` | Émetteur d'événements typés (run_start, tool_call_*, run_end…). Voir §4.5. |
| `memory` | `Memory \| None` | `None` | Appelé via `memory.compact(messages)` UNE FOIS avant la boucle. Voir §4.6. |
| `parallel_tool_calls` | `bool` | `False` | Les tool calls d'un même tour s'exécutent en pool de threads (opt-in : handlers thread-safe). Voir §16.6. |
| `token_budget` | `int \| None` | `None` | Cap dur sur les tokens du run ; `TokenBudgetExceeded` au-delà. Voir §16.5. |
| `tool_policy` | `ToolPolicy \| None` | `None` | Politique d'exécution des outils : allow / deny / approbation humaine, fail-closed. Voir §20. |

NB : `system_prompt` accepte aussi un **callable** `() -> str`, réévalué à chaque
run (état vivant dans le prompt — voir §16.4).

**Méthodes** :

```python
    @classmethod
    def from_model(cls, provider: str, model: str, **kwargs) -> "Agent":
        """Raccourci : Agent.from_model('openai', 'gpt-4o-mini', max_steps=12, ...)."""

    @classmethod
    def from_model_config(cls, config: ModelConfig, **kwargs) -> "Agent":
        """Variante quand tu construis déjà ton ModelConfig (base_url custom, etc)."""

    def tool(self, func=None, *, name=None, description=None,
             input_schema=None, permissions=None):
        """Décorateur (cf §6.1) — délègue à self.registry.register."""

    def add_tool(self, func) -> Callable:
        """Enregistre une fonction déjà décorée par @tool (top-level)."""

    def enable_dynamic_tools(self, builder: DynamicToolBuilder) -> None:
        """Active le méta-outil create_python_tool. Voir §11."""

    def enable_evolution(self, runtime, *, capabilities: set[str] | None = None):
        """Branche les outils d'évolution sur un projet hôte. Voir §10."""

    def register_recall_tool(
        self,
        *,
        name: str = "recall",
        description: str | None = None,
        default_k: int = 5,
    ) -> None:
        """Enregistre un tool `recall(query, k)` qui wrap self.memory.recall.
        No-op si self.memory is None. Voir §4.6."""

    def run(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
        checkpoint: CheckpointHook | None = None,   # 0.11.0 — snapshot RunState par étape (§19)
    ) -> AgentResult: ...

    def run_messages(
        self,
        messages: list[Message],
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
        checkpoint: CheckpointHook | None = None,   # 0.11.0
    ) -> AgentResult:
        """Continue une conversation. Si self.memory est défini, appelle
        memory.compact(messages) une seule fois avant la boucle."""

    def resume(self, state: RunState, *, context=None, cancel_token=None,
               checkpoint=None) -> AgentResult:
        """0.11.0 — reprend un run interrompu à state.step + 1 (§19).
        resume_stream(state, ...) = jumeau streaming."""
```

**Exemple complet — tout branché** :

```python
import threading
from autoagent import (
    Agent, BufferMemory, ImageAttachment, Message,
    ModelConfig, TraceEmitter, create_provider,
)

provider = create_provider(ModelConfig(provider="openai", model="gpt-4o-mini"))

def my_verifier(ctx) -> Message | None:
    # ctx.tool_calls = appels de cette user-turn ; ctx.correction_count = 0,1,...
    if not any(tc.name == "write_file" for tc in ctx.tool_calls):
        return Message(role="user", content="N'oublie pas de sauvegarder.")
    return None

cancel = threading.Event()

with TraceEmitter(file="run.jsonl") as trace:
    agent = Agent(
        provider,
        system_prompt="Tu es un assistant de code.",
        max_steps=12,
        memory=BufferMemory(max_messages=30),
        trace=trace,
        post_turn_hook=my_verifier,
        max_corrections_per_run=2,
    )
    agent.register_recall_tool()       # no-op ici car BufferMemory.recall = []
    result = agent.run("Refactor ./api.py", cancel_token=cancel)

print(result.output, result.steps)
```

### 4.2 `AgentResult`

```python
@dataclass
class AgentResult:
    output: str               # texte final de l'assistant (peut être vide)
    messages: list[Message]   # historique complet après run
    steps: int                # nombre de tours LLM consommés
```

### 4.3 `ToolRegistry`

Tu n'as généralement pas à manipuler le registry directement, mais pour les cas avancés :

```python
class ToolRegistry:
    def register(self, spec: ToolSpec, handler: Callable) -> None:
        """Enregistre un tool."""

    def replace(self, spec: ToolSpec, handler: Callable) -> None:
        """Remplace un tool existant du même nom (utilisé par DynamicToolBuilder)."""

    def add_function(self, func) -> Tool:
        """Enregistre une fonction qui a déjà été décorée."""

    def specs(self) -> list[ToolSpec]:
        """Retourne tous les ToolSpec (pour envoyer au LLM)."""

    def execute(self, call: ToolCall, context: dict | None = None) -> ToolResult:
        """Appelle le tool par son nom avec les arguments JSON désérialisés."""
```

**Hook intéressant** : tu peux **wrapper `registry.execute`** pour logger ou intercepter chaque appel :

```python
original_execute = agent.registry.execute
def execute_with_log(call, context=None):
    print(f"[tool] {call.name}({call.arguments})")
    result = original_execute(call, context=context)
    print(f"[tool] → ok={result.ok}")
    return result
agent.registry.execute = execute_with_log
```

C'est exactement ce qu'on fait dans le dashboard pour le progress toast SSE.

### 4.4 `ModelConfig` et `create_provider`

```python
@dataclass
class ModelConfig:
    provider: str                 # "openai" | "anthropic" | "deepseek" | "gemini"
    model: str                    # "gpt-4o-mini", "claude-sonnet-4-5", ...
    api_key: str | None = None    # fallback : variable d'env <PROVIDER>_API_KEY
    base_url: str | None = None   # pour endpoints custom (proxy, gateway)
    timeout: float = 60.0         # timeout HTTP global

def create_provider(config: ModelConfig) -> Provider: ...
```

Exemple :
```python
from autoagent import Agent, ModelConfig, create_provider

provider = create_provider(ModelConfig(
    provider="openai",
    model="gpt-4o-mini",
    timeout=180.0,                      # plus long pour les gros widgets
    base_url="https://my-proxy/v1",      # si tu as un proxy interne
))
agent = Agent(provider, system_prompt="Tu es ...", max_steps=12)
```

### 4.5 Tracing : `TraceEmitter` / `TraceEvent`

Ajouté en **0.5.0**. Module : `autoagent/trace.py`.

Un `TraceEmitter` reçoit des **événements typés** émis par `Agent.run_messages` à chaque point de cycle (début/fin de run, requête LLM, réponse LLM, appel/résultat tool, hook, annulation…). Tu peux l'écrire en JSONL et/ou passer un callback Python — les deux sont indépendants.

**Aucun overhead** si tu n'instancies pas de `TraceEmitter` : tous les emit sites sont guardés (`if self.trace is None: return None`).

#### 4.5.1 `TraceEvent`

```python
@dataclass
class TraceEvent:
    type: str                       # ex: "tool_call_start"
    span_id: str                    # token hex 16 chars, unique par event
    parent_id: str | None           # span_id du parent logique, None pour racine
    ts: float                       # time.time() (ou clock injecté)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]: ...
```

#### 4.5.2 `TraceEmitter`

```python
class TraceEmitter:
    def __init__(
        self,
        *,
        file: str | Path | IO[str] | None = None,    # JSONL append
        on_event: OnEvent | None = None,              # callback synchrone
        clock: Callable[[], float] | None = None,     # injection pour tests
    ): ...

    def emit(self, type_: str, payload: dict[str, Any] | None = None,
             *, parent_id: str | None = None) -> str:
        """Émet un event, retourne le span_id. NE LÈVE JAMAIS."""

    def close(self) -> None: ...      # idempotent
    def __enter__(self) -> "TraceEmitter": ...
    def __exit__(self, *exc) -> None: ...   # close()
```

- Si `file` est un **path**, l'émetteur ouvre le fichier en `append` et le ferme à `close()` / sortie du `with`.
- Si `file` est un **file-like déjà ouvert** (`.write(str)`), l'hôte garde la propriété — l'émetteur ne le ferme pas.
- `on_event` est synchrone et **isolé d'exception** : un callback qui lève est loggé puis ignoré.
- **Sérialisation** : un seul `RLock` interne sérialise la génération du `span_id`, le timestamp, l'écriture fichier ET le callback. Donc même sous deux threads concurrents, les events apparaissent dans le JSONL en true generation order.
- `emit()` ne lève jamais : `secrets.token_hex` et `clock` sont enveloppés dans des fallbacks.

#### 4.5.3 Catalogue d'événements (10 types)

Émis par `Agent.run_messages` :

| `type` | `parent_id` | `payload` |
|---|---|---|
| `run_start` | `None` (racine) | `{max_steps, model, message_count, tool_count}` |
| `llm_request` | span de `run_start` | `{step, message_count, tool_count}` |
| `llm_response` | span de `llm_request` | `{step, content_preview, tool_call_count, has_reasoning}` |
| `tool_call_start` | span de `llm_request` | `{name, call_id, arguments_preview}` |
| `tool_call_end` | span de `tool_call_start` | `{name, call_id, status: "ok"\|"error", duration_ms, content_preview}` |
| `post_turn_hook_invoked` | span de `llm_request` | `{correction_count}` |
| `post_turn_hook_correction` | span de `post_turn_hook_invoked` | `{content_preview}` |
| `cancelled` | span de `run_start` | `{step}` |
| `max_steps_exceeded` | span de `run_start` | `{max_steps}` |
| `run_end` | span de `run_start` | `{status: "ok"\|"cancelled"\|"max_steps"\|"error", steps?, output_preview?}` |

Chaque event a la forme **`{type, span_id, parent_id, ts, payload}`** — schéma stable.

#### 4.5.4 Redaction de secrets

**Tous les `*_preview`** passent par `truncate_preview` (et donc par `redact()` de `autoagent.logging`). Patterns nettoyés :

- `Bearer <token>` (tout header-shape)
- `x-api-key` / `x-goog-api-key` / `api_key` / `api-key` (JSON, dict, header)
- URL form `?key=...`

Donc un tool qui reçoit `{"authorization": "Bearer sk-..."}` en argument verra son `arguments_preview` redacté avant émission. Pareil pour les contents du LLM et les corrections du hook.

```python
from autoagent.trace import truncate_preview

# Pour formatter tes propres previews avec la même redaction :
safe = truncate_preview({"auth": "Bearer sk-xxx"}, limit=200)
# → '{"auth": "Bearer [REDACTED]"}'
```

#### 4.5.5 `OnEvent`

```python
OnEvent = Callable[[TraceEvent], None]
```

Cas typique : forward vers un backend externe (Langfuse, Phoenix, OTLP) :

```python
def push_to_langfuse(event: TraceEvent) -> None:
    if event.type == "tool_call_end":
        langfuse_client.log_span(
            name=event.payload["name"],
            duration_ms=event.payload["duration_ms"],
            status=event.payload["status"],
            span_id=event.span_id,
            parent_id=event.parent_id,
        )

trace = TraceEmitter(file="run.jsonl", on_event=push_to_langfuse)
agent = Agent(provider, trace=trace)
```

#### 4.5.6 Exemple : rendu live d'un tool call dans une UI

```python
import queue, threading

ui_queue: queue.Queue[TraceEvent] = queue.Queue()

with TraceEmitter(on_event=ui_queue.put) as trace:
    agent = Agent(provider, trace=trace)
    threading.Thread(target=lambda: agent.run("…"), daemon=True).start()

    while True:
        ev = ui_queue.get()
        if ev.type == "tool_call_start":
            ui.show_tool(ev.payload["name"], ev.payload["arguments_preview"])
        elif ev.type == "tool_call_end":
            ui.complete_tool(ev.payload["call_id"], ev.payload["status"])
        elif ev.type == "run_end":
            break
```

### 4.6 Mémoire : `Memory` / `BufferMemory`

Ajouté en **0.6.0**. Module : `autoagent/memory.py`.

`Memory` est un **Protocol** (≠ classe abstraite), donc tout objet qui implémente les deux méthodes `compact` + `recall` est accepté. `@runtime_checkable` permet l'`isinstance(..., Memory)`.

#### 4.6.1 Protocole

```python
@runtime_checkable
class Memory(Protocol):
    def compact(self, messages: list[Message]) -> list[Message]:
        """Retourne une liste de messages remodelée pour le prochain appel
        provider. L'implémentation décide : tronquer, résumer, projeter,
        ou ne rien faire. La liste retournée DOIT rester bien formée
        (toute Message(role='tool') doit suivre un assistant avec
        le même tool_call_id)."""

    def recall(self, query: str, k: int = 5) -> list[Message]:
        """Retrouver des messages passés pertinents pour `query`. Utilisé
        par le tool host-registered `recall`. Renvoie [] si pas de
        retrieval sémantique."""
```

#### 4.6.2 Sémantique d'intégration côté Agent

```python
def run_messages(self, messages, *, context=None, cancel_token=None):
    working_messages = list(messages)
    if self.memory is not None:
        try:
            working_messages = list(self.memory.compact(working_messages))
        except Exception:
            _log.exception("memory.compact raised; using messages unchanged")
    # ... boucle
```

- **UNE seule fois** avant la boucle (pas par-itération). Garde simple le post_turn_hook accounting et `turn_start`.
- **Erreurs isolées** : si `compact()` lève, on log et on poursuit avec les messages d'origine. Une mémoire boguée ne casse pas l'agent.
- Pour de la compaction mid-run, c'est à l'hôte d'appeler `memory.compact()` lui-même entre deux `run_messages`.

#### 4.6.3 `BufferMemory`

```python
class BufferMemory:
    def __init__(self, max_messages: int = 20) -> None: ...
    def compact(self, messages: list[Message]) -> list[Message]: ...
    def recall(self, query: str, k: int = 5) -> list[Message]:
        return []
```

Règles :

1. **Hard cap** : au plus `max_messages` messages non-système dans le retour. Non négociable.
2. **Ancrage sur user** : la queue de `max_messages` est avancée jusqu'au premier `role=="user"` pour éviter qu'un `tool` orphelin se retrouve en tête (les providers stricts rejettent).
3. **Drop si aucun user dans le budget** : on retourne **uniquement les system messages** — préférable à une conversation malformée.
4. `max_messages < 1` lève `ValueError` au constructeur.

#### 4.6.4 Quand l'utiliser

- ✅ Chat persistant avec un cap doux sur le coût des tokens
- ✅ Démos / scripts où une mémoire est OK pour 95% des cas
- ❌ Quand il faut **retrouver** un détail oublié → écris une `Memory` vectorielle (voir `examples/memory_vector.py`)
- ❌ Quand il faut **résumer** plutôt que tronquer → idem

#### 4.6.5 `agent.register_recall_tool()`

Si ta `Memory` implémente vraiment `recall` (ex : vector store), tu peux exposer un tool `recall(query, k)` à l'agent :

```python
agent = Agent(provider, memory=my_vector_memory)
agent.register_recall_tool(default_k=5)
# Le LLM voit maintenant un tool 'recall' qu'il peut appeler quand il a
# perdu un détail de la conversation.
```

Implémentation (`autoagent/agent.py` ~l.224-295) :

- **No-op silencieux** si `self.memory is None`.
- **Lookup dynamique** de `self.memory` à chaque call (pas de capture par closure) — réassigner `agent.memory = nouvelle_memory` après registration est honoré.
- **Erreurs absorbées** : si `recall()` lève, le tool retourne `{matches: [], error: "..."}` au lieu de propager — le LLM peut réagir gracieusement.
- **Truncation/redaction** : chaque `match["content"]` passe par `truncate_preview(..., limit=2000)` (mêmes patterns que `trace`).

#### 4.6.6 Mémoire vectorielle — voir `examples/memory_vector.py`

L'exemple fournit `VectorMemory(provider, embed_fn=None, keep_recent=6, summary_temperature=0.2)` (~520 LOC, no chromadb / no faiss — numpy en mémoire) :

- `embed_fn` par défaut : OpenAI `text-embedding-3-small` via `urllib` stdlib (cohérent avec le no-deps style d'autoagent). Lève si pas de `OPENAI_API_KEY`.
- Chunks par turn = (user + assistant qui suit + tool messages associés). Embeddés, indexés.
- **Lock TOCTOU-safe** : le check + embed + store est sous un même lock.
- **Résumé** des turns anciens via 1 appel LLM, retourne `[system…, summary, last keep_recent]`.
- **Redaction au chokepoint rendu** : `_render_chunk` retourne `redact("\n".join(out))` → cascade sur chunks stockés, input du résumé LLM, preview fallback.
- Recall : cosine similarity (dot product numpy sur vecteurs normalisés).

À utiliser comme base, pas comme prod-ready : tu vas typiquement le porter sur ton vector store (Chroma, Qdrant, pgvector).

### 4.7 `post_turn_hook` — boucle de vérification hôte

Ajouté en **0.2.0**. Permet à l'hôte (toi) d'inspecter ce que vient de faire l'agent et de **forcer une itération supplémentaire** en injectant un faux message user. Cas typique : vérifier qu'un tool précis a bien été appelé, qu'un fichier a été écrit, qu'une condition métier est remplie.

#### 4.7.1 Types

```python
@dataclass
class AgentTurnContext:
    messages: list[Message]              # historique complet (immutable snapshot)
    new_messages: list[Message]          # messages depuis le dernier user/system input
    tool_calls: list[ToolCall]           # tous les tool_calls de new_messages
    correction_count: int                # 0 au premier appel, +1 par correction

PostTurnHook = Callable[[AgentTurnContext], Message | None]
```

#### 4.7.2 Sémantique

- Le hook est appelé **uniquement** quand le LLM produit une réponse **sans tool_calls** (texte final). Sinon la boucle continue normalement.
- Retourne `None` → la run se termine, `AgentResult.output` est le content du LLM.
- Retourne `Message(role="user", content="...")` → la boucle reprend, le message est injecté dans `working_messages`, `correction_count += 1`, et `turn_start` est avancé.
- **Cap dur** : `max_corrections_per_run` (défaut 1) — au-delà, le hook n'est plus appelé. Évite la boucle infinie en cas de hook bavard.
- **Exceptions isolées** : un hook qui lève est loggé, et c'est comme s'il avait retourné `None`. Un hôte cassé n'empêche pas l'agent de répondre.

#### 4.7.3 Exemple — vérifier qu'un fichier a été sauvegardé

```python
from autoagent import Agent, AgentTurnContext, Message

def must_have_saved(ctx: AgentTurnContext) -> Message | None:
    wrote = any(tc.name in {"write_file", "replace_text"} for tc in ctx.tool_calls)
    if not wrote:
        return Message(
            role="user",
            content="Je n'ai pas vu d'appel à write_file. Sauvegarde la modification.",
        )
    return None

agent = Agent(
    provider,
    post_turn_hook=must_have_saved,
    max_corrections_per_run=1,
)
```

Le hook ne **ré-exécute pas** le LLM ni les tools ; il dit juste "ajoute ce nouveau prompt et fais un tour de plus". La logique réelle est dans le LLM qui reçoit la correction.

### 4.8 `cancel_token` — annulation coopérative

Ajouté en **0.2.0**. Mécanisme : tu passes un `threading.Event` à `run` / `run_messages` ; l'agent vérifie `cancel_token.is_set()` **entre deux itérations** et lève `AgentCancelled` si oui.

```python
import threading
from autoagent import AgentCancelled

cancel = threading.Event()

# Dans un thread d'UI :
threading.Timer(10.0, cancel.set).start()

try:
    result = agent.run("Long task...", cancel_token=cancel)
except AgentCancelled as exc:
    print(f"Annulé : {exc}")
```

#### 4.8.1 Contrat précis

- **Vérification** : au début de chaque itération de la boucle, AVANT l'appel `provider.complete(...)`. Si `is_set()` → emit `cancelled`, emit `run_end(status="cancelled")`, raise `AgentCancelled`.
- **HTTP en vol non interrompu** : un appel LLM déjà parti n'est PAS coupé. La lib n'utilise pas async/threads sur les sockets ; donc si le LLM répond en 30s, tu attends 30s. La granularité de l'annulation est donc **entre les tours LLM**.
- **Pas d'interruption des tools** : un tool qui boucle dans son code ne sera pas tué par le `cancel_token`. À l'auteur du tool de respecter lui-même un `threading.Event` injecté via `context`.
- `AgentCancelled` est une sous-classe d'`AutoAgentError` exportée publiquement depuis `autoagent`.

### 4.9 Messages multimodaux : `ImageAttachment`

Ajouté en **0.4.0**. Permet d'attacher des images à un `Message(role="user")`. Chaque provider sérialise vers son propre format wire.

```python
@dataclass
class ImageAttachment:
    data: str                # data URL OU base64 brut OU https URL
    mime_type: str | None = None

    def as_data_url(self) -> str: ...
    def as_base64(self) -> tuple[str, str]: ...   # → (mime, base64)
```

#### 4.9.1 Trois formes acceptées pour `data`

| Forme | `mime_type` requis ? |
|---|---|
| `"data:image/jpeg;base64,/9j/..."` (data URL complète) | non |
| `"/9j/4AAQSkZJ..."` (base64 brut) | **oui** |
| `"https://cdn.example/photo.jpg"` | non (le provider la fetch) |

#### 4.9.2 Sérialisation par provider

| Provider | Sérialisation |
|---|---|
| OpenAI | content user = liste `[{type: "text", text: ...}, {type: "image_url", image_url: {url: <data_url>}}, ...]` |
| Anthropic | bloc `{type: "image", source: {type: "base64", media_type, data}}` (recoder via `as_base64()`) |
| Gemini | part `{inline_data: {mime_type, data}}` |

Toi tu ne vois jamais ça : tu fournis l'`ImageAttachment`, le provider fait le bon mapping. Tests dans `tests/test_providers.py`.

#### 4.9.3 Exemple

```python
from autoagent import Agent, ImageAttachment, Message

img = ImageAttachment(
    data=raw_base64_from_my_paste,
    mime_type="image/png",
)

result = agent.run_messages([
    Message(role="system", content="Tu décris des images."),
    Message(
        role="user",
        content="Que vois-tu sur cette capture ?",
        attachments=[img],
    ),
])
```

L'exemple `examples/web_app_evolution.py` montre le pattern UI complet (bouton 📎, paste, drag-drop, 4 images max, 8 MB chacune, whitelist MIME `image/{jpeg,png,webp,gif}`, thumbnails avant envoi, bulles d'image dans le chat).

### 4.10 `reasoning_content` (DeepSeek / o-series)

Ajouté en **0.3.2**. Certains modèles (DeepSeek v4 pro en thinking mode, o-series OpenAI avec reveal) émettent une trace de raisonnement séparée du `content` final. **Et exigent qu'elle leur soit ré-injectée au tour suivant**, sinon ils rejettent avec :

> `reasoning_content in thinking mode must be passed back`

#### 4.10.1 Surface API

- `LLMResponse.reasoning_content: str | None` — extrait par le provider à partir du payload renvoyé.
- `Message.reasoning_content: str | None` — sur les messages `role="assistant"`. La boucle agent propage `response.reasoning_content` dans le message qu'elle ajoute à `working_messages`.
- Le provider re-sérialise ce champ lors du tour suivant (cf. `providers/openai.py` `_message_to_wire`, ligne 97-98 : `if message.role == "assistant" and message.reasoning_content: data["reasoning_content"] = message.reasoning_content`).

Tu n'as **rien à faire** côté hôte. La chaîne est :

```
provider.complete() → LLMResponse(reasoning_content=...)
                  → Message(role="assistant", reasoning_content=...)
                  → re-sérialisé au prochain appel provider
```

---

## 5. La boucle interne (run_messages)

Pseudo-code de `Agent.run_messages` (extrait simplifié de `agent.py`) :

```python
def run_messages(self, messages):
    history = list(messages)
    steps = 0
    while steps < self.max_steps:
        response = self.provider.complete(
            LLMRequest(
                messages=history,
                tools=self.registry.specs(),
                tool_choice="auto",
            )
        )
        steps += 1

        # 1. ajoute la réponse assistant à l'historique
        history.append(Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
        ))

        # 2. si pas de tool calls → on s'arrête, c'est la réponse finale
        if not response.tool_calls:
            return AgentResult(
                output=response.content,
                messages=history,
                steps=steps,
            )

        # 3. sinon, exécute chaque tool et ajoute les résultats
        for call in response.tool_calls:
            result = self.registry.execute(call, context={...})
            history.append(Message(
                role="tool",
                tool_call_id=call.id,
                name=call.name,
                content=result.to_text(),    # serialise dict → JSON string
            ))

    # max_steps dépassé
    raise MaxStepsExceeded(f"Agent exceeded max_steps={self.max_steps}")
```

**Points importants** :

- `max_steps` borne le nombre de **tours LLM**, pas le nombre de tool calls (un tour peut faire plusieurs tool calls en parallèle)
- Si le LLM produit une réponse mixte `content` + `tool_calls`, on garde **les deux** dans l'historique (Anthropic le permet, OpenAI rarement)
- `result.to_text()` sérialise en JSON. Si tu retournes des objets non-sérialisables (dataclass, Decimal…), tu auras une erreur — toujours retourner `dict[str, Any]` JSON-safe
- `MaxStepsExceeded` est une exception explicite, pas un retour silencieux

---

## 6. Écrire des tools

### 6.1 Trois manières d'enregistrer un tool

**A. Décorateur immédiat** (le plus courant) :

```python
@agent.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
```

**B. Avec override explicite** :

```python
@agent.tool(
    name="compute_sum",
    description="Sums two integers efficiently.",
    permissions=["filesystem.read"],
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
)
def _add(a: int, b: int) -> int:
    return a + b
```

**C. Top-level pour partager entre agents** :

```python
# tools.py
from autoagent import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# main.py
from tools import add
agent.add_tool(add)
```

### 6.2 Anatomie d'un tool généré automatiquement

Quand tu écris :
```python
@agent.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
```

la lib produit :
```python
ToolSpec(
    name="add",
    description="Add two integers.",          # de la docstring
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
    permissions=[],
)
```

Et stocke `handler=add` pour l'exécution.

### 6.3 Types Python supportés pour le schema auto

| Annotation Python | JSON schema généré |
|---|---|
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `str` | `{"type": "string"}` |
| `bool` | `{"type": "boolean"}` |
| `list[T]` | `{"type": "array", "items": <schema(T)>}` |
| `dict[str, T]` | `{"type": "object", "additionalProperties": <schema(T)>}` |
| `Literal["a", "b"]` | `{"type": "string", "enum": ["a", "b"]}` |
| `Optional[T]` / `T \| None` | `<schema(T)>` + ajout dans non-required |
| `Union[A, B]` | `{"anyOf": [<schema(A)>, <schema(B)>]}` |
| `Enum` | `{"type": "string", "enum": [<values>]}` |
| Aucune annotation | `{}` (schema vide — le LLM peut envoyer n'importe quoi) |

Voir `autoagent/registry.py` → `schema_from_annotation()`.

### 6.4 Tools avec contexte

Un tool peut recevoir un paramètre nommé `context` qui contient des infos du run :

```python
@agent.tool
def get_user_settings(context: dict | None = None) -> dict:
    """Return the current user's settings."""
    user_id = context.get("user_id") if context else None
    return load_settings(user_id)
```

Tu passes le contexte via `agent.run_messages(history, context={"user_id": 42})` — ou tu wrap `registry.execute` pour l'injecter.

### 6.5 Permissions — convention

Les `permissions` sont des **strings libres** que **ton code** interprète. Convention dans cette lib :

| Permission | Sens |
|---|---|
| `"filesystem.read"` | Le tool lit des fichiers |
| `"filesystem.write"` | Le tool écrit des fichiers |
| `"network"` | Le tool fait des appels réseau (HTTP, socket…) |
| `"db.read"`, `"db.write"` | Accès base de données |

C'est **toi** qui valides ou non en lisant `tool.spec.permissions`. La lib ne fait rien automatiquement avec ces tags **sauf** pour les tools dynamiques (voir §11), où les permissions filtrent les imports autorisés dans le code généré.

### 6.6 Retour d'un tool — règles

- Doit être **JSON-sérialisable** (dict, list, str, int, float, bool, None)
- Si exception levée → renvoyée au LLM dans le tool result comme `{"error": "..."}`
- Si retour `None` → sérialisé en `"null"` côté LLM
- Pas de générateurs, pas de Future/Promise, pas de dataclass non-asdict

Pattern recommandé :
```python
def my_tool(x: int) -> dict:
    try:
        result = do_stuff(x)
        return {"value": result, "ok": True}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
```

---

## 7. Génération automatique du JSON schema

### 7.1 Pourquoi c'est important

Les LLM modernes (OpenAI, Anthropic) attendent un JSON Schema strict pour chaque tool. Sans schema correct, le LLM peut envoyer des arguments mal typés et tu plantes.

`autoagent` génère ce schema à partir des annotations Python. Plus tu types proprement, plus le LLM appelle correctement tes tools.

### 7.2 Cas avancés

**Literal pour des enums string** :
```python
from typing import Literal

@agent.tool
def set_status(status: Literal["pending", "active", "done"]) -> dict:
    """Set the status."""
    return {"status": status}
```
→ Schema : `{"type": "string", "enum": ["pending", "active", "done"]}` → le LLM ne peut envoyer **que** ces valeurs.

**Nested dict** :
```python
@agent.tool
def search(query: dict, size: int = 10) -> dict:
    """..."""
```
→ `query` aura `{"type": "object"}` sans `properties` — le LLM peut envoyer n'importe quoi. Si tu veux contraindre, passe un `input_schema` explicite.

**Override total** :
```python
@agent.tool(input_schema={
    "type": "object",
    "properties": {
        "query": {
            "type": "object",
            "properties": {
                "match": {"type": "object"},
                "term":  {"type": "object"},
            },
        },
        "size": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["query"],
})
def es_search(query: dict, size: int = 10) -> dict:
    ...
```

### 7.3 Forcer `additionalProperties: false`

Par défaut, la lib génère `additionalProperties: false` au top-level pour empêcher le LLM d'ajouter des champs imprévus. Si tu veux l'autorisation, passe un `input_schema` explicite.

---

## 8. Providers

### 8.1 Interface

```python
class Provider(Protocol):
    config: ModelConfig
    def complete(self, request: LLMRequest) -> LLMResponse: ...
```

Une **seule méthode** : `complete`. Pas d'async, pas de streaming, pas de batch.

### 8.2 LLMRequest / LLMResponse

```python
@dataclass
class LLMRequest:
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    tool_choice: str | None = "auto"    # "auto" | "none" | "required"

@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None                      # réponse provider brute (debug)
```

### 8.3 Providers fournis

Tous dans `autoagent/providers/` :

| Provider | Fichier | Modèles testés |
|---|---|---|
| OpenAI | `providers/openai.py` | gpt-4o-mini, gpt-4o, gpt-4.1-mini |
| Anthropic | `providers/anthropic.py` | claude-sonnet-4-5, claude-opus-4 |
| DeepSeek | `providers/deepseek.py` | deepseek-chat |
| Gemini | `providers/gemini.py` | gemini-2.0-flash |

Chacun fait :
1. Traduit `LLMRequest` (format autoagent) → format du provider
2. Envoie via `urllib.request` (pas de SDK)
3. Parse la réponse → `LLMResponse` avec `content` + `tool_calls`

### 8.4 Différences entre providers à connaître

| Aspect | OpenAI | Anthropic | DeepSeek | Gemini |
|---|---|---|---|---|
| Format messages | `messages[]` | `system` séparé + `messages[]` | OpenAI-compat | `contents[]` différent |
| Tool calls | `tool_calls[]` sur assistant | `tool_use` content blocks | OpenAI-compat | `functionCall` |
| Tool results | `role="tool"` | `tool_result` content block | OpenAI-compat | `functionResponse` |
| Fiabilité tool chains | ✅ Très bonne | ✅ Très bonne | ⚠️ Échoue souvent > 3 tools | ✅ Bonne |
| Contexte max | 128k | 200k | 1M | 1M |
| Coût | ~$0.15/M input | ~$3/M input | ~$0.14/M input | ~$0.075/M input |

### 8.5 OpenAI : nouveaux modèles & `max_completion_tokens` (0.3.1)

Depuis 0.3.1, `providers/openai.py` détecte automatiquement les modèles qui rejettent `max_tokens` au profit de `max_completion_tokens` :

```python
def _uses_max_completion_tokens(model: str) -> bool:
    m = model.lower()
    return m.startswith(("o1", "o3", "o4", "gpt-5"))
```

Familles concernées : **`o1*`**, **`o3*`**, **`o4*`**, **`gpt-5*`**.

- Modèles legacy (`gpt-4o*`, `gpt-4.1*`, `gpt-4*`) : continuent d'utiliser `max_tokens`.
- Modèles modernes : la lib substitue à `max_completion_tokens` côté payload, transparent côté hôte.

Tu peux donc faire `Agent.from_model("openai", "o3-mini", max_tokens=2000)` sans te soucier du nommage du champ.

### 8.6 DeepSeek : `reasoning_content` en thinking mode (0.3.2)

DeepSeek v4 pro en thinking mode renvoie un champ `reasoning_content` à côté du `content`. **Et exige** qu'on le ré-injecte dans le tour suivant — sinon il refuse la requête :

```
{"error": "reasoning_content in thinking mode must be passed back"}
```

`autoagent` gère ça automatiquement (cf §4.10) : `LLMResponse.reasoning_content` est extrait, propagé sur le `Message(role="assistant")`, et re-sérialisé par le provider au tour d'après. Tu n'as rien à faire.

Si tu écris un provider custom OpenAI-compat qui parle à un modèle "thinking", n'oublie pas :

```python
# Parse :
reasoning_content=message.get("reasoning_content"),

# Re-serialize :
if message.role == "assistant" and message.reasoning_content:
    data["reasoning_content"] = message.reasoning_content
```

### 8.7 Ajouter un provider custom

Voir [§13.1](#131-ton-propre-provider).

---

## 9. ProjectWorkspace — lectures/écritures bornées

### 9.1 Pourquoi

Le LLM peut demander n'importe quel chemin. Sans bornement, c'est une faille (`/etc/passwd`, `../../secrets.env`, fichiers `.exe`…). `ProjectWorkspace` impose :

- **Allowlist d'extensions** : refuse tout `.py` si tu n'autorises que `.tsx`
- **Anti path-traversal** : refuse `..`, chemins absolus, symlinks qui sortent
- **Historique des écritures** : pour rollback en cas de validation KO

### 9.2 API

```python
class ProjectWorkspace:
    def __init__(
        self,
        root: str | Path,
        allowed_write_extensions: set[str] | None = None,  # None = tout autorisé
        max_write_chars: int = 200_000,
    ): ...

    def resolve(self, path: str) -> Path:
        """Lève ValueError si path remonte hors root."""

    def read_file(self, path: str) -> dict:
        """{'content': str, 'path': str}"""

    def write_file(self, path: str, content: str, reason: str = "") -> dict:
        """{'written': str, 'changed_id': int}. Refuse si extension hors allowlist."""

    def replace_text(self, path: str, old: str, new: str, reason: str = "") -> dict:
        """Remplace une chaîne exacte dans le fichier."""

    def list_changes(self) -> list[dict]:
        """[{'id', 'path', 'reason', 'timestamp', 'before_size', 'after_size'}]"""

    def rollback_change(self, change_id: int) -> dict: ...
    def rollback_last_change(self) -> dict: ...

    def list_files(self, subdir: str = "") -> list[str]: ...
```

### 9.3 Cas d'usage typique

```python
from autoagent import Agent, ProjectWorkspace

workspace = ProjectWorkspace(
    "./my_app/src",
    allowed_write_extensions={".py", ".json"},
)

@agent.tool
def list_files(subdir: str = "") -> dict:
    return {"files": workspace.list_files(subdir)}

@agent.tool
def read_file(path: str) -> dict:
    return workspace.read_file(path)

@agent.tool(permissions=["filesystem.write"])
def write_file(path: str, content: str, reason: str = "") -> dict:
    return workspace.write_file(path, content, reason)

@agent.tool(permissions=["filesystem.write"])
def rollback() -> dict:
    return workspace.rollback_last_change()
```

### 9.4 Comportement des refus

Si le LLM appelle `write_file('/etc/passwd', '...')`, la méthode `workspace.write_file` lève `ValueError("Path escapes workspace")`. La lib `Agent` catch et renvoie au LLM :
```json
{"error": "ValueError: Path escapes workspace: /etc/passwd"}
```
Le LLM voit l'erreur et corrige (typiquement il choisit un autre chemin ou abandonne).

---

## 10. EvolutionRuntime — l'agent pilote un projet entier

### 10.1 Quand l'utiliser

Quand tu veux que l'agent **modifie ton app vivante** : lire l'état, choisir un nouveau module Python, le brancher dans une pipeline déclarée, lancer la validation, rollback si KO.

Cas typique : un simulateur, un outil métier avec plugins, un jeu avec mécaniques évolutives.

### 10.2 Architecture

```python
class EvolutionRuntime:
    def __init__(
        self,
        workspace_root: str | Path,
        *,                                                  # tout le reste est keyword-only
        pipeline_path: str | None = None,                   # défaut "pipeline.json"
        validation_command: str | list[str] | None = None, # ["python", "-m", "pytest"]
        allow_custom_validation_command: bool = False,
        allowed_write_extensions: set[str] | None = None,
        state_reader: Callable[[], Any] | None = None,
        max_write_chars: int = 200_000,
    ) -> None: ...

    # Les fonctions host ne sont PAS un paramètre du constructeur : on les ajoute après.
    def register_host_function(self, name: str, func: Callable) -> None: ...
```

Tu fournis 3 choses :

1. **`state_reader`** : fonction qui retourne l'état actuel (snapshot)
2. **`host_functions`** : actions sûres déjà implémentées
3. **`pipeline.json`** : déclaration des slots remplaçables

### 10.3 pipeline.json

```json
{
  "slots": {
    "traffic_light_policy": {
      "module": "default_policy",
      "description": "Decides green/red duration for each light."
    },
    "enemy_spawner": {
      "module": "basic_spawner",
      "description": "..."
    }
  }
}
```

L'agent peut remplacer `traffic_light_policy.module` par `"adaptive_policy"` après avoir écrit le fichier `adaptive_policy.py`.

### 10.4 Capabilities exposées

Quand tu appelles `agent.enable_evolution(runtime, capabilities=...)`, les tools suivants sont ajoutés au registry :

| Capability | Tools ajoutés |
|---|---|
| `"read"` | `list_project_files`, `read_project_file`, `list_pipeline_slots`, `get_pipeline_slot`, `list_changes` |
| `"write"` | `write_project_file`, `replace_project_text`, `rollback_change`, `rollback_last_change` |
| `"host_state"` | `get_runtime_state` |
| `"host_call"` | `list_host_functions`, `call_host_function` |
| `"pipeline"` | `replace_pipeline_slot` |
| `"validate"` | `run_validation` |

Défaut `capabilities=None` = tout activé.

### 10.5 Exemple complet

```python
from autoagent import Agent, EvolutionRuntime

class MyGame:
    def __init__(self):
        self.score = 0
        self.enemies = []

    def snapshot(self) -> dict:
        return {"score": self.score, "enemies": len(self.enemies)}

    def spawn_enemy(self, kind: str = "basic") -> dict:
        # ...
        return {"enemies": len(self.enemies)}

game = MyGame()

runtime = EvolutionRuntime(
    "./game_workspace",
    pipeline_path="pipeline.json",
    validation_command=["python", "-m", "pytest", "-x"],
    state_reader=game.snapshot,
    allowed_write_extensions={".py", ".json"},
)
runtime.register_host_function("spawn_enemy", game.spawn_enemy)

agent = Agent.from_model("openai", "gpt-4o-mini", max_steps=14)
agent.enable_evolution(runtime, capabilities={"read", "write", "pipeline", "validate"})

result = agent.run("Ajoute un boss niveau 5 et augmente la difficulté.")
print(result.output)
```

L'agent peut :
1. Lire l'état via `get_runtime_state()`
2. Écrire `level5_boss.py` via `write_project_file()`
3. Remplacer le slot `enemy_spawner` via `replace_pipeline_slot()`
4. Lancer `run_validation()` → `pytest -x`
5. Si KO → `rollback_last_change()` et raisonne

### 10.6 Sécurité de `validation_command`

`validation_command` **doit** être passée comme **liste**, pas comme string. C'est imposé par la lib (refuse `str` au constructor) pour empêcher l'injection shell.

```python
# ✗ INTERDIT
runtime = EvolutionRuntime(..., validation_command="pytest -x")

# ✓ OK
runtime = EvolutionRuntime(..., validation_command=["python", "-m", "pytest", "-x"])
```

Si tu veux laisser l'agent passer une commande custom au runtime, mets `allow_custom_validation_command=True` — mais **fais-le seulement** si tu fais confiance au prompt et au LLM. Risque d'exécution arbitraire sinon.

---

## 11. Tools dynamiques

### 11.1 Vue d'ensemble

Quand `enable_dynamic_tools()` est actif, l'agent voit un **méta-outil** `create_python_tool` qui lui permet de demander la création d'un nouveau tool en plein run.

```python
from autoagent import Agent, DynamicToolBuilder, ModelConfig, create_provider

manager = create_provider(ModelConfig(provider="openai", model="gpt-4o-mini"))
builder = create_provider(ModelConfig(provider="anthropic", model="claude-sonnet-4-5"))

agent = Agent(manager, max_dynamic_tools_per_run=3)
agent.enable_dynamic_tools(DynamicToolBuilder(builder, tools_dir="./tools_dyn"))

result = agent.run("Lis ./access.log et donne-moi le top 5 des URLs visitées.")
```

L'agent décide qu'il a besoin d'un compteur, appelle `create_python_tool(name="count_paths", permissions=[...])`, le builder écrit le code, validation AST, exécution sandbox, et l'agent l'utilise.

### 11.2 Le pipeline interne

```
1. agent.run() → la boucle commence
2. LLM appelle create_python_tool({name, description, permissions})
3. DynamicToolBuilder demande au builder LLM d'écrire le code
4. Code reçu → parsing AST :
   - refus si eval/exec/__import__/getattr suspect
   - refus si import d'un module hors allowlist selon permissions
5. Code passé → écrit dans tools_dir/<name>.py
6. Wrapper créé : execute_in_sandbox(<name>.py, args) → subprocess
7. Tool enregistré dans registry → visible au prochain tour LLM
8. Le LLM appelle alors create_python_tool reste OU le nouveau tool directement
```

### 11.3 Validation statique (AST) — `validate_generated_tool_code`

Avant toute exécution, le code est parsé avec `ast.parse()` et doit définir une fonction
`run(args, context)` **et** un dict `TOOL` (clés `name`, `description`, `input_schema` ;
lu par `extract_tool_metadata`). C'est une **denylist** (tout ce qui n'est pas interdit
passe), pas une allowlist. Constantes dans `autoagent/sandbox.py` :

| Catégorie | Contenu refusé |
|---|---|
| `ALWAYS_BANNED_CALLS` | `eval`, `exec`, `compile`, `__import__`, `input`, `breakpoint` |
| Appels shell | `system`, `popen` |
| `PROCESS_SPAWN_CALLS` | `fork`, `forkpty`, `kill`, `startfile`, `putenv`, `execl*`, `execv*`, `spawn*`, `posix_spawn*` |
| `ALWAYS_BANNED_MODULES` (quelles que soient les permissions) | `subprocess`, `ctypes`, `multiprocessing`, `signal`, `importlib`, `os`, `posix`, `nt`, `sys`, `pty` |
| `DANGEROUS_NAMES` (référencés par `Name` **ou** attribut) | `__builtins__`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`, `importlib`, et les dunders d'introspection `__class__`, `__bases__`, `__subclasses__`, `__mro__`, `__globals__`, `__dict__`, `__code__`… |

Le blocage des dunders d'introspection ferme l'évasion CPython classique
`().__class__.__bases__[0].__subclasses__()` (qui atteint `subprocess.Popen` / fonctions
`os` sans aucun import). Certaines familles de modules sont **rouvertes** par permission :

| Permission (dans `TOOL["permissions"]`) | Débloque |
|---|---|
| `"network"` | `NETWORK_MODULES` : `socket`, `urllib`, `http`, `ftplib`, `smtplib`, `imaplib`, `poplib`, `requests` |
| `"filesystem.*"` (p.ex. `filesystem.read`) | `FILESYSTEM_MODULES` : `pathlib`, `glob`, `shutil`, `tempfile`, et l'appel `open()` |

> ⚠️ La denylist AST **durcit** mais n'est pas une frontière à elle seule (Python est trop
> dynamique). La vraie isolation vient du `DockerSandbox` (§11.4).

### 11.4 Deux sandboxes — `SubprocessSandbox` vs `DockerSandbox`

Les deux exposent le même contrat
`run_python_tool(file_path, args, context=None, *, allow_network=False, host_functions=None)`
et sont interchangeables derrière `make_sandbox()`.

**`SubprocessSandbox(timeout=10.0)`** — le repli. Lance `python -X utf8 -I -S -c <runner> <tool>`
(mode isolé : pas de `PYTHONPATH`, pas de site-packages, `env={}`), args en JSON sur stdin,
résultat JSON sur stdout.
> ⚠️ Un simple subprocess **ne peut PAS isoler le réseau** — `allow_network` n'est accepté que
> pour parité de signature ; seule la denylist AST joue. C'est du durcissement, pas une frontière.

**`DockerSandbox(image="python:3.11-slim", timeout=10.0, memory="256m", cpus="1.0", pids_limit=128)`**
— la VRAIE frontière (isolation OS). Par appel :
```
docker run --rm -i --read-only --tmpfs /tmp:size=32m \
  --memory 256m --cpus 1.0 --pids-limit 128 \
  --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges \
  [--network none]            # sauf si le tool a la permission "network"
  python:3.11-slim python -X utf8 -I -S -c <runner>
```
Le **code du tool voyage par stdin** (pas de volume monté → portable Windows/macOS/Linux, zéro
piège de montage). Conteneur jetable, FS racine read-only, non-root, toutes capabilities
supprimées, limites mémoire/CPU/pids.

**`make_sandbox(prefer_docker=True, timeout=10.0, image="python:3.11-slim")`** renvoie un
`DockerSandbox` si un démon Docker répond (`docker_available()`, mis en cache une fois), sinon le
`SubprocessSandbox`. C'est le point d'entrée recommandé :
```python
from autoagent.sandbox import make_sandbox
sandbox = make_sandbox()          # Docker si dispo, sinon subprocess
```

**Prérequis & setup Docker** — un démon Docker doit tourner (`docker info` doit répondre ;
`docker_available()` le teste et met le résultat en cache). L'image `python:3.11-slim` est **tirée
une seule fois** au premier appel (`docker pull`, via `_ensure_image()`), puis réutilisée — aucun
`Dockerfile` ni build de ta part. Aucun montage de volume (le code du tool passe par stdin), donc
rien à configurer côté chemins. Si Docker est absent ou arrêté, `make_sandbox()` retombe
**silencieusement** sur `SubprocessSandbox` : l'app continue de tourner, mais sans isolation réseau
— surveille le mode renvoyé par `load_tools()` (`"native"` / `"sandbox"`) et, si l'isolation forte
est requise en prod, vérifie explicitement `docker_available()` au démarrage.

```python
from autoagent.sandbox import docker_available
assert docker_available(), "Docker requis pour l'isolation des tools dynamiques en prod"
```

### 11.5 Pont host-function — `call_host` (accès contrôlé au host)

Un tool sandboxé n'a ni réseau ni objet de l'application. Pour lui donner un accès **contrôlé**
à des capacités du host (une requête SQL read-only, un GET HTTP allowlisté…), passe
`host_functions={"nom": callable}` au sandbox. Le tool les appelle ainsi :

```python
def run(args, context):
    rows = context["call_host"]("sql_query", {"sql": "SELECT count(*) FROM events"})
    return {"rows": rows}
```

Protocole (`_drive_bridge`) : JSON ligne-à-ligne sur les **pipes stdio** de l'enfant. Comme il
chevauche stdin/stdout — pas le réseau — il fonctionne **même sous `--network none`**. Seuls les
noms whitelistés répondent (`fn(**args)`) ; tout autre nom est refusé. Le tool n'obtient jamais
l'objet réel (DB, secrets) : seulement le résultat retourné par la fonction host.

C'est le mécanisme central de `examples/traffic_incident_agent/` (host functions `http_get`,
`avatar_station_proche`, `ask_user`, `connecter_service`…) et de `sql_react_dashboard` (`sql_query`).

### 11.6 Promotion humaine sandbox → natif — `autoagent/approval.py`

Cycle de confiance :
```
généré   ──▶  SANDBOX (context JSON-only, accès host via le seul pont)
   │  un humain relit le code + permissions, lance `approve`
   ▼
approuvé ──▶  NATIF (in-process, reçoit les vrais handles via context)
```

La porte de confiance = le **hash sha256** du source, épinglé dans un manifeste
(`approved_tools.json`, **committé en git**). Un tool tourne en natif UNIQUEMENT si le hash de son
source courant est dans le manifeste — un octet changé → retour sandbox jusqu'à re-validation
(ferme le trou TOCTOU « swap après approbation »).

`load_tools()` est le **point de câblage unique** : il enregistre chaque `*.py` d'un dossier sur
l'agent et choisit le mode par tool.

```python
from autoagent.approval import ToolManifest, load_tools
from autoagent.sandbox import make_sandbox

manifest = ToolManifest.load("approved_tools.json")
modes = load_tools(
    agent, "./dynamic_tools", manifest,
    host_context={"db": db},                          # injecté aux tools NATIFS (approuvés)
    sandbox=make_sandbox(),
    sandbox_host_functions={"sql_query": db.query},   # le pont des tools SANDBOX
)
# modes -> [("sql_aggregate", "native"), ("foo", "sandbox"), ...]
```

- hash dans le manifeste → **natif** : reçoit `host_context` (vrais objets) **plus** un `call_host`
  in-process — un tool écrit pour le pont marche à l'identique une fois promu.
- sinon → **sandbox** : `context` vide, accès host SEULEMENT via `sandbox_host_functions`.
- code qui échoue la validation AST → mode `"invalid"`, jamais enregistré.

API : `approve_tool(file, manifest, *, approved_by=…)` (valide statiquement puis épingle le hash),
`reject_tool(file, rejected_dir=None)`, `review_card(file)` (name/description/permissions/sha256/code
pour la relecture humaine).

CLI :
```bash
python -m autoagent.approval list    ./dynamic_tools [--manifest approved_tools.json]
python -m autoagent.approval show    ./dynamic_tools/sql_aggregate.py
python -m autoagent.approval approve ./dynamic_tools/sql_aggregate.py --by alice
python -m autoagent.approval reject  ./dynamic_tools/foo.py [--to ./rejected]
```

### 11.7 Persistance entre runs

| Niveau | Effet |
|---|---|
| Même `agent.run()` | Tool reste dans le registry pour les tours suivants ✅ gratuit |
| Différents `agent.run()`, même instance d'Agent | Tool reste dans le registry ✅ gratuit |
| Restart process Python | Tools écrits sur disque ; rechargés au démarrage par **`load_tools()`** (§11.6) qui choisit natif/sandbox par hash |

### 11.8 Limites

- **`max_dynamic_tools_per_run` (défaut 3)** : au-delà, refusé pour éviter l'inflation
- **Pas de pip install** : seuls les modules stdlib autorisés sont disponibles dans le sandbox
- **Pas de retour binaire** : le résultat doit être JSON. Pour traiter des images, l'agent doit encoder en base64

---

## 12. Tests

### 12.1 Lancer la suite

```bash
python -m unittest discover -s tests
```

339 tests couvrent (≈+110 depuis 0.3.0) :
- La boucle `agent.run()` / `run_messages()`
- Les providers (mocks HTTP des payloads OpenAI/Anthropic/Gemini/DeepSeek, **round-trip d'`ImageAttachment` sur les 3 providers**, `_uses_max_completion_tokens`, `reasoning_content`)
- Le sandbox subprocess
- Les schemas auto-générés (Literal, Optional, Union, Enum…)
- Le workspace + path traversal + rollback
- Le DynamicToolBuilder (validation AST, refus correctement, codes valides acceptés)
- Les capabilities d'évolution
- **`tests/test_trace.py`** — 36 tests : TraceEmitter (file/callback/clock), event shapes, redaction, exception isolation, threading
- **`tests/test_memory.py`** — 31 tests : BufferMemory (hard cap, ancrage user, drop-tail), Protocol shape, intégration agent
- **`tests/test_agent_post_turn_hook.py`** — 11 tests : injection, max_corrections, isolation d'exception
- **`tests/test_agent_cancel.py`** — 8 tests : cancel entre tours, exception levée, run_end emitté
- **`tests/test_qt_3d_parser.py`** — 7 tests : `_split_top_level_args`, `_classify_mesh_arg`, `describe_scene`

### 12.2 Écrire un test pour un tool custom

```python
import unittest
from autoagent import Agent
from autoagent.providers.fake import FakeProvider

class MyToolTest(unittest.TestCase):
    def test_add(self):
        # FakeProvider simule un LLM scripté
        provider = FakeProvider([
            # Tour 1 : appelle add(21, 21)
            LLMResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 21, "b": 21})]),
            # Tour 2 : répond avec le résultat
            LLMResponse(content="42"),
        ])
        agent = Agent(provider)

        @agent.tool
        def add(a: int, b: int) -> int:
            return a + b

        result = agent.run("21+21 ?")
        self.assertEqual(result.output, "42")
        self.assertEqual(result.steps, 2)
```

### 12.3 Tester un workspace borné

```python
def test_workspace_refuses_traversal(self):
    ws = ProjectWorkspace("./tmp", allowed_write_extensions={".txt"})
    with self.assertRaises(ValueError):
        ws.write_file("../../etc/passwd", "hack")
    with self.assertRaises(ValueError):
        ws.write_file("ok.exe", "binary")        # extension hors allowlist
    ws.write_file("ok.txt", "hello")             # OK
```

---

## 13. Extension : ton propre provider, ton propre runtime

### 13.1 Ton propre provider

```python
from autoagent.providers.base import Provider
from autoagent.schema import LLMRequest, LLMResponse, ToolCall, ModelConfig
from autoagent.http import post_json

class MyCustomProvider(Provider):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.my-llm.com/v1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        # 1. Traduit l'historique au format de mon LLM
        payload = {
            "model": self.config.model,
            "messages": [self._serialize_msg(m) for m in request.messages],
            "tools": [self._serialize_tool(t) for t in request.tools],
        }
        # 2. Envoie
        raw = post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=self.config.timeout,
        )
        # 3. Parse le retour
        msg = raw["choices"][0]["message"]
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"]),
            ))
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
            raw=raw,
        )

    def _serialize_msg(self, m): ...   # dépend du format de ton LLM
    def _serialize_tool(self, t): ...
```

Puis enregistre-le dans `create_provider` ou utilise-le directement :

```python
agent = Agent(MyCustomProvider(config))
```

### 13.2 Ton propre runtime / pattern

Si ni `ProjectWorkspace` seul ni `EvolutionRuntime` ne te suffisent, tu peux composer tes propres tools comme un runtime :

```python
class MyAppRuntime:
    def __init__(self, app):
        self.app = app

    def install(self, agent: Agent):
        @agent.tool
        def get_users() -> list[dict]:
            return [u.to_dict() for u in self.app.users]

        @agent.tool
        def get_orders(user_id: int) -> list[dict]:
            return [o.to_dict() for o in self.app.orders_of(user_id)]

        @agent.tool(permissions=["app.write"])
        def update_setting(key: str, value: str) -> dict:
            self.app.settings[key] = value
            return {"ok": True}

runtime = MyAppRuntime(my_app)
agent = Agent.from_model("openai", "gpt-4o-mini")
runtime.install(agent)
agent.run("Pour chaque user inactif depuis 90j, regarde s'il a une commande en cours…")
```

---

## 14. Pièges fréquents et FAQ

### 14.1 Mon tool n'est jamais appelé

- **Vérifie la docstring** : c'est elle qui décrit le tool au LLM. Sans description, le LLM ne sait pas quand l'utiliser.
- **Vérifie les noms** : `name` du tool doit matcher entre le décorateur et le call (la lib le gère automatiquement, mais si tu mets `name=` explicite, attention aux typos).
- **Vérifie `tool_choice`** : par défaut `"auto"`. Si tu mets `"none"`, le LLM ne peut pas appeler de tools.

### 14.2 J'ai `MaxStepsExceeded`

- Augmente `max_steps` (défaut 8). Pour des chaînes complexes, 14-25 est raisonnable.
- Logge les tool calls pour comprendre ce que l'agent boucle. Souvent c'est un tool qui retourne une erreur que le LLM ne sait pas corriger.

### 14.3 Mon LLM hallucine des chemins / arguments

- **Renforce les docstrings** : précise les formats attendus, donne des exemples.
- **Utilise `Literal`** pour restreindre les valeurs possibles.
- **Validation côté tool** : retourne `{"error": "..."}` si l'argument est mauvais — le LLM corrige.

### 14.4 Mon historique explose en tokens

Depuis 0.6.0, branche une `Memory` :

```python
from autoagent import Agent, BufferMemory
agent = Agent(provider, memory=BufferMemory(max_messages=30))
# compact() est appelé une fois avant chaque run_messages().
```

`BufferMemory` impose un hard cap sur les non-system messages et ancre la queue sur le premier user message (pas d'orphan tool).

Pour une mémoire **vectorielle avec recall** (le LLM peut explicitement aller chercher un vieux détail) :

```python
# Voir examples/memory_vector.py
from examples.memory_vector import VectorMemory

memory = VectorMemory(provider, keep_recent=6)
agent = Agent(provider, memory=memory)
agent.register_recall_tool()   # expose un tool recall(query, k)
```

Si tu préfères tout faire à la main :

- **Sliding window** côté hôte (voir `HistoryStore` dans `examples/react_dashboard_agent.py`).
- Tronque les `tool_call.arguments` et `tool_message.content` après une certaine taille.
- Important : **garde toujours le system message et ne casse pas une paire `assistant.tool_calls` ↔ `tool.tool_call_id`** (sinon les providers rejettent la requête).

### 14.5 DeepSeek/Gemini me donne des résultats étranges

- DeepSeek est moins fiable sur les chaînes de >3 tool calls. Préfère OpenAI/Anthropic pour les workflows complexes.
- Gemini a parfois un format de tool_calls différent — vérifie que `providers/gemini.py` parse bien ta version.

### 14.6 Le sandbox subprocess est lent

C'est attendu : ~200-500ms par appel à cause du fork + boot Python. Pour des tools très fréquemment utilisés, **promeus-les** en tools officiels (voir presentation.md §1.6 promotion).

### 14.7 Comment je débugge un run ?

```python
result = agent.run("...")
for m in result.messages:
    print(f"[{m.role}] {m.content[:200]}")
    for tc in m.tool_calls:
        print(f"  → {tc.name}({tc.arguments})")
```

Ou wrap `registry.execute` pour logger chaque tool call (voir §4.3).

### 14.8 Comment je passe une clé API custom (pas dans .env) ?

```python
provider = create_provider(ModelConfig(
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-...",   # passe-la explicitement
))
```

### 14.9 Comment j'augmente le timeout HTTP ?

```python
provider = create_provider(ModelConfig(
    provider="openai",
    model="gpt-4o-mini",
    timeout=180.0,   # défaut 60s
))
```

### 14.10 Puis-je utiliser un proxy interne / Azure OpenAI ?

```python
provider = create_provider(ModelConfig(
    provider="openai",
    model="gpt-4o-mini",
    base_url="https://my-azure-proxy.openai.azure.com/v1",
    api_key="...",
))
```

Si le format diffère trop (ex: Azure utilise un path `deployments/<name>/chat/completions`), tu peux soit étendre `OpenAIProvider`, soit écrire un provider custom (§13.1).

### 14.11 Comment je trace ce que fait l'agent ?

Branche un `TraceEmitter` (§4.5). Pour persistence + UI live, donne-lui un path JSONL ET un callback :

```python
from autoagent import Agent, TraceEmitter

with TraceEmitter(file="run.jsonl", on_event=push_to_ui) as trace:
    agent = Agent(provider, trace=trace)
    agent.run("…")
# Le JSONL contient un event par ligne ; `push_to_ui` est appelé en synchrone.
```

Les `*_preview` sont déjà redactés (Bearer, api_key, etc.). Pour redacter du PII custom, wrap `on_event`.

### 14.12 Comment j'annule un run en cours ?

Passe un `threading.Event` à `run` / `run_messages` et set-le depuis ton thread d'UI (§4.8).

```python
import threading
from autoagent import AgentCancelled

cancel = threading.Event()
try:
    result = agent.run("…", cancel_token=cancel)
except AgentCancelled:
    ...

# Depuis ailleurs :
cancel.set()
```

Granularité = **entre les tours LLM**. Un HTTP en vol n'est pas coupé ; un tool bloquant n'est pas tué (à toi d'écrire des tools qui respectent un `threading.Event` injecté via `context`).

### 14.13 Comment je passe une image à l'agent ?

```python
from autoagent import ImageAttachment, Message

img = ImageAttachment(data=b64, mime_type="image/png")
agent.run_messages([
    Message(role="system", content="..."),
    Message(role="user", content="Décris-la.", attachments=[img]),
])
```

Le provider (OpenAI / Anthropic / Gemini) sérialise tout seul vers son format wire (§4.9). Voir `examples/web_app_evolution.py` pour le pattern UI complet (paste, drag-drop, thumbnails).

### 14.14 Comment je force l'agent à valider quelque chose avant de répondre ?

Passe un `post_turn_hook` (§4.7) qui regarde `ctx.tool_calls` et retourne un `Message(role="user", content="...")` quand la condition métier n'est pas remplie. Cap dur via `max_corrections_per_run` (défaut 1).

---

## Annexes

### Annexe A — Liste des fichiers de la lib

```
autoagent/
├── __init__.py              # exports publics, __version__ = "0.6.0"
├── agent.py                 # Agent, AgentResult, AgentTurnContext, PostTurnHook
├── schema.py                # ToolSpec, ToolCall, Message, ModelConfig, ImageAttachment
├── registry.py              # ToolRegistry + schema_from_callable
├── workspace.py             # ProjectWorkspace + path traversal protection
├── pipeline.py              # PipelineManager (pipeline.json)
├── evolution.py             # EvolutionRuntime, EVOLUTION_CAPABILITIES
├── dynamic.py               # DynamicToolBuilder, ToolBuildRequest
├── sandbox.py               # SubprocessSandbox, DockerSandbox, make_sandbox, pont host-function
├── http.py                  # post_json / post_sse (urllib wrapper + retry/backoff)
├── errors.py                # MaxStepsExceeded, AgentCancelled, ProviderError, TokenBudgetExceeded,
│                            # MCPError + ApprovalRequired (0.11.0), ...
├── logging.py               # get_logger + SecretRedactingFilter + redact()
├── trace.py                 # 0.5.0 — TraceEmitter, TraceEvent, OnEvent, truncate_preview
├── memory.py                # 0.6.0 — Memory (Protocol), BufferMemory ; 0.10.0 — SummarizingMemory ;
│                            # 0.12.0 — FactMemory (§21)
├── mcp.py                   # 0.11.0 — MCPClient (serveur MCP stdio → tools locaux, §17)
├── otel.py                  # 0.11.0 — OTelTraceExporter (trace → spans OTel, §18)
└── providers/
    ├── __init__.py
    ├── base.py              # Protocol LLMProvider
    ├── openai.py            # 0.3.1 _uses_max_completion_tokens, 0.3.2 reasoning_content,
    │                        # 0.4.0 image_url multi-part
    ├── anthropic.py         # 0.4.0 image content block
    ├── deepseek.py          # OpenAI-compat (hérite _uses_max_completion_tokens)
    ├── gemini.py            # 0.4.0 inline_data parts
    └── fake.py              # mock pour tests
```

### Annexe B — Imports publics

```python
from autoagent import (
    # Cœur
    Agent,
    AgentResult,
    Message,
    ModelConfig,
    ProjectWorkspace,
    EvolutionRuntime,
    EVOLUTION_CAPABILITIES,
    DynamicToolBuilder,
    ToolBuildRequest,
    PipelineManager,
    tool,
    create_provider,
    get_logger,
    __version__,

    # Schema partagé
    ImageAttachment,        # 0.4.0
    LLMRequest, LLMResponse,
    ToolCall, ToolSpec,

    # 0.2.0 — post_turn_hook + cancel
    AgentTurnContext,
    PostTurnHook,
    AgentCancelled,

    # 0.5.0 — tracing
    TraceEmitter, TraceEvent, OnEvent,

    # 0.6.0 — memory
    Memory, BufferMemory,

    # 0.8.0 → 0.10.0 — streaming, mémoire résumante, multi-agent, budget, routing
    StreamChunk, StreamEvent,
    SummarizingMemory,
    FactMemory,   # 0.12.0 (§21)
    TokenUsage,
    RoutingProvider,
    Orchestrator, Step, TurnEvent, InterpretOutcome, PhraseSignals,   # 0.9.0

    # 0.11.0 — MCP, OpenTelemetry, checkpoint/resume, politique d'outils
    MCPClient,
    OTelTraceExporter,
    RunState, CheckpointHook,
    ToolPolicy, ToolPolicyContext, ApprovalRequired,

    # Providers (instances directes si besoin)
    AnthropicProvider, DeepSeekProvider, GeminiProvider, OpenAIProvider, LLMProvider,

    # Erreurs
    AutoAgentError, MaxStepsExceeded, ProviderError, ToolError, ToolValidationError,
    TokenBudgetExceeded, MCPError,
)
from autoagent.trace import truncate_preview        # helper public pour previews redactés
```

### Annexe C — Cheat-sheet

```python
# Init rapide
agent = Agent.from_model("openai", "gpt-4o-mini", max_steps=12)

# Tool simple
@agent.tool
def my_tool(x: int) -> dict:
    """Description."""
    return {"result": x * 2}

# Tool avec permissions et schema custom
@agent.tool(
    permissions=["filesystem.write"],
    input_schema={"type": "object", "properties": {...}, "required": [...]}
)
def my_other_tool(...) -> dict: ...

# Workspace borné
workspace = ProjectWorkspace("./src", allowed_write_extensions={".py"})

# Multi-provider (1 ligne)
agent2 = Agent.from_model("anthropic", "claude-sonnet-4-5")
agent3 = Agent.from_model("deepseek", "deepseek-chat")

# Tools dynamiques
builder_p = create_provider(ModelConfig(provider="anthropic", model="claude-sonnet-4-5"))
agent.enable_dynamic_tools(DynamicToolBuilder(builder_p))

# --- 0.4.0 — Image attachment ---
from autoagent import ImageAttachment, Message
img = ImageAttachment(data=base64_payload, mime_type="image/png")
agent.run_messages([
    Message(role="system", content="..."),
    Message(role="user", content="Décris l'image.", attachments=[img]),
])

# --- 0.5.0 — TraceEmitter (JSONL + callback) ---
from autoagent import TraceEmitter
with TraceEmitter(file="run.jsonl", on_event=lambda ev: print(ev.type)) as trace:
    a = Agent(provider, trace=trace)
    a.run("…")

# --- 0.6.0 — BufferMemory ---
from autoagent import BufferMemory
a = Agent(provider, memory=BufferMemory(max_messages=30))
# compact() est appelé une fois avant chaque run_messages()

# --- 0.2.0 — post_turn_hook ---
def verify(ctx):
    if not any(tc.name == "write_file" for tc in ctx.tool_calls):
        return Message(role="user", content="Tu n'as pas sauvegardé.")
    return None
a = Agent(provider, post_turn_hook=verify, max_corrections_per_run=1)

# --- 0.2.0 — cancel_token ---
import threading
from autoagent import AgentCancelled
cancel = threading.Event()
try:
    a.run("…", cancel_token=cancel)
except AgentCancelled:
    print("annulé")

# Run
result = agent.run("Que fais-tu ?")
print(result.output, result.steps)
```

---

## 15. `Orchestrator` — flux déterministe piloté par le host

*(`autoagent/orchestrator.py`, 0.9.0)*

### 15.1 Quand l'utiliser

Pour un flux dont **la machine à états appartient au host** : questionnaire CATI, formulaire
guidé, parcours d'onboarding — là où le LLM ne doit **JAMAIS** faire avancer, sauter ou inventer
une étape. Le LLM est cantonné à deux micro-tâches : (1) **interpréter** la réponse de
l'utilisateur en valeurs (JSON strict), (2) **reformuler** joliment l'étape courante (streamée).
Le host décide tout le reste. (Utilisé par `examples/cati_chat/`.)

Contraste avec `Agent` : `Agent` = boucle où le LLM **choisit** les tools à appeler.
`Orchestrator` = le **host** pilote, le LLM n'interprète/reformule que l'étape courante.

### 15.2 Le contrat (2 callbacks obligatoires)

```python
from autoagent.orchestrator import Orchestrator, Step

orch = Orchestrator(
    provider,                       # un LLMProvider (create_provider(...))
    current_steps=current_steps,    # () -> Sequence[Step] : étape courante en 1er (+ petit horizon)
    record=record,                  # (step_id, value) -> str|None : None=accepté, str=message de rejet
)
```

- `current_steps()` est rappelée après chaque `record`. Elle renvoie l'étape **courante en
  premier**, suivie optionnellement d'un petit horizon d'étapes à venir (que l'interpréteur peut
  remplir depuis une réponse composée). Séquence vide ⇒ **flux terminé**.
- `record(step_id, value)` valide + stocke. Renvoie `None` pour **accepter**, ou une **chaîne
  d'erreur lisible** pour **rejeter** (l'étape reste courante, l'erreur est reformulée à
  l'utilisateur).

Options keyword-only utiles : `describe`, `phrase_context`, `interpret_payload`, `parse_values`,
`interpret_system` / `phrase_system` (prompts), `closing_text`, hooks `on_offtopic` / `on_refused`,
`accept_extra` (autoriser la correction d'un slot déjà répondu), `interpret_temperature=0.0`,
`phrase_temperature=0.6`, et l'état anti-boucle `stuck_slot` / `stuck_count`.

### 15.3 Un tour : `turn(user_text) -> Iterator[TurnEvent]`

```python
for ev in orch.turn(user_message):
    if ev.type == "text":            # morceau de réponse à streamer à l'utilisateur
        send(ev.text)
    elif ev.type == "recorded":      # une valeur validée + stockée (observabilité)
        log(ev.step_id, ev.value)
    elif ev.type == "done":
        if ev.flow_complete:         # plus aucune étape -> flux fini
            finish()
```

`TurnEvent.type ∈ {"text", "recorded", "done"}`. `interpret()` est **failure-safe** : toute sortie
LLM malformée retombe en `unclear` (le flux ne bouge pas). `phrase_stream(step, signals)` streame
la reformulation. Dataclasses exposées : `Step(id, payload)`, `TurnEvent`, `PhraseSignals`,
`InterpretOutcome`.

### 15.4 Anti-boucle + persistance HTTP

`stuck_count` compte les non-réponses consécutives sur la même étape ; à 2+, la reformulation
change de stratégie. Entre deux requêtes HTTP, **persiste puis restaure** `orch.stuck_slot` et
`orch.stuck_count` (sinon ils repartent de zéro à chaque requête).

### 15.5 Exemple complet

```python
from autoagent.orchestrator import Orchestrator, Step

fields = ["name", "age", "city"]
answers: dict[str, object] = {}

def current_steps():
    todo = [f for f in fields if f not in answers]
    return [Step(id=f, payload={"ask": f}) for f in todo[:2]]   # courante + 1 d'horizon

def record(step_id, value):
    answers[step_id] = value
    return None          # None = accepté ; renvoyer une str rejette + fait reformuler l'erreur

orch = Orchestrator(provider, current_steps=current_steps, record=record)
for ev in orch.turn("je m'appelle Ana et j'ai 30 ans"):
    if ev.type == "text":
        print(ev.text, end="")
    elif ev.type == "recorded":
        print(f"\n[enregistré {ev.step_id}={ev.value}]")
```

---

## 16. Nouveautés 0.8.0 → 0.10.0

> Documenté depuis le code (2026-07-06). Huit ajouts majeurs : le **streaming**,
> la **mémoire résumante**, le **multi-agent minimal** (`as_tool`), le **budget de
> tokens**, le **prompt système dynamique**, les **tool calls parallèles**, la
> **sortie structurée native** (`response_format`) et le **routage multi-provider**.

### 16.1 Streaming : `run_stream` / `run_messages_stream` *(0.8.0)*

```python
def run_stream(self, prompt, *, context=None, cancel_token=None, checkpoint=None) -> Iterator[StreamEvent]: ...
def run_messages_stream(self, messages, *, context=None, cancel_token=None, checkpoint=None) -> Iterator[StreamEvent]: ...
# (checkpoint= ajouté en 0.11.0 — voir §19)
```

Contrepartie streaming de `run`/`run_messages` — **itérateurs synchrones** (pas d'async) :

```python
for ev in agent.run_stream("Analyse ce rapport…"):
    if ev.type == "text":        ui.append(ev.text)          # delta de texte
    elif ev.type == "tool_start": ui.show_spinner(ev.tool_name)
    elif ev.type == "tool_end":   ui.done(ev.tool_name, ev.tool_status)  # "ok"|"error"
    elif ev.type == "correction": ui.notice(ev.text)          # post_turn_hook a relancé
    elif ev.type == "done":
        save(ev.messages)         # ⚠️ PERSISTE ça : l'historique complet
        print(ev.output, ev.steps, ev.usage)
    elif ev.type == "error":      ui.fail(ev.error)           # "cancelled" | "max_steps=…" | exception
```

`StreamEvent` (schema.py) :

| champ | type | présent sur |
|---|---|---|
| `type` | `"text" \| "tool_start" \| "tool_end" \| "correction" \| "done" \| "error"` | tous |
| `text` | str | text, correction |
| `tool_name` / `tool_status` | str | tool_start / tool_end |
| `output` / `messages` / `steps` | str / list[Message] / int | done |
| `usage` | `TokenUsage \| None` | done *(0.10.0)* |
| `error` | str | error |
| `state` | `RunState \| None` | error `"approval_required: …"` *(0.11.0)* — snapshot à passer à `resume_stream` (§20) |

**Sémantique d'erreur inversée** : `run_messages` LÈVE (`AgentCancelled`,
`MaxStepsExceeded`…) ; `run_messages_stream` **ne lève jamais** — les échecs deviennent un
événement terminal `error` (un consommateur de stream lit des événements, il ne catch pas).

**Dégradation gracieuse** : un provider sans streaming natif retombe sur le fallback
`LLMProvider.stream()` — la réponse entière arrive comme UN événement `text` puis `done`.
Même code hôte dans les deux cas.

**Interne** *(0.10.0)* : les deux entrées publiques partagent UNE seule boucle `_run_loop`
(avant : deux quasi-jumelles de ~150 lignes à éditer en parallèle). Le tracing est
identique sur les deux chemins (payload `run_start` enrichi de `"streaming": true`).

### 16.2 `SummarizingMemory` — compaction par résumé *(0.10.0)*

```python
SummarizingMemory(provider, *, max_messages=40, keep_recent=12, summary_max_tokens=600)
```

Là où `BufferMemory` **tronque** (les vieux tours disparaissent), `SummarizingMemory`
**replie** les tours au-delà de `max_messages` dans un résumé LLM injecté comme message
système — contexte borné SANS perdre les décisions établies.

- **Incrémental** : chaque compaction ne résume que les tours pas encore couverts
  (fusionnés au résumé précédent) → UN appel LLM par compaction, jamais de re-synthèse
  totale. Le `provider` du résumé peut être un modèle moins cher que celui de l'agent.
- **Réabsorption in-band** : l'hôte qui persiste `result.messages` (le pattern courant)
  repasse le résumé comme message système ; il est détecté par son marqueur
  (`[Résumé de la conversation antérieure]`) et réabsorbé comme graine au lieu d'être
  empilé en double. Historique raccourci sans marqueur = nouvelle conversation → reset.
- **Sécurité d'échec** : résumé LLM qui échoue (réseau, quota) → compaction SAUTÉE ce
  tour-ci (le contexte grossit temporairement) plutôt que troncature silencieuse.
- **`recall(query)`** : recherche LEXICALE (recouvrement de termes, zéro dépendance) dans
  les messages déjà repliés — brancher `agent.register_recall_tool()` permet à l'agent de
  retrouver un détail sorti de sa fenêtre.
- La coupe `keep_recent` est **alignée sur un message `user`** (jamais de `tool` orphelin
  en tête — les providers stricts rejettent).

### 16.3 `agent.as_tool()` — le multi-agent minimal *(0.10.0)*

```python
expert = Agent(cheap_provider, system_prompt="Expert comptage routier…", max_steps=6)
supervisor.add_tool(expert.as_tool(
    name="analyser_comptage",
    description="Délègue les questions de comptage à l'expert.",
))
```

Expose UN agent comme OUTIL d'un autre — hiérarchies superviseur/spécialiste en deux
lignes, sans framework de « crew ». Sémantique précise :

- chaque appel = **conversation neuve** chez le sous-agent (délégation stateless ; donne-lui
  une `memory` s'il doit se souvenir entre les appels) ;
- le sous-agent garde SES provider, outils, `token_budget` et `trace` — **partage un même
  `TraceEmitter`** pour voir tout l'essaim dans un seul arbre de spans ;
- le `context` du parent est **forwardé** au run du sous-agent (les handles hôte restent
  accessibles) ;
- un échec du sous-agent (`MaxStepsExceeded`, `ProviderError`…) remonte comme **tool error**
  au LLM parent — qui peut réagir — jamais comme crash du run parent ;
- le retour porte `output`, `steps` et `tokens` → le parent (et ton transcript) voient le
  **coût de la délégation**.

⚠️ **Thread-safety** : un `Agent` sert UN appelant à la fois. Avec
`parallel_tool_calls=True` côté parent, donne à chaque outil de délégation son PROPRE
sous-agent.

### 16.4 `token_budget` + `TokenUsage` *(0.10.0)*

```python
agent = Agent(provider, token_budget=50_000)      # cap DUR sur le run
try:
    res = agent.run("…")
    print(res.usage.total_tokens)                  # TokenUsage sur AgentResult
except TokenBudgetExceeded as exc:
    print("budget crevé après", exc.spent, "tokens")
```

- Vérifié **entre les tours** : dès que le cumul rapporté par le provider atteint le
  budget → `TokenBudgetExceeded` (ou événement `error` en streaming), event de trace
  `token_budget_exceeded`.
- `TokenUsage(input_tokens, output_tokens, total_tokens)` : `None` quand le provider ne
  rapporte pas (« jamais inventé ») ; `total_tokens` retombe sur la somme si le total
  explicite manque. Présent sur `AgentResult.usage`, l'événement `done`, et les payloads
  de trace `llm_response`.

### 16.5 `system_prompt` dynamique (callable) + `render_system_prompt()` *(0.10.0)*

```python
def prompt_du_jour() -> str:
    return f"Tu es l'assistant. Nous sommes le {date.today():%d/%m/%Y}. Stock: {stock_courant()}."

agent = Agent(provider, system_prompt=prompt_du_jour)   # str OU Callable[[], str]
```

`render_system_prompt()` résout au moment du run : chaîne → telle quelle ; callable →
invoqué sans argument (un callable qui LÈVE est loggé et remplacé par le prompt par
défaut — même contrat de résilience que hook/trace). **Pattern HTTP** : un hôte qui
persiste l'historique entre requêtes doit appeler `render_system_prompt()` à chaque tour
et REMPLACER le message système stocké — le LLM voit toujours l'état frais.

### 16.6 `parallel_tool_calls` *(0.10.0, opt-in)*

```python
agent = Agent(provider, parallel_tool_calls=True)
```

Quand le modèle demande PLUSIEURS outils dans un même tour, ils s'exécutent en
**thread pool** au lieu de séquentiellement — gain direct quand les outils sont I/O-bound
(HTTP, DB). Opt-in car : les handlers doivent être **thread-safe** et ils partagent le même
dict `context`. Les résultats sont réinsérés **dans l'ordre d'appel du modèle** (pas l'ordre
de complétion) → transcript déterministe.

### 16.7 `RoutingProvider` — dispatch multi-provider par requête *(providers/routing.py)*

```python
from autoagent.providers.routing import RoutingProvider

provider = RoutingProvider(
    default=create_provider(ModelConfig(provider="deepseek", model="deepseek-chat")),
    vision=create_provider(ModelConfig(provider="gemini", model="gemini-3.5-flash")),
)
agent = Agent(provider)     # l'Agent ne voit RIEN : contrat LLMProvider standard
```

- **Défaut** : le dernier message user porte une image → route `vision` ; sinon `default`
  ET **strippe les pièces jointes de l'historique** (un provider texte crashe sur les
  `image_url` passés : `unknown variant image_url`).
- **Politique custom** : `router=lambda req: petit if court(req) else gros` — le strip
  s'applique toujours sauf si le provider choisi est le `vision` configuré
  (`strip_attachments_for_default=False` pour désactiver).
- `stream()` est routé aussi (sans cet override, le fallback de la base perdrait le
  streaming NATIF du provider choisi). `self.config` proxifie `default.config` — les hôtes
  qui lisent `agent.provider.config.model` continuent de marcher.

### 16.8 Sortie structurée native : `LLMRequest.response_format` *(0.10.0)*

Le JSON strict est demandé au PROVIDER (capacité native quand elle existe),
plus besoin de parser la prose du modèle :

```python
from autoagent import LLMRequest, Message

resp = provider.complete(LLMRequest(
    messages=[Message(role="user", content="3 villes de France, clés: nom, region. JSON.")],
    response_format={"type": "json_object"},
))
data = json.loads(resp.content)      # fiable — le mode JSON est garanti par l'API
```

Mapping par provider :

| provider | mécanisme |
|---|---|
| OpenAI / DeepSeek / Groq | `response_format` transmis VERBATIM (accepte aussi `{"type": "json_schema", "json_schema": {...}}` strict) |
| Gemini | `generationConfig.responseMimeType = application/json` (PAS `responseSchema` : dialecte OpenAPI divergent) |
| Anthropic | pas de mode natif → consigne système stricte « JSON only, no fences » (best effort — garder un parseur tolérant) |

`DynamicToolBuilder` l'utilise depuis la 0.10 : le builder demande le JSON mode
à la source, ce qui a tué la classe de bugs « fences ```json autour du JSON »
(le parseur tolérant reste en filet). NB : pour un VERDICT (décision typée),
le pattern « verdict = appel d'outil » (§14 / README) reste supérieur au JSON
parsé — le modèle ne peut pas répondre mal formé.

### 16.9 Récap des versions

| Version | Ajouts |
|---|---|
| 0.8.0 | `run_stream` / `run_messages_stream`, `StreamEvent`, `stream()` sur les providers (SSE) + fallback |
| 0.9.0 | `Orchestrator` (§15) |
| 0.10.0 | `_run_loop` unifié, `SummarizingMemory`, `as_tool()`, `token_budget` + `TokenUsage`, `system_prompt` callable, `parallel_tool_calls`, usage sur `done`/`AgentResult` |
| 0.11.0 | `MCPClient` (§17) + `MCPError`, `OTelTraceExporter` (§18), `RunState` + `checkpoint=` + `Agent.resume` (§19), `tool_policy` + `ApprovalRequired` (§20) |
| 0.12.0 | `FactMemory` + `register_remember_tool` (§21) |
| — | `RoutingProvider` (providers/routing.py) |

## 17. `MCPClient` — outils MCP branchés comme des tools locaux

> `autoagent/mcp.py`, zéro dépendance. Transport **stdio uniquement** (le
> serveur MCP est un sous-processus local, JSON-RPC 2.0 ligne par ligne).
> Pas de HTTP/SSE pour l'instant.

```python
from autoagent import Agent, MCPClient

agent = Agent.from_model("gemini", "gemini-3.5-flash", system_prompt="...")

with MCPClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]) as mcp:
    mcp.mount(agent, prefix="fs_")          # chaque tool serveur → tool autoagent
    print(agent.run("Liste les fichiers du projet.").output)
```

**API** :

| membre | rôle |
|---|---|
| `MCPClient(command, *, env=, cwd=, timeout=60.0, client_name=)` | `command` = argv liste (recommandé) ou str ; `env` FUSIONNÉ sur `os.environ` (clé API du serveur) |
| `start()` / `close()` / context manager | lance le process + handshake `initialize` ; `close()` idempotent |
| `list_tools()` | définitions brutes du serveur (pagination `nextCursor` suivie) |
| `call_tool(name, arguments, timeout=)` | 1 appel ; `structuredContent` renvoyé tel quel, sinon `{"text": ...}` |
| `tools(include=, exclude=, prefix=)` | handlers portant `__autoagent_tool_spec__` (schéma = `inputSchema` du serveur) |
| `mount(agent, include=, exclude=, prefix=)` | `add_tool` de chaque handler ; accepte aussi un `ToolRegistry` nu |
| `server_info` / `server_capabilities` / `alive` | état après handshake |

**Sémantique** :
* Les arguments sont validés par le `ToolRegistry` (JSON-Schema du serveur)
  AVANT d'atteindre le serveur — même chemin qu'un `@agent.tool` local.
* Résultat `isError` → `ToolError` → *tool error* pour le LLM (jamais un crash).
* Échec transport/protocole (process mort, timeout, erreur JSON-RPC) → `MCPError`
  (le bout de stderr du serveur est joint au message).
* Thread-safe : corrélation par id → compatible `parallel_tool_calls=True`.
* Pings serveur → répondus ; notifications → ignorées ; requêtes serveur
  (sampling/roots) → refusées proprement (`-32601`).
* Windows : donner le vrai exécutable (`npx.cmd`, pas `npx`) ; pipes forcés UTF-8.
* `include`/`exclude` filtrent sur les noms CÔTÉ SERVEUR (avant `prefix`) —
  monter 3 outils précis vaut mieux que 40 (contexte + surface d'attaque).

## 18. `OTelTraceExporter` — traces vers OpenTelemetry

> `autoagent/otel.py`. Dépendance `opentelemetry-api` **optionnelle** (import
> paresseux à la construction — le cœur reste zéro-dépendance ; sans le paquet,
> `AutoAgentError` explicite).

Callback `on_event` pour `TraceEmitter` qui reconstruit l'arbre de spans de
l'agent en vrais spans OTel (visibles dans Jaeger / Tempo / Langfuse / Phoenix) :

```python
from autoagent import Agent, TraceEmitter, OTelTraceExporter

# le HOST configure OTel comme d'habitude (TracerProvider + OTLP exporter)…
with OTelTraceExporter() as exporter:                 # tracer global par défaut
    trace = TraceEmitter(file="trace.jsonl", on_event=exporter)  # JSONL + OTel
    agent = Agent.from_model("gemini", "gemini-3.5-flash", trace=trace)
    agent.run("...")
```

**Mapping** (calqué sur l'émission réelle de `agent.py`) :

| événement | effet OTel |
|---|---|
| `run_start` / `llm_request` / `tool_call_start` | OUVRE un span (`agent.run`, `llm`, `tool.<nom>`), parenté via `parent_id` |
| `run_end` / `llm_response` / `tool_call_end` | FERME le span visé par son `parent_id` ; statut ERROR si `status` ∈ {error, cancelled, max_steps} |
| tout le reste (`cancelled`, hooks, événements custom du host) | span *event* ponctuel sur le span ouvert le plus proche |
| payload | attributs `autoagent.*` (previews déjà redactées des secrets) |

**Garanties** : un backend OTel cassé ne casse JAMAIS la boucle agent (mêmes
règles que les callbacks de `TraceEmitter`) ; `close()` ferme les spans laissés
ouverts par un run interrompu ; garde anti-fuite à 10 000 spans ouverts ;
partager UN `TraceEmitter` avec les sous-agents `as_tool` = un seul arbre.

## 19. `RunState` — checkpoint / resume (agents longue durée)

> Un run n'est plus prisonnier de son processus : snapshot JSON à chaque
> frontière d'étape, reprise après crash/redémarrage, ou au-delà d'un
> `max_steps` / `token_budget` relevé.

```python
from autoagent import Agent, RunState
import json, pathlib

CHECKPOINT = pathlib.Path("run_state.json")

def save(state: RunState):                       # appelé après CHAQUE étape complétée
    CHECKPOINT.write_text(json.dumps(state.to_dict()), encoding="utf-8")

result = agent.run("Longue mission…", checkpoint=save)

# … crash / redémarrage du process …
state = RunState.from_dict(json.loads(CHECKPOINT.read_text(encoding="utf-8")))
result = agent.resume(state)                     # reprend à state.step + 1
```

**API** :

| membre | rôle |
|---|---|
| `run` / `run_messages` / `run_stream` / `run_messages_stream` (`checkpoint=`) | callback `RunState -> None` appelé à chaque frontière d'étape (résultats d'outils inclus) et après chaque correction du hook |
| `RunState.to_dict()` / `from_dict()` | aller-retour JSON sans perte (s'appuie sur `Message.to_dict` 0.7.0 — tool_calls, attachments, reasoning inclus) |
| `Agent.resume(state, context=, cancel_token=, checkpoint=)` | continue la boucle à `state.step + 1`, compteurs restaurés |
| `Agent.resume_stream(state, ...)` | jumeau streaming (même contrat d'events que `run_messages_stream`) |
| `exc.state` sur `MaxStepsExceeded` / `TokenBudgetExceeded` / `AgentCancelled` | snapshot prêt à reprendre — relever la limite puis `agent.resume(exc.state)` |

**Sémantique** :
* Le comptage CONTINUE : `max_steps` et `token_budget` gardent leur sens
  « pour le run entier » à travers les reprises (relever la limite pour aller plus loin).
* Un callback `checkpoint` qui lève est loggué et IGNORÉ (même contrat que la
  trace : la persistance ne tue pas le run qu'elle protège).
* `memory.compact` est SAUTÉ à la reprise (un snapshot est en plein run ;
  compacter décalerait `turn_start`). La compaction reprend au run suivant.
* Le step final (réponse texte) ne produit pas de checkpoint : le résultat
  EST la persistance (`result.messages`, comme avant).
* Pause volontaire = `cancel_token` + le `.state` de l'`AgentCancelled` —
  c'est la moitié « pause/reprise » d'un approval gate.

## 20. `tool_policy` — politique d'exécution des outils & approval gate

> UNE primitive pour les quatre besoins entreprise : autoriser / refuser /
> demander validation humaine / auditer-quotas. Consultée pour CHAQUE appel
> d'outil, AVANT tout effet de bord du tour.

```python
from autoagent import Agent, ApprovalRequired, ToolPolicyContext

APPROUVES: set[str] = set()          # store d'approbations (fichier/DB en prod)

def politique(ctx: ToolPolicyContext):
    perms = ctx.spec.permissions if ctx.spec else []
    if "filesystem.write" not in perms:
        return None                                    # ALLOW (cas normal)
    if ctx.context.get("user") != "admin":
        return "écriture réservée aux admins"          # DENY motivé → le modèle re-planifie
    if ctx.call.id not in APPROUVES:
        raise ApprovalRequired(f"{ctx.call.name}({ctx.call.arguments})")   # ASK → pause

agent = Agent(provider, tool_policy=politique)

try:
    resultat = agent.run("Nettoie les vieux logs.", context={"user": "admin"})
except ApprovalRequired as exc:
    sauvegarder(exc.state.to_dict())                   # snapshot JSON reprenable
    prevenir_operateur(exc.calls)                      # les appels en attente (rien n'a tourné)
# … l'humain valide (APPROUVES.add(call.id)) — même process ou un autre :
resultat = agent.resume(RunState.from_dict(charger()))
```

**Le contrat, en 6 règles :**

| règle | détail |
|---|---|
| Verdicts | `None` = allow ; `str` = deny motivé (le modèle voit `ToolPolicyDenied: <raison>` en erreur d'outil et re-planifie) ; lever `ApprovalRequired` = pause reprenable |
| Pré-passe sur TOUT le tour | la politique est évaluée pour tous les appels du tour AVANT d'en exécuter un seul — une pause ne tombe JAMAIS après un effet de bord (y compris en `parallel_tool_calls`) |
| Fail-CLOSED | une politique qui plante REFUSE l'appel (c'est une frontière de sécurité — contrat inverse des callbacks trace/checkpoint, qui fail-open) |
| Reprise idempotente | au `resume()`, les appels en attente repassent par la politique : non approuvé → re-pause ; rejeté (`str`) → le modèle voit le refus ; approuvé → exécution UNE seule fois |
| `ctx` | `call` (l'`id` est stable à travers pause/reprise — clé du store d'approbations), `spec` (dont `permissions`), `step`, `messages` (lecture seule), `context` (user, quotas…) |
| Observabilité | événements de trace `tool_policy_deny` et `approval_required` ; `run_end` porte `status="approval_required"` |

**Streaming** : l'`ApprovalRequired` devient un événement terminal
`error` (`"approval_required: …"`) qui porte le snapshot dans `ev.state` →
`agent.resume_stream(ev.state)` après validation.

**Quota / audit** = le même hook, en code hôte : compter dans `ctx.context`,
logger, retourner un `str` quand le quota est dépassé. Pas de primitive
dédiée — c'est du Python.

## 21. `FactMemory` — mémoire factuelle tenue à jour

> `autoagent/memory.py`, 0.12.0. Inspirée du cœur de Mem0
> (extraction + consolidation add/update/delete) SANS la dépendance : LLM
> pas cher + JSON, zéro embedding, zéro service.

Là où `SummarizingMemory` replie les vieux tours en prose (une contradiction
s'EMPILE), `FactMemory` maintient des **faits atomiques à jour** :
« préfère le matin » REMPLACE « préfère le soir ».

```python
from autoagent import Agent, FactMemory

memoire = FactMemory(
    resumeur,                                 # LLM pas cher (extraction)
    path=f"faits/{numero_appelant}.json",     # 1 fichier JSON par identité
    max_messages=40, keep_recent=12,
)
agent = Agent(provider, memory=memoire)
agent.register_recall_tool()      # l'agent LIT sa mémoire (recherche lexicale sur les faits)
agent.register_remember_tool()    # l'agent ÉCRIT volontairement (« notez que… »), tracé
```

**API** :

| membre | rôle |
|---|---|
| `FactMemory(provider, *, path=, max_messages=40, keep_recent=12, max_context_facts=20, max_facts=500)` | mêmes bornes de compaction que SummarizingMemory ; `path` = persistance JSON lisible (audit main, RGPD = supprimer le fichier) |
| `compact(messages)` | replie les vieux tours → extraction LLM (JSON mode) → opérations `add`/`update`/`delete` sur la base ; injecte `[Faits mémorisés]` (les `max_context_facts` plus récents) |
| `recall(query, k)` | recherche lexicale sur les faits (courts et denses — le lexical y marche bien) |
| `remember(fait, subject=)` | ajout DIRECT sans LLM, dédupliqué à l'identique |
| `forget(id)` / `facts()` | suppression ciblée / copie de la base pour audit |
| `Agent.register_remember_tool(name=, description=)` | expose `remember` comme outil ; no-op si la mémoire n'a pas de `.remember` |

**Contrats** : échec d'extraction → compaction SAUTÉE (rien de tronqué en
silence) ; opérations mal formées ignorées (id inconnu, op inconnue, non-JSON,
fences ```json tolérées) ; les faits SURVIVENT aux conversations (un historique
qui raccourcit ne vide pas la base — c'est le but) ; réabsorption du message
`[Faits mémorisés]` in-band quand l'hôte persiste l'historique compacté.

**Ce que ça ne fait PAS** (assumé) : pas de recherche sémantique (« véhicule »
≠ « voiture ») ni de raisonnement temporel à la Zep — pour ça, brancher un
backend lourd via le protocole `Memory` ou un serveur mémoire MCP (§17).

---

*Doc maintenue par l'équipe Alyce R&D. Pour questions, ouvrir une issue sur le repo interne ou taper l'auteur sur Slack.*
