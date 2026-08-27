"""L'agent unique qui fait vivre l'app — il gagne des capacités au fil du temps.

Il y avait trois agents (orchestrateur + Données + HTML reliés par `as_tool`). Le
contrat entre l'orchestrateur et l'agent HTML n'existait QUE dans des prompts, et
c'est de là que venait presque tout ce qui a dysfonctionné. Un seul agent
maintenant, et publier une page est un OUTIL : la signature EST le contrat.

Trois niveaux de capacité, l'humain fait monter d'un niveau :
  1. l'agent écrit un outil    -> BAC À SABLE (jetable, sans réseau)
  2. tu valides son empreinte  -> NATIF (dans le process, accès au contexte hôte)
  3. tu accordes le niveau 3   -> EvolutionRuntime : il écrit le CODE SOURCE d'un
     projet dans `data/projet/` (allowlist d'extensions, anti-traversée, journal
     réversible) et lance LA commande de validation de l'hôte. Il ne peut pas
     EXÉCUTER ce qu'il écrit : lancer du code produit par un modèle reste un geste
     manuel, hors de cette application.

Ce qui est PARTAGÉ entre conversations (donc ce qui s'accumule) : les outils
qu'il s'est écrits, les pages qu'il a publiées, et sa mémoire factuelle.

Provider : Gemini si GEMINI_API_KEY (chargée depuis .env), sinon factice hors-ligne.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RACINE))

from . import capacites  # noqa: E402
from .paths import MANIFESTE, MEMOIRE, OUTILS, PAGES, PROJET, dossier_session  # noqa: E402
from .paths import slug as _slug  # noqa: E402


def _charger_env() -> None:
    """Charge le .env de la racine (os.environ.setdefault : n'écrase rien, rien
    n'est affiché). .env est gitignoré — jamais commité."""
    env = RACINE / ".env"
    if not env.is_file():
        return
    for ligne in env.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        cle, _, val = ligne.partition("=")
        val = val.strip()
        if val[:1] not in ("'", '"'):
            val = re.split(r"\s+#", val, maxsplit=1)[0].rstrip()
        os.environ.setdefault(cle.strip(), val.strip('"').strip("'"))


_charger_env()

from autoagent import (  # noqa: E402
    Agent,
    ApprovalRequired,
    DynamicToolBuilder,
    EvolutionRuntime,
    FactMemory,
    LLMRequest,
    LLMResponse,
    ProjectWorkspace,
    StreamChunk,
    ToolBuildRequest,
    TraceEmitter,
)
from autoagent.approval import ToolManifest, load_tools  # noqa: E402
from autoagent.logging import get_logger  # noqa: E402
from autoagent.providers.base import LLMProvider  # noqa: E402
from autoagent.sandbox import SubprocessSandbox, load_generated_tool  # noqa: E402
from autoagent.schema import ModelConfig  # noqa: E402


def _normaliser_types_schema(node):
    """Rends un JSON Schema tolérant aux types « à la Gemini » (OBJECT, INTEGER…).

    Quand l'orchestrateur Gemini FOURNIT lui-même un input_schema à
    create_python_tool, il écrit les types en MAJUSCULES ; jsonschema (qui valide
    ensuite les appels de l'outil) ne connaît que les types minuscules du standard.
    On les normalise récursivement — angle mort de la lib côté schéma ENTRANT."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            elif k == "type" and isinstance(v, list):
                out[k] = [t.lower() if isinstance(t, str) else t for t in v]
            else:
                out[k] = _normaliser_types_schema(v)
        return out
    if isinstance(node, list):
        return [_normaliser_types_schema(x) for x in node]
    return node


class _BuilderTolerant(DynamicToolBuilder):
    """DynamicToolBuilder qui normalise l'input_schema fourni par le modèle."""

    def build(self, request: ToolBuildRequest):
        if request.input_schema:
            request = ToolBuildRequest(
                capability=request.capability,
                tool_name=request.tool_name,
                input_schema=_normaliser_types_schema(request.input_schema),
                permissions=request.permissions,
            )
        return super().build(request)

# Outils dont l'exécution passe par une VALIDATION HUMAINE (pause reprenable).
# create_python_tool = le méta-outil qui fait ÉCRIRE un nouvel outil à l'agent.
OUTILS_A_APPROUVER = {"create_python_tool"}

_log = get_logger("showcase")


# ── Provider factice (offline) ───────────────────────────────────────────────
class _FakeProvider(LLMProvider):
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self._reponse(request), model="fake")

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        texte = self._reponse(request)
        for mot in texte.split(" "):
            yield StreamChunk(type="text", text=mot + " ")
            time.sleep(0.02)
        yield StreamChunk(type="final", response=LLMResponse(content=texte, model="fake"))

    @staticmethod
    def _reponse(request: LLMRequest) -> str:
        dernier = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        return (f"(mode démo hors-ligne) Reçu : « {dernier.strip()[:100]} ». "
                "Pose GEMINI_API_KEY dans .env pour activer l'agent (mémoire, outils, pages).")


# Le modèle réel utilisé par l'agent (surchargeable par la variable
# d'env AUTOAGENT_MODEL, ex. gemini-2.5-flash, gemini-3-pro…).
PROVIDER_LLM = "gemini"
MODELE_LLM = os.getenv("AUTOAGENT_MODEL", "gemini-3.7-flash")


def mode_provider() -> str:
    return "factice" if (os.getenv("AUTOAGENT_FAKE") == "1" or not os.getenv("GEMINI_API_KEY")) else "reel"


def infos_modele() -> dict:
    """Provider + modèle réellement utilisés (pour l'affichage dans l'UI)."""
    if mode_provider() == "factice":
        return {"provider": "factice", "modele": "démo hors-ligne", "reel": False}
    return {"provider": PROVIDER_LLM, "modele": MODELE_LLM, "reel": True}


@functools.lru_cache(maxsize=4)
def _provider_cache(mode: str, modele: str) -> LLMProvider:
    """Un provider est SANS ÉTAT (HTTP pur) : inutile de le reconstruire à chaque
    requête. Caché par (mode, modèle) pour rester correct si le modèle change."""
    if mode == "factice":
        return _FakeProvider(ModelConfig(provider="fake", model="fake", api_key="x"))
    from autoagent import create_provider
    return create_provider(ModelConfig(provider=PROVIDER_LLM, model=modele,
                                       api_key_env="GEMINI_API_KEY"))


def _provider() -> LLMProvider:
    return _provider_cache(mode_provider(), MODELE_LLM)


# ── Puits de canvas : passe le HTML généré (thread agent) → SSE (boucle async) ─
class CanvasSink:
    """Écrans générés en attente d'envoi. Chaque item porte le HTML (pour le SSE)
    ET le nom du fichier dans le workspace (pour ne persister qu'une référence)."""

    def __init__(self) -> None:
        self._q: list[dict] = []
        self._lock = threading.Lock()

    def push(self, titre: str, html: str, fichier: str = "") -> None:
        with self._lock:
            self._q.append({"titre": titre, "html": html, "fichier": fichier})

    def drain(self) -> list[dict]:
        with self._lock:
            items, self._q = self._q, []
            return items


class TraceSink:
    """Reçoit les TraceEvent de l'agent (thread agent) pour les relayer en SSE
    (boucle async). Même mécanique thread-safe que CanvasSink."""

    def __init__(self) -> None:
        self._q: list[dict] = []
        self._lock = threading.Lock()

    def push(self, event_dict: dict) -> None:
        with self._lock:
            self._q.append(event_dict)

    def drain(self) -> list[dict]:
        with self._lock:
            items, self._q = self._q, []
            return items


# ── Outil RÉEL de l'Agent Données : météo (open-meteo, sans clé) ─────────────
_WMO = {0: "☀️ Ciel clair", 1: "🌤️ Plutôt clair", 2: "⛅ Partiellement nuageux",
        3: "☁️ Couvert", 45: "🌫️ Brouillard", 48: "🌫️ Brouillard givrant",
        51: "🌦️ Bruine légère", 53: "🌦️ Bruine", 55: "🌦️ Bruine dense",
        61: "🌧️ Pluie faible", 63: "🌧️ Pluie", 65: "🌧️ Pluie forte",
        71: "🌨️ Neige faible", 73: "🌨️ Neige", 75: "🌨️ Neige forte",
        80: "🌦️ Averses", 81: "🌦️ Averses", 82: "⛈️ Fortes averses",
        95: "⛈️ Orage", 96: "⛈️ Orage grêle", 99: "⛈️ Violent orage"}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "autoagent-showcase"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


@functools.lru_cache(maxsize=256)
def _geocoder(ville: str) -> tuple[float, float, str] | None:
    """(lat, lon, libellé) d'une ville — CACHÉ : les coordonnées d'une ville ne
    changent pas, alors que la météo, elle, est toujours refetchée."""
    geo = _get_json("https://geocoding-api.open-meteo.com/v1/search?"
                    + urllib.parse.urlencode({"name": ville, "count": 1, "language": "fr"}))
    if not geo.get("results"):
        return None
    lieu = geo["results"][0]
    return (lieu["latitude"], lieu["longitude"],
            f"{lieu['name']} ({lieu.get('country', '')})".strip())


def meteo(ville: str, jours: int = 7) -> dict:
    """Prévisions météo RÉELLES pour une ville (source : open-meteo, sans clé).

    Renvoie les prévisions journalières (date, min, max, condition). Utilise
    ces données telles quelles — ne les invente jamais."""
    jours = max(1, min(int(jours), 16))
    try:
        trouve = _geocoder(ville)
        if trouve is None:
            return {"erreur": f"ville introuvable : {ville}"}
        lat, lon, libelle = trouve
        prev = _get_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "forecast_days": jours, "timezone": "auto"}))
        d = prev["daily"]
        jours_out = [{"date": d["time"][i], "min": d["temperature_2m_min"][i],
                      "max": d["temperature_2m_max"][i],
                      "condition": _WMO.get(d["weather_code"][i], "?")}
                     for i in range(len(d["time"]))]
        return {"ville": libelle, "jours": jours_out}
    except Exception as exc:  # noqa: BLE001 — l'agent voit l'erreur et s'adapte
        return {"erreur": f"{type(exc).__name__}: {exc}"}


# ── Prompts système ──────────────────────────────────────────────────────────
SYSTEME_ORCH = (
    "Tu es un assistant personnel qui GAGNE DES CAPACITÉS au fil du temps. Tu "
    "parles à une seule personne, toujours la même, et tu te souviens d'elle "
    "d'une conversation à l'autre.\n"
    "\n"
    "TES MOYENS, dans l'ordre où il faut y penser :\n"
    "1. Tes outils DÉJÀ acquis. Regarde d'abord ce que tu sais faire — beaucoup "
    "d'outils viennent de conversations passées.\n"
    "2. Ta mémoire : `recall` pour retrouver un fait, `remember` pour noter "
    "durablement ce qui compte (une préférence, une décision, un identifiant), "
    "`forget` si on te demande d'oublier.\n"
    "3. `publier_page(titre, html)` dès qu'un tableau, une fiche, un graphique ou "
    "un formulaire serait plus lisible qu'un paragraphe. La page s'affiche à "
    "l'écran. Écris une page COMPLÈTE et AUTONOME : tout le CSS dans une balise "
    "<style>, le JS en ligne, AUCUN CDN ni librairie distante.\n"
    "4. Si aucun outil ne permet de faire ce qu'on te demande, ÉCRIS-EN UN avec "
    "`create_python_tool`. Une validation humaine est demandée avant chaque "
    "création — c'est normal, attends-la. L'outil créé est CONSERVÉ : il servira "
    "aux prochaines conversations.\n"
    "\n"
    "RÈGLES :\n"
    "- Ne fabrique un outil que si c'est vraiment nécessaire. Réutiliser vaut "
    "toujours mieux que recréer, et un outil de plus est un outil à maintenir.\n"
    "- N'invente JAMAIS une donnée. Ce que tu affiches vient d'un outil.\n"
    "- Réponses courtes dans le chat. Le détail va dans une page publiée.\n"
    "- Si une création d'outil est refusée, termine avec les moyens existants "
    "sans la retenter."
)


SYSTEME_NIVEAU3 = (
    "CAPACITÉ ACCORDÉE — ÉCRIRE DU CODE SOURCE.\n"
    # Le prompt système est reconstruit à CHAQUE run depuis l'état courant ; le
    # transcript, lui, garde les refus d'avant. Il faut donc dire explicitement
    # que l'historique est périmé, sinon le modèle continue d'obéir à un refus
    # qui n'a plus cours (constaté en direct : il refusait sans appeler l'outil).
    "Si un message PLUS ANCIEN de cette conversation dit que cette capacité "
    "n'est pas accordée, il est PÉRIMÉ : elle l'est maintenant. Ne t'y fie pas, "
    "appelle les outils.\n"
    "Tu peux lire et écrire les fichiers d'un projet dans un dossier dédié : "
    "`list_project_files`, `read_project_file`, `write_project_file`, "
    "`replace_project_text`. Tu peux annuler tes propres modifications "
    "(`rollback_last_change`, `list_changes`) — sers-t'en si tu casses quelque "
    "chose.\n"
    "Après avoir écrit du Python, appelle TOUJOURS `run_validation` : elle compile "
    "le projet et te renvoie les erreurs de syntaxe. Corrige jusqu'à ce que ça "
    "passe. Tu ne choisis pas la commande — c'est l'hôte qui la fixe.\n"
    "Tu ne peux PAS exécuter ce que tu écris, et c'est voulu. Quand un service est "
    "prêt, dis à l'utilisateur quel fichier lancer et comment ; c'est lui qui le "
    "démarrera. Écris donc du code autonome, avec un point d'entrée clair et les "
    "instructions de lancement dans un README du projet."
)


def _politique_approbation(ctx) -> str | None:
    """tool_policy : met en PAUSE (ApprovalRequired) toute création d'outil tant
    qu'un humain n'a pas tranché. La décision arrive par le context de resume
    ({"approbations": {call_id: "allow"|"deny"}}), keyé sur l'id STABLE de l'appel.
    Fail-closed : une politique qui plante refuse l'appel."""
    if ctx.call.name in OUTILS_A_APPROUVER:
        decision = (ctx.context or {}).get("approbations", {}).get(ctx.call.id)
        if decision == "allow":
            return None
        if decision == "deny":
            return ("L'utilisateur a REFUSÉ la création de cet outil. "
                    "Termine la demande avec les moyens existants, sans le recréer.")
        cible = ctx.call.arguments.get("tool_name") or ctx.call.arguments.get("capability", "")[:50]
        raise ApprovalRequired(f"création de l'outil « {cible} » soumise à validation humaine")
    return None

# ── Ressources PARTAGÉES entre toutes les conversations ──────────────────────
# C'est le cœur du « de plus en plus puissant » : avant, les outils générés
# vivaient dans le dossier de la SESSION, donc rien ne s'accumulait — chaque
# conversation repartait de zéro. Ici les outils et la mémoire sont communs, et
# un outil forgé aujourd'hui sert la conversation de demain.
# (les chemins vivent dans paths.py — source de vérité unique)
OUTILS_PARTAGES = OUTILS
PAGES_PARTAGEES = PAGES
MEMOIRE_FAITS = MEMOIRE


def construire_agent(session_id: str, canvas: CanvasSink,
                     trace_sink: "TraceSink | None" = None) -> Agent:
    """UN agent, qui gagne des capacités au fil du temps.

    Changement d'architecture (août 2026) : il y avait trois agents — un
    orchestrateur, un agent DONNÉES et un agent HTML reliés par `as_tool`. Le
    contrat entre l'orchestrateur et l'agent HTML n'existait QUE dans des prompts
    (« mets le tableau dans l'écran, pas dans le chat », « appelle appelerAPI avec
    le nom CONCRET de l'outil »), et c'est de là que venait presque tout ce qui a
    dysfonctionné. Or la thèse de la lib est : le bornement est du CODE, pas du
    prompt.

    Ici, publier une page est un OUTIL — `publier_page(titre, html)`. La signature
    EST le contrat : il n'y a plus rien à espérer d'une consigne.

    Trois niveaux de capacité, et c'est toi qui fais monter d'un niveau :
      1. l'agent écrit un outil     → il tourne en BAC À SABLE (jetable, sans réseau)
      2. tu valides son empreinte   → le MÊME outil tourne en NATIF, dans le process
      3. (plus tard) EvolutionRuntime → il écrit le code source d'un vrai service
    """
    prov = _provider()
    OUTILS_PARTAGES.mkdir(parents=True, exist_ok=True)
    PAGES_PARTAGEES.mkdir(parents=True, exist_ok=True)
    MEMOIRE_FAITS.parent.mkdir(parents=True, exist_ok=True)

    # Le dossier de la conversation était créé par effet de bord du
    # ProjectWorkspace, qui pointe désormais sur les pages PARTAGÉES : il faut donc
    # le créer explicitement, sinon TraceEmitter ne peut pas ouvrir son fichier.
    dossier = dossier_session(session_id)
    dossier.mkdir(parents=True, exist_ok=True)
    trace = TraceEmitter(
        file=str(dossier / "trace.jsonl"),
        on_event=(lambda ev: trace_sink.push(ev.to_dict())) if trace_sink else None)

    # Les pages sont écrites sous garde : allowlist .html, anti-traversée, journal
    # réversible. C'est l'hôte qui possède la frontière, pas l'agent.
    pages = ProjectWorkspace(PAGES_PARTAGEES, allowed_write_extensions={".html"})

    # Mémoire factuelle PERSISTANTE : un seul utilisateur, donc une seule identité.
    # Elle survit aux conversations — c'est ce qui fait qu'il te reconnaît.
    memoire = FactMemory(prov, path=str(MEMOIRE_FAITS), max_messages=30, keep_recent=10)

    agent = Agent(
        prov,
        system_prompt=SYSTEME_ORCH,
        temperature=0.3,
        max_steps=14,
        memory=memoire,
        trace=trace,
        tool_policy=_politique_approbation,
        max_dynamic_tools_per_run=3,
        # ── INVARIANTS (des invariants, pas des réglages) ────────────────────
        trifecta_guard="deny",         # rien ne sort si du contenu non fiable est entré
        max_tool_result_chars=6000,    # un outil ne peut pas noyer le contexte
        max_repeated_tool_calls=3,     # il ne peut pas boucler indéfiniment
    )

    # Mémoire : lire, écrire, oublier. L'oubli reste en DRY RUN — effacer les
    # données de quelqu'un sur la seule décision d'un modèle, non.
    agent.register_recall_tool()
    agent.register_remember_tool()
    agent.register_forget_tool()

    # Données réelles. `untrusted=True` : ça vient d'internet, donc c'est traité
    # comme des DONNÉES, jamais comme des instructions — et le run devient teinté.
    agent.tool(meteo, untrusted=True)

    @agent.tool
    def publier_page(titre: str, html: str) -> dict:
        """Publie une page HTML autonome et l'affiche à l'écran.

        Écris une page COMPLÈTE (<style> inline, aucun CDN, aucune librairie
        distante). Utilise-la dès qu'un tableau, une fiche, un graphique ou un
        formulaire serait plus lisible qu'un paragraphe."""
        nom = f"{_slug(titre) or 'page'}.html"
        res = pages.write_file(nom, html, reason=f"page publiée : {titre}")
        canvas.push(titre, html, nom)
        return {"publie": True, "fichier": res.get("path", nom), "octets": len(html)}

    # ── Niveau 1 → 2 : les outils déjà écrits sont rechargés à chaque run.
    # `load_tools` choisit le mode PAR OUTIL selon le manifeste : empreinte
    # validée par toi → NATIF (dans le process, accès au contexte hôte) ; sinon
    # → BAC À SABLE. C'est l'échelle de capacité, appliquée par du code.
    modes = charger_outils_acquis(agent)

    # L'agent peut écrire de NOUVEAUX outils (LLM → validation AST → sandbox),
    # sous validation humaine via tool_policy. Ils atterrissent dans le dossier
    # PARTAGÉ, donc ils survivent à la conversation.
    agent.enable_dynamic_tools(_BuilderTolerant(
        prov, tools_dir=str(OUTILS_PARTAGES), timeout=12))

    # ── NIVEAU 3 : écrire du CODE SOURCE, si l'humain a accordé la capacité ──
    if capacites.niveau3_actif():
        brancher_niveau3(agent)
        # On ne décrit cette capacité au modèle QUE si elle lui est accordée :
        # sinon il proposerait d'écrire du code qu'il ne peut pas écrire.
        agent.system_prompt = f"{SYSTEME_ORCH}\n\n{SYSTEME_NIVEAU3}"
    else:
        brancher_refus_niveau3(agent)

    # Quand la bibliothèque grossit, on n'envoie plus tous les schémas : l'agent
    # cherche ses outils. Sans ça, 40 outils acquis = un préfixe énorme à chaque tour.
    if len(agent.registry.specs()) > 12:
        agent.enable_tool_search(threshold=12)

    agent.derniers_modes = modes  # exposé pour l'UI (qui est natif, qui est en bac à sable)
    return agent


class CapaciteNonAccordee(PermissionError):
    """Une capacité EXISTE mais n'a pas été accordée. Ce n'est pas une panne."""


def brancher_refus_niveau3(agent: Agent) -> None:
    """NIVEAU 3 ÉTEINT — on déclare quand même l'outil, pour qu'il REFUSE.

    Sans ça, l'agent n'a aucun outil d'écriture : il improvise une explication en
    prose, et rien ne distingue « je n'ai pas le droit » de « je ne veux pas ».
    Le refus doit passer par le canal des OUTILS — daté dans la trace, avec un
    motif, et sur le canal que le modèle sait déjà lire. C'est la thèse de la
    lib appliquée à une capacité absente : un refus est du code, pas de la prose.

    ATTENTION, leçon apprise à la dure : ce message reste dans le transcript. Une
    première version disait « propose-lui d'accorder le niveau 3 » et « n'essaie
    pas de contourner » — des ORDRES. Une fois la capacité accordée, le modèle
    relisait son propre passé, y retrouvait la consigne, et continuait de refuser
    SANS MÊME APPELER l'outil réel. Un message d'outil doit donc énoncer un FAIT
    daté, jamais une règle de conduite durable.
    """
    @agent.tool(name="write_project_file",
                description="Écrit un fichier de code source dans le projet "
                            "(nécessite la capacité « écrire du code source »).")
    def _refus_ecriture(path: str, content: str = "", reason: str = "") -> dict:
        raise CapaciteNonAccordee(
            f"À CET INSTANT, la capacité « écrire du code source » n'est pas "
            f"accordée : « {path} » n'a pas été écrit. Cet état peut changer — si "
            f"l'utilisateur l'accorde, cet outil fonctionnera et tu devras "
            f"RÉESSAYER au lieu de te fier à ce message, qui ne vaut que pour "
            f"l'appel qui vient d'échouer.")


def brancher_niveau3(agent: Agent) -> None:
    """NIVEAU 3 — l'agent écrit le CODE SOURCE d'un projet, sous garde.

    Ce que ça lui ouvre :
      * lire / écrire / remplacer des fichiers dans UN dossier (`data/projet/`),
        avec allowlist d'extensions, anti-traversée et journal réversible ;
      * annuler ses propres modifications (`rollback_last_change`) ;
      * lancer LA commande de validation — pas une commande de son choix.

    Ce que ça ne lui ouvre PAS, volontairement :
      * `host_call` (le pont vers des fonctions de l'hôte) : surface inutile ici ;
      * EXÉCUTER le service qu'il écrit. `allow_custom_validation_command` reste
        False, donc `run_validation` ne peut lancer que `compileall` — qui COMPILE
        sans exécuter le code. Démarrer un service écrit par un modèle reste un
        geste manuel, hors de cette application. C'est le seul endroit où j'ai
        refusé d'automatiser quelque chose.
    """
    PROJET.mkdir(parents=True, exist_ok=True)
    runtime = EvolutionRuntime(
        PROJET,
        # La commande est FIXÉE par l'hôte. `compileall` prouve que le code parse
        # (et compile) sans jamais l'exécuter — l'agent a donc une boucle de
        # retour pour corriger ses erreurs de syntaxe, sans pouvoir rien lancer.
        validation_command=[sys.executable, "-m", "compileall", "-q", str(PROJET)],
        allow_custom_validation_command=False,
        allowed_write_extensions={".py", ".html", ".css", ".js", ".json", ".md",
                                  ".txt", ".toml", ".cfg", ".ini"},
    )
    runtime.register_tools(agent, capabilities={"read", "write", "validate"})


def charger_outils_acquis(agent: Agent) -> list[tuple[str, str]]:
    """Recharge les outils que l'agent s'est écrits lors des conversations passées.

    Renvoie [(nom, mode)] où mode vaut "native" (empreinte validée par l'humain),
    "sandbox" (pas encore validé) ou "invalid" (ne passe plus la validation :
    ignoré, jamais enregistré).
    """
    if not OUTILS_PARTAGES.is_dir():
        return []
    try:
        manifeste = ToolManifest.load(MANIFESTE)
        return load_tools(agent, OUTILS_PARTAGES, manifeste,
                          sandbox=SubprocessSandbox(timeout=12))
    except Exception:  # noqa: BLE001 — un outil cassé ne doit pas tuer le chat
        _log.exception("chargement des outils acquis impossible")
        return []


# ── API vivante : outils exécutables en direct par le front ──────────────────
# Charger un outil généré construit un sandbox et relit/valide le fichier. Un
# dashboard qui rafraîchit toutes les 2 s le refaisait à CHAQUE tick — on cache
# l'objet chargé, invalidé par le mtime du fichier (un outil régénéré est rechargé).
_CACHE_OUTILS: dict[tuple[str, str], tuple[float, object]] = {}
_CACHE_LOCK = threading.Lock()


def _charger_outil(chemin: Path):
    cle = (str(chemin), "")
    mtime = chemin.stat().st_mtime
    with _CACHE_LOCK:
        cache = _CACHE_OUTILS.get(cle)
        if cache and cache[0] == mtime:
            return cache[1]
    outil = load_generated_tool(chemin, sandbox=SubprocessSandbox(timeout=12))
    with _CACHE_LOCK:
        _CACHE_OUTILS[cle] = (mtime, outil)
    return outil


def executer_outil_live(session_id: str, tool: str, args: dict | None = None) -> dict:
    """Exécute EN DIRECT un outil de la session — c'est l'« API vivante » que le
    front appelle (via window.appelerAPI). Sert le `meteo` intégré ou un outil créé
    dynamiquement (chargé depuis tools_dir, exécuté en SANDBOX). Read-only par nature."""
    args = args or {}
    if tool == "meteo":
        return meteo(**{k: v for k, v in args.items() if k in ("ville", "jours")})
    tools_dir = OUTILS_PARTAGES   # outils partagés entre conversations
    f = tools_dir / f"{_slug(tool).replace('-', '_')}.py"
    if not f.is_file() and tools_dir.is_dir():
        for cand in tools_dir.glob("*.py"):          # repli : match par nom de spec
            try:
                if _charger_outil(cand).spec.name == tool:
                    f = cand
                    break
            except Exception:  # noqa: BLE001
                continue
    if not f.is_file():
        return {"erreur": f"outil « {tool} » inconnu pour cette session"}
    try:
        res = _charger_outil(f)(**args)
        return res if isinstance(res, dict) else {"resultat": res}
    except Exception as exc:  # noqa: BLE001
        return {"erreur": f"{type(exc).__name__}: {exc}"}


def info_approbation_en_attente(state) -> dict | None:
    """Depuis un RunState en pause, décrit l'appel d'outil à valider (pour l'UI)."""
    for m in reversed(state.messages):
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                if tc.name in OUTILS_A_APPROUVER:
                    a = tc.arguments or {}
                    return {"call_id": tc.id, "outil": tc.name,
                            "tool_name": a.get("tool_name") or "(nom automatique)",
                            "capability": a.get("capability", "")}
            break
    return None
