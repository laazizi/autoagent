"""L'essaim qui fait vivre l'app : Orchestrateur + Agent Données + Agent HTML.

Architecture demandée (pattern multi-agent autoagent, `as_tool`) :

    orchestrateur ──► obtenir_donnees   (Agent Données : vraies API/outils)
          │      └──► construire_ecran  (Agent HTML : génère du VRAI HTML,
          │                              l'écrit dans un workspace borné et
          │                              l'affiche dans l'écran de droite)
          └── le CHAT ne sert qu'à de courtes phrases ; TOUT affichage
              (tableau, carte, fiche, formulaire, graphe) devient du HTML.

Bounding is code : le HTML généré est (1) écrit dans un ProjectWorkspace
restreint aux .html (audit + réversible) et (2) rendu côté navigateur dans un
iframe SANDBOXÉ (isolé du reste de la page).

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

from .paths import dossier_session  # noqa: E402
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
    LLMRequest,
    LLMResponse,
    ProjectWorkspace,
    StreamChunk,
    ToolBuildRequest,
    TraceEmitter,
)
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
                "Pose GEMINI_API_KEY dans .env pour activer l'essaim (données + HTML généré).")


# Le modèle réel utilisé par tout l'essaim (surchargeable par la variable
# d'env AUTOAGENT_MODEL, ex. gemini-2.5-flash, gemini-3-pro…).
PROVIDER_LLM = "gemini"
MODELE_LLM = os.getenv("AUTOAGENT_MODEL", "gemini-3.6-flash")


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
    """Reçoit les TraceEvent de l'essaim (thread agent) pour les relayer en SSE
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
    "Tu es l'ORCHESTRATEUR d'une application web vivante. Règle d'or : le CHAT "
    "(tes réponses en texte) ne sert QU'À de courtes phrases conversationnelles "
    "(1 à 2 phrases). Tu ne mets JAMAIS de tableau, de longue liste, de carte ou "
    "de mise en page dans le chat.\n"
    "Pour TOUT affichage riche (tableau, carte, fiche, formulaire, graphe, "
    "dashboard…), procède ainsi :\n"
    "  1. si des données réelles sont nécessaires, appelle obtenir_donnees ;\n"
    "  2. puis appelle construire_ecran en décrivant précisément l'écran voulu "
    "ET en lui passant les données (JSON) à afficher.\n"
    "construire_ecran rend du VRAI HTML dans l'écran de droite. Après ça, réponds "
    "juste une phrase courte, p.ex. « Voilà, c'est affiché à droite. » N'invente "
    "jamais de données : celles qui s'affichent viennent d'obtenir_donnees.\n"
    "Si AUCUN outil existant ne permet de traiter une demande (un calcul précis, "
    "une transformation de texte, une conversion…), tu peux FABRIQUER un petit "
    "outil Python via create_python_tool, PUIS l'appeler. Une validation humaine "
    "sera demandée avant toute création — c'est normal. Fais-le avec parcimonie, "
    "seulement quand c'est vraiment nécessaire.\n"
    "CAS SPÉCIAL — une API qui ALIMENTE le front (vue vivante / rechargeable / "
    "tableau de bord qui s'actualise) : ne fige PAS les données dans la page. "
    "Demande à construire_ecran une page qui appelle en JavaScript "
    "`window.appelerAPI('<outil>', {<arguments>})` — ça renvoie du JSON FRAIS du "
    "serveur — au chargement ET sur un bouton « Actualiser ».\n"
    "  IMPORTANT : dans appelerAPI, le <outil> doit être le nom CONCRET d'un outil "
    "de DONNÉES appelable en direct. Ceux-ci sont : « meteo » (arguments : ville, "
    "jours) et TOUT outil que TU crées via create_python_tool. N'utilise JAMAIS "
    "« obtenir_donnees » ni « construire_ecran » dans appelerAPI : ce sont tes "
    "délégations internes, PAS des API. Si la donnée voulue n'a pas d'outil concret, "
    "crée-le d'abord (create_python_tool), puis passe SON nom à la page.\n"
    "  CONTRAT DE SORTIE : quand tu wires un outil au front, DÉCRIS à construire_ecran "
    "la FORME EXACTE de ce que l'outil renvoie (un mini-exemple JSON), et les "
    "arguments attendus — sinon la page ne saura pas parser la réponse. Pour un "
    "outil que tu crées, impose-lui une sortie JSON simple et documente-la à la page.")


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

_SYS_DONNEES = (
    "Tu es l'agent DONNÉES. Tu réponds uniquement avec des données vérifiées par "
    "tes outils (jamais inventées). Renvoie les données brutes, structurées et "
    "complètes, prêtes à être affichées. Pas de mise en forme HTML ici.")

_SYS_HTML = (
    "Tu es l'agent HTML/UI. Tu génères une page HTML COMPLÈTE et AUTONOME, design "
    "moderne, sombre, épuré, responsive, agréable. RÈGLE STRICTE : ZÉRO dépendance "
    "externe — INTERDIT d'utiliser <script src=...> ou <link href=...> vers un CDN "
    "(pas de Tailwind CDN, pas de Google Fonts, pas de librairie distante). Écris "
    "TOUT le CSS toi-même dans une balise <style> et le JS dans <script> inline. "
    "Les images/cartes distantes sont tolérées mais prévois un repli si elles ne "
    "chargent pas. Tu n'écris PAS le HTML en texte de réponse : tu appelles "
    "afficher_ecran(titre, html) avec le HTML final. Utilise fidèlement les données "
    "fournies, sans en inventer.\n"
    "API VIVANTE : un pont JS est TOUJOURS disponible — `const r = await "
    "window.appelerAPI(nom_outil, args)`. CONTRAT : il renvoie DIRECTEMENT le "
    "résultat de l'outil (l'objet/typé que l'outil produit, PAS enveloppé) ; en cas "
    "d'erreur serveur il LÈVE une exception dont le message est la vraie cause — "
    "entoure donc l'appel de try/catch et affiche `e.message`. Ne suppose PAS de clé "
    "'result'/'resultat' : utilise directement ce que renvoie appelerAPI, selon la "
    "forme que l'orchestrateur t'aura décrite. Quand on te demande une vue vivante, "
    "n'écris PAS les données en dur : appelle appelerAPI au chargement ET sur le "
    "bouton « Actualiser », avec un état « chargement… » et un message d'erreur clair. "
    "Ne mets pas de <form> qui se soumet (pas de navigation) : gère tout en JS "
    "(addEventListener), preventDefault si tu utilises un formulaire.")


def construire_orchestrateur(session_id: str, canvas: CanvasSink,
                             trace_sink: "TraceSink | None" = None) -> Agent:
    """Assemble l'essaim pour UNE requête et renvoie l'orchestrateur (point d'entrée).

    La MÊME TraceEmitter est passée aux 3 agents → tout l'arbre de l'essaim
    (orchestrateur → sous-agents → outils) dans une seule trace, relayée en direct
    à l'UI via ``trace_sink`` et archivée en JSONL (audit)."""
    prov = _provider()
    dossier = dossier_session(session_id)
    ws = ProjectWorkspace(dossier, allowed_write_extensions={".html"})
    trace = TraceEmitter(
        file=str(dossier / "trace.jsonl"),
        on_event=(lambda ev: trace_sink.push(ev.to_dict())) if trace_sink else None)

    # Agent DONNÉES — ses propres outils (as_tool ne partage pas ceux du parent).
    agent_donnees = Agent(prov, system_prompt=_SYS_DONNEES, temperature=0.0, max_steps=6, trace=trace)
    agent_donnees.tool(meteo)

    # Agent HTML — l'outil d'affichage borné (workspace + canvas).
    agent_html = Agent(prov, system_prompt=_SYS_HTML, temperature=0.4, max_steps=4, trace=trace)

    @agent_html.tool
    def afficher_ecran(titre: str, html: str) -> dict:
        """Affiche une page HTML dans l'écran de droite (et l'archive, sous garde)."""
        nom = f"{_slug(titre) or 'ecran'}.html"
        res = ws.write_file(nom, html, reason=f"écran généré : {titre}")
        canvas.push(titre, html, nom)   # le HTML part en SSE, le NOM sert à la persistance
        return {"affiche": True, "fichier": res.get("path", nom), "octets": len(html)}

    # ORCHESTRATEUR — délègue via as_tool + peut forger ses propres outils (sous
    # validation humaine grâce à tool_policy). max_steps large : création d'outil
    # + appel + affichage tiennent dans un seul run.
    orch = Agent(prov, system_prompt=SYSTEME_ORCH, temperature=0.3, max_steps=14,
                 tool_policy=_politique_approbation, max_dynamic_tools_per_run=3, trace=trace)
    orch.add_tool(agent_donnees.as_tool(
        name="obtenir_donnees",
        description="Récupère des données RÉELLES (ex. météo d'une ville) via des outils.",
        request_description="Décris les données voulues en langage naturel (ville, période…)."))
    orch.add_tool(agent_html.as_tool(
        name="construire_ecran",
        description="Génère et AFFICHE du vrai HTML dans l'écran de droite.",
        request_description="Décris précisément l'écran à construire ET fournis les données (JSON) à afficher."))
    # Outils dynamiques : l'agent ÉCRIT un outil (LLM → validation AST → sandbox).
    # Chaque appel de l'outil généré s'exécute DANS le sandbox (isolé).
    orch.enable_dynamic_tools(_BuilderTolerant(
        prov, tools_dir=str(dossier / "tools"), timeout=12))
    return orch


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
    tools_dir = dossier_session(session_id) / "tools"
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
