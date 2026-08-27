"""API de la démo « app vivante » — FastAPI + streaming SSE.

P0 : chat en streaming (Server-Sent Events) + historique persisté par session.
La boucle d'agent (autoagent) est SYNCHRONE ; on l'itère dans un threadpool pour
ne pas bloquer la boucle asyncio (``iterate_in_threadpool``).

Lancer :  uvicorn showcase.backend.main:app --reload   (depuis la racine du dépôt)
Sans GEMINI_API_KEY → mode démo hors-ligne (provider factice), tout reste testable.
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

# autoagent vit à la racine du dépôt (démo non installée) : garantir le path
# AVANT d'importer autoagent, quel que soit le CWD d'où uvicorn est lancé.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from autoagent import Message, RunState

from . import capacites, sessions
from .agent_factory import (
    SYSTEME_ORCH,
    CanvasSink,
    TraceSink,
    construire_agent,
    executer_outil_live,
    info_approbation_en_attente,
    infos_modele,
    mode_provider,
)

app = FastAPI(title="autoagent — app vivante", version="0.1.0")
# Le front généré tourne dans un iframe SANDBOXÉ (origine « null ») : pour qu'il
# puisse appeler l'API vivante (window.appelerAPI → /api/live), CORS ouvert.
# Démo locale ; l'API vivante est read-only et bornée aux outils de la session.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def sans_cache(request, call_next):
    """Aucune reponse d'API ne doit etre mise en cache.

    Sans en-tete, un `fetch()` peut servir une reponse perimee alors qu'un F5
    revalide : le panneau des capacites restait donc vide jusqu'au rechargement,
    alors que l'outil venait d'etre cree. Le SSE a ses propres en-tetes, on n'y
    touche pas.
    """
    reponse = await call_next(request)
    if request.url.path.startswith("/api/") and "text/event-stream" not in             reponse.headers.get("content-type", ""):
        reponse.headers["Cache-Control"] = "no-store, must-revalidate"
    return reponse

FRONT = Path(__file__).resolve().parent.parent / "frontend"


class MessageIn(BaseModel):
    session_id: str | None = None
    message: str


class OutilIn(BaseModel):
    fichier: str


class Niveau3In(BaseModel):
    actif: bool


class ApproveIn(BaseModel):
    session_id: str
    call_id: str
    decision: str  # "allow" | "deny"


def _sse(event: str, data: dict) -> str:
    """Encode un événement SSE (une ligne event: + une ligne data:)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _fermer_trace(orch) -> None:
    """Ferme le TraceEmitter de l'agent → libère trace.jsonl (Windows : un fichier
    ouvert bloque la suppression du workspace)."""
    try:
        if getattr(orch, "trace", None) is not None:
            orch.trace.close()
    except Exception:  # noqa: BLE001
        pass


async def _relais(session_id: str, gen, canvas: CanvasSink, trace: TraceSink):
    """Relaie un flux de StreamEvent (run OU resume) en SSE, gère l'écran, la
    trace vivante, la galerie de pages et la persistance ; transforme
    un ApprovalRequired en événement ``approval`` (le point de pause du gate).

    Les écrans générés sont accumulés en mémoire et persistés en UNE écriture à la
    fin du run (le disque ne bloque plus la boucle asyncio : tout passe par
    ``run_in_threadpool``).
    """
    ecrans: list[dict] = []          # métadonnées des pages produites par ce run
    outil_cree = False               # ce run a-t-il coûté un nouvel outil ?

    def _rendus() -> str:
        s = ""
        for item in canvas.drain():
            ecrans.append(item)
            s += _sse("render", item)
        return s

    def _traces() -> str:
        return "".join(_sse("trace", te) for te in trace.drain())

    async for ev in iterate_in_threadpool(gen):
        if (t := _traces()):          # trace accumulée depuis le dernier event
            yield t
        if ev.type == "text":
            yield _sse("text", {"delta": ev.text})
        elif ev.type == "tool_start":
            yield _sse("tool", {"phase": "start", "name": ev.tool_name})
        elif ev.type == "tool_end":
            if ev.tool_name == "create_python_tool" and ev.tool_status == "ok":
                outil_cree = True
            yield _sse("tool", {"phase": "end", "name": ev.tool_name, "status": ev.tool_status})
            if (r := _rendus()):
                yield r
        elif ev.type == "done":
            if (r := _rendus()):
                yield r
            await run_in_threadpool(sessions.sauver, session_id, ev.messages,
                                    canvas=ecrans[-1] if ecrans else None,
                                    pages_ajout=ecrans, pending=None)
            if (t := _traces()):
                yield t
            # La mesure de montée en puissance : cette demande a-t-elle exigé un
            # nouvel outil ? Comptée seulement sur un run ABOUTI — un run en
            # attente d'approbation ou en erreur n'est pas une demande traitée.
            await run_in_threadpool(capacites.enregistrer_demande, outil_cree=outil_cree)
            yield _sse("done", {"output": ev.output, "steps": ev.steps,
                                "outil_cree": outil_cree,
                                "mesure": await run_in_threadpool(capacites.mesure)})
        elif ev.type == "error":
            if (r := _rendus()):
                yield r
            if (t := _traces()):
                yield t
            if ev.error.startswith("approval_required") and ev.state is not None:
                await run_in_threadpool(sessions.sauver_pending, session_id, ev.state.to_dict())
                info = info_approbation_en_attente(ev.state) or {"call_id": "?", "outil": "?"}
                yield _sse("approval", info)
            else:
                if ev.messages:
                    await run_in_threadpool(sessions.sauver, session_id, ev.messages,
                                            canvas=ecrans[-1] if ecrans else None,
                                            pages_ajout=ecrans)
                yield _sse("error", {"error": ev.error})


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "provider": mode_provider(), **infos_modele()}


@app.get("/api/live/{session_id}/{tool}")
async def live(session_id: str, tool: str, args: str = "{}") -> dict:
    """API VIVANTE : exécute un outil de la session en direct et renvoie du JSON.
    C'est ce que la page générée appelle via window.appelerAPI(outil, args).

    Enveloppe STABLE pour que le front ait toujours le même contrat :
      succès → {"ok": true,  "resultat": <ce que renvoie l'outil>}
      échec  → {"ok": false, "erreur": "<message>"}
    Le pont appelerAPI déballe `resultat` et lève une exception sur erreur.
    """
    try:
        a = json.loads(args) if args else {}
    except json.JSONDecodeError:
        a = {}
    # Exécution BLOQUANTE (réseau + sous-processus sandbox) → hors event loop.
    res = await run_in_threadpool(executer_outil_live, session_id, tool,
                                 a if isinstance(a, dict) else {})
    if isinstance(res, dict) and set(res.keys()) == {"erreur"}:
        return {"ok": False, "erreur": res["erreur"]}
    return {"ok": True, "resultat": res}


@app.get("/api/capacites")
async def get_capacites() -> dict:
    """Ce que l'agent a acquis + la preuve chiffrée qu'il monte en puissance."""
    return {
        "outils": await run_in_threadpool(capacites.inventaire),
        "mesure": await run_in_threadpool(capacites.mesure),
    }


@app.post("/api/capacites/promouvoir")
async def post_promouvoir(entree: OutilIn) -> dict:
    """Niveau 1 -> 2 : l'outil validé tournera en NATIF (dans le process).

    C'est le seul endroit de l'app où une capacité s'élargit, et il exige ce geste
    humain explicite. L'empreinte du code est validée telle quelle : si l'agent
    réécrit l'outil, il retombe en bac à sable tout seul.
    """
    return await run_in_threadpool(capacites.promouvoir, entree.fichier)


@app.post("/api/capacites/retrograder")
async def post_retrograder(entree: OutilIn) -> dict:
    """Niveau 2 -> 1 : l'outil repasse en bac à sable, sans être supprimé."""
    return await run_in_threadpool(capacites.retrograder, entree.fichier)


@app.post("/api/capacites/niveau3")
async def post_niveau3(entree: Niveau3In) -> dict:
    """Niveau 3 : accorde (ou retire) le droit d'écrire du CODE SOURCE.

    Ouvre lire/écrire dans `data/projet/` (allowlist d'extensions, anti-traversée,
    journal réversible) et LA commande de validation de l'hôte — jamais une
    commande choisie par l'agent. N'ouvre PAS l'exécution du service écrit :
    lancer du code produit par un modèle reste un geste manuel, hors de l'app.
    """
    return await run_in_threadpool(capacites.activer_niveau3, entree.actif)


@app.get("/api/sessions")
async def get_sessions() -> list[dict]:
    return await run_in_threadpool(sessions.lister)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """Chat + MÉTADONNÉES des pages générées. Le HTML n'est PAS renvoyé ici : le
    front le charge à la demande via /api/pages (une seule page à la fois)."""
    msgs = await run_in_threadpool(sessions.charger, session_id)
    if not msgs:
        raise HTTPException(404, "session inconnue")
    # On n'expose pas le message system (bruit pour l'UI).
    visibles = [{"role": m.role, "content": m.content}
                for m in msgs if m.role in ("user", "assistant") and m.content]
    # Une pause d'approbation DOIT survivre à un rechargement : sans ça, la carte
    # Autoriser/Refuser disparaît alors que le run, lui, attend toujours — et la
    # conversation est bloquée sans que rien ne le dise.
    pending = None
    if (brut := await run_in_threadpool(sessions.charger_pending, session_id)):
        try:
            pending = info_approbation_en_attente(RunState.from_dict(brut))
        except Exception:  # noqa: BLE001 — un état illisible ne doit pas casser l'ouverture
            pending = None
    return {"id": session_id, "messages": visibles, "pending": pending,
            "pages": await run_in_threadpool(sessions.charger_pages, session_id)}


@app.get("/api/pages/{session_id}/{fichier}")
async def get_page(session_id: str, fichier: str) -> dict:
    """HTML d'une page générée, relu depuis le workspace de la session."""
    html = await run_in_threadpool(sessions.lire_page, session_id, fichier)
    if html is None:
        raise HTTPException(404, "page introuvable")
    return {"fichier": fichier, "html": html}


@app.post("/api/reset")
async def reset() -> dict:
    """Repart d'un projet VIERGE : efface tout ce que l'agent a produit.

    Aucun paramètre — les cibles viennent de `paths.py`. Un client ne peut donc
    pas orienter la suppression, et il n'y a rien à valider côté entrée.
    """
    return {"supprime": await run_in_threadpool(sessions.remise_a_zero)}


@app.delete("/api/sessions/{session_id}")
async def del_session(session_id: str) -> dict:
    return {"supprime": await run_in_threadpool(sessions.supprimer, session_id)}


_ENTETES_SSE = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _reponse_sse(session_id: str, orch, demarrer, canvas: CanvasSink, trace: TraceSink,
                 prologue: dict | None = None) -> StreamingResponse:
    """Emballe un run/resume en réponse SSE : prologue optionnel, relais des
    événements, fermeture de la trace garantie. Partagé par /chat et /approve.

    ``demarrer`` est un callable sans argument qui rend l'itérateur de
    StreamEvent (``run_messages_stream`` ou ``resume_stream``) — appelé à
    l'intérieur du générateur pour que rien ne s'exécute avant le streaming.
    """
    async def flux():
        if prologue:
            yield _sse("session", prologue)
        try:
            async for chunk in _relais(session_id, demarrer(), canvas, trace):
                yield chunk
        finally:
            _fermer_trace(orch)  # libère trace.jsonl (sinon rmtree bloque sous Windows)

    return StreamingResponse(flux(), media_type="text/event-stream", headers=_ENTETES_SSE)


@app.post("/api/chat")
async def chat(entree: MessageIn):
    """Envoie un message, streame la réponse en SSE, persiste à la fin.

    Les événements SSE : ``session`` (id), ``text`` (deltas), ``tool``
    (délégations de l'orchestrateur), ``render`` (VRAI HTML poussé dans l'écran
    central par l'agent HTML), ``trace``, ``approval``, ``done`` / ``error``.
    """
    session_id = entree.session_id or secrets.token_hex(8)
    historique = await run_in_threadpool(sessions.charger, session_id)
    if not historique:
        historique = [Message(role="system", content=SYSTEME_ORCH)]
    historique.append(Message(role="user", content=entree.message))

    canvas, trace = CanvasSink(), TraceSink()
    orch = construire_agent(session_id, canvas, trace)
    return _reponse_sse(
        session_id, orch,
        lambda: orch.run_messages_stream(historique, context={"approbations": {}}),
        canvas, trace, prologue={"session_id": session_id})


@app.post("/api/approve")
async def approve(entree: ApproveIn):
    """Reprend un run mis en pause, avec la décision humaine (allow/deny)."""
    pend = await run_in_threadpool(sessions.charger_pending, entree.session_id)
    if not pend:
        raise HTTPException(404, "aucune approbation en attente pour cette session")
    state = RunState.from_dict(pend)
    canvas, trace = CanvasSink(), TraceSink()
    orch = construire_agent(entree.session_id, canvas, trace)
    contexte = {"approbations": {entree.call_id: entree.decision}}
    return _reponse_sse(entree.session_id, orch,
                        lambda: orch.resume_stream(state, context=contexte), canvas, trace)


# ── Front statique (une page autonome) ───────────────────────────────────────
@app.get("/")
def index() -> FileResponse:
    # no-store : en démo on retouche le front en direct ; un cache navigateur
    # qui garde l'ancienne version fait perdre un quart d'heure à chercher un
    # bug déjà corrigé.
    return FileResponse(FRONT / "index.html",
                        headers={"Cache-Control": "no-store, must-revalidate"})


if (FRONT / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONT / "assets"), name="assets")
