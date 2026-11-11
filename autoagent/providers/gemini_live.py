"""Gemini Live : la voix en continu, sous la boucle d'agent habituelle.

Une conversation vocale n'est pas une suite de requêtes : l'audio du micro part en continu, le modèle répond
quand il a fini d'écouter, et il peut demander un outil en plein milieu. La bibliothèque est synchrone et sans
dépendance : ce fournisseur garde une connexion WebSocket (`autoagent.ws`) ouverte, un fil lit le serveur, et la
boucle d'agent le voit comme n'importe quel autre fournisseur — `complete(request)` rend UNE réponse (texte
et/ou appels d'outils). Conséquence voulue : **chaque appel d'outil demandé par la voix traverse la même boucle**
que ceux d'un fournisseur texte (politique, gardes, journal, trace, bornes). Aucun second chemin d'exécution :
c'est ce que `tests/test_gate.py` garde, et pourquoi Live est un fournisseur de plus et non un module à part.

Usage :

    agent = Agent.from_model("gemini_live", "gemini-3.8-live", system_prompt="Tu es un petit robot.")
    agent.provider.on_audio = haut_parleur.write            # PCM 16 bits mono 24 kHz, au fil de l'eau
    agent.provider.on_interrupted = haut_parleur.vider      # l'utilisateur coupe la parole : vider le tampon
    agent.provider.open(agent.render_system_prompt(), agent.registry.specs())   # connecter AVANT d'écouter
    # un fil du micro appelle agent.provider.send_audio(pcm_16k)  (16 bits mono 16 kHz)
    while conversation:
        resultat = agent.run(GeminiLiveProvider.LISTEN)     # attend que l'utilisateur parle, exécute les outils

`LISTEN` est le message utilisateur qui n'envoie RIEN : le run attend la prochaine réponse du modèle. Un autre
texte est envoyé tel quel (`realtimeInput.text`). Une image jointe au message part en `realtimeInput.video`.

Ce qui a été OBSERVÉ sur l'API réelle (2026-11-07, `gemini-3.8-live`) et qui fixe la conception — la page de
documentation, elle, était fausse ou muette sur trois points :
* la configuration valide est `setup.generationConfig.responseModalities` (la forme de la page est REFUSÉE, 1007) ;
* le JSON du serveur arrive en trames BINAIRES (texte accepté aussi) ;
* un `toolCall` est suivi IMMÉDIATEMENT de `generationComplete` puis `turnComplete` : ils terminent le tour qui a
  produit l'appel, avant que l'outil n'ait tourné. Les prendre pour la fin de la réponse finale rendrait une
  réponse VIDE après chaque outil. Le fournisseur absorbe cette paire (jusqu'à `GRACE_S`) avant de rendre les
  appels ; la réponse parlée vient après la `toolResponse`, et son `turnComplete` n'arrive qu'environ la durée de
  l'audio plus tard, avec `usageMetadata` — c'est lui qui termine un run.

Limites assumées (v1) : une session perdue en pleine réponse est une erreur TYPÉE (`ProviderError`, `retryable`
selon le code de fermeture), jamais une réponse tronquée rendue comme un succès ; le run suivant se reconnecte
avec le jeton de reprise du serveur s'il en a donné un (sinon session neuve, mémoire de la conversation perdue).
Les jetons ne sont rapportés qu'avec `turnComplete` : un run interrompu avant n'a pas de consommation.
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from autoagent.errors import ProviderError
from autoagent.http import is_retryable_status
from autoagent.logging import get_logger
from autoagent.schema import LLMRequest, LLMResponse, Message, TokenUsage, ToolCall, ToolSpec
from autoagent.ws import WebSocketClient, WebSocketClosed, WebSocketError

from .base import LLMProvider, parse_tool_arguments, synthetic_call_id
from .gemini import _gemini_fix_arrays

_log = get_logger("gemini_live")

__all__ = ["GeminiLiveProvider"]


def _entier(x: Any) -> int | None:
    return x if isinstance(x, int) and not isinstance(x, bool) else None


def _usage_live(meta: Any) -> TokenUsage | None:
    """`usageMetadata` du `turnComplete`. La sortie est celle FACTURÉE, pensées comprises (même règle que le
    Gemini REST, 0.23.1) ; rien n'est inventé : sans aucun compteur, `None`."""
    if not isinstance(meta, dict):
        return None
    entree, sortie, pensees = (_entier(meta.get("promptTokenCount")), _entier(meta.get("responseTokenCount")),
                               _entier(meta.get("thoughtsTokenCount")))
    if entree is None and sortie is None and pensees is None:
        return None
    facturee = None if sortie is None and pensees is None else (sortie or 0) + (pensees or 0)
    return TokenUsage(input_tokens=entree, output_tokens=facturee)


class GeminiLiveProvider(LLMProvider):
    default_base_url = ("wss://generativelanguage.googleapis.com/ws/"
                        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
    LISTEN = "<live:listen>"                 # message utilisateur qui n'envoie rien : le run attend le modèle
    GRACE_S = 2.0                            # attente max de la paire generationComplete/turnComplete d'un toolCall
    ENTREE_MIME = "audio/pcm;rate=16000"     # PCM 16 bits mono 16 kHz, little-endian
    _CODES_REESSAYABLES = frozenset({1000, 1001, 1006, 1011, 1012, 1013, 1014})

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # Crochets de l'hôte (tous optionnels). Ils tournent dans le fil de lecture : un crochet LENT retarde
        # la lecture ; un crochet qui LÈVE est journalisé et ignoré (fail-open : l'observation ne casse pas la voix).
        self.on_audio: Callable[[bytes], None] | None = None          # PCM 24 kHz 16 bits mono, au fil de l'eau
        self.on_interrupted: Callable[[], None] | None = None         # l'utilisateur coupe la parole
        self.on_input_text: Callable[[str], None] | None = None       # transcription de ce que dit l'utilisateur
        self.on_output_text: Callable[[str], None] | None = None      # transcription de ce que dit le modèle
        self.on_go_away: Callable[[str | None], None] | None = None   # le serveur va fermer (temps restant)
        self.resume_handle: str | None = None                         # jeton de reprise donné par le serveur
        self.go_away: str | None = None
        self._ws: WebSocketClient | None = None
        self._lecteur: threading.Thread | None = None
        self._file: queue.Queue = queue.Queue()
        self._signature: tuple | None = None
        self._pret = threading.Event()
        self._ouverture = threading.RLock()
        self._ferme_par_l_hote = False

    # ── état et cycle de vie ───────────────────────────────────────────────────────────
    @property
    def is_open(self) -> bool:
        ws = self._ws
        return ws is not None and not ws.closed and self._pret.is_set()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Attend que la session soit ouverte (`setupComplete` reçu). Pour un fil du micro qui démarre avec le run."""
        return self._pret.wait(timeout)

    def open(self, system_prompt: str, tools: list[ToolSpec], *, temperature: float | None = None) -> bool:
        """Ouvre la session si besoin. Rend `True` si une session NEUVE vient d'être ouverte.

        Idempotent : une session vivante dont le prompt système et les outils n'ont pas changé est gardée. Les
        outils d'une session sont figés à son ouverture (`setup`) : si l'agent en a gagné ou perdu, une nouvelle
        session est ouverte plutôt que de laisser le modèle appeler un outil qui n'existe plus.
        """
        signature = (system_prompt, temperature,
                     json.dumps([t.as_gemini_declaration() for t in tools], sort_keys=True, default=str))
        with self._ouverture:
            self._ferme_par_l_hote = False
            if self.is_open and self._signature == signature:
                return False
            self._fermer_session()
            try:
                self._connecter(system_prompt, tools, temperature, avec_reprise=True)
            except ProviderError:
                if self.resume_handle is None:
                    raise
                _log.warning("gemini_live: reprise refusée, nouvelle session sans mémoire de la précédente")
                self.resume_handle = None
                self._connecter(system_prompt, tools, temperature, avec_reprise=False)
            self._signature = signature
            return True

    def close(self) -> None:
        """Ferme la session (idempotent) et réveille un run qui attendait. Un run suivant en rouvre une."""
        with self._ouverture:
            self._ferme_par_l_hote = True
            self._fermer_session()

    def _fermer_session(self) -> None:
        ws, lecteur = self._ws, self._lecteur
        self._pret.clear()
        if ws is not None:
            ws.close()
        if lecteur is not None and lecteur is not threading.current_thread():
            lecteur.join(timeout=2.0)
        self._ws, self._lecteur, self._signature = None, None, None

    # ── envoi vers le serveur (appelable depuis le fil du micro) ───────────────────────
    def send_audio(self, pcm: bytes) -> None:
        """Un bloc d'audio du micro : PCM 16 bits, mono, 16 kHz, little-endian."""
        self._envoyer({"realtimeInput": {"audio": {"data": base64.b64encode(pcm).decode("ascii"),
                                                   "mimeType": self.ENTREE_MIME}}})

    def end_audio(self) -> None:
        """Le micro se tait (pause, bouton relâché) : dit au serveur de vider son tampon d'entrée."""
        self._envoyer({"realtimeInput": {"audioStreamEnd": True}})

    def send_image(self, donnees: bytes, mime_type: str = "image/jpeg") -> None:
        """Une image de la caméra : le modèle la voit pendant qu'il parle avec l'utilisateur."""
        self._envoyer({"realtimeInput": {"video": {"data": base64.b64encode(donnees).decode("ascii"),
                                                   "mimeType": mime_type}}})

    def send_text(self, texte: str) -> None:
        """Une phrase de l'HÔTE glissée dans la session en cours (`realtimeInput.text`) : le modèle y répond comme à
        l'utilisateur, et le run qui attendait (`LISTEN`) rend cette réponse — outils compris. Pour un événement que
        l'hôte détecte lui-même (un réflexe : « ta caméra voit une pomme »)."""
        if not isinstance(texte, str) or not texte.strip():
            raise ValueError("send_text : texte vide")
        self._envoyer({"realtimeInput": {"text": texte}})

    def _envoyer(self, objet: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            raise ProviderError("gemini_live : pas de session ouverte (appeler open(), ou lancer un run avant "
                                "d'envoyer de l'audio)", retryable=True)
        try:
            ws.send_text(json.dumps(objet))
        except WebSocketError as exc:
            raise self._erreur(exc) from None

    # ── connexion ──────────────────────────────────────────────────────────────────────
    def _connecter(self, systeme: str, tools: list[ToolSpec], temperature: float | None, *, avec_reprise: bool) -> None:
        cle = self.config.resolved_api_key()          # ValueError claire AVANT toute connexion
        entetes = {"x-goog-api-key": cle, **self.config.extra_headers}   # en-tête, jamais dans l'URL (journaux)
        ws = WebSocketClient(self.config.base_url or self.default_base_url, headers=entetes,
                             timeout=self.config.timeout)
        try:
            ws.connect()
        except WebSocketError as exc:
            raise self._erreur(exc) from None
        modele = self.config.model
        configuration: dict[str, Any] = {
            "model": modele if modele.startswith("models/") else f"models/{modele}",
            # La forme valide : `generationConfig.responseModalities`. La page de documentation montre
            # `setup.responseModalities`, que le serveur REFUSE (fermeture 1007) — observé le 2026-11-07.
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
        if temperature is not None:
            configuration["generationConfig"]["temperature"] = temperature
        if systeme:
            configuration["systemInstruction"] = {"parts": [{"text": systeme}]}
        if tools:
            configuration["tools"] = [{"functionDeclarations": [
                _gemini_fix_arrays(t.as_gemini_declaration()) for t in tools]}]
        if avec_reprise and self.resume_handle:
            configuration["sessionResumption"] = {"handle": self.resume_handle}
        charge = self._with_extra_body({"setup": configuration})
        try:
            ws.send_text(json.dumps(charge))
            fin = time.monotonic() + self.config.timeout
            while True:
                recu = ws.recv(timeout=max(0.0, fin - time.monotonic()))
                if recu is None:
                    raise ProviderError(f"gemini_live : pas de setupComplete en {self.config.timeout:g} s",
                                        retryable=True)
                message = self._decoder(recu)
                if message is not None and "setupComplete" in message:
                    break
        except WebSocketError as exc:
            ws.close()
            raise self._erreur(exc) from None
        except ProviderError:
            ws.close()
            raise
        file: queue.Queue = queue.Queue()
        self._ws, self._file = ws, file
        self.go_away = None
        self._lecteur = threading.Thread(target=self._lire, args=(ws, file), name="gemini-live-lecture", daemon=True)
        self._lecteur.start()
        self._pret.set()

    # ── lecture : un fil, des crochets, une file pour le run ───────────────────────────
    @staticmethod
    def _decoder(recu: str | bytes) -> dict[str, Any] | None:
        try:
            texte = recu.decode("utf-8") if isinstance(recu, bytes) else recu
            objet = json.loads(texte)
        except (UnicodeDecodeError, json.JSONDecodeError):
            _log.warning("gemini_live : message du serveur illisible (%d octets), ignoré", len(recu))
            return None
        return objet if isinstance(objet, dict) else None

    def _lire(self, ws: WebSocketClient, file: queue.Queue) -> None:
        try:
            while True:
                recu = ws.recv(timeout=None)
                message = self._decoder(recu) if recu is not None else None
                if message is None:
                    continue
                self._effets(message)
                file.put(("message", message))
        except WebSocketError as exc:
            self._pret.clear() if self._ws is ws else None
            file.put(("ferme", exc))
        except Exception as exc:                                  # un bogue ici ne doit pas laisser un run attendre sans fin
            _log.exception("gemini_live : le fil de lecture s'arrête")
            file.put(("ferme", exc))

    def _crochet(self, nom: str, *arguments: Any) -> None:
        fonction = getattr(self, nom, None)
        if fonction is None:
            return
        try:
            fonction(*arguments)
        except Exception:
            _log.exception("gemini_live : le crochet %s a levé ; la lecture continue", nom)

    def _effets(self, message: dict[str, Any]) -> None:
        """Ce qui doit se passer TOUT DE SUITE, hors du run : audio vers le haut-parleur, interruption, transcriptions,
        jeton de reprise. Le run, lui, lit la file."""
        contenu = message.get("serverContent")
        if isinstance(contenu, dict):
            for partie in (contenu.get("modelTurn") or {}).get("parts", []):
                brut = (partie.get("inlineData") or {}).get("data")
                if brut:
                    try:
                        self._crochet("on_audio", base64.b64decode(brut))
                    except ValueError:
                        _log.warning("gemini_live : bloc audio illisible, ignoré")
            if contenu.get("interrupted"):
                self._crochet("on_interrupted")
            entree = (contenu.get("inputTranscription") or {}).get("text")
            if entree:
                self._crochet("on_input_text", entree)
            sortie = (contenu.get("outputTranscription") or {}).get("text")
            if sortie:
                self._crochet("on_output_text", sortie)
        reprise = message.get("sessionResumptionUpdate")
        if isinstance(reprise, dict) and reprise.get("resumable") and reprise.get("newHandle"):
            self.resume_handle = str(reprise["newHandle"])
        if isinstance(message.get("goAway"), dict):
            self.go_away = message["goAway"].get("timeLeft")
            self._crochet("on_go_away", self.go_away)

    # ── erreurs typées ─────────────────────────────────────────────────────────────────
    def _erreur(self, exc: BaseException) -> ProviderError:
        """Une erreur de transport -> `ProviderError`. `retryable` suit le code de fermeture (1008 politique et
        1007 charge invalide : NON ; 1011/1001/1006/1013… : oui) ou le statut HTTP d'un refus de poignée de main."""
        if isinstance(exc, WebSocketClosed):
            code, raison = exc.code, exc.reason
            detail = f" (code {code}{': ' + raison[:300] if raison else ''})" if code is not None else ""
            return ProviderError(f"gemini_live : session fermée par le serveur{detail}",
                                 retryable=code is None or code in self._CODES_REESSAYABLES)
        statut = getattr(exc, "status", None)
        return ProviderError(f"gemini_live : {exc}", status_code=statut,
                             retryable=True if statut is None else is_retryable_status(statut))

    # ── le run : UNE réponse par appel ─────────────────────────────────────────────────
    def complete(self, request: LLMRequest) -> LLMResponse:
        systeme = "\n\n".join(m.content for m in request.messages if m.role == "system" and m.content)
        nouvelle = self.open(systeme, request.tools, temperature=request.temperature)
        genre, charge = self._entree(request.messages)
        if genre == "outils" and nouvelle:
            raise ProviderError("gemini_live : la session a été perdue pendant un appel d'outil ; ses résultats ne "
                                "s'appliquent plus à la nouvelle session", retryable=True)
        file = self._file
        if genre == "outils":
            self._envoyer({"toolResponse": {"functionResponses": charge}})
        elif genre == "texte":
            for image in charge[1]:
                self._envoyer({"realtimeInput": {"video": {"data": image[1], "mimeType": image[0]}}})
            if charge[0]:
                self._envoyer({"realtimeInput": {"text": charge[0]}})
        return self._attendre(file, ecoute=genre == "ecoute")

    def _entree(self, messages: list[Message]) -> tuple[str, Any]:
        """Ce que la boucle vient d'ajouter et que le serveur doit recevoir : des résultats d'outils (la dernière
        suite de messages `tool`), un texte de l'utilisateur, ou rien (`LISTEN`)."""
        dernier = messages[-1]
        if dernier.role == "tool":
            debut = len(messages)
            while debut > 0 and messages[debut - 1].role == "tool":
                debut -= 1
            noms = {}
            if debut > 0:
                noms = {c.id: c.name for c in messages[debut - 1].tool_calls}
            reponses = [{"id": m.tool_call_id, "name": noms.get(m.tool_call_id or "") or m.name or "",
                         "response": {"result": m.content}} for m in messages[debut:]]
            return "outils", reponses
        if dernier.role == "user":
            texte = None if dernier.content == self.LISTEN else dernier.content
            images = [att.as_base64() for att in dernier.attachments]
            if texte or images:
                return "texte", (texte, images)
        return "ecoute", None

    def _prochain(self, file: queue.Queue, delai: float | None) -> tuple[str, Any] | None:
        """Le prochain événement, ou `None` au délai. Réveillé toutes les 0,5 s pour voir si l'hôte a fermé."""
        fin = None if delai is None else time.monotonic() + delai
        while True:
            if self._ferme_par_l_hote:
                raise ProviderError("gemini_live : session fermée par l'hôte (close())", retryable=True)
            tranche = 0.5 if fin is None else max(0.0, min(0.5, fin - time.monotonic()))
            try:
                evenement: tuple[str, Any] | None = file.get(timeout=tranche)
                return evenement
            except queue.Empty:
                if fin is not None and time.monotonic() >= fin:
                    return None

    def _attendre(self, file: queue.Queue, *, ecoute: bool = False) -> LLMResponse:
        contenu: list[str] = []
        appels: list[ToolCall] = []
        usage: TokenUsage | None = None
        interrompu = False
        actif = not ecoute               # en écoute, on attend SANS limite le premier signe de vie du modèle
        echeance_grace: float | None = None
        while True:
            if echeance_grace is not None:
                delai: float | None = max(0.0, echeance_grace - time.monotonic())
            elif actif:
                delai = self.config.timeout
            else:
                delai = None
            evenement = self._prochain(file, delai)
            if evenement is None:
                if echeance_grace is not None:
                    break                # la paire de fin de tour n'est pas venue : on rend ce qu'on a
                raise ProviderError(f"gemini_live : aucun événement du serveur en {self.config.timeout:g} s",
                                    retryable=True)
            genre, charge = evenement
            if genre == "ferme":
                raise self._erreur(charge) if isinstance(charge, WebSocketError) else ProviderError(
                    f"gemini_live : lecture interrompue ({type(charge).__name__})", retryable=True)
            message = charge
            if "toolCall" in message:
                for i, c in enumerate((message["toolCall"] or {}).get("functionCalls") or []):
                    appels.append(ToolCall(id=str(c.get("id") or synthetic_call_id("gemini_live", i)),
                                           name=str(c.get("name") or ""),
                                           arguments=parse_tool_arguments(c.get("args"))))
                actif = True
                echeance_grace = time.monotonic() + self.GRACE_S
            annulation = message.get("toolCallCancellation")
            if isinstance(annulation, dict):
                ids = set(annulation.get("ids") or [])
                appels = [a for a in appels if a.id not in ids]
            if "usageMetadata" in message:
                usage = _usage_live(message["usageMetadata"]) or usage
            contenu_serveur = message.get("serverContent")
            if isinstance(contenu_serveur, dict):
                for partie in (contenu_serveur.get("modelTurn") or {}).get("parts", []):
                    if "inlineData" in partie or partie.get("text"):
                        actif = True
                    if partie.get("text") and not partie.get("thought"):
                        contenu.append(partie["text"])
                sortie = (contenu_serveur.get("outputTranscription") or {}).get("text")
                if sortie:
                    contenu.append(sortie)
                    actif = True
                if contenu_serveur.get("interrupted"):
                    interrompu = True
                    actif = True
                    echeance_grace = time.monotonic() + self.GRACE_S
                if contenu_serveur.get("generationComplete"):
                    actif = True
                if contenu_serveur.get("turnComplete"):
                    break
        texte = "".join(contenu).strip()
        if appels:
            return LLMResponse(content=texte, tool_calls=appels, model=self.config.model, usage=usage,
                               finish_reason="tool_calls", raw={"interrupted": interrompu})
        return LLMResponse(content=texte, model=self.config.model, usage=usage, finish_reason="stop",
                           raw={"interrupted": interrompu})
