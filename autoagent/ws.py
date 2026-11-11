"""Client WebSocket synchrone (RFC 6455), bibliothèque standard seulement.

Pourquoi ce module existe : un fournisseur « Live » (voix en continu) parle en WebSocket, et la bibliothèque
n'a ni dépendance ni `async` (voir AGENTS.md). Ce client est donc écrit ici, petit, et soumis aux mêmes
contrats que `http.py` :

* un échec est TYPÉ (`WebSocketError`, `WebSocketClosed`) — jamais un message tronqué rendu comme un succès ;
* **jamais la requête dans un message d'erreur** : la clé d'API d'un fournisseur voyage parfois dans la chaîne
  de requête (`?key=`), donc toute étiquette d'erreur est « schéma://hôte:port/chemin » SANS requête ;
* des limites en CODE : taille maximale d'un message (refusée dès l'en-tête, avant d'attendre la charge),
  longueur maximale de la réponse de poignée de main.

Ce que le client fait : poignée de main (clé aléatoire par connexion, `Sec-WebSocket-Accept` vérifiée), trames
masquées (obligatoire côté client), réassemblage des messages fragmentés, pong automatique à un ping, réponse à
une fermeture, refus des trames invalides avec le bon code de fermeture (1002 protocole, 1007 UTF-8, 1009 trop
grand). Ce qu'il ne fait PAS : extensions (permessage-deflate), sous-protocoles, proxy, redirections.

Fils d'exécution : `send_*` est protégé par un verrou et peut être appelé depuis plusieurs fils (le micro qui
envoie pendant que la boucle lit) ; `recv` est pour UN seul fil lecteur ; `close()` peut venir de n'importe où et
réveille un `recv` bloqué. L'attente en lecture passe par `select`, pas par un délai posé sur la socket : un
délai posé par le fil lecteur s'appliquerait AUSSI aux envois de l'autre fil et en ferait échouer un en plein
milieu d'une trame.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import select
import socket
import ssl
import struct
import threading
import time
from collections.abc import Mapping
from typing import NoReturn
from urllib.parse import urlsplit

__all__ = ["WebSocketClient", "WebSocketClosed", "WebSocketError"]

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_OP_CONT, _OP_TEXT, _OP_BINARY, _OP_CLOSE, _OP_PING, _OP_PONG = 0, 1, 2, 8, 9, 10
_OPCODES = frozenset({_OP_CONT, _OP_TEXT, _OP_BINARY, _OP_CLOSE, _OP_PING, _OP_PONG})
_REPONSE_MAX = 65_536          # taille maximale de la réponse de poignée de main
_SECRET = re.compile(r"(key|token|access_token|api_key)=[^&\s\"']+", re.IGNORECASE)


class WebSocketError(Exception):
    """Échec du client : connexion, poignée de main, trame invalide, message trop grand.

    `status` : le statut HTTP quand le serveur a REFUSÉ la poignée de main (403, 429, 503…), sinon `None` —
    l'hôte décide « réessayable ? » sur ce champ, sans lire le texte (même idée que `ProviderError.status_code`)."""

    def __init__(self, message: str = "", *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class WebSocketClosed(WebSocketError):
    """La connexion est fermée (par le serveur, par `close()`, ou coupée). `code` suit la RFC :
    1000 normal, 1006 coupure sans trame de fermeture, 1008 politique, etc. ; `None` si non fourni."""

    def __init__(self, code: int | None = None, reason: str = "", message: str | None = None) -> None:
        self.code = code
        self.reason = reason
        detail = f"code {code}" if code is not None else "sans code"
        super().__init__(message or f"WebSocket fermé ({detail}{', ' + reason if reason else ''})")


def _xor_masque(charge: bytes, masque: bytes) -> bytes:
    """XOR de la charge avec le masque de 4 octets répété — par un grand entier : bien plus rapide qu'une
    boucle d'octets sur un bloc audio."""
    n = len(charge)
    if n == 0:
        return b""
    repete = (masque * (n // 4 + 1))[:n]
    return (int.from_bytes(charge, "big") ^ int.from_bytes(repete, "big")).to_bytes(n, "big")


class WebSocketClient:
    def __init__(self, url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 10.0,
                 max_message_bytes: int = 16 * 1024 * 1024, ssl_context: ssl.SSLContext | None = None) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("ws", "wss"):
            raise ValueError(f"URL WebSocket attendue (ws:// ou wss://), reçu {parts.scheme or 'aucun schéma'!r}")
        if not parts.hostname:
            raise ValueError("URL WebSocket sans nom d'hôte")
        self._secure = parts.scheme == "wss"
        self._host = parts.hostname
        self._port = parts.port or (443 if self._secure else 80)
        self._target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        self._query = parts.query
        # L'étiquette des erreurs : SANS la requête (elle peut porter une clé d'API).
        self._label = f"{parts.scheme}://{self._host}:{self._port}{parts.path or '/'}"
        self._headers = dict(headers or {})
        for nom, valeur in self._headers.items():
            if any(c in f"{nom}{valeur}" for c in "\r\n"):
                raise ValueError(f"en-tête {nom!r} invalide : retour à la ligne interdit")
        self._timeout = timeout
        self._max = max_message_bytes
        self._ssl_context = ssl_context
        self._sock: socket.socket | None = None
        self._buf = bytearray()
        self._frag: tuple[int, bytearray] | None = None
        self._envoi = threading.Lock()
        self._etat = threading.Lock()
        self._closed = True                     # tant qu'on n'est pas connecté
        self._close_sent = False

    # ── état ──────────────────────────────────────────────────────────────────────────
    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> WebSocketClient:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── connexion ─────────────────────────────────────────────────────────────────────
    def connect(self) -> None:
        if self._sock is not None:
            raise WebSocketError("déjà connecté : un client WebSocket ne se reconnecte pas, en créer un autre")
        try:
            brute = socket.create_connection((self._host, self._port), timeout=self._timeout)
        except OSError as exc:
            raise WebSocketError(f"connexion à {self._label} impossible : {exc.strerror or type(exc).__name__}") from None
        try:
            sock: socket.socket = brute
            if self._secure:
                contexte = self._ssl_context or ssl.create_default_context()
                sock = contexte.wrap_socket(brute, server_hostname=self._host)
            sock.settimeout(self._timeout)
            self._sock = sock
            self._poignee_de_main(sock)
        except (OSError, ssl.SSLError) as exc:
            self._fermer_socket()
            try:
                brute.close()
            except OSError:
                pass
            raise WebSocketError(
                f"connexion à {self._label} : échec {'TLS ' if self._secure else ''}"
                f"({type(exc).__name__}: {exc.__class__.__name__ if not str(exc) else str(exc)[:120]})") from None
        except WebSocketError:
            self._fermer_socket()
            raise
        self._closed = False

    def _poignee_de_main(self, sock: socket.socket) -> None:
        cle = base64.b64encode(os.urandom(16)).decode()
        hote = self._host if self._port in (80, 443) else f"{self._host}:{self._port}"
        lignes = [f"GET {self._target} HTTP/1.1", f"Host: {hote}", "Upgrade: websocket", "Connection: Upgrade",
                  f"Sec-WebSocket-Key: {cle}", "Sec-WebSocket-Version: 13"]
        lignes += [f"{nom}: {valeur}" for nom, valeur in self._headers.items()]
        sock.sendall(("\r\n".join(lignes) + "\r\n\r\n").encode("utf-8"))

        recu = bytearray()
        while b"\r\n\r\n" not in recu:
            if len(recu) > _REPONSE_MAX:
                raise WebSocketError(f"poignée de main avec {self._label} : réponse trop grande")
            morceau = sock.recv(4096)
            if not morceau:
                raise WebSocketError(f"poignée de main avec {self._label} : connexion fermée avant la réponse")
            recu += morceau
        tete, _, reste = bytes(recu).partition(b"\r\n\r\n")
        lignes_reponse = tete.decode("latin-1").split("\r\n")
        statut = lignes_reponse[0].split(" ", 2)
        entetes = {}
        for ligne in lignes_reponse[1:]:
            nom, _, valeur = ligne.partition(":")
            entetes[nom.strip().lower()] = valeur.strip()
        if len(statut) < 2 or statut[1] != "101":
            code = statut[1] if len(statut) > 1 else "?"
            corps = self._expurger(reste[:300].decode("utf-8", "replace")).strip()
            raise WebSocketError(f"poignée de main refusée par {self._label} : HTTP {code}"
                                 + (f" — {corps}" if corps else ""),
                                 status=int(code) if code.isdecimal() else None)
        attendue = base64.b64encode(hashlib.sha1((cle + _GUID).encode()).digest()).decode()
        if entetes.get("sec-websocket-accept") != attendue:
            raise WebSocketError(f"poignée de main avec {self._label} : Sec-WebSocket-Accept incorrect "
                                 "(ce serveur ne parle pas WebSocket, ou une machine intermédiaire l'altère)")
        if entetes.get("upgrade", "").lower() != "websocket" or "upgrade" not in entetes.get("connection", "").lower():
            raise WebSocketError(f"poignée de main avec {self._label} : en-têtes Upgrade/Connection incorrects")
        self._buf = bytearray(reste)             # des trames ont pu arriver avec la réponse

    def _expurger(self, texte: str) -> str:
        """Retire d'un texte venu du réseau tout ce qui ressemble à un secret (la requête, `key=…`)."""
        if self._query:
            texte = texte.replace(self._query, "***")
        return _SECRET.sub(lambda m: f"{m.group(1)}=***", texte)

    # ── envoi ─────────────────────────────────────────────────────────────────────────
    def send_text(self, texte: str) -> None:
        self._envoyer_trame(_OP_TEXT, texte.encode("utf-8"))

    def send_binary(self, donnees: bytes) -> None:
        self._envoyer_trame(_OP_BINARY, bytes(donnees))

    def _envoyer_trame(self, opcode: int, charge: bytes) -> None:
        sock = self._sock
        if sock is None or self._closed:
            raise WebSocketClosed(message="WebSocket non connecté ou déjà fermé : envoi impossible")
        n = len(charge)
        if n < 126:
            tete = bytes([0x80 | opcode, 0x80 | n])
        elif n < 65_536:
            tete = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack("!H", n)
        else:
            tete = bytes([0x80 | opcode, 0x80 | 127]) + struct.pack("!Q", n)
        masque = os.urandom(4)
        trame = tete + masque + _xor_masque(charge, masque)
        with self._envoi:                         # une trame entière à la fois, même avec plusieurs fils
            try:
                sock.sendall(trame)
            except OSError as exc:
                self._fermer_socket()
                raise WebSocketClosed(1006, "", f"envoi impossible vers {self._label} : "
                                                f"{exc.strerror or type(exc).__name__}") from None

    # ── réception ─────────────────────────────────────────────────────────────────────
    def recv(self, timeout: float | None = None) -> str | bytes | None:
        """Le prochain message COMPLET : `str` (texte) ou `bytes` (binaire).

        `None` si `timeout` s'écoule sans message complet (la connexion reste utilisable, et une trame déjà
        entamée est gardée en mémoire) ; `timeout=None` attend sans limite. Lève `WebSocketClosed` à la
        fermeture et `WebSocketError` à une trame invalide.
        """
        fin = None if timeout is None else time.monotonic() + timeout
        while True:
            trame = self._extraire_trame()
            if trame is None:
                if self._closed and self._sock is None:
                    raise WebSocketClosed(message="WebSocket non connecté ou déjà fermé : lecture impossible")
                if not self._lire_plus(fin):
                    return None
                continue
            message = self._traiter(trame)
            if message is not None:
                return message

    def _lire_plus(self, fin: float | None) -> bool:
        """Ajoute des octets au tampon. `False` si le délai s'est écoulé sans rien recevoir."""
        sock = self._sock
        if sock is None:
            raise WebSocketClosed(1006, "", "WebSocket fermé pendant la lecture")
        en_attente = getattr(sock, "pending", None)
        if not (callable(en_attente) and en_attente()):      # TLS : des octets déchiffrés peuvent déjà attendre
            restant = None if fin is None else max(0.0, fin - time.monotonic())
            try:
                pret, _, _ = select.select([sock], [], [], restant)
            except (OSError, ValueError):                    # fermée par close() depuis un autre fil
                raise WebSocketClosed(1006, "", "WebSocket fermé pendant la lecture") from None
            if not pret:
                return False
        try:
            morceau = sock.recv(65_536)
        except (ssl.SSLWantReadError, BlockingIOError):
            return True                                      # pas encore un enregistrement TLS complet : on réessaie
        except OSError as exc:
            self._fermer_socket()
            raise WebSocketClosed(1006, "", f"lecture impossible sur {self._label} : "
                                            f"{exc.strerror or type(exc).__name__}") from None
        if not morceau:
            self._fermer_socket()
            raise WebSocketClosed(1006, "", f"{self._label} a coupé la connexion sans trame de fermeture")
        self._buf += morceau
        return True

    def _extraire_trame(self) -> tuple[bool, int, bytes] | None:
        """Une trame complète du tampon, ou `None` s'il en manque. Les contrôles d'en-tête se font DÈS que
        l'en-tête est lisible : une longueur absurde est refusée sans attendre une charge qui n'arrivera pas."""
        b = self._buf
        if len(b) < 2:
            return None
        fin, rsv, opcode = bool(b[0] & 0x80), b[0] & 0x70, b[0] & 0x0F
        masquee, n, pos = bool(b[1] & 0x80), b[1] & 0x7F, 2
        if n == 126:
            if len(b) < 4:
                return None
            n, pos = struct.unpack("!H", bytes(b[2:4]))[0], 4
        elif n == 127:
            if len(b) < 10:
                return None
            n, pos = struct.unpack("!Q", bytes(b[2:10]))[0], 10
        if rsv:
            self._echec(1002, f"{self._label} : bit RSV posé alors qu'aucune extension n'est négociée")
        if masquee:
            self._echec(1002, f"{self._label} : trame du serveur masquée (interdit par la RFC 6455)")
        if opcode not in _OPCODES:
            self._echec(1002, f"{self._label} : opcode inconnu {opcode}")
        if opcode >= _OP_CLOSE and (not fin or n > 125):
            self._echec(1002, f"{self._label} : trame de contrôle invalide (fragmentée ou trop longue)")
        if n > self._max:
            self._echec(1009, f"{self._label} : message trop grand ({n} octets annoncés, limite {self._max})")
        if len(b) < pos + n:
            return None
        charge = bytes(b[pos:pos + n])
        del b[:pos + n]
        return fin, opcode, charge

    def _traiter(self, trame: tuple[bool, int, bytes]) -> str | bytes | None:
        fin, opcode, charge = trame
        if opcode == _OP_PING:
            self._repondre_pong(charge)
            return None
        if opcode == _OP_PONG:
            return None
        if opcode == _OP_CLOSE:
            code, raison = None, ""
            if len(charge) == 1:
                self._echec(1002, f"{self._label} : trame de fermeture invalide")
            if len(charge) >= 2:
                code = struct.unpack("!H", charge[:2])[0]
                raison = charge[2:].decode("utf-8", "replace")
            self._repondre_fermeture(code)
            self._fermer_socket()
            raise WebSocketClosed(code, raison)
        if opcode in (_OP_TEXT, _OP_BINARY):
            if self._frag is not None:
                self._echec(1002, f"{self._label} : nouveau message avant la fin du message fragmenté")
            if fin:
                return self._livrer(opcode, charge)
            self._frag = (opcode, bytearray(charge))
            return None
        # continuation
        if self._frag is None:
            self._echec(1002, f"{self._label} : trame de continuation sans message en cours")
        initial, morceaux = self._frag
        morceaux += charge
        if len(morceaux) > self._max:
            self._echec(1009, f"{self._label} : message trop grand (fragments cumulés au-delà de {self._max} octets)")
        if not fin:
            return None
        self._frag = None
        return self._livrer(initial, bytes(morceaux))

    def _livrer(self, opcode: int, donnees: bytes) -> str | bytes:
        if opcode == _OP_BINARY:
            return donnees
        try:
            return donnees.decode("utf-8")
        except UnicodeDecodeError:
            self._echec(1007, f"{self._label} : message texte qui n'est pas de l'UTF-8")

    # ── fermeture et fautes ───────────────────────────────────────────────────────────
    def _repondre_pong(self, charge: bytes) -> None:
        try:
            self._envoyer_trame(_OP_PONG, charge)
        except WebSocketClosed:
            pass

    def _repondre_fermeture(self, code: int | None) -> None:
        if self._close_sent:
            return
        self._close_sent = True
        try:
            self._envoyer_trame(_OP_CLOSE, struct.pack("!H", code) if code is not None else b"")
        except WebSocketClosed:
            pass

    def _echec(self, code: int, message: str) -> NoReturn:
        """Faute de protocole : fermeture avec `code` (au mieux), socket coupée, erreur typée."""
        if not self._close_sent:
            self._close_sent = True
            try:
                self._envoyer_trame(_OP_CLOSE, struct.pack("!H", code))
            except WebSocketClosed:
                pass
        self._fermer_socket()
        raise WebSocketError(message)

    def close(self, code: int = 1000, reason: str = "") -> None:
        """Ferme la connexion (idempotent). Réveille un `recv` bloqué dans un autre fil."""
        if self._sock is not None and not self._closed and not self._close_sent:
            self._close_sent = True
            try:
                self._envoyer_trame(_OP_CLOSE, struct.pack("!H", code) + reason.encode("utf-8")[:120])
            except WebSocketClosed:
                pass
        self._fermer_socket()

    def _fermer_socket(self) -> None:
        with self._etat:
            self._closed = True
            sock = self._sock
            if sock is None:
                return
            try:
                sock.shutdown(socket.SHUT_RDWR)           # réveille le fil bloqué dans `select`
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
