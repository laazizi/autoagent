"""Garder le CANAL DE LA RÉPONSE : neutraliser, en code, les URL que la réponse d'un run TEINTÉ n'a pas le droit de porter.

Les gardes de la bibliothèque (teinte, `trifecta_guard`, `tool_policy`) regardent les APPELS d'outil. Aucune ne lisait
la réponse finale : un modèle qui a lu une page piégée peut écrire le secret dans l'URL d'une image de sa réponse
(`![logo](https://hôte/p.png?d=SECRET)`), et l'interface qui l'affiche part chercher l'image sans un clic — la forme
en ligne du canal d'EchoLeak. Le banc d'injection (`redteam.py`, attaque `image_markdown`) le mesure : 1 compromis sur 1
dans les trois configurations de la démo 38.

**La règle** (`OutputGuard`, activée par `Agent(output_guard="neutralize")`). Dans un run TEINTÉ seulement, toute URL de
la réponse doit se retrouver MOT POUR MOT dans le transcript, sinon elle est retirée :

* une IMAGE (chargée sans clic : le canal le plus dangereux) n'est gardée que si son URL figure dans un contenu FIABLE
  (le message de l'utilisateur, le prompt système, le résultat d'un outil non `untrusted`) ou si son hôte est dans
  `allowed_hosts` ;
* un LIEN ou une URL nue (il faut un clic) est gardé aussi s'il figure mot pour mot dans le contenu non fiable (une
  citation de source) — `links_from_untrusted=False` le refuse. Une URL CONSTRUITE par le modèle (le secret collé dans
  la requête ou le chemin, encodé, découpé) n'est jamais mot pour mot dans le transcript : elle part.

Les formes couvertes : image et lien en ligne (`![](u)`, `[t](u)`, `[t](<u>)`), références (`[id]: u`, avec l'image
`![x][id]`), lien automatique `<u>`, balises HTML (`src`, `href`, `srcset`, `action`, `poster`, `data`…), URL nue
(`http`, `ftp`, `//hôte`, `www.`), schéma en majuscules, `\\/` échappés, entités numériques (`&#104;ttps`), tout
schéma (`javascript:`, `mailto:`). Une URL qui échappe à la lecture structurée est attrapée par le balayage des URL nues.

Contrats. **Fail-closed** : un doute (un contenu non fiable compacté par la mémoire, `TAINT_SENTINEL`) retire l'URL ;
une valeur inconnue de `output_guard` lève à la construction. **Observabilité fail-open** : le texte neutralisé ne dit
JAMAIS l'URL (elle peut contenir le secret) ; l'événement de trace donne le genre, la longueur et une empreinte.

Ce que ça ne couvre PAS (écrit ici pour que personne ne le croie) : un secret écrit en CLAIR dans le texte de la
réponse (l'utilisateur en est le lecteur légitime) ; un canal par SÉLECTION entre des URL préparées par l'attaquant et
présentes dans la page (un bit par choix, `links_from_untrusted=False` le ferme) ; une URL de LECTURE (`lire_page(url)`
portant le secret : c'est un appel d'outil, pas la réponse — l'audit `audit_trifecta` le signale) ; la voix (TTS) ;
l'interface qui rend un Markdown que le modèle n'a pas écrit ; le texte déjà DIFFUSÉ avant la garde si `output_guard`
est mis après coup (le flux est retenu seulement quand la garde est active ET le run déjà teinté à la requête).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .schema import TAINT_SENTINEL, UNTRUSTED_OPEN, Message

__all__ = ["OutputFinding", "OutputGuard", "guard_output", "resolve_output_guard", "split_transcript"]

IMAGE_RETIREE = "[image retirée]"
LIEN_RETIRE = "[lien retiré]"
REFERENCE_RETIREE = "[référence retirée]"

_SCHEMA = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.I)
_ENTITE = re.compile(r"&(?:#(\d{1,7})|#[xX]([0-9a-fA-F]{1,6})|(colon|sol|period|bsol|lpar|rpar|quest|equals|num|percnt|commat"
                     r"|lowbar|hyphen|dash|semi|excl|comma|Tab|NewLine|lbrack|rbrack));", re.I)
_ENTITES_NOMMEES = {"colon": ":", "sol": "/", "period": ".", "bsol": "\\", "lpar": "(", "rpar": ")", "quest": "?",
                    "equals": "=", "num": "#", "percnt": "%", "commat": "@", "lowbar": "_", "hyphen": "-", "dash": "-",
                    "semi": ";", "excl": "!", "comma": ",", "tab": "", "newline": "", "lbrack": "[", "rbrack": "]"}
_ECHAPPE = re.compile(r"\\([!-/:-@\[-`{-~])")
_PONCT_FINALE = ".,;:!?)]}>'\"*_`"

# Une URL « nue » : avec schéma, `//hôte.tld` (relative au schéma) ou `www.hôte.tld`. `\/` toléré (échappé en Markdown).
_NUE = re.compile(
    r"(?:(?<![\w])[a-z][a-z0-9+.\-]*:(?:\\?/){2}"
    r"|(?<![\w:/\\])(?:\\?/){2}(?=[a-z0-9\-]+\.[a-z0-9\-]+)"
    r"|(?<![\w/.\\])(?=www\.[a-z0-9\-]+\.[a-z]))"
    r"[^\s<>\"'`]+", re.I)
# `schéma:` seul (javascript:, mailto:, data:…) dans une destination : traité comme une URL.
_DEST_URL = re.compile(r"^(?:[a-z][a-z0-9+.\-]*:|(?:\\?/){2})", re.I)

_LIEN_EN_LIGNE = re.compile(
    r"(?P<bang>!?)\[(?P<texte>[^\]\n]{0,300})\]\(\s*(?P<dest><[^>\n]*>|[^()\s]*(?:\([^()\s]*\)[^()\s]*)*)"
    r"(?P<titre>\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)")
_DEFINITION = re.compile(r"(?m)^(?P<tete> {0,3}\[(?P<id>[^\]\n]+)\]:[ \t]*)<?(?P<dest>[^\s>]+)>?(?P<fin>[^\n]*)$")
_AUTOLIEN = re.compile(r"<(?P<dest>(?:[a-z][a-z0-9+.\-]*:|(?:\\?/){2})[^\s<>]*)>", re.I)
_BALISE = re.compile(r"<(?P<nom>[a-z][a-z0-9:\-]*)(?P<attrs>(?:\s+[^<>]*?)?)\s*/?>", re.I)
_ATTR_URL = re.compile(
    r"\b(?:src|href|srcset|data|action|poster|background|formaction|xlink:href|cite|ping|content)\s*=\s*"
    r"(?:\"(?P<d>[^\"]*)\"|'(?P<s>[^']*)'|(?P<n>[^\s>]+))", re.I)
_BALISES_IMAGE = {"img", "image", "source", "video", "audio", "picture", "embed", "object", "svg", "input", "track"}


@dataclass(frozen=True)
class OutputGuard:
    """Réglage de la garde de réponse. `Agent(output_guard="neutralize")` = les valeurs par défaut.

    `allowed_hosts` : hôtes (et leurs sous-domaines) dont l'URL passe quelle que soit sa forme — C'EST À L'HÔTE de ne
    nommer que des domaines qui ne rendent pas une requête qu'il ne contrôle pas. `links_from_untrusted` : un LIEN (pas
    une image) qui figure mot pour mot dans le contenu non fiable est gardé (citer une source)."""

    allowed_hosts: tuple[str, ...] = ()
    links_from_untrusted: bool = True

    def __post_init__(self) -> None:
        hotes = tuple(str(h).strip().lower().lstrip(".") for h in self.allowed_hosts)
        if any(not h or "/" in h or ":" in h for h in hotes):
            raise ValueError("allowed_hosts : des noms d'hôte nus (« exemple.fr »), sans schéma, port ni chemin")
        object.__setattr__(self, "allowed_hosts", hotes)


@dataclass(frozen=True)
class OutputFinding:
    """Une URL retirée. JAMAIS l'URL elle-même (elle peut porter le secret) : genre, longueur, empreinte."""

    kind: str          # "image" | "link" | "reference" | "html" | "bare"
    length: int
    digest: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "length": self.length, "digest": self.digest}


def resolve_output_guard(valeur: object) -> OutputGuard | None:
    """`"off"`/`None` → pas de garde ; `"neutralize"` → réglage par défaut ; un `OutputGuard` → tel quel. Autre : erreur
    (fail-closed : une faute de frappe ne désactive pas la garde en silence)."""
    if valeur is None or valeur == "off":
        return None
    if valeur == "neutralize":
        return OutputGuard()
    if isinstance(valeur, OutputGuard):
        return valeur
    raise ValueError(f"output_guard : « off », « neutralize » ou un OutputGuard, pas {valeur!r}")


def split_transcript(messages: Sequence[Message]) -> tuple[str, str]:
    """(contenu FIABLE, contenu NON FIABLE) du transcript. Fiable : prompt système, utilisateur, résultats d'outils non
    encadrés. Non fiable : résultats d'outils encadrés (`UNTRUSTED_OPEN`). Les messages de l'assistant ne comptent dans
    aucun des deux (le modèle ne se cite pas lui-même), ni un résumé de mémoire teinté (le contenu d'origine a disparu :
    ce qu'il cite n'est plus vérifiable, donc refusé)."""
    fiable: list[str] = []
    non_fiable: list[str] = []
    for m in messages:
        contenu = m.content or ""
        if m.role == "tool":
            (non_fiable if UNTRUSTED_OPEN in contenu else fiable).append(contenu)
        elif m.role in ("system", "user") and TAINT_SENTINEL not in contenu:
            fiable.append(contenu)
    return "\n".join(fiable), "\n".join(non_fiable)


def _decoder_entites(texte: str) -> str:
    def un(m: re.Match[str]) -> str:
        try:
            if m.group(1):
                return chr(int(m.group(1)))
            if m.group(2):
                return chr(int(m.group(2), 16))
        except (ValueError, OverflowError):
            return m.group(0)
        return _ENTITES_NOMMEES.get(m.group(3).lower(), m.group(0))
    return _ENTITE.sub(un, texte)


def _normaliser(dest: str) -> str:
    dest = dest.strip().strip("<>").strip()
    return _ECHAPPE.sub(r"\1", _decoder_entites(dest))


def _hote(url: str) -> str:
    u = url if _SCHEMA.match(url) else ("https:" + url if url.startswith("//") else "https://" + url)
    try:
        return (urlsplit(u).hostname or "").lower()
    except ValueError:
        return ""


def _empreinte(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8", "replace")).hexdigest()[:10]


class _Passe:
    """L'état d'UNE neutralisation : le transcript lu une fois, les URL retirées."""

    def __init__(self, politique: OutputGuard, fiable: str, non_fiable: str) -> None:
        self.p = politique
        self.fiable, self.non_fiable = fiable, non_fiable
        self.trouvailles: list[OutputFinding] = []

    def permis(self, brut: str, *, image: bool) -> bool:
        url = _normaliser(brut)
        candidats = [url]
        while url and url[-1] in _PONCT_FINALE:              # ponctuation de phrase collée à l'URL
            url = url[:-1]
            candidats.append(url)
        for c in candidats:
            if not c:
                continue
            if c in self.fiable:
                return True
            if not image and self.p.links_from_untrusted and c in self.non_fiable:
                return True
        hote = _hote(candidats[-1] if candidats[-1] else candidats[0])
        return bool(hote) and any(hote == h or hote.endswith("." + h) for h in self.p.allowed_hosts)

    def decide(self, brut: str, genre: str, *, image: bool) -> bool:
        """True = garder. Sinon enregistre la trouvaille."""
        if self.permis(brut, image=image):
            return True
        self.trouvailles.append(OutputFinding(genre, len(brut), _empreinte(brut)))
        return False


def _nettoyer(texte: str, passe: _Passe) -> str:
    texte = _decoder_entites(texte) if _ENTITE.search(texte) else texte

    # 1. Lien ou image en ligne : on réécrit la construction entière, le texte alternatif reste.
    def en_ligne(m: re.Match[str]) -> str:
        dest = m.group("dest")
        image = bool(m.group("bang"))
        if not (image or _DEST_URL.match(_normaliser(dest))):
            return m.group(0)                                # lien relatif : pas une sortie
        if passe.decide(dest, "image" if image else "link", image=image):
            return m.group(0)
        alt = m.group("texte").strip()
        if image:
            return f"{IMAGE_RETIREE}" + (f" {alt}" if alt else "")
        return f"{alt} {LIEN_RETIRE}".strip()
    texte = _LIEN_EN_LIGNE.sub(en_ligne, texte)

    # 2. Définitions de référence : image si une image les cite, sinon lien.
    def definition(m: re.Match[str]) -> str:
        dest = m.group("dest")
        if not _DEST_URL.match(_normaliser(dest)):
            return m.group(0)
        ident = re.escape(m.group("id").strip())
        image = bool(re.search(rf"!\[[^\]\n]*\]\[\s*{ident}\s*\]|!\[{ident}\](?:\[\s*\])?", texte, re.I))
        if passe.decide(dest, "reference", image=image):
            return m.group(0)
        return f"{m.group('tete')}{REFERENCE_RETIREE}"
    texte = _DEFINITION.sub(definition, texte)

    # 3. Lien automatique <https://…>.
    def auto(m: re.Match[str]) -> str:
        return m.group(0) if passe.decide(m.group("dest"), "link", image=False) else LIEN_RETIRE
    texte = _AUTOLIEN.sub(auto, texte)

    # 4. Balises HTML : une URL refusée dans un attribut retire la balise ENTIÈRE.
    def balise(m: re.Match[str]) -> str:
        image = m.group("nom").lower() in _BALISES_IMAGE
        garder = True
        for a in _ATTR_URL.finditer(m.group("attrs")):
            valeur = next(g for g in (a.group("d"), a.group("s"), a.group("n"), "") if g is not None)
            candidats = re.split(r"[\s,]+", valeur) if valeur else []   # srcset : plusieurs URL
            for c in candidats:
                if _DEST_URL.match(_normaliser(c)) and not passe.decide(c, "html", image=image):
                    garder = False
        return m.group(0) if garder else (IMAGE_RETIREE if image else LIEN_RETIRE)
    texte = _BALISE.sub(balise, texte)

    # 5. Filet : toute URL nue restante (y compris dans un `url(...)` CSS ou une construction mal formée).
    def nue(m: re.Match[str]) -> str:
        brut = m.group(0)
        if passe.decide(brut, "bare", image=False):
            return brut
        coeur = brut.rstrip(_PONCT_FINALE)
        return LIEN_RETIRE + brut[len(coeur):]
    return _NUE.sub(nue, texte)


def guard_output(
    texte: str, messages: Iterable[Message], politique: OutputGuard | None = None,
) -> tuple[str, list[OutputFinding]]:
    """Rend `(texte neutralisé, trouvailles)` pour une réponse d'un run teinté ; `messages` = le transcript.

    Pure et déterministe, sans effet de bord. Appelée seulement quand le run est teinté : sur un run sain, la
    réponse n'est jamais touchée. Idempotente : relancée sur son propre résultat, elle ne trouve plus rien."""
    if not texte:
        return texte, []
    fiable, non_fiable = split_transcript(list(messages))
    passe = _Passe(politique or OutputGuard(), fiable, non_fiable)
    return _nettoyer(texte, passe), passe.trouvailles
