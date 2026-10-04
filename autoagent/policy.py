"""Politique d'outils DÉCLARATIVE — des données, plus du code (0.18.0).

`tool_policy` est une fonction Python : puissante, mais elle ne se versionne pas
en revue, ne se relit pas en diff, ne se transporte pas dans un snapshot et ne se
génère pas. `ToolPolicySpec` exprime la même chose en JSON, et `compile()` en fait
un callable de la signature `tool_policy` EXISTANTE — le code de production ne
bouge donc pas d'une ligne.

    spec = ToolPolicySpec.from_dict({
        "default": "allow",
        "rules": [
            {"tool": "write_file", "action": "deny",
             "when": {"args": {"path": {"not": {"path_within": "rapports/"}}}},
             "reason": "écriture limitée à rapports/"},
            {"tool": "*", "action": "deny", "when": {"tainted": True, "egress": True},
             "reason": "sortie interdite après lecture de contenu non fiable"},
            {"tool": "supprimer_compte", "action": "approve"},
        ],
    })
    agent = Agent(provider, tool_policy=spec.compile())

Pour confiner un chemin ou une URL, `path_within` et `url_host` (0.23.1) : `starts_with`
compare des chaînes et se contourne (`rapports/../../etc/…`, `https://api.exemple.fr.evil.example`,
`https://api.exemple.fr@evil.example`). Un `deny` l'emporte toujours sur un `allow` : « écrire
seulement sous rapports/ » s'écrit donc « REFUSE si le chemin n'est PAS sous rapports/ » (`not`).

Trois choix de conception :

* **Précédence par ACTION, pas par ordre** : parmi les règles qui matchent,
  `deny` gagne, puis `approve`, puis `allow`, sinon `default`. Une politique n'a
  donc pas de comportement caché dépendant de l'ordre des lignes — un refus ne
  peut jamais être « masqué » par une autorisation placée plus haut.

* **Fail-CLOSED partout** : une structure invalide est refusée dès
  `from_dict` (l'erreur sort au démarrage, pas au premier appel sensible) ; et si
  l'évaluation d'une condition lève malgré tout, l'appel est REFUSÉ. C'est le
  contrat de `tool_policy`, à l'opposé de la trace qui échoue en douceur.

* **Confinement monotone** : `narrow()` n'accepte que des règles qui RESTREIGNENT
  (`deny`/`approve`) et s'applique librement ; tout ce qui pourrait ÉLARGIR passe
  par `expand()`, qui lève `ApprovalRequired` sans `approved=True`. On ne tente
  pas de *prouver* qu'un changement est une restriction (l'état de l'art utilise
  un solveur SMT, hors périmètre d'une lib zéro-dépendance) : on classe par le
  type d'action, ce qui est conservateur — dans le doute, on demande.
"""

from __future__ import annotations

import ipaddress
import posixpath
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .errors import ApprovalRequired
from .logging import get_logger, warn_once

__all__ = ["ToolPolicySpec"]

_log = get_logger("policy")

_ACTIONS = ("allow", "deny", "approve")
_RESTRICTIVE = ("deny", "approve")          # ne peuvent que retirer des droits
# `source` (D1) : d'où vient l'action — "tool" (appel direct du modèle), "host_function" (fonction
# de l'hôte appelée par du code du modèle), "subagent" (outil d'un sous-agent qui hérite de la
# politique). Permet de dire en DONNÉES « aucun envoi depuis un programme », sans toucher au code.
_CONTEXT_KEYS = ("args", "tainted", "egress", "step", "permissions", "source")


def _as_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# ── Opérateurs de CONFINEMENT (0.23.1) ───────────────────────────────────────────────────────
#
# `starts_with` compare des CHAÎNES. Pour confiner un chemin ou une URL, c'est une fausse
# sécurité — reproduit sur la 0.23.0, avec la règle que notre propre doc enseignait :
#   `{"path": {"starts_with": "rapports/"}}`  laisse passer  `rapports/../../etc/cron.d/x`
#   `{"url": {"starts_with": "https://api.exemple.fr"}}`  laisse passer
#       `https://api.exemple.fr.evil.example/…`  et  `https://api.exemple.fr@evil.example/…`
# (même classe de faille que CVE-2025-53110, une « naive string prefix-matching check » dans le
# serveur MCP filesystem d'Anthropic — Cymulate, 17 mars 2026).
# `path_within` et `url_host` comparent la chose elle-même, une fois NORMALISÉE. Ils sont
# fail-closed : tout ce qu'ils ne savent pas lire proprement ne correspond pas.

_CONTROLE = re.compile(r"[\x00-\x1f\x7f]")
_ENCODAGE_AMBIGU = re.compile(r"%(?:2e|2f|5c)", re.IGNORECASE)    # « . », « / », « \ » encodés
_LECTEUR = re.compile(r"^([A-Za-z]):")                            # « C: » : Windows ; pour `posixpath`, un chemin RELATIF
# Les noms de périphériques de Windows (`CON`, `NUL`, `COM1`, `LPT1.txt`…) ne sont pas des fichiers : un chemin qui
# passe par l'un d'eux n'est dans aucun répertoire. Refusés partout (aucun fichier légitime ne s'appelle ainsi).
_PERIPHERIQUE = re.compile(r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$", re.IGNORECASE)


def _chemin_dans(chemin: str, base: str) -> bool:
    """`chemin`, une fois normalisé (`..`, `.`, `//`, `\\`), est-il `base` elle-même ou dessous ?"""
    # Une lettre de lecteur (`C:\Windows\win.ini`) est un chemin ABSOLU sous Windows : `posixpath` le voit relatif, et
    # contre la base « . » il passait (reproduit sous Windows par la relecture du 4 oct. 2026). Même lecteur des deux
    # côtés, ou aucun : on compare alors le reste.
    lc, lb = _LECTEUR.match(chemin), _LECTEUR.match(base)
    if (lc.group(1).lower() if lc else None) != (lb.group(1).lower() if lb else None):
        return False
    if lc and lb:
        chemin, base = chemin[2:], base[2:] or "."
    c = posixpath.normpath(chemin.replace("\\", "/"))
    b = posixpath.normpath(base.replace("\\", "/"))
    if any(_PERIPHERIQUE.match(morceau) for morceau in c.split("/")):
        return False
    if c.startswith("/") != b.startswith("/"):
        return False                      # absolu contre relatif : on ne devine pas le répertoire courant
    if b == ".":
        return c != ".." and not c.startswith("../")
    if c == b:
        return True
    return c.startswith(b if b.endswith("/") else b + "/")


def _liste_de_textes(expected: Any) -> list[str]:
    if isinstance(expected, str):
        return [expected] if expected else []
    if isinstance(expected, (list, tuple)):
        return [e for e in expected if isinstance(e, str) and e]
    return []


def _path_within(actual: Any, expected: Any) -> bool:
    """Le chemin est-il DANS un des répertoires attendus ? Purement syntaxique : les liens symboliques
    sont l'affaire de l'outil. Refuse (fail-closed) un chemin vide, à caractère de contrôle (NUL…),
    ou contenant `%2e` / `%2f` / `%5c` (un `..` encodé qu'un outil pourrait décoder). La vérification se
    fait aussi sur la forme NFKC : des points « pleine chasse » (U+FF0E) valent un point pour certains outils."""
    bases = _liste_de_textes(expected)
    if not isinstance(actual, str) or not actual or not bases:
        return False
    formes = {actual, unicodedata.normalize("NFKC", actual)}
    # Le contrôle des caractères de contrôle et des `%2e` porte sur CHAQUE forme : « %2e » écrit en pleine chasse vaut
    # `%2e` après NFKC, et un outil qui normalise puis décode y verrait un « . ».
    if any(_CONTROLE.search(f) or _ENCODAGE_AMBIGU.search(f) for f in formes):
        return False
    return any(all(_chemin_dans(f, base) for f in formes) for base in bases)


# Les caractères que IDNA 2003 (le codec `idna` de la bibliothèque standard) PLIE et que IDNA 2008 (requests, urllib3,
# Node, curl) GARDE : « straße.example » devient `strasse.example` d'un côté et `xn--strae-oqa.example` de l'autre —
# deux domaines différents, dont l'un peut être enregistré par un attaquant. On ne parie pas : un hôte qui en contient
# ne correspond JAMAIS (écrire la forme punycode `xn--…` dans la règle). Mesuré par la relecture sur 123 827 URLs :
# toutes les divergences avec Node (WHATWG) étaient de cette classe.
_AMBIGUS_IDNA = frozenset({chr(0x00DF), chr(0x1E9E), chr(0x03C2), chr(0x200C), chr(0x200D)})


def _hote_ascii(hote: str) -> str | None:
    hote = hote.strip().lower().rstrip(".")
    if not hote or len(hote) > 253:                     # 253 : la longueur maximale d'un nom DNS — et l'encodeur
        return None                                     # punycode est quadratique sur un nom géant (relecture)
    if hote.isascii():
        return hote
    if _AMBIGUS_IDNA & set(hote):
        return None
    try:
        return hote.encode("idna").decode("ascii")      # punycode : un « a » cyrillique (U+0430) n'est pas un « a » latin
    except UnicodeError:
        return None


def _url_host(actual: Any, expected: Any) -> bool:
    """L'hôte RÉEL de l'URL est-il un des hôtes attendus ? `api.exemple.fr` exact, ou `*.exemple.fr`
    (sous-domaines, pas le domaine nu). http(s) seulement ; le port n'est pas examiné. Refuse l'URL à
    identifiants (`user@hôte`), à antislash, à espace ou caractère de contrôle, sans schéma ou sans hôte :
    les analyseurs d'URL ne s'accordent pas sur ces formes, et c'est là que se cachent les contournements."""
    motifs = _liste_de_textes(expected)
    if not isinstance(actual, str) or not motifs:
        return False
    if _CONTROLE.search(actual) or "\\" in actual or " " in actual:
        return False
    try:
        parts = urlsplit(actual)
        hote = parts.hostname
        _ = parts.port                     # lève ValueError si le port est invalide
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.netloc or "@" in parts.netloc or not hote:
        return False
    hote_ascii = _hote_ascii(hote)
    if hote_ascii is None:
        return False
    for motif in motifs:
        if motif.strip().startswith("*."):
            socle = _hote_ascii(motif.strip()[2:])
            if socle and hote_ascii.endswith("." + socle):
                return True
        elif _hote_ascii(motif) == hote_ascii:
            return True
    return False


def _match_operator(operator: str, actual: Any, expected: Any) -> bool:
    """Applique UN opérateur. Lève ValueError si l'opérateur est inconnu."""
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "in":
        return isinstance(expected, (list, tuple, set)) and actual in expected
    if operator == "not_in":
        return isinstance(expected, (list, tuple, set)) and actual not in expected
    if operator == "starts_with":
        return isinstance(actual, str) and actual.startswith(str(expected))
    if operator == "ends_with":
        return isinstance(actual, str) and actual.endswith(str(expected))
    if operator == "contains":
        if isinstance(actual, str):
            return str(expected) in actual
        return isinstance(actual, (list, tuple, set)) and expected in actual
    if operator == "matches":
        return isinstance(actual, str) and re.search(str(expected), actual) is not None
    if operator == "path_within":
        return _path_within(actual, expected)
    if operator == "url_host":
        return _url_host(actual, expected)
    if operator == "not":
        # Négation (0.23.1) : sans elle, « écrire seulement sous rapports/ » est inexprimable —
        # un `deny` l'emporte toujours sur un `allow` (précédence par ACTION), donc il faut pouvoir
        # écrire « REFUSE si le chemin n'est PAS sous rapports/ ». Valeur absente (None) : le
        # prédicat interne est faux, sa négation est vraie — le refus s'applique (fail-closed).
        return not _match_predicate(actual, expected)
    if operator == "max_length":
        return hasattr(actual, "__len__") and len(actual) <= int(expected)
    if operator == "max_items":
        return isinstance(actual, (list, tuple, set)) and len(actual) <= int(expected)
    if operator == "exists":
        return (actual is not None) is bool(expected)
    if operator in ("lt", "le", "gt", "ge"):
        left, right = _as_number(actual), _as_number(expected)
        if left is None or right is None:
            return False
        return {"lt": left < right, "le": left <= right,
                "gt": left > right, "ge": left >= right}[operator]
    raise ValueError(f"opérateur de condition inconnu : {operator!r}")


def _match_predicate(actual: Any, predicate: Any) -> bool:
    """`predicate` est soit une valeur brute (égalité), soit {opérateur: attendu}."""
    if isinstance(predicate, dict):
        return all(_match_operator(op, actual, exp) for op, exp in predicate.items())
    return actual == predicate


@dataclass(frozen=True)
class ToolPolicySpec:
    """Politique d'outils sérialisable. Immuable : `narrow`/`expand` rendent une copie."""

    rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    default: str = "allow"

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolPolicySpec":
        """Valide et construit. Lève `ValueError` avec la position fautive.

        La validation est STRICTE et précoce à dessein : une politique de sécurité
        qui contient une faute de frappe doit échouer au démarrage, pas laisser
        passer un appel sensible parce qu'une règle ne matchait jamais.
        """
        if not isinstance(data, dict):
            raise ValueError("la politique doit être un objet JSON")
        default = data.get("default", "allow")
        if default not in _ACTIONS:
            raise ValueError(f"default doit valoir un de {_ACTIONS}, reçu {default!r}")
        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            raise ValueError("rules doit être une liste")
        return cls(rules=tuple(cls._validate_rule(r, i) for i, r in enumerate(raw_rules)),
                   default=default)

    @staticmethod
    def _validate_rule(rule: Any, index: int) -> dict[str, Any]:
        where = f"rules[{index}]"
        if not isinstance(rule, dict):
            raise ValueError(f"{where} doit être un objet")
        tool = rule.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError(f"{where}.tool doit être un nom d'outil ou '*'")
        action = rule.get("action")
        if action not in _ACTIONS:
            raise ValueError(f"{where}.action doit valoir un de {_ACTIONS}, reçu {action!r}")
        when = rule.get("when", {})
        if not isinstance(when, dict):
            raise ValueError(f"{where}.when doit être un objet")
        for key, predicate in when.items():
            if key not in _CONTEXT_KEYS:
                raise ValueError(
                    f"{where}.when.{key} inconnu — clés acceptées : {_CONTEXT_KEYS}")
            if key == "args":
                if not isinstance(predicate, dict):
                    raise ValueError(f"{where}.when.args doit être un objet")
                for arg_name, arg_pred in predicate.items():
                    _check_operators(arg_pred, f"{where}.when.args.{arg_name}")
                    _avertir_prefixe_fragile(arg_name, arg_pred, f"{where}.when.args.{arg_name}")
            else:
                _check_operators(predicate, f"{where}.when.{key}")
        out = {"tool": tool, "action": action, "when": when}
        if rule.get("reason"):
            out["reason"] = str(rule["reason"])
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe — versionnable en git, diffable en revue, archivable."""
        return {"default": self.default, "rules": [dict(r) for r in self.rules]}

    # ── confinement monotone ────────────────────────────────────────────────

    def narrow(self, rules: Sequence[dict[str, Any]]) -> "ToolPolicySpec":
        """Ajoute des règles qui ne peuvent que RESTREINDRE. S'applique librement.

        Les nouvelles règles sont mises en tête, mais la précédence par action
        rend cela sans effet de bord : un `deny` gagne où qu'il soit.
        """
        valides = [self._validate_rule(r, i) for i, r in enumerate(rules)]
        fautives = [r["tool"] for r in valides if r["action"] not in _RESTRICTIVE]
        if fautives:
            raise ValueError(
                f"narrow() n'accepte que {_RESTRICTIVE} — règles élargissantes pour "
                f"{fautives}. Utilise expand() (qui demande une approbation)."
            )
        return ToolPolicySpec(rules=tuple(valides) + self.rules, default=self.default)

    def expand(
        self,
        rules: Sequence[dict[str, Any]] = (),
        *,
        default: str | None = None,
        approved: bool = False,
    ) -> "ToolPolicySpec":
        """Ajoute des droits (ou change le défaut) — approbation REQUISE.

        Lève `ApprovalRequired` tant que `approved=True` n'est pas passé : c'est
        la garantie de confinement. Un agent qui pourrait élargir sa propre
        politique en cours de route n'aurait aucune politique.
        """
        valides = [self._validate_rule(r, i) for i, r in enumerate(rules)]
        nouveau_defaut = default if default is not None else self.default
        if nouveau_defaut not in _ACTIONS:
            raise ValueError(f"default doit valoir un de {_ACTIONS}")
        if not approved:
            quoi = [f"{r['action']} {r['tool']}" for r in valides]
            if default is not None and default != self.default:
                quoi.append(f"default {self.default} -> {default}")
            raise ApprovalRequired(
                "élargissement de la politique d'outils soumis à approbation : "
                + (", ".join(quoi) or "(aucun changement)")
            )
        return ToolPolicySpec(rules=self.rules + tuple(valides), default=nouveau_defaut)

    # ── évaluation ──────────────────────────────────────────────────────────

    def compile(self) -> Callable[[Any], str | None]:
        """Rend un callable de la signature `tool_policy` : `(ctx) -> str | None`.

        `None` autorise, une chaîne refuse avec ce motif (le modèle la voit comme
        une erreur d'outil et replanifie), `ApprovalRequired` met le run en pause
        de façon reprenable — exactement les trois issues du hook natif.
        """
        spec = self

        def politique(ctx: Any) -> str | None:
            try:
                decision, reason = spec.decide(ctx)
            except Exception as exc:
                # Fail-CLOSED : une politique qui ne sait pas conclure refuse.
                _log.exception("ToolPolicySpec: évaluation impossible; refus")
                return f"policy error: {type(exc).__name__}: {exc}"
            if decision == "allow":
                return None
            if decision == "approve":
                raise ApprovalRequired(
                    reason or f"l'appel à `{getattr(ctx.call, 'name', '?')}` "
                              f"requiert une approbation humaine"
                )
            return reason or f"`{getattr(ctx.call, 'name', '?')}` refusé par la politique"

        return politique

    def decide(self, ctx: Any) -> tuple[str, str]:
        """`(action, motif)` pour un `ToolPolicyContext`. Utile pour tester à sec."""
        matches = [rule for rule in self.rules if self._rule_matches(rule, ctx)]
        for action in ("deny", "approve", "allow"):      # précédence explicite
            for rule in matches:
                if rule["action"] == action:
                    return action, rule.get("reason", "")
        return self.default, ""

    @staticmethod
    def _rule_matches(rule: dict[str, Any], ctx: Any) -> bool:
        name = getattr(ctx.call, "name", None)
        if rule["tool"] != "*" and rule["tool"] != name:
            return False
        arguments = getattr(ctx.call, "arguments", None) or {}
        spec = getattr(ctx, "spec", None)
        for key, predicate in rule["when"].items():
            if key == "args":
                for arg_name, arg_pred in predicate.items():
                    if not _match_predicate(arguments.get(arg_name), arg_pred):
                        return False
            elif key == "tainted":
                if not _match_predicate(bool(getattr(ctx, "tainted", False)), predicate):
                    return False
            elif key == "egress":
                if not _match_predicate(bool(getattr(ctx, "egress", False)), predicate):
                    return False
            elif key == "step":
                if not _match_predicate(getattr(ctx, "step", 0), predicate):
                    return False
            elif key == "permissions":
                perms = list(getattr(spec, "permissions", None) or [])
                if not _match_predicate(perms, predicate):
                    return False
            elif key == "source" and not _match_predicate(getattr(ctx, "source", "tool"), predicate):
                return False
        return True


def _motif_d_hote_valide(motif: str) -> bool:
    """Un motif `url_host` est un nom d'hôte (éventuellement `*.domaine`) ou un littéral IPv6 — rien d'autre."""
    nu = motif.strip()
    if re.search(r"[/@\\\s]", nu):
        return False
    if ":" in nu:
        try:
            ipaddress.IPv6Address(nu)
            return True
        except ValueError:
            return False
    return bool(nu) and nu != "*."


def _check_operators(predicate: Any, where: str) -> None:
    """Refuse un opérateur inconnu DÈS la construction (fail-closed précoce)."""
    if not isinstance(predicate, dict):
        return  # valeur brute = égalité, rien à valider
    for operator, expected in predicate.items():
        try:
            _match_operator(operator, None, None)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
        except Exception:
            pass  # l'opérateur existe ; l'échec vient des valeurs de test None
        # Une règle de CONFINEMENT mal écrite ne doit pas se contenter de ne jamais correspondre :
        # dans une règle `allow` sous `default: "allow"`, ce serait une ouverture silencieuse.
        if operator in ("path_within", "url_host") and not _liste_de_textes(expected):
            raise ValueError(
                f"{where}.{operator}: attendu un texte non vide ou une liste de textes, reçu {expected!r}")
        if operator == "url_host":
            for motif in _liste_de_textes(expected):
                if not _motif_d_hote_valide(motif):
                    raise ValueError(
                        f"{where}.url_host: {motif!r} n'est pas un NOM D'HÔTE (`api.exemple.fr` ou `*.exemple.fr`) : "
                        "ni schéma, ni port, ni chemin, ni identifiants. Un tel motif ne correspondrait jamais — "
                        "dans une règle `not`, il refuserait tout.")
        if operator == "not":
            _check_operators(expected, f"{where}.not")


# `starts_with` sur un argument qui ressemble à un chemin ou une URL : la 0.23.1 le SIGNALE (journal
# d'avertissement, aucun changement de comportement) — c'est le motif que notre doc enseignait.
_NOM_CHEMIN_OU_URL = re.compile(
    r"path|file|dir|folder|url|uri|href|link|endpoint|host|chemin|fichier|dossier|r[eé]pertoire|lien|adresse|cible"
    r"|emplacement|racine", re.IGNORECASE)


def _avertir_prefixe_fragile(arg_name: str, predicate: Any, where: str) -> None:
    if isinstance(predicate, dict) and "starts_with" in predicate and _NOM_CHEMIN_OU_URL.search(arg_name):
        # Une fois par argument et par processus : une politique reconstruite à chaque requête (une par
        # utilisateur) ne doit pas inonder le journal.
        warn_once(
            _log, f"policy.starts_with.{arg_name}",
            f"{where}: `starts_with` sur l'argument `{arg_name}` ne confine NI un chemin NI une URL "
            "(`..`, `@`, sous-domaine) — utilise `path_within` / `url_host`.")
