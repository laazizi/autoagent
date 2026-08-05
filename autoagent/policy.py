"""Politique d'outils DÉCLARATIVE — des données, plus du code (0.18.0).

`tool_policy` est une fonction Python : puissante, mais elle ne se versionne pas
en revue, ne se relit pas en diff, ne se transporte pas dans un snapshot et ne se
génère pas. `ToolPolicySpec` exprime la même chose en JSON, et `compile()` en fait
un callable de la signature `tool_policy` EXISTANTE — le code de production ne
bouge donc pas d'une ligne.

    spec = ToolPolicySpec.from_dict({
        "default": "allow",
        "rules": [
            {"tool": "write_file", "action": "allow",
             "when": {"args": {"path": {"starts_with": "rapports/"}}}},
            {"tool": "write_file", "action": "deny",
             "reason": "écriture limitée à rapports/"},
            {"tool": "*", "action": "deny", "when": {"tainted": True, "egress": True},
             "reason": "sortie interdite après lecture de contenu non fiable"},
            {"tool": "supprimer_compte", "action": "approve"},
        ],
    })
    agent = Agent(provider, tool_policy=spec.compile())

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

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import ApprovalRequired
from .logging import get_logger

__all__ = ["ToolPolicySpec"]

_log = get_logger("policy")

_ACTIONS = ("allow", "deny", "approve")
_RESTRICTIVE = ("deny", "approve")          # ne peuvent que retirer des droits
_CONTEXT_KEYS = ("args", "tainted", "egress", "step", "permissions")


def _as_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


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
        return True


def _check_operators(predicate: Any, where: str) -> None:
    """Refuse un opérateur inconnu DÈS la construction (fail-closed précoce)."""
    if not isinstance(predicate, dict):
        return  # valeur brute = égalité, rien à valider
    for operator in predicate:
        try:
            _match_operator(operator, None, None)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
        except Exception:
            pass  # l'opérateur existe ; l'échec vient des valeurs de test None
