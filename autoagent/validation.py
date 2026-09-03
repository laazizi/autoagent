"""Validateur JSON Schema interne — la promesse « zéro dépendance » rendue VRAIE (0.21.0).

Jusqu'ici la bibliothèque annonçait « zero dependencies for the core » et
déclarait `jsonschema>=4.0` — qui installe six paquets, dont un binaire Rust
compilé. La première ligne du README ne survivait pas à `pip install`. Ce module
remplace `jsonschema` pour le seul usage que la lib en faisait : valider les
ARGUMENTS d'un appel d'outil contre son `input_schema`, avant d'exécuter le code
de l'hôte.

Ce qui est couvert — le sous-ensemble que la lib GÉNÈRE (`schema_from_callable`)
plus ce qu'on rencontre dans les schémas écrits par un modèle ou fournis par un
serveur MCP :

    type (simple ou liste, avec "null")      enum · const
    properties · required                    additionalProperties (bool ou schéma)
    patternProperties · propertyNames        minProperties · maxProperties
    items · prefixItems                      minItems · maxItems · uniqueItems
    minimum · maximum · exclusive*           multipleOf
    minLength · maxLength · pattern          anyOf · oneOf · allOf · not
    $ref local (#/$defs/… , #/definitions/…) schémas booléens (true / false)

Ce qui est volontairement IGNORÉ, comme le fait `jsonschema` par défaut :
`format` (annotation, non vérifiée), `default`, `description`, `title`,
`examples`, `$schema`, `$id`, `deprecated`, `readOnly`. Et ce qui n'est PAS
implémenté — `if`/`then`/`else`, `contains`, `dependentRequired`,
`unevaluated*`, `$ref` distant — est ignoré aussi : un mot-clé inconnu ne fait
jamais échouer une validation. C'est un choix fail-OPEN sur la QUALITÉ des
arguments, pas sur la sécurité : la frontière de sécurité, c'est `tool_policy`,
la teinte et le bac à sable, pas ce fichier.

Deux comportements de `jsonschema` sont reproduits exprès :

  * la VALIDITÉ DU SCHÉMA est vérifiée (`check_schema`) : un `type` inconnu ou
    un `$ref` insoluble est signalé — c'est ce qui avait révélé les schémas
    Gemini en MAJUSCULES (0.18.0) ;
  * la FORME DES MESSAGES est la même (« 'x' is a required property », « 1 is
    not of type 'string' »…) : le modèle les lit dans le résultat d'outil, et
    les tests des consommateurs peuvent les attendre.

`tests/test_validation.py` compare ce module à `jsonschema` sur un corpus quand
celui-ci est installé (extra `dev`) — l'équivalence sur le sous-ensemble est
donc mesurée, pas affirmée.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["check_schema", "validate", "format_errors"]

_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})


# ── validité du schéma ──────────────────────────────────────────────────────

def check_schema(schema: Any) -> str | None:
    """Rend une phrase décrivant le premier défaut du schéma, ou None.

    Volontairement peu bavard : on attrape ce qui rendrait la validation
    absurde (un `type` inconnu, une regex invalide, un `$ref` insoluble, un
    nœud qui n'est ni un dict ni un booléen), pas la conformité au méta-schéma.
    """
    return _check(schema, schema, "#")


def _check(node: Any, root: Any, path: str) -> str | None:
    if isinstance(node, bool):
        return None
    if not isinstance(node, dict):
        return f"{path}: a schema must be an object or a boolean, got {type(node).__name__}"

    types = node.get("type")
    if types is not None:
        liste = types if isinstance(types, list) else [types]
        for t in liste:
            if not isinstance(t, str) or t not in _TYPES:
                return f"Unknown type {t!r}"

    for cle in ("pattern",):
        motif = node.get(cle)
        if motif is not None:
            try:
                re.compile(motif)
            except (re.error, TypeError) as exc:
                return f"{path}/{cle}: invalid regular expression: {exc}"

    ref = node.get("$ref")
    if isinstance(ref, str) and _resolve(ref, root) is _UNRESOLVED:
        return f"Unresolvable JSON pointer: {ref!r}"

    for cle in ("properties", "patternProperties", "$defs", "definitions"):
        sous = node.get(cle)
        if isinstance(sous, dict):
            for nom, s in sous.items():
                pb = _check(s, root, f"{path}/{cle}/{nom}")
                if pb:
                    return pb
    for cle in ("items", "additionalProperties", "propertyNames", "not"):
        if cle in node:
            pb = _check(node[cle], root, f"{path}/{cle}")
            if pb:
                return pb
    for cle in ("anyOf", "oneOf", "allOf", "prefixItems"):
        sous = node.get(cle)
        if isinstance(sous, list):
            for i, s in enumerate(sous):
                pb = _check(s, root, f"{path}/{cle}/{i}")
                if pb:
                    return pb
    return None


# ── résolution des $ref locaux ──────────────────────────────────────────────

_UNRESOLVED = object()


def _resolve(ref: str, root: Any) -> Any:
    if not ref.startswith("#"):
        return _UNRESOLVED                      # $ref distant : hors périmètre
    node = root
    for brut in ref[1:].split("/"):
        if brut == "":
            continue
        cle = brut.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and cle in node:
            node = node[cle]
        elif isinstance(node, list) and cle.isdigit() and int(cle) < len(node):
            node = node[int(cle)]
        else:
            return _UNRESOLVED
    return node


# ── typage JSON ─────────────────────────────────────────────────────────────

def _is_type(value: Any, t: str) -> bool:
    if t == "null":
        return value is None
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":                          # 2020-12 : 1.0 EST un integer
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    return False


def _repr(value: Any) -> str:
    """Le rendu de `jsonschema` : repr Python, tel que le modèle le lisait déjà."""
    return repr(value)


# ── validation ──────────────────────────────────────────────────────────────

def validate(instance: Any, schema: Any, *, root: Any = None) -> list[tuple[str, str]]:
    """Rend la liste des (emplacement, message). Vide = valide.

    L'emplacement est le chemin pointé dans l'INSTANCE (« a.b.0 »), ou vide
    à la racine — le registre le rend comme `<root>`.
    """
    erreurs: list[tuple[str, str]] = []
    _validate(instance, schema, schema if root is None else root, (), erreurs)
    return erreurs


def _validate(inst: Any, schema: Any, root: Any, path: tuple, out: list) -> None:
    if schema is True or schema == {}:
        return
    if schema is False:
        out.append((_loc(path), f"False schema does not allow {_repr(inst)}"))
        return
    if not isinstance(schema, dict):
        return                                  # défaut de schéma : check_schema l'a dit

    ref = schema.get("$ref")
    if isinstance(ref, str):
        cible = _resolve(ref, root)
        if cible is not _UNRESOLVED:
            _validate(inst, cible, root, path, out)

    types = schema.get("type")
    if types is not None:
        liste = types if isinstance(types, list) else [types]
        if not any(_is_type(inst, t) for t in liste if isinstance(t, str)):
            attendu = (repr(liste[0]) if len(liste) == 1
                       else ", ".join(repr(t) for t in liste))
            out.append((_loc(path), f"{_repr(inst)} is not of type {attendu}"))
            return                              # le reste n'aurait pas de sens

    if "enum" in schema and not any(_json_equal(inst, v) for v in _as_list(schema["enum"])):
        out.append((_loc(path), f"{_repr(inst)} is not one of {_repr(schema['enum'])}"))
    if "const" in schema and not _json_equal(inst, schema["const"]):
        out.append((_loc(path), f"{_repr(schema['const'])} was expected"))

    if isinstance(inst, dict):
        _validate_object(inst, schema, root, path, out)
    elif isinstance(inst, list):
        _validate_array(inst, schema, root, path, out)
    elif isinstance(inst, str):
        _validate_string(inst, schema, path, out)
    elif isinstance(inst, (int, float)) and not isinstance(inst, bool):
        _validate_number(inst, schema, path, out)

    _validate_combinators(inst, schema, root, path, out)


def _validate_object(inst: dict, schema: dict, root: Any, path: tuple, out: list) -> None:
    brut = schema.get("properties")
    props: dict[str, Any] = brut if isinstance(brut, dict) else {}
    for nom in _as_list(schema.get("required", [])):
        if nom not in inst:
            out.append((_loc(path), f"{nom!r} is a required property"))
    for nom, s in props.items():
        if nom in inst:
            _validate(inst[nom], s, root, (*path, nom), out)

    motifs = schema.get("patternProperties")
    couverts: set[str] = {nom for nom in inst if nom in props}
    if isinstance(motifs, dict):
        for motif, s in motifs.items():
            try:
                rx = re.compile(motif)
            except re.error:
                continue
            for nom in inst:
                if rx.search(nom):
                    couverts.add(nom)
                    _validate(inst[nom], s, root, (*path, nom), out)

    extra = schema.get("additionalProperties", True)
    if extra is not True:
        restes = [nom for nom in inst if nom not in couverts]
        if extra is False and restes:
            liste = ", ".join(repr(n) for n in restes)
            mot = "was" if len(restes) == 1 else "were"
            out.append((_loc(path), f"Additional properties are not allowed ({liste} {mot} unexpected)"))
        elif isinstance(extra, dict):
            for nom in restes:
                _validate(inst[nom], extra, root, (*path, nom), out)

    noms = schema.get("propertyNames")
    if noms is not None:
        for nom in inst:
            _validate(nom, noms, root, path, out)

    mn, mx = schema.get("minProperties"), schema.get("maxProperties")
    if isinstance(mn, int) and len(inst) < mn:
        out.append((_loc(path), f"{_repr(inst)} does not have enough properties"))
    if isinstance(mx, int) and len(inst) > mx:
        out.append((_loc(path), f"{_repr(inst)} has too many properties"))


def _validate_array(inst: list, schema: dict, root: Any, path: tuple, out: list) -> None:
    prefixe = schema.get("prefixItems")
    n_prefixe = 0
    if isinstance(prefixe, list):
        for i, (val, s) in enumerate(zip(inst, prefixe, strict=False)):
            _validate(val, s, root, (*path, i), out)
        n_prefixe = len(prefixe)
    items = schema.get("items")
    if items is not None:
        for i in range(n_prefixe, len(inst)):
            _validate(inst[i], items, root, (*path, i), out)

    mn, mx = schema.get("minItems"), schema.get("maxItems")
    if isinstance(mn, int) and len(inst) < mn:
        out.append((_loc(path), f"{_repr(inst)} is too short"))
    if isinstance(mx, int) and len(inst) > mx:
        out.append((_loc(path), f"{_repr(inst)} is too long"))
    if schema.get("uniqueItems") and _has_duplicates(inst):
        out.append((_loc(path), f"{_repr(inst)} has non-unique elements"))


def _validate_string(inst: str, schema: dict, path: tuple, out: list) -> None:
    mn, mx = schema.get("minLength"), schema.get("maxLength")
    if isinstance(mn, int) and len(inst) < mn:
        out.append((_loc(path), f"{_repr(inst)} is too short"))
    if isinstance(mx, int) and len(inst) > mx:
        out.append((_loc(path), f"{_repr(inst)} is too long"))
    motif = schema.get("pattern")
    if isinstance(motif, str):
        try:
            if not re.search(motif, inst):
                out.append((_loc(path), f"{_repr(inst)} does not match {motif!r}"))
        except re.error:
            pass                                # check_schema l'a signalé


def _validate_number(inst: float, schema: dict, path: tuple, out: list) -> None:
    mn, mx = schema.get("minimum"), schema.get("maximum")
    emn, emx = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
    if isinstance(mn, (int, float)) and inst < mn:
        out.append((_loc(path), f"{_repr(inst)} is less than the minimum of {_repr(mn)}"))
    if isinstance(mx, (int, float)) and inst > mx:
        out.append((_loc(path), f"{_repr(inst)} is greater than the maximum of {_repr(mx)}"))
    if isinstance(emn, (int, float)) and inst <= emn:
        out.append((_loc(path), f"{_repr(inst)} is less than or equal to the minimum of {_repr(emn)}"))
    if isinstance(emx, (int, float)) and inst >= emx:
        out.append((_loc(path), f"{_repr(inst)} is greater than or equal to the maximum of {_repr(emx)}"))
    mult = schema.get("multipleOf")
    if isinstance(mult, (int, float)) and mult and abs((inst / mult) - round(inst / mult)) > 1e-9:
        out.append((_loc(path), f"{_repr(inst)} is not a multiple of {_repr(mult)}"))


def _validate_combinators(inst: Any, schema: dict, root: Any, path: tuple, out: list) -> None:
    sous = schema.get("allOf")
    if isinstance(sous, list):
        for s in sous:
            _validate(inst, s, root, path, out)

    sous = schema.get("anyOf")
    if isinstance(sous, list) and sous and not any(_valid(inst, s, root) for s in sous):
        out.append((_loc(path), f"{_repr(inst)} is not valid under any of the given schemas"))

    sous = schema.get("oneOf")
    if isinstance(sous, list) and sous:
        ok = [i for i, s in enumerate(sous) if _valid(inst, s, root)]
        if not ok:
            out.append((_loc(path), f"{_repr(inst)} is not valid under any of the given schemas"))
        elif len(ok) > 1:
            out.append((_loc(path), f"{_repr(inst)} is valid under each of the given schemas"))

    if "not" in schema and _valid(inst, schema["not"], root):
        out.append((_loc(path), f"{_repr(inst)} should not be valid under {_repr(schema['not'])}"))


def _valid(inst: Any, schema: Any, root: Any) -> bool:
    tampon: list = []
    _validate(inst, schema, root, (), tampon)
    return not tampon


# ── utilitaires ─────────────────────────────────────────────────────────────

def _loc(path: tuple) -> str:
    return ".".join(str(p) for p in path)


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else []


def _json_equal(a: Any, b: Any) -> bool:
    """Égalité au sens JSON Schema, pas au sens Python.

    Le piège que le test différentiel a attrapé : en Python `True == 1`, donc
    `True in [1]`. En JSON Schema un booléen n'est jamais égal à un nombre ;
    en revanche `1.0` et `1` SONT égaux (même valeur numérique).
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _has_duplicates(items: list) -> bool:
    vus: list = []
    for it in items:
        if any(_json_equal(it, v) for v in vus):
            return True
        vus.append(it)
    return False


def format_errors(errors: list[tuple[str, str]]) -> str:
    """Le rendu exact que le registre produisait avec `jsonschema` :
    « ValidationError: <emplacement>: <message>; … », emplacements triés."""
    parts = [f"{loc or '<root>'}: {msg}" for loc, msg in sorted(errors, key=lambda e: e[0])]
    return "ValidationError: " + "; ".join(parts)
