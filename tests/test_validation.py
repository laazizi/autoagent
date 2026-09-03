"""Validateur JSON Schema interne (0.21.0) — la promesse « zéro dépendance » prouvée.

Trois choses sont verrouillées ici :

  1. LA LIB S'IMPORTE SANS `jsonschema`. Mesuré dans un sous-processus : après
     `import autoagent`, le module n'est pas dans `sys.modules`. C'est la preuve
     de la première ligne du README, pas une affirmation.
  2. LE SOUS-ENSEMBLE ANNONCÉ EST COUVERT, et les MESSAGES gardent la forme que
     le modèle lisait déjà (« 'x' is a required property »…) : le préfixe
     `ValidationError:` est attendu par le test MCP et par les consommateurs.
  3. L'ÉQUIVALENCE AVEC `jsonschema` EST MESURÉE sur un corpus quand il est
     installé (extra `dev`) — valide/invalide doit coïncider pour chaque cas.
     Sans lui, ce test est SAUTÉ, pas réussi en silence.
"""

from __future__ import annotations

import subprocess
import sys
from typing import ClassVar

import pytest

from autoagent.registry import ToolRegistry, _validate_args
from autoagent.schema import ToolCall, ToolSpec
from autoagent.validation import check_schema, format_errors, validate


def _messages(instance, schema) -> list[str]:
    return [m for _, m in validate(instance, schema)]


class TestZeroDependance:
    def test_import_autoagent_ne_charge_pas_jsonschema(self) -> None:
        code = (
            "import sys, autoagent\n"
            "print('CHARGE' if 'jsonschema' in sys.modules else 'ABSENT')"
        )
        sortie = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, check=True).stdout.strip()
        assert sortie == "ABSENT", "la lib importe encore jsonschema quelque part"

    def test_le_registre_valide_sans_jsonschema(self) -> None:
        registre = ToolRegistry()
        registre.add(ToolSpec(name="t", description="t", input_schema={
            "type": "object", "properties": {"n": {"type": "integer"}},
            "required": ["n"], "additionalProperties": False}), lambda n: n * 2)
        assert registre.execute(ToolCall(id="1", name="t", arguments={"n": 4})).result == 8
        mauvais = registre.execute(ToolCall(id="2", name="t", arguments={"n": "x"}))
        assert not mauvais.ok and mauvais.error.startswith("ValidationError: ")


class TestTypes:
    @pytest.mark.parametrize("t,ok,ko", [
        ("string", "a", 1), ("integer", 3, 3.5), ("integer", 3.0, "3"),
        ("number", 2.5, "2.5"), ("boolean", True, 1), ("null", None, 0),
        ("array", [1], {"a": 1}), ("object", {}, []),
    ])
    def test_chaque_type(self, t, ok, ko) -> None:
        assert not validate(ok, {"type": t})
        assert validate(ko, {"type": t})

    def test_bool_n_est_ni_integer_ni_number(self) -> None:
        """Le piège classique : en Python `True == 1`. Pas en JSON Schema."""
        assert validate(True, {"type": "integer"})
        assert validate(False, {"type": "number"})

    def test_liste_de_types_avec_null(self) -> None:
        s = {"type": ["string", "null"]}
        assert not validate("a", s) and not validate(None, s)
        assert _messages(1, s) == ["1 is not of type 'string', 'null'"]

    def test_message_type(self) -> None:
        assert _messages(1, {"type": "string"}) == ["1 is not of type 'string'"]


class TestObjets:
    S: ClassVar[dict] = {"type": "object",
         "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
         "required": ["a"], "additionalProperties": False}

    def test_required(self) -> None:
        assert _messages({}, self.S) == ["'a' is a required property"]

    def test_propriete_imbriquee_et_emplacement(self) -> None:
        erreurs = validate({"a": "x"}, self.S)
        assert erreurs == [("a", "'x' is not of type 'integer'")]

    def test_additional_properties_false(self) -> None:
        assert _messages({"a": 1, "z": 0}, self.S) == [
            "Additional properties are not allowed ('z' was unexpected)"]

    def test_additional_properties_schema(self) -> None:
        s = {"type": "object", "additionalProperties": {"type": "integer"}}
        assert not validate({"x": 1}, s)
        assert validate({"x": "a"}, s) == [("x", "'a' is not of type 'integer'")]

    def test_pattern_properties(self) -> None:
        s = {"type": "object", "patternProperties": {"^n_": {"type": "integer"}},
             "additionalProperties": False}
        assert not validate({"n_1": 1}, s)
        assert validate({"n_1": "a"}, s)
        assert validate({"z": 1}, s), "hors motif ET additionalProperties=False"

    def test_min_max_properties(self) -> None:
        assert validate({}, {"minProperties": 1})
        assert validate({"a": 1, "b": 2}, {"maxProperties": 1})


class TestTableaux:
    def test_items(self) -> None:
        s = {"type": "array", "items": {"type": "integer"}}
        assert not validate([1, 2], s)
        assert validate([1, "x"], s) == [("1", "'x' is not of type 'integer'")]

    def test_prefix_items_puis_items(self) -> None:
        s = {"prefixItems": [{"type": "string"}], "items": {"type": "integer"}}
        assert not validate(["a", 1, 2], s)
        assert validate([1, 1], s) and validate(["a", "b"], s)

    def test_bornes_et_unicite(self) -> None:
        assert _messages([], {"minItems": 1}) == ["[] is too short"]
        assert _messages([1, 2], {"maxItems": 1}) == ["[1, 2] is too long"]
        assert _messages([1, 1], {"uniqueItems": True}) == ["[1, 1] has non-unique elements"]
        assert not validate([1, True], {"uniqueItems": True}), "1 et True sont distincts"


class TestChainesEtNombres:
    def test_longueurs_et_motif(self) -> None:
        assert _messages("ab", {"minLength": 3}) == ["'ab' is too short"]
        assert _messages("abcd", {"maxLength": 3}) == ["'abcd' is too long"]
        assert _messages("x1", {"pattern": "^[a-z]+$"}) == ["'x1' does not match '^[a-z]+$'"]
        assert not validate("abc", {"pattern": "^[a-z]+$"})

    def test_bornes_numeriques(self) -> None:
        assert _messages(0, {"minimum": 1}) == ["0 is less than the minimum of 1"]
        assert _messages(101, {"maximum": 100}) == ["101 is greater than the maximum of 100"]
        assert validate(1, {"exclusiveMinimum": 1})
        assert validate(10, {"exclusiveMaximum": 10})
        assert not validate(1.5, {"minimum": 1, "maximum": 2})

    def test_multiple_of(self) -> None:
        assert not validate(9, {"multipleOf": 3})
        assert validate(10, {"multipleOf": 3})
        assert not validate(0.3, {"multipleOf": 0.1}), "tolérance flottante"


class TestEnumConstCombinateurs:
    def test_enum_et_const(self) -> None:
        assert _messages("Z", {"enum": ["A", "B"]}) == ["'Z' is not one of ['A', 'B']"]
        assert _messages(2, {"const": 1}) == ["1 was expected"]

    def test_any_of(self) -> None:
        s = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert not validate("a", s) and not validate(1, s)
        assert _messages(1.5, s) == ["1.5 is not valid under any of the given schemas"]

    def test_one_of(self) -> None:
        s = {"oneOf": [{"type": "integer"}, {"minimum": 0}]}
        assert not validate(-1, s), "integer seulement"
        assert not validate(0.5, s), "minimum seulement"
        assert _messages(2, s) == ["2 is valid under each of the given schemas"]

    def test_all_of_et_not(self) -> None:
        assert validate("a", {"allOf": [{"type": "string"}, {"minLength": 2}]})
        assert not validate("ab", {"allOf": [{"type": "string"}, {"minLength": 2}]})
        assert validate(1, {"not": {"type": "integer"}})

    def test_schemas_booleens(self) -> None:
        assert not validate("anything", True)
        assert validate("x", False) == [("", "False schema does not allow 'x'")]


class TestRef:
    def test_ref_local_defs(self) -> None:
        s = {"$defs": {"n": {"type": "integer"}},
             "type": "object", "properties": {"a": {"$ref": "#/$defs/n"}}}
        assert not validate({"a": 1}, s)
        assert validate({"a": "x"}, s) == [("a", "'x' is not of type 'integer'")]

    def test_ref_legacy_definitions(self) -> None:
        s = {"definitions": {"n": {"type": "integer"}}, "$ref": "#/definitions/n"}
        assert not validate(1, s) and validate("x", s)

    def test_ref_insoluble_est_un_defaut_de_schema(self) -> None:
        assert check_schema({"$ref": "#/$defs/absent"}) == "Unresolvable JSON pointer: '#/$defs/absent'"

    def test_ref_distant_ignore_sans_planter(self) -> None:
        assert check_schema({"$ref": "https://x/y.json"}) is not None
        assert not validate(1, {"$ref": "https://x/y.json"}), "ignoré, jamais bloquant"


class TestValiditeDuSchema:
    def test_type_inconnu(self) -> None:
        """Ce qui avait révélé les schémas Gemini en MAJUSCULES (0.18.0)."""
        assert check_schema({"type": "OBJECT"}) == "Unknown type 'OBJECT'"
        assert check_schema({"type": "object", "properties": {"a": {"type": "STRING"}}}) \
            == "Unknown type 'STRING'"

    def test_regex_invalide(self) -> None:
        assert "invalid regular expression" in (check_schema({"pattern": "("}) or "")

    def test_schema_valide(self) -> None:
        assert check_schema({"type": "object", "properties": {"a": {"type": ["string", "null"]}},
                             "required": ["a"], "additionalProperties": False}) is None

    def test_mots_cles_inconnus_ignores(self) -> None:
        """`format`, `if/then`… ne bloquent ni la validité ni la validation."""
        s = {"type": "string", "format": "date-time", "if": {}, "x-custom": 1}
        assert check_schema(s) is None and not validate("n'importe quoi", s)


class TestFormatDesErreurs:
    def test_tri_par_emplacement_et_racine(self) -> None:
        texte = format_errors([("b", "m2"), ("", "m0"), ("a", "m1")])
        assert texte == "ValidationError: <root>: m0; a: m1; b: m2"

    def test_validate_args_conserve_le_contrat(self) -> None:
        assert _validate_args({}, None) is None
        assert _validate_args({"a": 1}, {"type": "object", "required": ["a"]}) is None
        assert _validate_args({}, {"type": "object", "required": ["a"]}) == \
            "ValidationError: <root>: 'a' is a required property"
        assert _validate_args({}, {"type": "NOPE"}) == \
            "SchemaError: tool input_schema is invalid: Unknown type 'NOPE'"


# ── Équivalence mesurée avec jsonschema, quand il est là ──────────────────────

jsonschema = pytest.importorskip("jsonschema", reason="extra dev absent : test différentiel sauté")

CORPUS: list[tuple[dict, list]] = [
    ({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
      "required": ["a"], "additionalProperties": False},
     [{"a": 1}, {"a": 1, "b": "x"}, {}, {"a": "1"}, {"a": 1, "z": 0}, {"a": 1.0}, {"a": True}]),
    ({"type": ["string", "null"], "minLength": 2},
     ["ab", None, "a", 1, ""]),
    ({"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 3, "uniqueItems": True},
     [[1], [1, 2.5], [], [1, 2, 3, 4], [1, 1], ["a"], [1, True]]),
    ({"enum": ["A", "B", 1]}, ["A", "B", 1, "C", 1.0, True]),
    ({"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]},
     ["s", 0, 5, -1, 1.5, None]),
    ({"oneOf": [{"type": "integer"}, {"minimum": 0}]}, [-1, 0.5, 2, "x"]),
    ({"type": "object", "patternProperties": {"^n_": {"type": "integer"}},
      "additionalProperties": False}, [{"n_1": 1}, {"n_1": "a"}, {"z": 1}, {}]),
    ({"type": "number", "multipleOf": 0.5, "exclusiveMaximum": 10}, [1, 1.5, 1.2, 10, 9.5]),
    ({"$defs": {"n": {"type": "integer"}}, "type": "object",
      "properties": {"a": {"$ref": "#/$defs/n"}}}, [{"a": 1}, {"a": "x"}, {}]),
    ({"prefixItems": [{"type": "string"}], "items": {"type": "integer"}},
     [["a", 1], ["a", "b"], [1], []]),
    ({"type": "string", "pattern": "^[a-z]{2,4}$"}, ["ab", "abcd", "a", "abcde", "AB"]),
    ({"not": {"type": "string"}}, [1, "a", None]),
    ({"const": {"k": [1, 2]}}, [{"k": [1, 2]}, {"k": [2, 1]}]),
]


@pytest.mark.parametrize("schema,instances", CORPUS)
def test_meme_verdict_que_jsonschema(schema, instances) -> None:
    ref = jsonschema.Draft202012Validator(schema)
    for inst in instances:
        attendu = ref.is_valid(inst)
        obtenu = not validate(inst, schema)
        assert obtenu == attendu, f"divergence sur {inst!r} contre {schema!r}"
