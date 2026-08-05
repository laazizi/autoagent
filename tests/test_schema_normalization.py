"""Normalisation des types de JSON Schema à la frontière `ToolSpec` (0.18.0).

Trouvé EN CONDITIONS RÉELLES : quand l'orchestrateur Gemini fabrique un outil
dynamique, il rédige l'`input_schema` dans SON dialecte — types en MAJUSCULES
(`OBJECT`, `INTEGER`). Conséquences avant le correctif :
  * `jsonschema` refusait de valider les arguments (« Unknown type 'OBJECT' »)
    → tout appel à l'outil créé échouait ;
  * le même schéma envoyé à OpenAI/Anthropic était rejeté → portabilité perdue.

Un schéma n'arrive pas toujours de `schema_from_callable` : il peut venir du
MODÈLE (outil dynamique) ou d'un serveur MCP tiers.
"""

from __future__ import annotations

from autoagent.registry import ToolRegistry
from autoagent.schema import ToolCall, ToolSpec, normalize_schema_types


class TestNormalisationUnitaire:
    def test_types_gemini_en_majuscules(self) -> None:
        assert normalize_schema_types({"type": "OBJECT"}) == {"type": "object"}
        assert normalize_schema_types({"type": "INTEGER"}) == {"type": "integer"}

    def test_recursif_dans_properties_et_items(self) -> None:
        source = {
            "type": "OBJECT",
            "properties": {
                "n": {"type": "INTEGER"},
                "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
                "profond": {"type": "OBJECT", "properties": {"ok": {"type": "BOOLEAN"}}},
            },
        }
        out = normalize_schema_types(source)
        assert out["type"] == "object"
        assert out["properties"]["n"]["type"] == "integer"
        assert out["properties"]["tags"]["items"]["type"] == "string"
        assert out["properties"]["profond"]["properties"]["ok"]["type"] == "boolean"

    def test_liste_de_types(self) -> None:
        assert normalize_schema_types({"type": ["STRING", "NULL"]})["type"] == ["string", "null"]

    def test_idempotent(self) -> None:
        deja = {"type": "object", "properties": {"a": {"type": "string"}}}
        assert normalize_schema_types(deja) == deja

    def test_conservateur_ne_touche_pas_le_reste(self) -> None:
        """Seuls les 7 types JSON Schema sont concernés — rien d'autre."""
        source = {"type": "MonTypeMaison", "description": "NE PAS TOUCHER",
                  "enum": ["ROUGE", "VERT"], "format": "DATE-TIME"}
        out = normalize_schema_types(source)
        assert out["type"] == "MonTypeMaison"        # type inconnu : laissé au validateur
        assert out["description"] == "NE PAS TOUCHER"
        assert out["enum"] == ["ROUGE", "VERT"]      # les valeurs d'enum ne sont pas des types
        assert out["format"] == "DATE-TIME"

    def test_ne_mute_pas_l_entree(self) -> None:
        source = {"type": "OBJECT"}
        normalize_schema_types(source)
        assert source == {"type": "OBJECT"}


class TestToolSpec:
    def test_normalise_a_la_construction(self) -> None:
        spec = ToolSpec(name="t", description="d", input_schema={
            "type": "OBJECT", "properties": {"n": {"type": "INTEGER"}}, "required": ["n"]})
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["properties"]["n"]["type"] == "integer"

    def test_portabilite_inter_provider(self) -> None:
        """Le schéma exporté vers CHAQUE provider est du JSON Schema valide."""
        spec = ToolSpec(name="t", description="d",
                        input_schema={"type": "OBJECT", "properties": {"s": {"type": "STRING"}}})
        assert spec.as_openai_tool()["function"]["parameters"]["type"] == "object"
        assert spec.as_anthropic_tool()["input_schema"]["type"] == "object"


class TestValidationDesArguments:
    def test_un_outil_a_schema_gemini_est_appelable(self) -> None:
        """LE bug d'origine : un outil créé par le modèle refusait tous les appels."""
        registry = ToolRegistry()

        def compter(n: int) -> dict:
            return {"resultat": n * 2}

        registry.add(
            spec=ToolSpec(name="compter", description="Double un entier.",
                          input_schema={"type": "OBJECT",
                                        "properties": {"n": {"type": "INTEGER"}},
                                        "required": ["n"]}),
            handler=compter,
        )
        res = registry.execute(ToolCall(id="c1", name="compter", arguments={"n": 21}))
        assert res.ok is True, f"appel refusé : {res.error}"
        assert res.result == {"resultat": 42}

    def test_arguments_invalides_toujours_refuses(self) -> None:
        """La normalisation ne relâche PAS la validation."""
        registry = ToolRegistry()
        registry.add(
            spec=ToolSpec(name="compter", description="d",
                          input_schema={"type": "OBJECT",
                                        "properties": {"n": {"type": "INTEGER"}},
                                        "required": ["n"]}),
            handler=lambda n: n,
        )
        res = registry.execute(ToolCall(id="c1", name="compter", arguments={"n": "pas un entier"}))
        assert res.ok is False and "ValidationError" in (res.error or "")
