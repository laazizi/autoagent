"""Tests for the schema dataclasses.

These cover:
  * `ToolSpec.as_openai_tool / as_anthropic_tool / as_gemini_declaration`
    — the wire formats each provider expects. A silent change in these
    methods would break production calls without any other test
    catching it.
  * `ModelConfig.resolved_api_key` — env-var fallback chain.
"""

from __future__ import annotations

import pytest

from autoagent.schema import (
    DEFAULT_API_KEY_ENVS,
    ImageAttachment,
    ModelConfig,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# ImageAttachment.as_base64 — providers (Anthropic/Gemini) depend on this
# returning correct (mime, raw_base64), even on edge-case data URLs.
# ---------------------------------------------------------------------------


class TestImageAttachmentAsBase64:
    def test_canonical_data_url(self) -> None:
        att = ImageAttachment(data="data:image/png;base64,iVBORw0KGgo=")
        assert att.as_base64() == ("image/png", "iVBORw0KGgo=")

    def test_canonical_data_url_with_charset(self) -> None:
        """Data URL with extra parameters before `;base64` (rare but legal)."""
        att = ImageAttachment(data="data:image/png;charset=utf-8;base64,iVBORw0KGgo=")
        mime, payload = att.as_base64()
        assert mime == "image/png"
        assert payload == "iVBORw0KGgo="

    def test_raw_base64_with_mime_type(self) -> None:
        att = ImageAttachment(data="iVBORw0KGgo=", mime_type="image/png")
        assert att.as_base64() == ("image/png", "iVBORw0KGgo=")

    def test_raw_base64_without_mime_raises(self) -> None:
        att = ImageAttachment(data="iVBORw0KGgo=")
        with pytest.raises(ValueError, match="mime_type is required"):
            att.as_base64()

    def test_non_base64_data_url_rejected(self) -> None:
        """data:image/png,<URL-encoded> is technically valid HTML5 but
        Anthropic/Gemini expect raw base64 — we reject explicitly rather
        than silently corrupting the LLM request."""
        att = ImageAttachment(data="data:image/png,%89PNG%0D%0A")
        with pytest.raises(ValueError, match="base64-encoded"):
            att.as_base64()

    def test_remote_url_rejected(self) -> None:
        att = ImageAttachment(data="https://example.com/cat.png")
        with pytest.raises(ValueError, match="Remote URLs"):
            att.as_base64()

    def test_http_url_rejected(self) -> None:
        att = ImageAttachment(data="http://example.com/cat.png")
        with pytest.raises(ValueError, match="Remote URLs"):
            att.as_base64()


class TestImageAttachmentAsDataUrl:
    def test_already_data_url_returned_as_is(self) -> None:
        url = "data:image/png;base64,iVBORw0KGgo="
        assert ImageAttachment(data=url).as_data_url() == url

    def test_remote_url_returned_as_is(self) -> None:
        url = "https://example.com/cat.png"
        assert ImageAttachment(data=url).as_data_url() == url

    def test_raw_base64_wrapped_with_mime(self) -> None:
        att = ImageAttachment(data="iVBORw0KGgo=", mime_type="image/png")
        assert att.as_data_url() == "data:image/png;base64,iVBORw0KGgo="

    def test_raw_base64_without_mime_raises(self) -> None:
        att = ImageAttachment(data="iVBORw0KGgo=")
        with pytest.raises(ValueError, match="mime_type is required"):
            att.as_data_url()

    def test_empty_data_raises(self) -> None:
        att = ImageAttachment(data="")
        with pytest.raises(ValueError, match="empty"):
            att.as_data_url()


@pytest.fixture
def echo_spec() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo a string back to the caller.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        permissions=["network"],
    )


# ---------------------------------------------------------------------------
# Wire formats per provider
# ---------------------------------------------------------------------------


class TestToolSpecOpenAI:
    def test_shape(self, echo_spec: ToolSpec) -> None:
        wire = echo_spec.as_openai_tool()
        assert wire == {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a string back to the caller.",
                "parameters": echo_spec.input_schema,
            },
        }

    def test_parameters_is_the_same_dict_reference(self, echo_spec: ToolSpec) -> None:
        """The provider hands the dict to json.dumps; aliasing is fine
        but we want to know if it ever stops being the source object."""
        wire = echo_spec.as_openai_tool()
        assert wire["function"]["parameters"] is echo_spec.input_schema


class TestToolSpecAnthropic:
    def test_shape(self, echo_spec: ToolSpec) -> None:
        wire = echo_spec.as_anthropic_tool()
        assert wire == {
            "name": "echo",
            "description": "Echo a string back to the caller.",
            "input_schema": echo_spec.input_schema,
        }


class TestToolSpecGemini:
    def test_shape_strips_additional_properties(self, echo_spec: ToolSpec) -> None:
        """Gemini's tool API rejects `additionalProperties` — verify it's
        stripped from the wire output even though the source schema sets it."""
        # The echo fixture intentionally has additionalProperties: False.
        assert echo_spec.input_schema["additionalProperties"] is False
        wire = echo_spec.as_gemini_declaration()
        assert wire["name"] == "echo"
        assert wire["description"] == "Echo a string back to the caller."
        # additionalProperties stripped...
        assert "additionalProperties" not in wire["parameters"]
        # ...but everything else (type, properties, required) preserved.
        assert wire["parameters"]["type"] == "object"
        assert wire["parameters"]["properties"] == echo_spec.input_schema["properties"]
        assert wire["parameters"]["required"] == ["value"]

    def test_strips_nested_additional_properties(self) -> None:
        """The sanitiser walks INTO nested schemas (properties.<x>, items)."""
        spec = ToolSpec(
            name="bulk_create",
            description="Create many items.",
            input_schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "additionalProperties": False,  # nested offender
                        },
                    },
                },
                "required": ["items"],
                "additionalProperties": False,  # top-level offender
            },
        )
        params = spec.as_gemini_declaration()["parameters"]
        assert "additionalProperties" not in params
        assert "additionalProperties" not in params["properties"]["items"]["items"]
        # Structure intact
        assert params["properties"]["items"]["items"]["properties"]["name"]["type"] == "string"

    def test_strips_other_jsonschema_keywords(self) -> None:
        """Other JSON Schema fields Gemini doesn't know ($schema, $id,
        definitions, $ref, $defs, patternProperties, unevaluatedProperties)
        are all stripped."""
        spec = ToolSpec(
            name="x",
            description="x",
            input_schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "https://example.com/schemas/x",
                "type": "object",
                "definitions": {"foo": {"type": "string"}},
                "$defs": {"bar": {"type": "string"}},
                "patternProperties": {"^x": {"type": "string"}},
                "unevaluatedProperties": False,
                "properties": {"a": {"type": "string"}},
            },
        )
        params = spec.as_gemini_declaration()["parameters"]
        for forbidden in (
            "$schema",
            "$id",
            "definitions",
            "$defs",
            "patternProperties",
            "unevaluatedProperties",
        ):
            assert forbidden not in params, f"{forbidden} should be stripped"
        assert params["properties"]["a"]["type"] == "string"

    def test_preserves_valid_openapi_fields(self) -> None:
        """`enum`, `format`, `minimum`, `maximum`, `description` and other
        OpenAPI 3.0 fields that Gemini DOES accept must pass through."""
        spec = ToolSpec(
            name="search",
            description="Search.",
            input_schema={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["a", "b", "c"],
                        "description": "Kind of search.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "since": {"type": "string", "format": "date-time"},
                },
                "required": ["kind"],
            },
        )
        params = spec.as_gemini_declaration()["parameters"]
        assert params["properties"]["kind"]["enum"] == ["a", "b", "c"]
        assert params["properties"]["kind"]["description"] == "Kind of search."
        assert params["properties"]["limit"]["minimum"] == 1
        assert params["properties"]["limit"]["maximum"] == 100
        assert params["properties"]["since"]["format"] == "date-time"
        assert params["required"] == ["kind"]


# ---------------------------------------------------------------------------
# ModelConfig.resolved_api_key
# ---------------------------------------------------------------------------


class TestResolvedApiKey:
    def test_explicit_api_key_used_first(self) -> None:
        config = ModelConfig(provider="openai", model="gpt-4", api_key="explicit-key")
        assert config.resolved_api_key() == "explicit-key"

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        config = ModelConfig(provider="openai", model="gpt-4")
        assert config.resolved_api_key() == "from-env"

    def test_custom_api_key_env_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "from-custom-env")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = ModelConfig(provider="openai", model="gpt-4", api_key_env="MY_KEY")
        assert config.resolved_api_key() == "from-custom-env"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for env in DEFAULT_API_KEY_ENVS.values():
            monkeypatch.delenv(env, raising=False)
        config = ModelConfig(provider="openai", model="gpt-4")
        with pytest.raises(ValueError, match="Missing API key"):
            config.resolved_api_key()

    def test_provider_aliases_resolve_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `google` is an alias for gemini and must use GEMINI_API_KEY.
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-from-env")
        config = ModelConfig(provider="google", model="gemini-2")
        assert config.resolved_api_key() == "gemini-from-env"

    def test_unknown_provider_without_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = ModelConfig(provider="cohere", model="m")
        with pytest.raises(ValueError):
            config.resolved_api_key()
