"""`ModelConfig.extra_body` — model-specific knobs `LLMRequest` does not map.

Why this exists: a voice pipeline needs Gemini's ``thinkingConfig`` to switch
reasoning OFF (measured on gemini-3.5-flash: 192 thought tokens and ~1.8 s for a
one-line answer, plus a truncated reply when ``max_tokens`` is tight). There was
no way to reach that knob, and the naive fix — letting callers pass a whole
``generationConfig`` — would silently drop the ``temperature`` and
``maxOutputTokens`` the request already set. Hence a DEEP merge.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from autoagent.providers.anthropic import AnthropicProvider
from autoagent.providers.base import deep_merge
from autoagent.providers.gemini import GeminiProvider
from autoagent.providers.openai import OpenAIProvider
from autoagent.schema import LLMRequest, Message, ModelConfig

REQUEST = LLMRequest(
    messages=[Message(role="user", content="hi")],
    temperature=0.3,
    max_tokens=400,
)

THINKING_OFF: dict[str, Any] = {"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}


class TestDeepMerge:
    def test_nested_dicts_keep_their_siblings(self) -> None:
        base = {"a": {"x": 1, "y": 2}}
        assert deep_merge(base, {"a": {"y": 3, "z": 4}}) == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_scalar_overlay_wins(self) -> None:
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_list_is_replaced_not_concatenated(self) -> None:
        assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}

    def test_dict_replaces_a_scalar(self) -> None:
        assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}

    def test_new_keys_are_added(self) -> None:
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


class TestGemini:
    def _config(self, **kw: Any) -> ModelConfig:
        return ModelConfig(provider="gemini", model="gemini-3.5-flash", api_key="k", **kw)

    def test_thinking_config_reaches_the_payload(self) -> None:
        payload = GeminiProvider(self._config(extra_body=THINKING_OFF))._build_payload(REQUEST)
        assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}

    def test_request_settings_survive_the_merge(self) -> None:
        # The whole point of merging deeply rather than overwriting.
        payload = GeminiProvider(self._config(extra_body=THINKING_OFF))._build_payload(REQUEST)
        gen = payload["generationConfig"]
        assert gen["temperature"] == 0.3 and gen["maxOutputTokens"] == 400

    def test_empty_extra_body_changes_nothing(self) -> None:
        plain = GeminiProvider(self._config())._build_payload(REQUEST)
        assert "thinkingConfig" not in plain["generationConfig"]
        assert set(plain["generationConfig"]) == {"temperature", "maxOutputTokens"}

    def test_payload_actually_sent_on_the_wire(self) -> None:
        captured: dict[str, Any] = {}

        def fake_post_json(url, payload, headers=None, timeout=None):
            captured["payload"] = payload
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

        provider = GeminiProvider(self._config(extra_body=THINKING_OFF))
        with patch("autoagent.providers.gemini.post_json", fake_post_json):
            assert provider.complete(REQUEST).content == "ok"
        assert captured["payload"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0


class TestOpenAIAndAnthropic:
    def test_openai_top_level_knob(self) -> None:
        config = ModelConfig(provider="openai", model="gpt-4o", api_key="k",
                             extra_body={"reasoning_effort": "low"})
        payload = OpenAIProvider(config)._build_payload(REQUEST)
        assert payload["reasoning_effort"] == "low"
        assert payload["temperature"] == 0.3        # untouched

    def test_anthropic_top_level_knob(self) -> None:
        config = ModelConfig(provider="anthropic", model="claude", api_key="k",
                             extra_body={"top_k": 5})
        payload = AnthropicProvider(config)._build_payload(REQUEST)
        assert payload["top_k"] == 5
        assert payload["temperature"] == 0.3

    def test_default_is_an_empty_dict_not_none(self) -> None:
        # Shared-mutable-default trap: each config must own its dict.
        a = ModelConfig(provider="openai", model="m")
        b = ModelConfig(provider="openai", model="m")
        a.extra_body["x"] = 1
        assert a.extra_body == {"x": 1} and b.extra_body == {}
