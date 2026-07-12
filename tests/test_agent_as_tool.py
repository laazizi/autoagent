"""Agent.as_tool() — la primitive multi-agents minimale (0.10.0).

Un agent s'expose comme OUTIL d'un autre : hiérarchie superviseur ->
spécialistes sans framework. Délégation sans état, erreurs du sous-agent
remontées comme erreurs d'outil, coût visible.
"""

from __future__ import annotations

from autoagent import Agent
from autoagent.errors import ProviderError
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, TokenUsage, ToolCall


class _ScriptedProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.config = ModelConfig(provider="fake", model="fake-model")
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _delegating_parent(tool_name: str) -> _ScriptedProvider:
    return _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name=tool_name, arguments={"request": "compte les camions"})
                ],
            ),
            LLMResponse(content="Rapport final basé sur l'expert."),
        ]
    )


def test_supervisor_delegates_to_specialist() -> None:
    expert_provider = _ScriptedProvider(
        [LLMResponse(content="42 camions", usage=TokenUsage(input_tokens=30, output_tokens=8))]
    )
    expert = Agent(expert_provider, system_prompt="Tu es l'expert comptage.")

    parent = Agent(_delegating_parent("analyser_comptage"))
    parent.add_tool(
        expert.as_tool(
            name="analyser_comptage",
            description="Délègue les questions de comptage à l'expert.",
        )
    )

    result = parent.run("Combien de camions sur l'A7 ?")

    assert result.output == "Rapport final basé sur l'expert."
    # L'expert a reçu une conversation FRAÎCHE : son system prompt + la demande.
    sub_request = expert_provider.requests[0]
    assert sub_request.messages[0].content == "Tu es l'expert comptage."
    assert sub_request.messages[1].content == "compte les camions"
    # Le résultat d'outil porte output + steps + tokens (coût visible).
    tool_msg = next(m for m in result.messages if m.role == "tool")
    assert '"output": "42 camions"' in tool_msg.content
    assert '"tokens": 38' in tool_msg.content


def test_specialist_failure_surfaces_as_tool_error_not_crash() -> None:
    expert_provider = _ScriptedProvider(
        [ProviderError("HTTP 500 from api", status_code=500, retryable=True)]  # type: ignore[list-item]
    )
    expert = Agent(expert_provider)

    parent = Agent(
        _ScriptedProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="expert", arguments={"request": "x"})],
                ),
                LLMResponse(content="Je conclus sans l'expert (il est en panne)."),
            ]
        )
    )
    parent.add_tool(expert.as_tool(name="expert", description="d"))

    result = parent.run("go")  # le run PARENT ne crashe pas
    tool_msg = next(m for m in result.messages if m.role == "tool")
    assert '"ok": false' in tool_msg.content and "ProviderError" in tool_msg.content
    assert result.output.startswith("Je conclus")


def test_spec_shape() -> None:
    expert = Agent(_ScriptedProvider([]))
    handler = expert.as_tool(name="expert_x", description="ma description")
    spec = handler.__autoagent_tool_spec__  # type: ignore[attr-defined]
    assert spec.name == "expert_x"
    assert spec.description == "ma description"
    assert spec.input_schema["required"] == ["request"]
