__all__ = [
    "AgentCancelled",
    "AutoAgentError",
    "MaxStepsExceeded",
    "ProviderError",
    "TokenBudgetExceeded",
    "ToolError",
    "ToolValidationError",
]


class AutoAgentError(Exception):
    """Base error for autoagent."""


class AgentCancelled(AutoAgentError):
    """Raised when an agent run is cancelled cooperatively via `cancel_token`.

    The lib checks the token at the start of every loop iteration. When the
    token is set, the next iteration raises `AgentCancelled` instead of
    issuing a new provider call. Pre-existing tool calls in flight are not
    interrupted — cancellation happens at the next safe boundary.
    """


class ProviderError(AutoAgentError):
    """Raised when an LLM provider request fails.

    Attributes:
        status_code: HTTP status when the failure came from an HTTP error
            response (``None`` for network-level failures — DNS, timeout,
            connection reset — and for non-HTTP failures like bad JSON).
        retryable: ``True`` when the failure class is worth retrying
            (429 / 5xx / transient network error). Hosts can branch on
            this instead of parsing the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ToolError(AutoAgentError):
    """Raised when a tool cannot be executed."""


class ToolValidationError(ToolError):
    """Raised when generated tool code is rejected."""


class MaxStepsExceeded(AutoAgentError):
    """Raised when the agent loop reaches its configured step limit."""


class TokenBudgetExceeded(AutoAgentError):
    """Raised when a run's cumulative token usage reaches ``token_budget``.

    Checked BEFORE each provider call (the call that crossed the budget is
    never truncated mid-flight). Only enforceable when the provider reports
    usage — unreported calls count as zero (best effort, never invented).
    ``messages`` (the conversation so far) and ``spent`` (input+output
    tokens) are attached as attributes.
    """
