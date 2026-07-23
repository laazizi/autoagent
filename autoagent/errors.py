__all__ = [
    "AgentCancelled",
    "ApprovalRequired",
    "AutoAgentError",
    "MCPError",
    "MaxStepsExceeded",
    "ProviderError",
    "ReplayMismatch",
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


class ApprovalRequired(AutoAgentError):
    """Raised BY a ``tool_policy`` hook to pause the run for human approval.

    The agent loop catches it before ANY tool of the turn has executed,
    attaches a resumable snapshot, and re-raises to the host:

    Attributes (attached by the loop):
        state: ``RunState`` snapshot — feed it to ``Agent.resume`` once
            the human has decided. On resume the pending tool calls go
            through the policy AGAIN: an unapproved call pauses again
            (idempotent), a rejected one should get a ``str`` verdict so
            the model sees the refusal and re-plans.
        calls: The turn's pending ``ToolCall`` list (nothing executed).
    """


class ReplayMismatch(AutoAgentError):
    """Raised during replay when the run DIVERGES from the recorded fixture
    (0.16.0).

    Either the request signature at position N no longer matches the recorded
    one (different tool requested, different message shape — with ``strict``),
    or the run asks for more calls than the fixture holds. This divergence is
    a FEATURE: it tells you the agent's behaviour changed since the recording
    (a prompt edit, a code change), and points at the exact step.
    """


class MCPError(AutoAgentError):
    """Raised when an MCP server interaction fails.

    Covers transport failures (server not launchable, closed pipe,
    response timeout) and JSON-RPC error responses (the code and message
    are included in the text). A tool result flagged ``isError`` is NOT
    an ``MCPError`` — it raises ``ToolError`` so the registry surfaces
    it to the LLM as an ordinary tool error.
    """


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
