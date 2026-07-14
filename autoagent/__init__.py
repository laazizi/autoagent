"""autoagent — LLM agent library with tool use, dynamic tool generation,
and controlled software-evolution capabilities.

The public API exposed here is covered by Semantic Versioning. Anything
imported from a private submodule (single-underscore prefixed names,
internal-only helpers) is NOT public and may change without notice.

Threading model: `Agent`, `ToolRegistry`, and `ProjectWorkspace` are
safe to use from multiple threads. Each one serializes its internal
mutable state with a per-instance `threading.RLock`. Provider HTTP
calls are stateless and can be issued concurrently.
"""

from __future__ import annotations

__version__ = "0.12.0"

from .agent import (
    Agent,
    AgentResult,
    AgentTurnContext,
    CheckpointHook,
    PostTurnHook,
    RunState,
    ToolPolicy,
    ToolPolicyContext,
)
from .dynamic import DynamicToolBuilder, ToolBuildRequest
from .errors import (
    AgentCancelled,
    ApprovalRequired,
    AutoAgentError,
    MCPError,
    MaxStepsExceeded,
    TokenBudgetExceeded,
    ProviderError,
    ToolError,
    ToolValidationError,
)
from .evolution import EVOLUTION_CAPABILITIES, EvolutionRuntime, enable_software_evolution
from .logging import get_logger
from .mcp import MCPClient
from .memory import BufferMemory, FactMemory, Memory, SummarizingMemory
from .otel import OTelTraceExporter
from .orchestrator import (
    InterpretOutcome,
    Orchestrator,
    PhraseSignals,
    Step,
    TurnEvent,
)
from .pipeline import PipelineManager
from .providers import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    LLMProvider,
    OpenAIProvider,
    RoutingProvider,
    create_provider,
)
from .registry import ToolRegistry, tool
from .schema import (
    ImageAttachment,
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    StreamChunk,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from .trace import OnEvent, TraceEmitter, TraceEvent
from .workspace import ProjectWorkspace

__all__ = [
    "EVOLUTION_CAPABILITIES",
    "Agent",
    "AgentCancelled",
    "AgentResult",
    "AgentTurnContext",
    "ApprovalRequired",
    "AnthropicProvider",
    "AutoAgentError",
    "BufferMemory",
    "CheckpointHook",
    "RunState",
    "SummarizingMemory",
    "DeepSeekProvider",
    "DynamicToolBuilder",
    "EvolutionRuntime",
    "FactMemory",
    "GeminiProvider",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ImageAttachment",
    "InterpretOutcome",
    "MCPClient",
    "MCPError",
    "MaxStepsExceeded",
    "TokenBudgetExceeded",
    "Memory",
    "Message",
    "ModelConfig",
    "OTelTraceExporter",
    "OnEvent",
    "OpenAIProvider",
    "Orchestrator",
    "PhraseSignals",
    "PipelineManager",
    "PostTurnHook",
    "ProjectWorkspace",
    "ProviderError",
    "RoutingProvider",
    "Step",
    "StreamChunk",
    "StreamEvent",
    "TokenUsage",
    "ToolBuildRequest",
    "ToolCall",
    "ToolError",
    "ToolPolicy",
    "ToolPolicyContext",
    "ToolRegistry",
    "ToolSpec",
    "ToolValidationError",
    "TraceEmitter",
    "TraceEvent",
    "TurnEvent",
    "__version__",
    "create_provider",
    "enable_software_evolution",
    "get_logger",
    "tool",
]
