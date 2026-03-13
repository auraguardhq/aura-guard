"""Aura Guard — Reliability middleware for tool-using agents: idempotency, circuit breaking, and loop detection.

Quick start:
    from aura_guard import AgentGuard

    guard = AgentGuard(secret_key=b"your-secret-key", max_cost_per_run=0.50)
    decision = guard.check_tool("search_kb", args={"query": "refund policy"})
"""

from .config import AuraGuardConfig, CostModel, ToolAccess, ToolPolicy
from .guard import AuraGuard, GuardState
from .middleware import AgentGuard, GuardDenied
from .async_middleware import AsyncAgentGuard
from .telemetry import (
    CompositeTelemetry,
    InMemoryTelemetry,
    LoggingTelemetry,
    SlackWebhookTelemetry,
    Telemetry,
    WebhookTelemetry,
)
from .types import CostEvent, PolicyAction, PolicyDecision, ToolCall, ToolCallSig, ToolResult

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("aura-guard")
except Exception:
    __version__ = "0.7.0"  # fallback for editable installs / development

__all__ = [
    # High-level API
    "AgentGuard",
    "AsyncAgentGuard",
    "GuardDenied",
    # Core engine
    "AuraGuard",
    "AuraGuardConfig",
    "GuardState",
    # Types
    "CostModel",
    "CostEvent",
    "PolicyAction",
    "PolicyDecision",
    "ToolAccess",
    "ToolPolicy",
    "ToolCall",
    "ToolCallSig",
    "ToolResult",
    # Telemetry
    "Telemetry",
    "LoggingTelemetry",
    "InMemoryTelemetry",
    "WebhookTelemetry",
    "SlackWebhookTelemetry",
    "CompositeTelemetry",
]
