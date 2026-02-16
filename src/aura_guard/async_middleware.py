"""aura_guard.async_middleware

Async-compatible wrapper for Aura Guard.

Since the guard engine is pure computation (no I/O, sub-millisecond),
the async methods call the synchronous engine directly on the event loop.
This is safe because the core engine never blocks on I/O. Note: if you attach WebhookTelemetry, its emit() is synchronous and may block briefly.

Usage with async agent loops:

    from aura_guard.async_middleware import AsyncAgentGuard, PolicyAction

    guard = AsyncAgentGuard(secret_key=b"your-secret-key", max_cost_per_run=0.50)

    decision = await guard.check_tool("search_kb", args={"query": "test"})
    if decision.action == PolicyAction.ALLOW:
        result = await execute_tool(...)
        await guard.record_result(ok=True, payload=result)

    stall = await guard.check_output("I apologize for the delay...")

Works with asyncio, FastAPI, Starlette, and any async framework.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import AuraGuardConfig
from .middleware import AgentGuard
from .telemetry import Telemetry
from .types import PolicyAction, PolicyDecision


class AsyncAgentGuard:
    """Async wrapper around AgentGuard for use in async agent loops.

    WARNING: AsyncAgentGuard holds per-run state and is NOT thread-safe.
    Do not share a single AsyncAgentGuard instance across threads, async tasks,
    or concurrent requests. Create one AsyncAgentGuard per agent run.
    For web servers, create a new instance per request.

    Provides the same 3-method API as AgentGuard but with async/await syntax.
    All guard operations are non-blocking (pure CPU, sub-millisecond), so they
    run directly on the event loop without thread offloading.

    Args:
        Same as AgentGuard — see :class:`aura_guard.middleware.AgentGuard`.
    """

    def __init__(
        self,
        *,
        max_cost_per_run: Optional[float] = None,
        tool_costs: Optional[Dict[str, float]] = None,
        default_tool_cost: float = 0.04,
        side_effect_tools: Optional[set] = None,
        max_calls_per_tool: Optional[int] = None,
        secret_key: Optional[bytes] = None,
        telemetry: Optional[Telemetry] = None,
        config: Optional[AuraGuardConfig] = None,
        shadow_mode: bool = False,
        strict_mode: bool = False,
    ):
        self._sync = AgentGuard(
            max_cost_per_run=max_cost_per_run,
            tool_costs=tool_costs,
            default_tool_cost=default_tool_cost,
            side_effect_tools=side_effect_tools,
            max_calls_per_tool=max_calls_per_tool,
            secret_key=secret_key,
            telemetry=telemetry,
            config=config,
            shadow_mode=shadow_mode,
            strict_mode=strict_mode,
        )

    # ─────────────────────────────────────────
    # Async 3-Method API
    # ─────────────────────────────────────────

    async def check_tool(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        ticket_id: Optional[str] = None,
        side_effect: Optional[bool] = None,
    ) -> PolicyDecision:
        """Check whether a tool call should proceed (async).

        See :meth:`AgentGuard.check_tool` for full documentation.
        """
        return self._sync.check_tool(
            name, args=args, ticket_id=ticket_id, side_effect=side_effect,
        )

    async def record_result(
        self,
        ok: bool = True,
        payload: Any = None,
        error_code: Optional[str] = None,
    ) -> None:
        """Record the result of the most recent tool call (async).

        See :meth:`AgentGuard.record_result` for full documentation.
        """
        self._sync.record_result(ok=ok, payload=payload, error_code=error_code)

    async def check_output(self, text: str) -> Optional[PolicyDecision]:
        """Check an assistant's text output for stall/loop behavior (async).

        See :meth:`AgentGuard.check_output` for full documentation.
        """
        return self._sync.check_output(text)

    async def record_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0,
                            cost_override: Optional[float] = None) -> None:
        """Report actual LLM token usage (async).

        See :meth:`AgentGuard.record_tokens` for full documentation.
        """
        self._sync.record_tokens(
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_override=cost_override,
        )

    # ─────────────────────────────────────────
    # Convenience (delegated to sync guard)
    # ─────────────────────────────────────────

    @property
    def cost_spent(self) -> float:
        return self._sync.cost_spent

    @property
    def reported_token_cost(self) -> float:
        return self._sync.reported_token_cost

    @property
    def cost_limit(self) -> Optional[float]:
        return self._sync.cost_limit

    @property
    def cost_remaining(self) -> Optional[float]:
        return self._sync.cost_remaining

    @property
    def quarantined_tools(self) -> Dict[str, str]:
        return self._sync.quarantined_tools

    @property
    def blocks(self) -> int:
        return self._sync.blocks

    @property
    def cache_hits(self) -> int:
        return self._sync.cache_hits

    @property
    def rewrite_decisions(self) -> int:
        return self._sync.rewrite_decisions

    @property
    def escalations(self) -> int:
        return self._sync.escalations

    @property
    def shadow_would_deny(self) -> int:
        return self._sync.shadow_would_deny

    @property
    def missed_results(self) -> int:
        return self._sync.missed_results

    @property
    def stats(self) -> Dict[str, Any]:
        return self._sync.stats

    @property
    def summary(self) -> Dict[str, Any]:
        return self._sync.summary

    async def reset(self) -> None:
        """Reset state for a new run (same config)."""
        self._sync.reset()
