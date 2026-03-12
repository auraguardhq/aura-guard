"""aura_guard.adapters.mcp_adapter

MCP (Model Context Protocol) integration for Aura Guard.

Wraps a FastMCP server so that every tool call is automatically protected
by AuraGuard enforcement (loop detection, idempotency, cost budget, etc.).

Requires: pip install mcp (or pip install aura-guard[mcp])

Usage:
    from aura_guard.adapters.mcp_adapter import GuardedMCP

    mcp = GuardedMCP(
        "Customer Support",
        secret_key=b"your-secret-key",
        side_effect_tools={"refund", "cancel"},
        max_cost_per_run=1.00,
    )

    @mcp.tool()
    def search_kb(query: str) -> str:
        return db.search(query)

    @mcp.tool()
    def refund(order_id: str, amount: float) -> str:
        return payments.refund(order_id, amount)

    if __name__ == "__main__":
        mcp.run(transport="stdio")

When a tool call is denied, GuardedMCP returns a structured error message
to the LLM instead of crashing the server. The LLM sees:

    [AURA GUARD] BLOCK: side_effect_limit_exceeded

This lets the LLM understand it was blocked and adjust its behavior.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from typing import Any, Callable, Dict, Optional, Set

try:
    from mcp.server.fastmcp import FastMCP

    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    FastMCP = None  # type: ignore

from ..config import AuraGuardConfig
from ..middleware import AgentGuard
from ..telemetry import Telemetry
from ..types import PolicyAction

logger = logging.getLogger("aura_guard.mcp")


class GuardedMCP:
    """MCP server with AuraGuard protection on all registered tools.

    Wraps FastMCP so that every @mcp.tool() call goes through AuraGuard's
    check_tool → execute → record_result cycle automatically.

    When a tool call is denied (BLOCK, REWRITE, ESCALATE), the server
    returns a structured error message to the LLM instead of raising an
    exception. This lets the LLM understand it was blocked and adjust.

    Args:
        name: Server name (passed to FastMCP).
        secret_key: HMAC key for AuraGuard signatures. Required.
        side_effect_tools: Set of tool names that perform mutations.
        max_cost_per_run: Optional USD spending cap per session.
        max_calls_per_tool: Max times any single tool can be called.
        tool_costs: Optional dict mapping tool names to per-call costs.
        default_tool_cost: Default per-call cost if not in tool_costs.
        telemetry: Optional Telemetry instance for event emission.
        guard_config: Optional full AuraGuardConfig (overrides other guard params).
        **fastmcp_kwargs: Additional keyword arguments passed to FastMCP.
    """

    def __init__(
        self,
        name: str = "AuraGuard MCP Server",
        *,
        secret_key: Optional[bytes] = None,
        side_effect_tools: Optional[Set[str]] = None,
        max_cost_per_run: Optional[float] = None,
        max_calls_per_tool: Optional[int] = None,
        tool_costs: Optional[Dict[str, float]] = None,
        default_tool_cost: float = 0.04,
        telemetry: Optional[Telemetry] = None,
        guard_config: Optional[AuraGuardConfig] = None,
        **fastmcp_kwargs: Any,
    ):
        if not HAS_MCP:
            raise ImportError(
                "The 'mcp' package is required for the MCP adapter. "
                "Install it with: pip install mcp\n"
                "Or: pip install aura-guard[mcp]"
            )

        self._server = FastMCP(name, **fastmcp_kwargs)
        self._guard = AgentGuard(
            secret_key=secret_key,
            side_effect_tools=side_effect_tools,
            max_cost_per_run=max_cost_per_run,
            max_calls_per_tool=max_calls_per_tool,
            tool_costs=tool_costs,
            default_tool_cost=default_tool_cost,
            telemetry=telemetry,
            config=guard_config,
        )
        self._tool_side_effects: Dict[str, bool] = {}

    def tool(
        self,
        *args: Any,
        side_effect: Optional[bool] = None,
        **kwargs: Any,
    ) -> Callable:
        """Register a tool with AuraGuard protection.

        Works like @mcp.tool() but wraps every call with guard enforcement.

        Args:
            *args: Positional args passed to FastMCP.tool().
            side_effect: If True, mark this tool as a side-effect for AuraGuard
                         (overrides the side_effect_tools set for this specific tool).
            **kwargs: Keyword args passed to FastMCP.tool().
        """

        def decorator(fn: Callable) -> Callable:
            tool_name = fn.__name__

            if side_effect is not None:
                self._tool_side_effects[tool_name] = side_effect

            if asyncio.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_guarded(**kw: Any) -> Any:
                    se = self._tool_side_effects.get(tool_name)
                    try:
                        decision = self._guard.check_tool(
                            tool_name,
                            args=kw,
                            side_effect=se,
                        )
                    except Exception as exc:
                        logger.error("AuraGuard check_tool error for '%s': %s", tool_name, exc)
                        return await fn(**kw)  # fail-open: execute if guard errors

                    if decision.action == PolicyAction.ALLOW:
                        try:
                            result = await fn(**kw)
                            self._guard.record_result(ok=True, payload=result)
                            return result
                        except Exception as exc:
                            self._guard.record_result(ok=False, error_code=type(exc).__name__)
                            raise

                    if decision.action == PolicyAction.CACHE:
                        payload = decision.cached_result.payload if decision.cached_result else None
                        if payload is None:
                            return "[AURA GUARD] CACHED: previous result (payload not available)"
                        try:
                            return f"[AURA GUARD] CACHED: {json.dumps(payload)}"
                        except (TypeError, ValueError):
                            return f"[AURA GUARD] CACHED: {str(payload)}"

                    logger.warning(
                        "AuraGuard %s tool '%s': %s",
                        decision.action.value,
                        tool_name,
                        decision.reason,
                    )
                    msg = f"[AURA GUARD] {decision.action.value.upper()}: {decision.reason}"
                    if decision.action == PolicyAction.REWRITE and decision.injected_system:
                        msg += f"\n\nSYSTEM: {decision.injected_system}"
                    elif decision.action == PolicyAction.ESCALATE and decision.escalation_packet:
                        msg += f"\n\nESCALATION: {json.dumps(decision.escalation_packet)}"
                    elif decision.action == PolicyAction.FINALIZE and decision.finalized_output:
                        msg += f"\n\nFINAL OUTPUT: {decision.finalized_output}"
                    return msg

                return self._server.tool(*args, **kwargs)(async_guarded)

            @functools.wraps(fn)
            def sync_guarded(**kw: Any) -> Any:
                se = self._tool_side_effects.get(tool_name)
                try:
                    decision = self._guard.check_tool(
                        tool_name,
                        args=kw,
                        side_effect=se,
                    )
                except Exception as exc:
                    logger.error("AuraGuard check_tool error for '%s': %s", tool_name, exc)
                    return fn(**kw)  # fail-open

                if decision.action == PolicyAction.ALLOW:
                    try:
                        result = fn(**kw)
                        self._guard.record_result(ok=True, payload=result)
                        return result
                    except Exception as exc:
                        self._guard.record_result(ok=False, error_code=type(exc).__name__)
                        raise

                if decision.action == PolicyAction.CACHE:
                    payload = decision.cached_result.payload if decision.cached_result else None
                    if payload is None:
                        return "[AURA GUARD] CACHED: previous result (payload not available)"
                    try:
                        return f"[AURA GUARD] CACHED: {json.dumps(payload)}"
                    except (TypeError, ValueError):
                        return f"[AURA GUARD] CACHED: {str(payload)}"

                logger.warning(
                    "AuraGuard %s tool '%s': %s",
                    decision.action.value,
                    tool_name,
                    decision.reason,
                )
                msg = f"[AURA GUARD] {decision.action.value.upper()}: {decision.reason}"
                if decision.action == PolicyAction.REWRITE and decision.injected_system:
                    msg += f"\n\nSYSTEM: {decision.injected_system}"
                elif decision.action == PolicyAction.ESCALATE and decision.escalation_packet:
                    msg += f"\n\nESCALATION: {json.dumps(decision.escalation_packet)}"
                elif decision.action == PolicyAction.FINALIZE and decision.finalized_output:
                    msg += f"\n\nFINAL OUTPUT: {decision.finalized_output}"
                return msg

            return self._server.tool(*args, **kwargs)(sync_guarded)

        return decorator

    def resource(self, *args: Any, **kwargs: Any) -> Callable:
        """Register an MCP resource (delegated to FastMCP)."""
        return self._server.resource(*args, **kwargs)

    def prompt(self, *args: Any, **kwargs: Any) -> Callable:
        """Register an MCP prompt template (delegated to FastMCP)."""
        return self._server.prompt(*args, **kwargs)

    def run(self, **kwargs: Any) -> None:
        """Run the MCP server (delegated to FastMCP)."""
        return self._server.run(**kwargs)

    @property
    def guard(self) -> AgentGuard:
        """Access the underlying AgentGuard instance."""
        return self._guard

    @property
    def guard_stats(self) -> Dict[str, Any]:
        """Guard activity statistics for this session."""
        return self._guard.stats

    @property
    def server(self) -> Any:
        """Access the underlying FastMCP server instance."""
        return self._server

    def reset_guard(self) -> None:
        """Reset guard state for a new session (same config)."""
        self._guard.reset()
