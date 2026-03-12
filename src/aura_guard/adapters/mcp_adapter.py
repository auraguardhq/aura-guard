"""aura_guard.adapters.mcp_adapter

MCP (Model Context Protocol) integration for Aura Guard.

Wraps a FastMCP server so that every tool call is automatically protected
by AuraGuard enforcement (loop detection, idempotency, cost budget, etc.).

Requires: pip install mcp (or pip install aura-guard[mcp])

Usage (single session — stdio):
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

    mcp.run(transport="stdio")

Usage (multi-session — HTTP):
    mcp = GuardedMCP(
        "Customer Support",
        secret_key=b"your-secret-key",
        side_effect_tools={"refund", "cancel"},
        session_mode="per_session",
        max_sessions=100,
        session_ttl_seconds=3600,
    )

    @mcp.tool()
    def search_kb(query: str) -> str:
        return db.search(query)

    mcp.run(transport="streamable-http")

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
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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


def _now() -> float:
    """Monotonic clock for TTL tracking."""
    return time.monotonic()


class _SessionRegistry:
    """Thread-safe registry of session-scoped AgentGuard instances.

    Each session gets its own guard with independent state. Sessions are
    identified by a string key (typically the async task ID). Expired
    sessions are cleaned up lazily on access.
    """

    def __init__(
        self,
        guard_factory: Callable[[], AgentGuard],
        max_sessions: int = 100,
        session_ttl_seconds: float = 3600,
    ):
        self._factory = guard_factory
        self._max_sessions = max_sessions
        self._ttl = session_ttl_seconds
        self._guards: Dict[str, Tuple[AgentGuard, float]] = {}  # key -> (guard, last_access)
        self._lock = threading.Lock()

    def get(self, session_id: str) -> AgentGuard:
        """Get or create a guard for the given session."""
        now = _now()
        with self._lock:
            self._cleanup_expired(now)

            if session_id in self._guards:
                guard, _ = self._guards[session_id]
                self._guards[session_id] = (guard, now)
                return guard

            # Evict oldest if at capacity
            if len(self._guards) >= self._max_sessions:
                oldest_key = min(self._guards, key=lambda k: self._guards[k][1])
                logger.info("Session registry full, evicting oldest session: %s", oldest_key)
                del self._guards[oldest_key]

            guard = self._factory()
            self._guards[session_id] = (guard, now)
            return guard

    def remove(self, session_id: str) -> None:
        """Remove a session's guard (e.g., on disconnect)."""
        with self._lock:
            self._guards.pop(session_id, None)

    def _cleanup_expired(self, now: float) -> None:
        """Remove sessions that haven't been accessed within TTL."""
        expired = [
            k for k, (_, last) in self._guards.items()
            if now - last > self._ttl
        ]
        for k in expired:
            logger.debug("Cleaning up expired session: %s", k)
            del self._guards[k]

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return len(self._guards)

    def all_stats(self) -> Dict[str, Any]:
        """Get stats for all active sessions."""
        with self._lock:
            return {
                "active_sessions": len(self._guards),
                "sessions": {
                    k: guard.stats for k, (guard, _) in self._guards.items()
                },
            }

    def reset_all(self) -> None:
        """Reset all session guards."""
        with self._lock:
            self._guards.clear()


def _get_session_id() -> str:
    """Get a session ID based on the current async task or thread.

    MCP HTTP servers run each client connection in a separate async task.
    Using the task ID as the session key naturally isolates sessions.
    Falls back to thread ID for sync contexts.
    """
    try:
        task = asyncio.current_task()
        if task is not None:
            return f"task-{id(task)}"
    except RuntimeError:
        pass
    return f"thread-{threading.get_ident()}"


class GuardedMCP:
    """MCP server with AuraGuard protection on all registered tools.

    Wraps FastMCP so that every @mcp.tool() call goes through AuraGuard's
    check_tool → execute → record_result cycle automatically.

    When a tool call is denied (BLOCK, REWRITE, ESCALATE), the server
    returns a structured error message to the LLM instead of raising an
    exception. This lets the LLM understand it was blocked and adjust.

    Session modes:
        "single" (default): One shared AgentGuard for all clients.
            Best for stdio transport where there's only one client.
        "per_session": A separate AgentGuard per MCP session.
            Best for HTTP/SSE transport serving multiple clients.
            Each session gets independent loop detection, cost budgets,
            idempotency, and quarantine state.

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
        session_mode: "single" (one guard) or "per_session" (guard per client).
        max_sessions: Max concurrent sessions in per_session mode (default 100).
        session_ttl_seconds: Seconds before an idle session is cleaned up (default 3600).
        session_id_fn: Optional callable returning a session ID string.
            Overrides the default async-task-based session detection.
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
        session_mode: str = "single",
        max_sessions: int = 100,
        session_ttl_seconds: float = 3600,
        session_id_fn: Optional[Callable[[], str]] = None,
        **fastmcp_kwargs: Any,
    ):
        if not HAS_MCP:
            raise ImportError(
                "The 'mcp' package is required for the MCP adapter. "
                "Install it with: pip install mcp\n"
                "Or: pip install aura-guard[mcp]"
            )

        if session_mode not in ("single", "per_session"):
            raise ValueError(f"session_mode must be 'single' or 'per_session', got '{session_mode}'")

        self._server = FastMCP(name, **fastmcp_kwargs)
        self._session_mode = session_mode
        self._session_id_fn = session_id_fn or _get_session_id
        self._tool_side_effects: Dict[str, bool] = {}

        # Store guard construction kwargs so we can create new guards per session
        self._guard_kwargs: Dict[str, Any] = dict(
            secret_key=secret_key,
            side_effect_tools=side_effect_tools,
            max_cost_per_run=max_cost_per_run,
            max_calls_per_tool=max_calls_per_tool,
            tool_costs=tool_costs,
            default_tool_cost=default_tool_cost,
            telemetry=telemetry,
            config=guard_config,
        )

        if session_mode == "single":
            self._shared_guard = AgentGuard(**self._guard_kwargs)
            self._registry: Optional[_SessionRegistry] = None
        else:
            self._shared_guard = None
            self._registry = _SessionRegistry(
                guard_factory=lambda: AgentGuard(**self._guard_kwargs),
                max_sessions=max_sessions,
                session_ttl_seconds=session_ttl_seconds,
            )

    def _get_guard(self) -> AgentGuard:
        """Get the appropriate guard for the current context."""
        if self._session_mode == "single":
            return self._shared_guard
        return self._registry.get(self._session_id_fn())

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
                    guard = self._get_guard()
                    se = self._tool_side_effects.get(tool_name)
                    try:
                        decision = guard.check_tool(
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
                            guard.record_result(ok=True, payload=result)
                            return result
                        except Exception as exc:
                            guard.record_result(ok=False, error_code=type(exc).__name__)
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
                guard = self._get_guard()
                se = self._tool_side_effects.get(tool_name)
                try:
                    decision = guard.check_tool(
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
                        guard.record_result(ok=True, payload=result)
                        return result
                    except Exception as exc:
                        guard.record_result(ok=False, error_code=type(exc).__name__)
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

    # ─────────────────────────────────────────
    # Delegate FastMCP methods
    # ─────────────────────────────────────────

    def resource(self, *args: Any, **kwargs: Any) -> Callable:
        """Register an MCP resource (delegated to FastMCP)."""
        return self._server.resource(*args, **kwargs)

    def prompt(self, *args: Any, **kwargs: Any) -> Callable:
        """Register an MCP prompt template (delegated to FastMCP)."""
        return self._server.prompt(*args, **kwargs)

    def run(self, **kwargs: Any) -> None:
        """Run the MCP server (delegated to FastMCP)."""
        return self._server.run(**kwargs)

    # ─────────────────────────────────────────
    # Guard access
    # ─────────────────────────────────────────

    @property
    def guard(self) -> AgentGuard:
        """Access the guard for the current session.

        In single mode, returns the shared guard.
        In per_session mode, returns the guard for the current async task/thread.
        """
        return self._get_guard()

    @property
    def guard_stats(self) -> Dict[str, Any]:
        """Guard activity statistics.

        In single mode, returns stats for the shared guard.
        In per_session mode, returns stats for all active sessions.
        """
        if self._session_mode == "single":
            return self._shared_guard.stats
        return self._registry.all_stats()

    @property
    def server(self) -> Any:
        """Access the underlying FastMCP server instance."""
        return self._server

    @property
    def session_mode(self) -> str:
        """Current session mode ("single" or "per_session")."""
        return self._session_mode

    @property
    def active_sessions(self) -> int:
        """Number of active sessions (always 1 in single mode)."""
        if self._session_mode == "single":
            return 1
        return self._registry.active_session_count

    def reset_guard(self) -> None:
        """Reset guard state.

        In single mode, resets the shared guard.
        In per_session mode, clears all session guards.
        """
        if self._session_mode == "single":
            self._shared_guard.reset()
        else:
            self._registry.reset_all()
