"""aura_guard.middleware

Simplified wrapper for Aura Guard.

Provides two levels of API:

1. **Convenience API** (simplest):
   guard.run(name, fn, **kwargs)  — one-liner, handles everything
   @guard.protect                 — decorator, even simpler

2. **3-method API** (full control):
   guard.check_tool(...)          — before each tool call
   guard.record_result(...)       — after each tool call
   guard.check_output(...)        — after each LLM output

Usage (convenience):
    from aura_guard import AgentGuard, GuardDenied

    guard = AgentGuard(secret_key=b"your-secret-key", side_effect_tools={"refund"})

    # One-liner
    result = guard.run("search_kb", search_kb, query="refund policy")

    # Decorator
    @guard.protect
    def refund(order_id, amount): ...

    try:
        refund(order_id="o1", amount=50)
    except GuardDenied as e:
        print(f"Blocked: {e.reason}")
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Dict, List, Optional

from .config import AuraGuardConfig, CostModel
from .guard import AuraGuard, GuardState
from .telemetry import Telemetry
from .types import PolicyAction, PolicyDecision, ToolCall, ToolResult

import logging

logger = logging.getLogger("aura_guard")

class GuardDenied(Exception):
    """Raised by guard.run() and @guard.protect when a tool call is denied.

    Attributes:
        action: The PolicyAction (BLOCK, REWRITE, ESCALATE, FINALIZE)
        reason: Human-readable reason for the denial
        decision: The full PolicyDecision object
    """

    def __init__(self, decision: PolicyDecision):
        self.action = decision.action
        self.reason = decision.reason
        self.decision = decision
        super().__init__(f"GuardDenied: {decision.action.value} — {decision.reason}")


class AgentGuard:
    """Simplified guard wrapper for custom agent loops.

    WARNING: AgentGuard holds per-run state and is NOT thread-safe.
    Do not share a single AgentGuard instance across threads, async tasks,
    or concurrent requests. Create one AgentGuard per agent run.
    For web servers, create a new instance per request.

    Wraps AuraGuard into three methods:
    - check_tool()     → call before each tool execution
    - record_result()  → call after each tool execution
    - check_output()   → call after each LLM output

    Args:
        max_cost_per_run: Optional USD spending cap per agent run.
        tool_costs: Optional dict mapping tool names to per-call costs (USD).
        default_tool_cost: Default per-call cost if not in tool_costs.
        side_effect_tools: Set of tool names that perform mutations.
        max_calls_per_tool: Max times any single tool can be called per run.
        secret_key: HMAC key for signatures. Change in production.
        telemetry: Optional Telemetry instance for event emission.
        config: Optional full AuraGuardConfig (overrides other params).
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
        self._creator_thread = threading.get_ident()

        if config is not None:
            self._cfg = config
        else:
            kwargs: Dict[str, Any] = {}
            if max_cost_per_run is not None:
                kwargs["max_cost_per_run"] = max_cost_per_run
            if side_effect_tools is not None:
                kwargs["side_effect_tools"] = side_effect_tools
            if max_calls_per_tool is not None:
                kwargs["max_calls_per_tool"] = max_calls_per_tool
            if secret_key is not None:
                kwargs["secret_key"] = secret_key
            if shadow_mode:
                kwargs["shadow_mode"] = True

            cost_model = CostModel(
                default_tool_call_cost=default_tool_cost,
                tool_cost_by_name=dict(tool_costs or {}),
            )
            kwargs["cost_model"] = cost_model
            self._cfg = AuraGuardConfig(**kwargs)

        self._shadow = self._cfg.shadow_mode
        self._strict_mode = strict_mode
        self._guard = AuraGuard(config=self._cfg, telemetry=telemetry)
        self._state = self._guard.new_state()
        self._last_call: Optional[ToolCall] = None

        # Public counters
        self.blocks: int = 0
        self.cache_hits: int = 0
        self.rewrite_decisions: int = 0   # guard returned REWRITE (tool denied with instruction)
        self.escalations: int = 0
        self.tool_calls_executed: int = 0  # actually sent to the tool (success + failure)
        self.tool_calls_failed: int = 0    # executed but returned error
        self.tool_calls_denied: int = 0    # guard prevented execution (block + cache + rewrite + escalate)
        self.shadow_would_deny: int = 0    # shadow mode: would have denied but allowed
        self.missed_results: int = 0       # check_tool called before prior ALLOW had record_result
        self._interventions: List[Dict[str, str]] = []

    # ─────────────────────────────────────────
    # 3-Method API
    # ─────────────────────────────────────────

    def _assert_same_thread(self) -> None:
        if threading.get_ident() != self._creator_thread:
            raise RuntimeError(
                "AgentGuard is not thread-safe. Create one instance per agent run. See docs."
            )

    def check_tool(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        ticket_id: Optional[str] = None,
        side_effect: Optional[bool] = None,
    ) -> PolicyDecision:
        """Check whether a tool call should proceed.

        Returns a PolicyDecision. Inspect decision.action:
        - "allow"    → execute the tool, then call record_result()
        - "cache"    → use decision.cached_result, skip execution
        - "block"    → skip execution
        - "rewrite"  → inject decision.injected_system into next model prompt
        - "escalate" → stop the agent run, use decision.escalation_packet
        - "finalize" → stop the agent run, use decision.finalized_output
        """
        self._assert_same_thread()
        if self._last_call is not None:
            if self._strict_mode:
                raise RuntimeError(
                    "check_tool() called without a preceding record_result() for "
                    f"tool '{self._last_call.name}'."
                )

            logger.warning(
                "check_tool() called without a preceding record_result() for tool '%s'. "
                "The previous result will not be recorded. This may cause the guard to "
                "undercount tool calls.",
                self._last_call.name,
            )
            self.missed_results += 1

        call = ToolCall(
            name=name,
            args=args or {},
            ticket_id=ticket_id,
            side_effect=side_effect,
        )
        if self._shadow and hasattr(self._guard, "_evaluate_tool_call"):
            decision = self._guard._evaluate_tool_call(state=self._state, call=call)
        else:
            decision = self._guard.on_tool_call_request(state=self._state, call=call)

        # Shadow mode: log the decision but override to ALLOW
        if self._shadow and decision.action != PolicyAction.ALLOW:
            self.shadow_would_deny += 1
            self._interventions.append({"tool": name, "action": "shadow_deny", "reason": decision.reason})
            self._guard._emit(
                "shadow_would_deny", state=self._state,
                tool=name, original_action=decision.action.value,
                original_reason=decision.reason,
            )
            # Let the tool execute — return ALLOW
            self._last_call = call
            return PolicyDecision(action=PolicyAction.ALLOW, reason="shadow_allow")

        # Only set _last_call on ALLOW — prevents sloppy integrators from
        # calling record_result() after a block and corrupting state.
        if decision.action == PolicyAction.ALLOW:
            self._last_call = call
        elif decision.action == PolicyAction.BLOCK:
            self._last_call = None
            self.blocks += 1
            self.tool_calls_denied += 1
            self._interventions.append({"tool": name, "action": "block", "reason": decision.reason})
        elif decision.action == PolicyAction.CACHE:
            self._last_call = None
            self.cache_hits += 1
            self.tool_calls_denied += 1
            self._interventions.append({"tool": name, "action": "cache", "reason": decision.reason})
            # Auto-record cached results
            if decision.cached_result:
                self._guard.on_tool_result(
                    state=self._state, call=call, result=decision.cached_result,
                )
        elif decision.action == PolicyAction.REWRITE:
            self._last_call = None
            self.rewrite_decisions += 1
            self.tool_calls_denied += 1
            self._interventions.append({"tool": name, "action": "rewrite", "reason": decision.reason})
        elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
            self._last_call = None
            self.escalations += 1
            self.tool_calls_denied += 1
            self._interventions.append({"tool": name, "action": decision.action.value, "reason": decision.reason})

        return decision

    def record_result(
        self,
        ok: bool = True,
        payload: Any = None,
        error_code: Optional[str] = None,
        side_effect_executed: Optional[bool] = None,
    ) -> None:
        """Record the result of the most recent tool call.

        Call this after executing a tool that was allowed by check_tool().

        Args:
            ok: Whether the tool call succeeded.
            payload: The tool's return value (may contain PII — cached in memory only).
            error_code: A coarse error classification (e.g. "429", "timeout", "ValueError").
                        Do not pass raw exception messages — error_code is persisted in
                        serialized state. Use type(exc).__name__ or a short category string.
            side_effect_executed: Set to True if the side effect was executed (even on timeout).
        """
        self._assert_same_thread()
        if self._last_call is None:
            if self._strict_mode:
                raise RuntimeError(
                    "record_result() called without a preceding check_tool(). "
                    "In strict_mode, every record_result() must follow a check_tool() "
                    "that returned ALLOW."
                )
            return
        result = ToolResult(ok=ok, payload=payload, error_code=error_code, side_effect_executed=side_effect_executed)
        self._guard.on_tool_result(state=self._state, call=self._last_call, result=result)
        self._last_call = None
        self.tool_calls_executed += 1
        if not ok:
            self.tool_calls_failed += 1

    def check_output(self, text: str) -> Optional[PolicyDecision]:
        """Check an assistant's text output for stall/loop behavior.

        Returns None if no intervention needed, or a PolicyDecision if the
        agent should be rewritten, escalated, or finalized.
        """
        self._assert_same_thread()
        if self._shadow and hasattr(self._guard, "_evaluate_llm_output"):
            decision = self._guard._evaluate_llm_output(state=self._state, text=text)
        else:
            decision = self._guard.on_llm_output(state=self._state, text=text)
        if decision is not None:
            # Shadow mode: log but don't enforce
            if self._shadow:
                self.shadow_would_deny += 1
                self._guard._emit(
                    "shadow_would_intervene", state=self._state,
                    original_action=decision.action.value,
                    original_reason=decision.reason,
                )
                return None

            if decision.action == PolicyAction.REWRITE:
                self.rewrite_decisions += 1
                self._interventions.append({"tool": "_llm_output", "action": "rewrite", "reason": decision.reason})
            elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                self.escalations += 1
                self._interventions.append({"tool": "_llm_output", "action": "escalate", "reason": decision.reason})
        return decision

    def record_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0,
                      cost_override: Optional[float] = None) -> None:
        """Report actual LLM token usage for accurate cost tracking.

        Call after each LLM API response with the real usage numbers.

        Example:
            guard.record_tokens(input_tokens=resp.usage.input_tokens,
                                output_tokens=resp.usage.output_tokens)
        """
        self._guard.record_tokens(
            state=self._state,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_override=cost_override,
        )

    def run(
        self,
        tool_name: str,
        fn: Callable[..., Any],
        *,
        ticket_id: Optional[str] = None,
        side_effect: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a tool function with full guard protection in one call.

        Handles the complete check_tool → execute → record_result cycle.

        Returns the tool's return value on ALLOW.
        Returns cached payload on CACHE (may be None if restored from serialization).
        Raises GuardDenied on BLOCK, REWRITE, ESCALATE, or FINALIZE.

        All tool arguments MUST be passed as keyword arguments. This ensures the
        guard's signature tracker sees the same arguments the function receives.
        Positional arguments are not supported — use the 3-method API
        (check_tool / record_result) if your tool requires positional args.

        Example:
            result = guard.run("search_kb", search_kb, query="refund policy")
            result = guard.run("refund", refund, order_id="o1", ticket_id="t1", side_effect=True)

        Args:
            tool_name: Name of the tool (used for guard tracking).
            fn: The callable to execute if the guard allows it.
            ticket_id: Optional ticket/session ID for idempotency.
            side_effect: Override whether this tool is a side-effect.
            **kwargs: Keyword arguments passed to fn AND used as tool args for guard signature tracking.

        .. note:: Reserved keyword arguments

           ``ticket_id`` and ``side_effect`` are consumed by the guard wrapper
           and are NOT forwarded to ``fn``. If your tool function has parameters
           named ``ticket_id`` or ``side_effect``, use the 3-method API instead.

        .. warning:: Side-effect timeout handling

           If a side-effect tool succeeds server-side but times out locally,
           run() cannot mark side_effect_executed=True on the error path.
           For tools where ambiguous execution is possible (payments, emails,
           cancellations), use the 3-method API (check_tool / record_result)
           so you can set side_effect_executed=True on timeout.
        """
        decision = self.check_tool(
            tool_name, args=kwargs, ticket_id=ticket_id, side_effect=side_effect,
        )

        if decision.action == PolicyAction.ALLOW:
            try:
                result = fn(**kwargs)
                self.record_result(ok=True, payload=result)
                return result
            except Exception as exc:
                self.record_result(ok=False, error_code=type(exc).__name__)
                raise

        if decision.action == PolicyAction.CACHE:
            return decision.cached_result.payload if decision.cached_result else None

        raise GuardDenied(decision)

    def protect(
        self,
        fn: Optional[Callable[..., Any]] = None,
        *,
        tool_name: Optional[str] = None,
        ticket_id: Optional[str] = None,
        side_effect: Optional[bool] = None,
    ) -> Any:
        """Decorator that wraps a function with guard protection.

        Every call to the decorated function goes through check_tool → execute → record_result.

        IMPORTANT: The decorated function must be called with keyword arguments only.
        Positional arguments are not tracked by the guard and will cause a TypeError.
        If your tool requires positional args, use the 3-method API instead.

        Can be used with or without arguments:

            @guard.protect
            def search_kb(query): ...

            @guard.protect(tool_name="custom_name", side_effect=True)
            def charge_card(customer_id, amount): ...

        Args:
            fn: The function to wrap (when used without parentheses).
            tool_name: Override the tool name (defaults to fn.__name__).
            ticket_id: Optional ticket/session ID for idempotency.
            side_effect: Override whether this tool is a side-effect.

        .. note:: Reserved keyword arguments

           ``ticket_id`` and ``side_effect`` are consumed by the guard wrapper
           and are NOT forwarded to ``fn``. If your tool function has parameters
           named ``ticket_id`` or ``side_effect``, use the 3-method API instead.

        .. warning:: Side-effect timeout handling

           If a side-effect tool succeeds server-side but times out locally,
           run() cannot mark side_effect_executed=True on the error path.
           For tools where ambiguous execution is possible (payments, emails,
           cancellations), use the 3-method API (check_tool / record_result)
           so you can set side_effect_executed=True on timeout.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            name = tool_name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if args:
                    raise TypeError(
                        f"@guard.protect requires keyword arguments only. "
                        f"Call {func.__name__}() with keyword args so the guard "
                        f"can track them for deduplication. "
                        f"Got {len(args)} positional arg(s)."
                    )
                return self.run(
                    name, func,
                    ticket_id=ticket_id, side_effect=side_effect,
                    **kwargs,
                )

            return wrapper

        if fn is not None:
            # Called without parentheses: @guard.protect
            return decorator(fn)
        # Called with parentheses: @guard.protect(tool_name="...", ...)
        return decorator


    # ─────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────

    @property
    def cost_spent(self) -> float:
        """Cumulative cost spent this run (USD) — includes tool + token costs."""
        return round(self._state.cumulative_cost, 4)

    @property
    def reported_token_cost(self) -> float:
        """Actual token costs reported by the integrator (USD)."""
        return round(self._state.reported_token_cost, 4)

    @property
    def cost_limit(self) -> Optional[float]:
        """Cost budget limit for this run (USD), or None."""
        return self._cfg.max_cost_per_run

    @property
    def cost_remaining(self) -> Optional[float]:
        """Remaining cost budget (USD), or None if no limit."""
        if self._cfg.max_cost_per_run is None:
            return None
        return round(self._cfg.max_cost_per_run - self._state.cumulative_cost, 4)

    @property
    def quarantined_tools(self) -> Dict[str, str]:
        """Tools that have been quarantined in this run, with reasons."""
        return dict(self._state.quarantined_tools)

    @property
    def rewrites(self) -> int:
        """Backward-compat alias for rewrite_decisions."""
        return self.rewrite_decisions

    @property
    def stats(self) -> Dict[str, Any]:
        """Summary statistics for this run."""
        return {
            "tool_calls_executed": self.tool_calls_executed,
            "tool_calls_failed": self.tool_calls_failed,
            "tool_calls_cached": self.cache_hits,
            "tool_calls_denied": self.tool_calls_denied,
            "blocks": self.blocks,
            "cache_hits": self.cache_hits,
            "rewrite_decisions": self.rewrite_decisions,
            "escalations": self.escalations,
            "cost_spent_usd": self.cost_spent,
            "reported_token_cost_usd": self.reported_token_cost,
            "cost_limit_usd": self.cost_limit,
            "cost_remaining_usd": self.cost_remaining,
            "quarantined_tools": self.quarantined_tools,
            "unique_tool_calls": len(self._state.unique_tool_calls_seen),
            "stall_streak": self._state.stall_streak,
            "shadow_mode": self._shadow,
            "shadow_would_deny": self.shadow_would_deny,
            "missed_results": self.missed_results,
            "interventions_count": len(self._interventions),
        }

    def report_data(self) -> Dict[str, Any]:
        """Structured run report for programmatic use / JSON export.

        Returns a dict with:
        - summary: overall call counts and efficiency
        - cost: spending breakdown
        - interventions_by_primitive: which enforcement rules triggered
        - interventions_by_tool: per-tool breakdown
        - side_effects: attempted vs executed
        - quarantined_tools: which tools were quarantined and why
        - top_interventions: most frequent intervention reasons
        - interventions: full intervention log
        """
        total_calls = self.tool_calls_executed + self.tool_calls_denied
        efficiency = round(self.tool_calls_executed / total_calls * 100, 1) if total_calls > 0 else 100.0

        by_primitive: Dict[str, int] = {}
        by_tool: Dict[str, Dict[str, int]] = {}
        reason_counts: Dict[str, int] = {}

        for iv in self._interventions:
            reason = iv["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            tool = iv["tool"]
            if tool not in by_tool:
                by_tool[tool] = {"block": 0, "cache": 0, "rewrite": 0, "escalate": 0, "finalize": 0, "shadow_deny": 0}
            action = iv["action"]
            if action in by_tool[tool]:
                by_tool[tool][action] += 1

            primitive = self._classify_primitive(reason)
            by_primitive[primitive] = by_primitive.get(primitive, 0) + 1

        top = sorted(reason_counts.items(), key=lambda x: -x[1])


        return {
            "summary": {
                "total_calls": total_calls,
                "executed": self.tool_calls_executed,
                "denied": self.tool_calls_denied,
                "efficiency_pct": efficiency,
                "blocks": self.blocks,
                "cache_hits": self.cache_hits,
                "rewrites": self.rewrite_decisions,
                "escalations": self.escalations,
            },
            "cost": {
                "spent_usd": self.cost_spent,
                "limit_usd": self.cost_limit,
                "remaining_usd": self.cost_remaining,
                "token_cost_usd": self.reported_token_cost,
            },
            "interventions_by_primitive": by_primitive,
            "interventions_by_tool": by_tool,
            "side_effects": {
                "attempted": dict(self._state.attempted_side_effect_calls),
                "executed": dict(self._state.executed_side_effect_calls),
            },
            "quarantined_tools": self.quarantined_tools,
            "top_interventions": [{"reason": r, "count": c} for r, c in top[:10]],
            "stall_streak": self._state.stall_streak,
            "shadow_mode": self._shadow,
            "shadow_would_deny": self.shadow_would_deny,
            "interventions": list(self._interventions),
        }

    @staticmethod
    def _classify_primitive(reason: str) -> str:
        """Map a decision reason to its enforcement primitive name."""
        if reason.startswith("identical_toolcall"):
            return "repeat_detection"
        if reason.startswith("arg_jitter") or reason.startswith("query_jitter"):
            return "jitter_detection"
        if reason.startswith("error_") or reason.startswith("quarantine"):
            return "circuit_breaker"
        if reason.startswith("idempotent") or reason.startswith("side_effect"):
            return "side_effect_gating"
        if reason.startswith("stall") or reason.startswith("no_state_change"):
            return "stall_detection"
        if reason.startswith("cost_") or reason.startswith("budget"):
            return "cost_budget"
        if reason.startswith("policy_") or reason.startswith("tool_denied"):
            return "tool_policy"
        if reason.startswith("sequence_loop"):
            return "sequence_detection"
        if reason.startswith("max_calls"):
            return "per_tool_cap"
        if reason == "shadow_allow":
            return "shadow_mode"
        return "other"

    def report(self) -> str:
        """Formatted text report of guard activity for this run.

        Returns a human-readable string ready to print or log.
        """
        d = self.report_data()
        s = d["summary"]
        c = d["cost"]
        lines = []

        lines.append("=" * 50)
        lines.append("AURAGUARD RUN REPORT")
        lines.append("=" * 50)
        lines.append("")

        lines.append(f"Run efficiency: {s['efficiency_pct']}% productive ({s['executed']} executed, {s['denied']} denied)")
        if s["denied"] > 0:
            parts = []
            if s["cache_hits"]:
                parts.append(f"Cached: {s['cache_hits']}")
            if s["blocks"]:
                parts.append(f"Blocked: {s['blocks']}")
            if s["rewrites"]:
                parts.append(f"Rewritten: {s['rewrites']}")
            if s["escalations"]:
                parts.append(f"Escalated: {s['escalations']}")
            lines.append(f"  {', '.join(parts)}")
        lines.append("")

        if c["limit_usd"] is not None:
            pct_remaining = round(c["remaining_usd"] / c["limit_usd"] * 100, 0) if c["limit_usd"] > 0 else 0
            lines.append(f"Cost: ${c['spent_usd']:.2f} spent / ${c['limit_usd']:.2f} budget ({pct_remaining:.0f}% remaining)")
        else:
            lines.append(f"Cost: ${c['spent_usd']:.2f} spent (no budget limit)")
        if c["token_cost_usd"] > 0:
            lines.append(f"  Token cost: ${c['token_cost_usd']:.2f}")
        lines.append("")

        se = d["side_effects"]
        if se["attempted"] or se["executed"]:
            executed_total = sum(se["executed"].values())
            attempted_total = sum(se["attempted"].values())
            blocked_total = attempted_total - executed_total
            lines.append(f"Side-effects: {executed_total} executed, {blocked_total} blocked")
            for tool, count in se["executed"].items():
                attempted = se["attempted"].get(tool, count)
                if attempted > count:
                    lines.append(f"  {tool}: {count} executed, {attempted - count} blocked")
                else:
                    lines.append(f"  {tool}: {count} executed")
            lines.append("")

        by_p = d["interventions_by_primitive"]
        if by_p:
            lines.append("Interventions by primitive:")
            for prim, count in sorted(by_p.items(), key=lambda x: -x[1]):
                lines.append(f"  [{count:>3}x] {prim}")
            lines.append("")

        top = d["top_interventions"]
        if top:
            lines.append("Top interventions:")
            for item in top[:5]:
                lines.append(f"  [{item['count']:>3}x] {item['reason']}")
            lines.append("")

        qt = d["quarantined_tools"]
        if qt:
            lines.append("Quarantined tools:")
            for tool, reason in qt.items():
                lines.append(f"  {tool} ({reason})")
            lines.append("")

        if d["stall_streak"] > 0:
            lines.append(f"Stall streak: {d['stall_streak']} consecutive stall detections")
            lines.append("")

        if d["shadow_mode"]:
            lines.append(f"Shadow mode: ON ({d['shadow_would_deny']} calls would have been denied)")
            lines.append("")

        return "\n".join(lines)

    @property
    def summary(self) -> Dict[str, Any]:
        """Alias for stats (for LangChain adapter compatibility)."""
        return self.stats

    def reset(self) -> None:
        """Reset state for a new run (same config)."""
        self._state = self._guard.new_state()
        self._last_call = None
        self.blocks = 0
        self.cache_hits = 0
        self.rewrite_decisions = 0
        self.escalations = 0
        self.tool_calls_executed = 0
        self.tool_calls_failed = 0
        self.tool_calls_denied = 0
        self.shadow_would_deny = 0
        self.missed_results = 0
        self._interventions = []
