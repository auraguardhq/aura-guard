"""aura_guard.guard

Aura Guard: reliability middleware and runtime enforcement for tool-using agents.

Design goals:
- Framework-agnostic: integrates via on_tool_call_request/on_tool_result/on_llm_output.
- Deterministic recovery primitives for common loop failure modes.
- Safe-by-default: no raw args, payloads, or ticket IDs persisted.
- Sub-millisecond overhead: no LLM calls, no network requests.

7 enforcement primitives:
1. Identical tool-call repeat protection (cache or block)
2. Argument-jitter loop detection for query-like tools
3. Error retry circuit breaker (quarantine failing tools)
4. Side-effect gating + idempotency ledger (no double refunds)
5. No-state-change stall detection (force finalize/escalate)
6. Cost budget enforcement (auto-escalate on budget overrun)
7. Tool policy layer (allow/deny/human-approval per tool + risk class)
"""

from __future__ import annotations

import hmac
import json
import re
import time as _time
import warnings
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from .config import AuraGuardConfig
from .telemetry import Telemetry
from .types import CostEvent, PolicyAction, PolicyDecision, ToolCall, ToolCallSig, ToolResult


# ================================
# Canonicalization & Signatures
# ================================

def _canonicalize(obj: Any) -> Any:
    """Convert `obj` into a JSON-serializable structure with deterministic ordering."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return {"__bytes__": sha256(obj).hexdigest()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, set):
        return sorted((_canonicalize(x) for x in obj), key=lambda x: str(x))
    if isinstance(obj, dict):
        items = [(str(k), _canonicalize(v)) for k, v in obj.items()]
        return {k: v for k, v in sorted(items, key=lambda kv: kv[0])}
    return str(obj)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(_canonicalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hmac_hex(secret_key: bytes, message: str) -> str:
    return hmac.new(secret_key, message.encode("utf-8"), sha256).hexdigest()


def _args_sig(cfg: AuraGuardConfig, args: Dict[str, Any], tool_name: str = "") -> str:
    filtered = args
    if tool_name:
        ignore = cfg.get_arg_ignore_keys(tool_name)
        if ignore:
            filtered = {k: v for k, v in args.items() if k not in ignore}
    return _hmac_hex(cfg.secret_key, "args:" + _stable_json_dumps(filtered))


def _ticket_sig(cfg: AuraGuardConfig, ticket_id: Optional[str], fallback_run_id: str) -> str:
    if ticket_id:
        return _hmac_hex(cfg.secret_key, "ticket:" + str(ticket_id))
    return _hmac_hex(cfg.secret_key, "run:" + fallback_run_id)


def _payload_sig(cfg: AuraGuardConfig, payload: Any) -> str:
    return _hmac_hex(cfg.secret_key, "payload:" + _stable_json_dumps(payload))


def _now() -> float:
    """Monotonic time for cache TTL."""
    return _time.monotonic()


# ================================
# Text Token Signatures
# ================================

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Pre-compiled stall pattern detectors (Primitive 5 enhancement)
# These catch "apology loops" and "still working" loops that token overlap misses
# because the phrasing varies but the intent is identical.
_STALL_APOLOGY_RE = re.compile(
    r"\b(i\s+apologize|i'?m\s+sorry|apologies|sorry\s+for)\b", re.I
)
_STALL_WORKING_RE = re.compile(
    r"\b(still\s+(looking|working|checking|searching|researching|investigating)"
    r"|let\s+me\s+(check|look|try|search|investigate)"
    r"|i'?m\s+(looking|working|checking|searching|researching|investigating)"
    r"|looking\s+into\s+(this|it|that)"
    r"|bear\s+with\s+me"
    r"|one\s+moment"
    r"|please\s+wait"
    r"|thank\s+you\s+for\s+(your\s+)?patience)\b", re.I
)
_STALL_FILLER_RE = re.compile(
    r"\b(inconvenience|delay|trouble|difficulty|issue|problem)\b", re.I
)


def _stall_pattern_score(text: str) -> float:
    """Score text for stall-like patterns (0.0 = no stall signals, 1.0 = maximum stall).

    Independent of token overlap — catches semantically similar but differently
    worded apology/filler loops.
    """
    score = 0.0
    if _STALL_APOLOGY_RE.search(text):
        score += 0.4
    if _STALL_WORKING_RE.search(text):
        score += 0.4
    if _STALL_FILLER_RE.search(text):
        score += 0.2
    return min(score, 1.0)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _token_sig_set(cfg: AuraGuardConfig, text: str) -> Set[str]:
    """Compute HMAC-signed token set for similarity without storing raw text."""
    toks = _tokenize(text)
    return {_hmac_hex(cfg.secret_key, "tok:" + t) for t in toks}


def _overlap_similarity(a: Set[str], b: Set[str]) -> float:
    """Overlap coefficient: |A∩B| / min(|A|,|B|).

    Better than Jaccard for "argument jitter" where a query accumulates
    extra qualifiers (A is a subset of B).
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0
    return inter / denom


# ================================
# Error Classification
# ================================

def _classify_error_code(error_code: Optional[str]) -> str:
    if not error_code:
        return "unknown"
    s = str(error_code).strip().lower()

    if s in {"timeout", "timed_out", "read_timeout", "write_timeout"}:
        return "timeout"
    if s in {"conn", "connection", "connection_error", "network_error"}:
        return "connection_error"
    if s.isdigit():
        if s.startswith("5"):
            return "5xx"
        return s
    if "timeout" in s:
        return "timeout"
    if "rate" in s or "429" in s:
        return "429"
    return s


# ================================
# Guard State
# ================================

@dataclass
class GuardState:
    """Per-run state. Create a new one for each agent run.

    .. warning:: NOT THREAD-SAFE

       GuardState is a plain dataclass with no locking. Do not share a single
       GuardState instance across threads or async tasks without external
       synchronization. For concurrent agent runs, create one GuardState per
       run. For distributed/multi-process agents, serialize and share state
       explicitly using :mod:`aura_guard.serialization`.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex)

    # Rolling history (request signatures)
    tool_stream: List[ToolCallSig] = field(default_factory=list)

    # Per-tool call counts (for max_calls_per_tool enforcement)
    tool_call_counts: Dict[str, int] = field(default_factory=dict)

    # Tool → list of hashed token sets for query-like args (rolling)
    tool_query_sigs: Dict[str, List[Set[str]]] = field(default_factory=dict)

    # Caches
    result_cache: Dict[Tuple[str, str], ToolResult] = field(default_factory=dict)
    result_cache_ts: Dict[Tuple[str, str], float] = field(default_factory=dict)  # monotonic timestamps

    # Side-effect idempotency ledger
    idempotency_ledger: Dict[Tuple[str, str, str], ToolResult] = field(default_factory=dict)

    # Error tracking
    error_streaks: Dict[Tuple[str, str], int] = field(default_factory=dict)
    quarantined_tools: Dict[str, str] = field(default_factory=dict)

    # Side-effect accounting
    attempted_side_effect_calls: Dict[str, int] = field(default_factory=dict)
    executed_side_effect_calls: Dict[str, int] = field(default_factory=dict)

    # Progress tracking (primitive 5)
    unique_tool_calls_seen: Set[Tuple[str, str]] = field(default_factory=set)
    unique_tool_results_seen: Set[str] = field(default_factory=set)
    last_progress_marker: Tuple[int, int] = (0, 0)
    last_assistant_token_sigs: Optional[Set[str]] = None
    stall_streak: int = 0
    stall_pattern_streak: int = 0
    stall_rewrite_attempts: int = 0

    # Cost tracking (primitive 6)
    cumulative_cost: float = 0.0
    reported_token_cost: float = 0.0   # actual token costs reported by integrator
    cost_events: List[CostEvent] = field(default_factory=list)
    budget_warning_emitted: bool = False


# ================================
# Aura Guard Engine
# ================================

class AuraGuard:
    """Cost governance and loop prevention for AI agents.

    Call on_tool_call_request() before each tool execution,
    on_tool_result() after each tool execution,
    and on_llm_output() after each assistant response.
    """

    def __init__(
        self,
        config: Optional[AuraGuardConfig] = None,
        telemetry: Optional[Telemetry] = None,
    ):
        self.cfg = config or AuraGuardConfig()
        self.telemetry = telemetry

        if self.cfg.secret_key == b"aura-guard-dev-key-CHANGE-ME":
            warnings.warn(
                "AuraGuard is using the default development secret_key. "
                "HMAC signatures will be predictable. Set a unique secret_key "
                "in AuraGuardConfig for production use.",
                stacklevel=2,
            )

    # -------------------------
    # Internal helpers
    # -------------------------

    def _emit(self, event: str, *, state: Optional[GuardState] = None, **fields: Any) -> None:
        if not self.telemetry:
            return
        if state is not None:
            fields["run_id"] = state.run_id
        self.telemetry.emit(event, **fields)

    def _estimate_tool_cost(self, tool_name: str) -> float:
        return self.cfg.cost_model.tool_cost(tool_name)

    def _cache_valid(self, state: GuardState, cache_key: Tuple[str, str]) -> bool:
        """Check if a cached result is still valid (TTL not expired)."""
        if self.cfg.cache_ttl_seconds is None:
            return True  # no TTL = valid forever within run
        ts = state.result_cache_ts.get(cache_key)
        if ts is None:
            return True  # no timestamp = legacy entry, treat as valid
        return (_now() - ts) < self.cfg.cache_ttl_seconds

    def record_tokens(self, *, state: GuardState, input_tokens: int = 0,
                      output_tokens: int = 0, cost_override: Optional[float] = None) -> None:
        """Report actual token usage for accurate cost tracking.

        Call this after each LLM call with the real token counts from the API
        response. If cost_override is provided, it's used directly instead of
        computing from the cost model.

        Example:
            guard.record_tokens(state=state, input_tokens=resp.usage.input_tokens,
                                output_tokens=resp.usage.output_tokens)
        """
        if cost_override is not None:
            actual = cost_override
        else:
            actual = self.cfg.cost_model.token_cost(input_tokens, output_tokens)
        state.reported_token_cost += actual
        state.cumulative_cost += actual
        state.cost_events.append(CostEvent(
            event="token_cost_reported", tool="llm",
            amount=actual, cumulative=state.cumulative_cost,
        ))
        self._emit(
            "token_cost_reported",
            state=state,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=round(actual, 6),
            cumulative_cost=round(state.cumulative_cost, 4),
        )

    def _forced_outcome_system_prompt(self) -> str:
        """System instruction used when we detect a stall/no-state-change loop."""
        return (
            "SYSTEM ALERT: You are stuck in a loop. Do NOT call any tools.\n"
            "Output exactly ONE JSON object and nothing else.\n"
            "Schema:\n"
            "{\n"
            '  "action": "finalize" | "escalate",\n'
            '  "reason": string,\n'
            '  "reply_draft": string | null,\n'
            '  "escalation": {\n'
            '    "route": string | null,\n'
            '    "summary": string,\n'
            '    "missing_info": [string],\n'
            '    "tags": [string]\n'
            '  } | null\n'
            "}\n"
            "Rules:\n"
            "- If you have enough facts to answer safely, use action=finalize and write reply_draft.\n"
            "- Otherwise use action=escalate with a concise escalation packet.\n"
            "- Do not include markdown, commentary, or additional text."
        )

    def _parse_forced_outcome(self, text: str) -> Optional[PolicyDecision]:
        """Parse JSON output from forced outcome prompt."""
        t = text.strip()
        if not (t.startswith("{") and t.endswith("}")):
            return None

        try:
            obj = json.loads(t)
        except Exception:
            return None

        if not isinstance(obj, dict):
            return None

        action = obj.get("action")
        reason = obj.get("reason")
        if action not in {"finalize", "escalate"}:
            return None
        if not isinstance(reason, str) or not reason.strip():
            return None

        if action == "finalize":
            reply = obj.get("reply_draft")
            if not isinstance(reply, str) or not reply.strip():
                return None
            return PolicyDecision(
                action=PolicyAction.FINALIZE,
                reason=f"forced_finalize:{reason.strip()}",
                finalized_output={"reply_draft": reply, "reason": reason.strip()},
            )

        # Escalate
        esc = obj.get("escalation")
        if not isinstance(esc, dict):
            esc = {"route": None, "summary": "(missing)", "missing_info": [], "tags": []}

        packet = {
            "route": esc.get("route"),
            "summary": esc.get("summary") if isinstance(esc.get("summary"), str) else "(missing)",
            "missing_info": esc.get("missing_info") if isinstance(esc.get("missing_info"), list) else [],
            "tags": esc.get("tags") if isinstance(esc.get("tags"), list) else [],
            "reason": reason.strip(),
        }
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            reason=f"forced_escalate:{reason.strip()}",
            escalation_packet=packet,
        )

    # -------------------------
    # Public API
    # -------------------------

    def new_state(self, run_id: Optional[str] = None) -> GuardState:
        """Create a fresh GuardState for a new agent run."""
        return GuardState(run_id=run_id or uuid4().hex)

    def on_tool_call_request(self, *, state: GuardState, call: ToolCall) -> PolicyDecision:
        """Evaluate a tool call before executing it.

        Returns a PolicyDecision telling the orchestrator what to do.
        """
        tool = call.name
        side_effect = self.cfg.is_side_effect_tool(tool, call.side_effect)
        call.side_effect = side_effect

        args_sig = _args_sig(self.cfg, call.args, tool_name=tool)
        t_sig = _ticket_sig(self.cfg, call.ticket_id, state.run_id)

        sig = ToolCallSig(name=tool, args_sig=args_sig, ticket_sig=t_sig, side_effect=side_effect)

        # ──────────────────────────────────────────
        # Primitive 7: Tool policy layer (deny/approve)
        # ──────────────────────────────────────────
        policy = self.cfg.get_tool_policy(tool)
        if policy is not None:
            from .config import ToolAccess
            if policy.access == ToolAccess.DENY:
                reason = policy.deny_reason or f"tool_denied_by_policy"
                self._emit("tool_policy_denied", state=state, tool=tool, reason=reason)
                return PolicyDecision(
                    action=PolicyAction.BLOCK,
                    reason=f"policy_deny:{reason}",
                )
            if policy.access == ToolAccess.HUMAN_APPROVAL:
                self._emit("tool_policy_human_approval", state=state, tool=tool, risk=policy.risk)
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason="policy_human_approval_required",
                    escalation_packet={
                        "route": "human_approval",
                        "summary": f"Tool '{tool}' requires human approval (risk: {policy.risk})",
                        "tool": tool,
                        "args_sig": args_sig,
                        "risk": policy.risk,
                        "tags": ["human_approval", f"risk:{policy.risk}"],
                    },
                )
            # Check policy-level required args
            if policy.require_args:
                missing = policy.require_args - set(call.args.keys())
                if missing:
                    return PolicyDecision(
                        action=PolicyAction.BLOCK,
                        reason=f"policy_missing_args:{','.join(sorted(missing))}",
                    )

        # Store request signature in rolling window
        state.tool_stream.append(sig)
        if len(state.tool_stream) > self.cfg.tool_loop_window:
            state.tool_stream = state.tool_stream[-self.cfg.tool_loop_window:]

        # ──────────────────────────────────────────
        # Primitive 6: Cost budget enforcement
        # ──────────────────────────────────────────
        if self.cfg.max_cost_per_run is not None:
            estimated = self._estimate_tool_cost(tool)
            projected = state.cumulative_cost + estimated

            if round(projected, 8) >= round(self.cfg.max_cost_per_run, 8):
                evt = CostEvent(
                    event="budget_exceeded", tool=tool,
                    amount=estimated, cumulative=state.cumulative_cost,
                    limit=self.cfg.max_cost_per_run,
                    pct=round(projected / self.cfg.max_cost_per_run * 100, 1),
                )
                state.cost_events.append(evt)
                self._emit(
                    "budget_exceeded_escalate",
                    state=state,
                    tool=tool,
                    cumulative_cost=round(state.cumulative_cost, 4),
                    projected_cost=round(projected, 4),
                    limit=self.cfg.max_cost_per_run,
                    estimated_cost_avoided=round(estimated, 4),
                )
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason="cost_budget_exceeded",
                    escalation_packet={
                        "route": None,
                        "summary": (
                            f"Agent run exceeded cost budget "
                            f"(${state.cumulative_cost:.2f} spent / "
                            f"${self.cfg.max_cost_per_run:.2f} limit)"
                        ),
                        "tags": ["budget_exceeded"],
                        "cumulative_cost": round(state.cumulative_cost, 4),
                        "limit": self.cfg.max_cost_per_run,
                    },
                )

            # Budget warning (emit once)
            warn_threshold = self.cfg.max_cost_per_run * self.cfg.cost_warning_threshold
            if round(projected, 8) >= round(warn_threshold, 8) and not state.budget_warning_emitted:
                state.budget_warning_emitted = True
                pct = round(projected / self.cfg.max_cost_per_run * 100, 1)
                evt = CostEvent(
                    event="budget_warning", tool=tool,
                    amount=estimated, cumulative=state.cumulative_cost,
                    limit=self.cfg.max_cost_per_run, pct=pct,
                )
                state.cost_events.append(evt)
                self._emit(
                    "budget_warning",
                    state=state,
                    tool=tool,
                    cumulative_cost=round(state.cumulative_cost, 4),
                    projected_cost=round(projected, 4),
                    limit=self.cfg.max_cost_per_run,
                    pct=pct,
                )

        # ──────────────────────────────────────────
        # Primitive 3 (enforcement): Quarantine check
        # ──────────────────────────────────────────
        if tool in state.quarantined_tools:
            reason = state.quarantined_tools[tool]
            injected = (
                f"SYSTEM ALERT: Tool '{tool}' is unavailable ({reason}). "
                "Do not call it again in this run. If you cannot proceed safely, "
                "output an escalation packet."
            )
            self._emit(
                "tool_quarantined_block",
                state=state,
                tool=tool, reason=reason, args_sig=args_sig,
                estimated_cost_avoided=round(self._estimate_tool_cost(tool), 4),
            )
            return PolicyDecision(
                action=PolicyAction.REWRITE,
                reason=f"tool_quarantined:{reason}",
                injected_system=injected,
            )

        # ──────────────────────────────────────────
        # Primitive 4: Side-effect gating + idempotency
        # ──────────────────────────────────────────
        if side_effect:
            state.attempted_side_effect_calls[tool] = (
                state.attempted_side_effect_calls.get(tool, 0) + 1
            )

            # Idempotency check
            key = (t_sig, tool, args_sig)
            if key in state.idempotency_ledger:
                cached = state.idempotency_ledger[key]
                cached2 = ToolResult(
                    ok=cached.ok, payload=cached.payload,
                    error_code=cached.error_code,
                    payload_sig=cached.payload_sig,
                    cached=True,
                    side_effect_executed=cached.side_effect_executed,
                )
                est = self._estimate_tool_cost(tool)
                self._emit(
                    "idempotent_replay_blocked",
                    state=state,
                    tool=tool, ticket_sig=t_sig, args_sig=args_sig,
                    estimated_cost_avoided=round(est, 4),
                )
                return PolicyDecision(
                    action=PolicyAction.CACHE,
                    reason="idempotent_replay",
                    cached_result=cached2,
                )

            # Max execution limit
            executed = state.executed_side_effect_calls.get(tool, 0)
            if executed >= self.cfg.side_effect_max_executed_per_run:
                self._emit(
                    "side_effect_limit_block",
                    state=state,
                    tool=tool, ticket_sig=t_sig, args_sig=args_sig,
                    executed=executed,
                    limit=self.cfg.side_effect_max_executed_per_run,
                    estimated_cost_avoided=round(self._estimate_tool_cost(tool), 4),
                )
                return PolicyDecision(
                    action=PolicyAction.BLOCK,
                    reason="side_effect_limit_exceeded",
                )

            # Set deterministic idempotency key
            call.idempotency_key = _hmac_hex(
                self.cfg.secret_key, f"idem:{t_sig}:{tool}:{args_sig}"
            )[:32]

        # ──────────────────────────────────────────
        # Primitive 1: Identical tool-call loop
        # ──────────────────────────────────────────
        repeats = sum(
            1 for s in state.tool_stream if s.name == tool and s.args_sig == args_sig
        )
        if repeats >= self.cfg.repeat_toolcall_threshold:
            cache_key = (tool, args_sig)
            if cache_key in state.result_cache and self._cache_valid(state, cache_key):
                cached = state.result_cache[cache_key]
                cached2 = ToolResult(
                    ok=cached.ok, payload=cached.payload,
                    error_code=cached.error_code,
                    payload_sig=cached.payload_sig,
                    cached=True,
                    side_effect_executed=cached.side_effect_executed,
                )
                est = self._estimate_tool_cost(tool)
                self._emit(
                    "tool_call_cache_hit",
                    state=state,
                    tool=tool, args_sig=args_sig,
                    repeats=repeats,
                    estimated_cost_avoided=round(est, 4),
                )
                return PolicyDecision(
                    action=PolicyAction.CACHE,
                    reason="identical_toolcall_loop_cache",
                    cached_result=cached2,
                )

            self._emit(
                "identical_toolcall_loop_block",
                state=state,
                tool=tool, args_sig=args_sig,
                repeats=repeats,
                estimated_cost_avoided=round(self._estimate_tool_cost(tool), 4),
            )
            return PolicyDecision(
                action=PolicyAction.BLOCK,
                reason="identical_toolcall_loop",
            )

        # ──────────────────────────────────────────
        # Primitive 2: Argument-jitter similarity
        # ──────────────────────────────────────────
        query_keys = self.cfg.query_arg_keys(tool)
        q_val: Optional[str] = None
        for k in query_keys:
            v = call.args.get(k)
            if isinstance(v, str) and v.strip():
                q_val = v
                break

        if q_val is not None:
            q_sig_set = _token_sig_set(self.cfg, q_val)
            hist = state.tool_query_sigs.setdefault(tool, [])

            similar = sum(
                1
                for prev in hist[-self.cfg.tool_loop_window:]
                if _overlap_similarity(prev, q_sig_set) >= self.cfg.arg_jitter_similarity_threshold
            )

            hist.append(q_sig_set)
            if len(hist) > self.cfg.tool_loop_window:
                state.tool_query_sigs[tool] = hist[-self.cfg.tool_loop_window:]

            if similar >= self.cfg.arg_jitter_repeat_threshold:
                # Quarantine the tool for the remainder of the run
                state.quarantined_tools[tool] = "arg_jitter_loop"

                injected = (
                    f"SYSTEM ALERT: You are repeatedly calling '{tool}' with near-identical queries. "
                    "Do not vary the query further and do not call this tool again. "
                    "Use the best available prior result to proceed. "
                    "If you cannot proceed safely, output an escalation packet."
                )
                self._emit(
                    "arg_jitter_loop_quarantine",
                    state=state,
                    tool=tool, args_sig=args_sig, similar=similar,
                    estimated_cost_avoided=round(self._estimate_tool_cost(tool), 4),
                )
                return PolicyDecision(
                    action=PolicyAction.REWRITE,
                    reason="arg_jitter_loop",
                    injected_system=injected,
                )

        # ──────────────────────────────────────────
        # Primitive 2b: Per-tool call cap
        # ──────────────────────────────────────────
        cap_decision = self._check_tool_call_cap(state, tool)
        if cap_decision is not None:
            return cap_decision

        return PolicyDecision(action=PolicyAction.ALLOW, reason="allow")

    def _check_tool_call_cap(self, state: GuardState, tool: str) -> Optional[PolicyDecision]:
        """Check per-tool call cap (catches smart reformulation loops).

        Uses per-tool policy limit if set, otherwise global max_calls_per_tool.
        Returns a REWRITE decision if the cap is exceeded, else None.
        """
        cap = self.cfg.get_tool_max_calls(tool)
        if cap is None:
            return None

        count = state.tool_call_counts.get(tool, 0)
        if count >= cap:
            state.quarantined_tools[tool] = "max_calls_per_tool"
            injected = (
                f"SYSTEM ALERT: You have called '{tool}' {count} times this run. "
                "You must now use the information you have already gathered. "
                "Do not call this tool again. Synthesize your answer from prior results."
            )
            self._emit(
                "tool_call_cap_quarantine",
                state=state,
                tool=tool, count=count,
                limit=cap,
            )
            return PolicyDecision(
                action=PolicyAction.REWRITE,
                reason="max_calls_per_tool",
                injected_system=injected,
            )

        state.tool_call_counts[tool] = count + 1
        return None

    def on_tool_result(self, *, state: GuardState, call: ToolCall, result: ToolResult) -> None:
        """Update state after a tool call completes (or is served from cache)."""
        tool = call.name
        side_effect = self.cfg.is_side_effect_tool(tool, call.side_effect)

        args_sig = _args_sig(self.cfg, call.args, tool_name=tool)
        t_sig = _ticket_sig(self.cfg, call.ticket_id, state.run_id)

        # Fill payload signature if missing
        if result.payload_sig is None and result.payload is not None:
            result.payload_sig = _payload_sig(self.cfg, result.payload)

        # Infer side_effect_executed when possible
        if result.side_effect_executed is None:
            result.side_effect_executed = bool(side_effect and result.ok and not result.cached)

        # Cache successful non-cached results (respecting never_cache_tools)
        if result.ok and not result.cached and self.cfg.is_cacheable(tool):
            # Enforce state growth cap on result_cache
            if len(state.result_cache) < self.cfg.max_cache_entries:
                state.result_cache[(tool, args_sig)] = result
                state.result_cache_ts[(tool, args_sig)] = _now()

            # Update "unique" progress markers (bounded)
            if len(state.unique_tool_calls_seen) < self.cfg.max_unique_calls_tracked:
                state.unique_tool_calls_seen.add((tool, args_sig))
            if result.payload_sig and len(state.unique_tool_results_seen) < self.cfg.max_unique_calls_tracked:
                state.unique_tool_results_seen.add(result.payload_sig)

        # Side-effect ledger and executed counts
        if side_effect:
            key = (t_sig, tool, args_sig)
            if result.ok:
                state.idempotency_ledger[key] = result

            if result.side_effect_executed:
                state.executed_side_effect_calls[tool] = (
                    state.executed_side_effect_calls.get(tool, 0) + 1
                )

        # Primitive 3: Error retry circuit breaker
        if not result.ok:
            err_class = _classify_error_code(result.error_code)
            streak_key = (tool, err_class)
            state.error_streaks[streak_key] = state.error_streaks.get(streak_key, 0) + 1

            if (
                state.error_streaks[streak_key] >= self.cfg.error_retry_threshold
                and self.cfg.should_quarantine_error(err_class)
            ):
                state.quarantined_tools[tool] = f"error_retry:{err_class}"
                self._emit(
                    "tool_quarantined_error_retry",
                    state=state,
                    tool=tool, error_class=err_class,
                    streak=state.error_streaks[streak_key],
                )

        # Primitive 6: Cost tracking
        # Every executed tool call costs money — failures are NOT free.
        # Only cached results (never sent to the API) are free.
        if not result.cached:
            est = self._estimate_tool_cost(tool)
            state.cumulative_cost += est
            state.cost_events.append(CostEvent(
                event="cost_incurred", tool=tool,
                amount=est, cumulative=state.cumulative_cost,
            ))

    def on_llm_output(self, *, state: GuardState, text: str) -> Optional[PolicyDecision]:
        """Inspect an assistant output to detect stall/no-state-change."""

        # If we previously asked for a forced outcome, try to parse it.
        if state.stall_rewrite_attempts > 0:
            parsed = self._parse_forced_outcome(text)
            if parsed is not None:
                self._emit(
                    "forced_outcome_parsed",
                    state=state,
                    action=parsed.action.value,
                    reason=parsed.reason,
                )
                return parsed

            # Exhausted rewrite attempts → deterministic escalation
            if state.stall_rewrite_attempts >= self.cfg.stall_rewrite_max_attempts:
                packet = {
                    "route": None,
                    "summary": "Agent failed to comply with forced finalize/escalate after stall detection.",
                    "missing_info": [],
                    "tags": ["loop_detected", "stall", "noncompliant"],
                    "reason": "stall",
                }
                self._emit("stall_deterministic_escalate_noncompliant", state=state)
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason="stall_noncompliant_forced_outcome",
                    escalation_packet=packet,
                )

        # Primitive 5: Stall detection (dual approach)
        #
        # Two independent signals, either can trigger intervention:
        # A) Token overlap: identical or near-identical text repeated (threshold 0.92)
        # B) Stall patterns: apology/filler language with no tool progress (catches
        #    "I apologize" → "Sorry for the delay" → "I'm still looking" which have
        #    LOW token overlap but HIGH stall intent)
        #
        cur_marker = (len(state.unique_tool_calls_seen), len(state.unique_tool_results_seen))
        if cur_marker != state.last_progress_marker:
            # Progress happened via tools; reset ALL stall counters.
            state.last_progress_marker = cur_marker
            state.stall_streak = 0
            state.stall_pattern_streak = 0
            state.last_assistant_token_sigs = _token_sig_set(self.cfg, text)
            return None

        # No new tool progress. Check both signals.

        # Signal A: Token overlap similarity
        cur_tokens = _token_sig_set(self.cfg, text)
        prev_tokens = state.last_assistant_token_sigs

        if prev_tokens is None:
            state.last_assistant_token_sigs = cur_tokens
            state.stall_streak = 0
            # Still check pattern on first message (no overlap baseline yet)
        else:
            sim = _overlap_similarity(prev_tokens, cur_tokens)
            if sim >= self.cfg.stall_text_similarity_threshold:
                state.stall_streak += 1
            else:
                state.stall_streak = 0
            state.last_assistant_token_sigs = cur_tokens

        # Signal B: Stall pattern detection (apology/filler/working language)
        pattern_score = _stall_pattern_score(text)
        if pattern_score >= self.cfg.stall_pattern_threshold:
            state.stall_pattern_streak += 1
        else:
            state.stall_pattern_streak = 0

        # Trigger intervention if EITHER signal exceeds threshold
        should_intervene = (
            state.stall_streak >= self.cfg.no_state_change_threshold
            or state.stall_pattern_streak >= self.cfg.stall_pattern_streak_threshold
        )

        if should_intervene:
            # First attempt: force the model to produce a final outcome.
            if state.stall_rewrite_attempts < self.cfg.stall_rewrite_max_attempts:
                state.stall_rewrite_attempts += 1
                # Capture trigger reason BEFORE resetting streaks
                trigger = "pattern" if state.stall_pattern_streak >= self.cfg.stall_pattern_streak_threshold else "similarity"
                state.stall_streak = 0
                state.stall_pattern_streak = 0

                injected = self._forced_outcome_system_prompt()
                self._emit(
                    "stall_forced_rewrite",
                    state=state,
                    trigger=trigger,
                    pattern_score=round(pattern_score, 3),
                    attempts=state.stall_rewrite_attempts,
                )
                return PolicyDecision(
                    action=PolicyAction.REWRITE,
                    reason="no_state_change_loop",
                    injected_system=injected,
                )

            # Deterministic fallback escalation.
            packet = {
                "route": None,
                "summary": "Agent stalled (no-state-change loop detected).",
                "missing_info": [],
                "tags": ["loop_detected", "stall"],
                "reason": "stall",
            }
            self._emit("stall_deterministic_escalate", state=state)
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                reason="stall_fallback_escalate",
                escalation_packet=packet,
            )

        return None

    def get_run_summary(self, state: GuardState) -> Dict[str, Any]:
        """Return a summary of guard activity for this run."""
        return {
            "run_id": state.run_id,
            "cumulative_cost_usd": round(state.cumulative_cost, 4),
            "cost_limit_usd": self.cfg.max_cost_per_run,
            "tools_quarantined": dict(state.quarantined_tools),
            "side_effects_attempted": dict(state.attempted_side_effect_calls),
            "side_effects_executed": dict(state.executed_side_effect_calls),
            "unique_tool_calls": len(state.unique_tool_calls_seen),
            "unique_tool_results": len(state.unique_tool_results_seen),
            "stall_streak": state.stall_streak,
            "stall_rewrite_attempts": state.stall_rewrite_attempts,
            "cost_events": len(state.cost_events),
        }
