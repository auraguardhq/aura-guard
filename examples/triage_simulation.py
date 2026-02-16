"""triage_simulation.py

Demonstrates Aura Guard's value vs. no protection and naive call limits.

Run (no API key needed):
    python examples/triage_simulation.py

Uses a scripted 'bad agent' that:
- Tries to refund the same order 3 times
- Spams a search tool with jittering queries
- Stalls by repeating the same apology
"""

from __future__ import annotations

import os
import sys

# Allow running this example from a repo clone without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from aura_guard import AgentGuard, PolicyAction, ToolCall


# -----------------
# Scripted bad agent
# -----------------

def bad_agent_steps() -> List[Tuple[str, Any]]:
    """Return a list of (kind, payload) steps simulating a broken agent."""
    steps: List[Tuple[str, Any]] = []

    # Phase 1: Side-effect spam (triple refund)
    for _ in range(3):
        steps.append(("tool", ("refund", {"order_id": "o1", "amount": 10}, "t1")))

    # Phase 2: Argument jitter spam on search
    queries = [
        "refund policy",
        "refund policy EU",
        "refund policy Germany",
        "refund policy EU Germany",
        "refund policy EU Germany 2024",
        "refund policy EU Germany 2024",
        "refund policy EU Germany 2024",
        "refund policy EU Germany 2024",
    ]
    for q in queries:
        steps.append(("tool", ("search_kb", {"query": q}, None)))

    # Phase 3: Stall loop
    for _ in range(6):
        steps.append(("llm", "I apologize for the inconvenience. We're looking into it."))

    # Phase 4: Forced outcome compliance
    steps.append((
        "llm",
        '{"action":"finalize","reason":"ready","reply_draft":"Your refund has been processed. Reply with your order number for further help.","escalation":null}',
    ))

    return steps


def mock_tool_execute(name: str, args: Dict[str, Any]) -> Any:
    """Simulate tool execution."""
    if name == "refund":
        return {"status": "refunded", "order_id": args.get("order_id"), "amount": args.get("amount")}
    if name == "search_kb":
        return {"hits": [f"KB:{args.get('query', '')}"]}
    return {"status": "ok"}


# -----------------
# Simulation variants
# -----------------

@dataclass
class RunReport:
    name: str
    tool_calls_executed: int
    side_effects_executed: int
    blocked: int
    cache_hits: int
    rewrites: int
    cost_spent: float
    terminated: Optional[str]


def run_no_guard() -> RunReport:
    """Run with no protection at all."""
    executed = 0
    side_effects = 0
    for kind, payload in bad_agent_steps():
        if kind == "tool":
            name, args, _ = payload
            mock_tool_execute(name, args)
            executed += 1
            if name == "refund":
                side_effects += 1
    return RunReport(
        name="no_guard", tool_calls_executed=executed,
        side_effects_executed=side_effects, blocked=0, cache_hits=0,
        rewrites=0, cost_spent=executed * 0.04, terminated=None,
    )


def run_call_limit(limit: int = 5) -> RunReport:
    """Run with a naive call count limit."""
    executed = 0
    side_effects = 0
    terminated = None
    for kind, payload in bad_agent_steps():
        if kind == "tool":
            if executed >= limit:
                terminated = "call_limit_reached"
                break
            name, args, _ = payload
            mock_tool_execute(name, args)
            executed += 1
            if name == "refund":
                side_effects += 1
    return RunReport(
        name=f"call_limit({limit})", tool_calls_executed=executed,
        side_effects_executed=side_effects, blocked=0, cache_hits=0,
        rewrites=0, cost_spent=executed * 0.04, terminated=terminated,
    )


def run_aura_guard() -> RunReport:
    """Run with Aura Guard protection."""
    guard = AgentGuard(
        secret_key=b"your-secret-key",
        max_cost_per_run=0.50,
        side_effect_tools={"refund", "send_reply", "cancel"},
    )

    executed = 0
    side_effects = 0
    terminated: Optional[str] = None

    for kind, payload in bad_agent_steps():
        if terminated:
            break

        if kind == "tool":
            name, args, ticket_id = payload
            decision = guard.check_tool(name, args=args, ticket_id=ticket_id)

            if decision.action == PolicyAction.ALLOW:
                result = mock_tool_execute(name, args)
                guard.record_result(ok=True, payload=result)
                executed += 1
                if name == "refund":
                    side_effects += 1
            elif decision.action == PolicyAction.CACHE:
                pass  # cached result, no execution
            elif decision.action == PolicyAction.BLOCK:
                pass  # blocked
            elif decision.action == PolicyAction.REWRITE:
                pass  # would inject system message
            elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                terminated = decision.action.value
        else:
            text = payload
            stall = guard.check_output(text)
            if stall:
                if stall.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                    terminated = stall.action.value
                # REWRITE: would inject system message

    return RunReport(
        name="aura_guard", tool_calls_executed=executed,
        side_effects_executed=side_effects,
        blocked=guard.blocks, cache_hits=guard.cache_hits,
        rewrites=guard.rewrites,
        cost_spent=guard.cost_spent,
        terminated=terminated,
    )


# -----------------
# Main
# -----------------

def main() -> None:
    TOOL_COST = 0.04

    a = run_no_guard()
    b = run_call_limit(5)
    c = run_aura_guard()

    print("=" * 60)
    print("  Aura Guard — Triage Simulation")
    print("=" * 60)
    print(f"  Assumed tool-call cost: ${TOOL_COST:.2f} per call\n")

    fmt = "  {:<24} {:>6} {:>8} {:>8} {:>8} {:>8}  {}"
    print(fmt.format("Variant", "Calls", "SideFX", "Blocks", "Cache", "Cost", "Terminated"))
    print("  " + "-" * 80)

    for r in (a, b, c):
        print(fmt.format(
            r.name, r.tool_calls_executed, r.side_effects_executed,
            r.blocked, r.cache_hits,
            f"${r.cost_spent:.2f}", r.terminated or "-",
        ))

    saved = a.cost_spent - c.cost_spent
    pct = (saved / a.cost_spent * 100) if a.cost_spent else 0
    print()
    print(f"  Cost saved vs no_guard: ${saved:.2f} ({pct:.0f}%)")
    print(f"  Side-effects prevented: {a.side_effects_executed - c.side_effects_executed}")
    print()


if __name__ == "__main__":
    main()
