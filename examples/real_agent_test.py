"""real_agent_test.py

REAL-WORLD integration test: connects to OpenAI API, runs an agent
with tools, deliberately provokes loops, and shows Aura Guard catching
them live.

Setup:
    pip install openai
    set OPENAI_API_KEY=sk-...     (Windows)
    export OPENAI_API_KEY=sk-...  (Mac/Linux)

Run:
    python examples/real_agent_test.py --max-turns 20

Optional:
    # Choose a model (or set OPENAI_MODEL)
    python examples/real_agent_test.py --model gpt-4o-mini

What this does:
1. Gives GPT-4o-mini a customer support task with 3 tools
2. The tools are rigged to return unhelpful results (forcing the agent to retry)
3. WITHOUT Aura Guard: the agent loops 15-25 times burning money
4. WITH Aura Guard: loops are caught after 3-4 calls, agent forced to resolve

You'll see real-time output of every tool call and every guard decision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: Install openai first:")
    print("  pip install openai")
    sys.exit(1)

# Add parent src to path if not installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aura_guard import AgentGuard, PolicyAction
from aura_guard.telemetry import InMemoryTelemetry, Telemetry


# ─────────────────────────────────────
# Tool definitions (for OpenAI function calling)
# ─────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the knowledge base for policy documents and FAQs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order by ID to get its status, items, and shipping info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID (e.g. ORD-12345)",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund",
            "description": "Process a refund for an order. This is irreversible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to refund",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Refund amount in USD",
                    },
                },
                "required": ["order_id", "amount"],
            },
        },
    },
]


# ─────────────────────────────────────
# Rigged tool implementations
# These deliberately return unhelpful results to provoke loops
# ─────────────────────────────────────

_search_call_count = 0


def execute_search_kb(args: Dict[str, Any]) -> str:
    """Returns vague results that tempt the agent to search again."""
    global _search_call_count
    _search_call_count += 1

    # First few results are deliberately unhelpful
    if _search_call_count <= 3:
        return json.dumps({
            "results": [
                {"title": "General FAQ", "snippet": "For refund inquiries, please check our policy page."},
                {"title": "Contact Us", "snippet": "Email support@example.com for assistance."},
            ],
            "total_hits": 47,
            "note": "Try refining your search for more specific results."
        })
    # Eventually return something useful (if the agent gets this far)
    return json.dumps({
        "results": [
            {"title": "Refund Policy", "snippet": "Full refunds available within 30 days. Partial refunds after 30 days at manager discretion."},
        ],
        "total_hits": 1,
    })


def execute_get_order(args: Dict[str, Any]) -> str:
    """Returns order info."""
    return json.dumps({
        "order_id": args.get("order_id", "ORD-12345"),
        "status": "delivered",
        "items": [{"name": "Wireless Headphones", "price": 79.99, "qty": 1}],
        "delivered_at": "2025-01-15",
        "total": 79.99,
    })


def execute_refund(args: Dict[str, Any]) -> str:
    """Processes a refund."""
    return json.dumps({
        "status": "refunded",
        "order_id": args.get("order_id"),
        "amount": args.get("amount"),
        "refund_id": "REF-98765",
    })


TOOL_EXECUTORS = {
    "search_kb": execute_search_kb,
    "get_order": execute_get_order,
    "refund": execute_refund,
}


# ─────────────────────────────────────
# Agent loop
# ─────────────────────────────────────

SYSTEM_PROMPT = """You are a customer support agent. You have access to tools to help customers.
Always search the knowledge base before answering policy questions.
Always look up the order before processing a refund.
Be thorough - if the search results aren't specific enough, try different queries."""

CUSTOMER_MESSAGE = """Hi, I bought wireless headphones (order ORD-12345) three weeks ago and
they stopped working. I'd like a full refund. What's your refund policy for defective items?"""


def run_agent(
    *,
    use_guard: bool,
    max_turns: int = 20,
    model: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the agent loop with or without Aura Guard.

    Returns a summary dict with costs and actions taken.
    """
    global _search_call_count
    _search_call_count = 0

    # Pick a model (CLI flag > env var > default)
    model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"

    client = OpenAI()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CUSTOMER_MESSAGE},
    ]

    guard: Optional[AgentGuard] = None
    telemetry_sink: Optional[InMemoryTelemetry] = None

    if use_guard:
        telemetry_sink = InMemoryTelemetry()
        guard = AgentGuard(
            secret_key=b"your-secret-key",
            max_cost_per_run=0.50,
            side_effect_tools={"refund"},
            tool_costs={"search_kb": 0.03, "get_order": 0.04, "refund": 0.05},
            telemetry=Telemetry(sink=telemetry_sink),
        )

    label = "WITH GUARD" if use_guard else "NO GUARD"
    tool_calls_executed = 0
    tool_calls_blocked = 0
    side_effects = 0
    turns = 0
    terminated = None
    api_calls = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Running agent: {label}")
        print(f"{'='*60}")

    for turn in range(max_turns):
        turns = turn + 1

        # Call the LLM
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            api_calls += 1
        except Exception as e:
            if verbose:
                print(f"  [Turn {turns}] API ERROR: {e}")
            terminated = f"api_error:{type(e).__name__}"
            break

        msg = response.choices[0].message

        # Check for stall (text output with no tool calls)
        if msg.content and not msg.tool_calls:
            if verbose:
                text_preview = msg.content[:100].replace('\n', ' ')
                print(f"  [Turn {turns}] ASSISTANT: {text_preview}...")

            if guard:
                stall = guard.check_output(msg.content)
                if stall:
                    if verbose:
                        print(f"  [Turn {turns}] GUARD: {stall.action.value} — {stall.reason}")
                    if stall.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                        terminated = stall.action.value
                        break
                    elif stall.action == PolicyAction.REWRITE:
                        messages.append({"role": "system", "content": stall.injected_system or ""})
                        continue

            # Agent gave a final response — done
            messages.append({"role": "assistant", "content": msg.content})
            break

        # Handle tool calls
        if msg.tool_calls:
            messages.append(msg)  # Add assistant message with tool calls

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                if verbose:
                    args_preview = json.dumps(fn_args)[:60]
                    print(f"  [Turn {turns}] TOOL CALL: {fn_name}({args_preview})")

                # Guard check
                if guard:
                    decision = guard.check_tool(
                        fn_name,
                        args=fn_args,
                        ticket_id="ticket-real-test",
                    )

                    if decision.action != PolicyAction.ALLOW:
                        if verbose:
                            print(f"  [Turn {turns}] GUARD: {decision.action.value} — {decision.reason}")
                        tool_calls_blocked += 1

                        if decision.action == PolicyAction.CACHE:
                            # Return cached result
                            cached_payload = decision.cached_result.payload if decision.cached_result else '{"cached": true}'
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": str(cached_payload),
                            })
                        elif decision.action == PolicyAction.BLOCK:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({"error": "Tool call blocked by guard", "reason": decision.reason}),
                            })
                        elif decision.action == PolicyAction.REWRITE:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({"error": "Tool quarantined", "reason": decision.reason}),
                            })
                            messages.append({"role": "system", "content": decision.injected_system or ""})
                        elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                            terminated = decision.action.value
                            # Still need to provide tool result for API consistency
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({"error": "Run terminated", "reason": decision.reason}),
                            })
                            break
                        continue

                # Execute the tool
                executor = TOOL_EXECUTORS.get(fn_name)
                if executor:
                    result = executor(fn_args)
                    tool_calls_executed += 1
                    if fn_name == "refund":
                        side_effects += 1

                    if guard:
                        guard.record_result(ok=True, payload=result)

                    if verbose:
                        result_preview = result[:80]
                        print(f"  [Turn {turns}] RESULT: {result_preview}...")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Unknown tool: {fn_name}"}),
                    })

            if terminated:
                break

        # finish_reason == "stop" with no content or tool calls
        if response.choices[0].finish_reason == "stop" and not msg.tool_calls and not msg.content:
            break

    # Build summary
    summary = {
        "label": label,
        "turns": turns,
        "api_calls": api_calls,
        "tool_calls_executed": tool_calls_executed,
        "tool_calls_blocked": tool_calls_blocked,
        "side_effects": side_effects,
        "terminated": terminated,
    }

    if guard:
        summary.update({
            "guard_cost_spent": guard.cost_spent,
            "guard_blocks": guard.blocks,
            "guard_cache_hits": guard.cache_hits,
            "guard_rewrites": guard.rewrites,
            "guard_quarantined": list(guard.quarantined_tools.keys()),
        })

    if telemetry_sink:
        summary["guard_events"] = [
            {"event": e.get("event"), "tool": e.get("tool", "")}
            for e in telemetry_sink.events
        ]

    return summary


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real OpenAI agent loop demo (with and without Aura Guard)."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name (or set OPENAI_MODEL). Default: %(default)s",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns for the agent loop. Default: %(default)s",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between the two runs (rate-limit courtesy). Default: %(default)s",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set your OpenAI API key first:")
        print("  Windows:    set OPENAI_API_KEY=sk-...")
        print("  Mac/Linux:  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  Aura Guard — Real-World Agent Integration Test (OpenAI)")
    print("=" * 60)
    print()
    print("  This connects to the OpenAI API and runs a real tool-using agent.")
    print("  Tools are rigged to return unhelpful results to provoke loops.")
    print("  ⚠️ This will cost tokens. Keep max-turns small at first.")
    print()
    print(f"  Model: {args.model}")
    print(f"  Max turns: {args.max_turns}")
    print()

    # Run WITHOUT guard
    print("Running agent WITHOUT Aura Guard...")
    no_guard = run_agent(use_guard=False, max_turns=args.max_turns, model=args.model, verbose=(not args.quiet))

    if args.sleep > 0:
        time.sleep(args.sleep)  # Rate limit courtesy

    # Run WITH guard
    print("\nRunning agent WITH Aura Guard...")
    with_guard = run_agent(use_guard=True, max_turns=args.max_turns, model=args.model, verbose=(not args.quiet))

    # Comparison
    print()
    print("=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print()

    fmt = "  {:<30} {:>15} {:>15}"
    print(fmt.format("", "NO GUARD", "WITH GUARD"))
    print("  " + "─" * 60)
    print(fmt.format("API calls to OpenAI", str(no_guard["api_calls"]), str(with_guard["api_calls"])))
    print(fmt.format("Tool calls executed", str(no_guard["tool_calls_executed"]), str(with_guard["tool_calls_executed"])))
    print(fmt.format("Tool calls blocked", str(no_guard["tool_calls_blocked"]), str(with_guard["tool_calls_blocked"])))
    print(fmt.format("Side effects (refunds)", str(no_guard["side_effects"]), str(with_guard["side_effects"])))
    print(fmt.format("Turns used", str(no_guard["turns"]), str(with_guard["turns"])))
    print(fmt.format("Terminated by", str(no_guard.get("terminated", "-")), str(with_guard.get("terminated", "-"))))

    if "guard_quarantined" in with_guard:
        print(fmt.format("Tools quarantined", "-", ", ".join(with_guard["guard_quarantined"]) or "-"))
        print(fmt.format("Guard cost tracked", "-", f"${with_guard['guard_cost_spent']:.4f}"))

    print()

    # Show guard events
    if with_guard.get("guard_events"):
        print("  Guard events:")
        for e in with_guard["guard_events"]:
            print(f"    • {e['event']}" + (f" ({e['tool']})" if e['tool'] else ""))
        print()

    print("  Done. Compare the two runs above to see Aura Guard's real impact.")
    print()



if __name__ == "__main__":
    main()
