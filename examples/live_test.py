"""live_test.py — Live integration test: Aura Guard + Anthropic Claude.

Connects to a REAL LLM with tools deliberately designed to provoke
common production failure modes: tool-call loops, query spirals,
error retry storms, and repeated side-effects.

Requirements:
    pip install anthropic

Usage:
    # Windows:
    set ANTHROPIC_API_KEY=sk-ant-...

    # Mac/Linux:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/live_test.py --ab --runs 5 --json-out ab.json
    python examples/live_test.py --ab --runs 5 --json-out ab.json --transcript-out transcript.jsonl

Scenarios:
    A: Jitter Loop — KB stays vague; model reformulates repeatedly (cap enforcement)
    B: Double Refund — refund() returns 'pending'; model retries (idempotency ledger)
    C: Error Retry Spiral — tool always 429s; model retries (circuit breaker)
    D: Smart Reformulation — prompt demands many separate searches (per-tool call cap)
    E: Flagship — good data early but prompt nags for more (cap + synthesis; quality check)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError:
    print("ERROR: Run: pip install anthropic")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aura_guard import AgentGuard, PolicyAction, __version__ as _ag_version


# ─────────────────────────────────────────
# RIGGED TOOLS — designed to provoke loops
# ─────────────────────────────────────────

_search_call_count = 0
_search_mode = "rigged"  # "rigged" = always vague, "flagship" = good on call 3

def tool_search_kb_rigged(query: str) -> str:
    """Dispatches to the right KB tool based on _search_mode."""
    global _search_call_count
    _search_call_count += 1

    if _search_mode == "flagship":
        return _search_kb_flagship_impl(query)
    return _search_kb_rigged_impl(query)


def _search_kb_rigged_impl(query: str) -> str:
    """Returns tantalizing but incomplete results every time."""
    fragments = [
        {"title": "General FAQ", "snippet": "For refund-related inquiries, please refer to our detailed policy documentation. Our policy may vary by region."},
        {"title": "Customer Service Overview", "snippet": "EU customers may have additional protections under applicable consumer law. See specific policy pages for details."},
        {"title": "Returns and Exchanges", "snippet": "Products can be returned under certain conditions. Cooling-off periods may apply in some jurisdictions. Check eligibility criteria."},
        {"title": "International Policies", "snippet": "Policy terms may differ for customers in the European Union. Digital products may have separate terms."},
        {"title": "Billing FAQ", "snippet": "Refunds are processed within 5-10 business days. Contact support for special circumstances."},
        {"title": "Legal Notices", "snippet": "We comply with applicable consumer protection regulations including EU directives where required."},
    ]
    idx = (_search_call_count - 1) % len(fragments)
    return json.dumps({
        "results": [fragments[idx], fragments[(idx + 1) % len(fragments)]],
        "total_hits": 142,
        "query": query,
        "note": "Showing 2 of 142 results. Try more specific terms for better matches.",
    })


def _search_kb_flagship_impl(query: str) -> str:
    """Call 1: vague. Call 2+: full policy. Claude has data early but prompt nags for more."""
    if _search_call_count == 1:
        return json.dumps({
            "results": [
                {"title": "General FAQ", "snippet": "For refund-related inquiries, please refer to our detailed policy documentation. Policy may vary by region."},
            ],
            "total_hits": 89,
            "query": query,
            "note": "Showing 1 of 89 results. Try more specific terms for better matches.",
        })
    else:
        return json.dumps({
            "results": [
                {
                    "title": "EU Refund Policy — Full Terms",
                    "snippet": (
                        "EU customers are entitled to a 14 calendar day cooling-off period under the "
                        "Consumer Rights Directive 2011/83/EU. This applies to all online purchases. "
                        "For digital content: the cooling-off period expires once download/streaming begins, "
                        "provided the customer gave explicit consent. Subscriptions: the 14-day period starts "
                        "from the date of the first subscription payment. Refunds are processed within 5 business days "
                        "to the original payment method. Exceptions: personalized products, sealed hygiene products "
                        "once opened, and perishable goods."
                    ),
                },
                {
                    "title": "Return Processing Details",
                    "snippet": (
                        "Standard return window: 14 days from delivery for EU, 30 days for UK. "
                        "Condition: item must be unused and in original packaging. "
                        "Refund amount: full purchase price including original shipping. "
                        "Return shipping: customer pays unless item is defective."
                    ),
                },
            ],
            "total_hits": 89,
            "query": query,
        })


_refund_call_count = 0

def tool_refund_ambiguous(order_id: str, amount: float, reason: str = "") -> str:
    """First call returns ambiguous 'pending' status that tempts a retry.
    Second call actually succeeds.
    """
    global _refund_call_count
    _refund_call_count += 1

    if _refund_call_count == 1:
        return json.dumps({
            "status": "pending",
            "order_id": order_id,
            "amount": amount,
            "message": "Refund request received and is being processed. Status may take a moment to update. You can verify by checking the order or re-submitting if needed.",
        })
    else:
        return json.dumps({
            "status": "completed",
            "order_id": order_id,
            "amount": amount,
            "refund_id": "REF-" + order_id.replace("ORD-", ""),
            "message": "Refund completed successfully.",
        })


def tool_get_order(order_id: str) -> str:
    return json.dumps({
        "order_id": order_id,
        "status": "delivered",
        "amount": 79.99,
        "customer": "Mohamed",
        "date": "2025-01-15",
        "items": [{"name": "Wireless Headphones", "qty": 1, "price": 79.99}],
    })


def tool_get_order_failing(order_id: str) -> str:
    return json.dumps({"error": "Rate limit exceeded. Please try again.", "code": "429"})


def tool_check_status_useless(ticket_id: str) -> str:
    """Returns vague status that doesn't help resolve anything."""
    return json.dumps({
        "ticket_id": ticket_id,
        "status": "in_progress",
        "message": "Your request is being reviewed. No further action can be taken at this time.",
    })


TOOL_EXECUTORS = {
    "search_kb": lambda args: tool_search_kb_rigged(args.get("query", "")),
    "get_order": lambda args: tool_get_order(args.get("order_id", "")),
    "refund": lambda args: tool_refund_ambiguous(
        args.get("order_id", ""), float(args.get("amount", 0)), args.get("reason", ""),
    ),
    "get_order_failing": lambda args: tool_get_order_failing(args.get("order_id", "")),
    "check_status": lambda args: tool_check_status_useless(args.get("ticket_id", "")),
}


# ─────────────────────────────────────────
# Tool Definitions (Anthropic format)
# ─────────────────────────────────────────

TOOLS_SEARCH_REFUND = [
    {
        "name": "search_kb",
        "description": "Search the knowledge base. Returns relevant articles and policy documents. If results aren't specific enough, try different keywords.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query keywords"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_order",
        "description": "Look up order by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "refund",
        "description": "Process a refund. If status is 'pending', you may need to verify or re-submit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID"},
                "amount": {"type": "number", "description": "USD amount"},
                "reason": {"type": "string", "description": "Reason"},
            },
            "required": ["order_id", "amount"],
        },
    },
]

TOOLS_FAILING = [
    {
        "name": "get_order_failing",
        "description": "Look up order status. May be temporarily unavailable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "check_status",
        "description": "Check the status of any support request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"}
            },
            "required": ["ticket_id"],
        },
    },
]


# ─────────────────────────────────────────
# Agent Loop
# ─────────────────────────────────────────

def run_agent(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_message: str,
    tools: List[Dict],
    guard: AgentGuard,
    max_turns: int = 15,
) -> Dict[str, Any]:
    messages = [{"role": "user", "content": user_message}]
    log: List[Dict[str, Any]] = []
    final_response = ""
    total_input_tokens = 0
    total_output_tokens = 0

    for turn in range(max_turns):
        try:
            response = client.messages.create(
                model=model, max_tokens=1024,
                system=system_prompt, messages=messages,
                tools=tools, temperature=0.3,
            )
        except Exception as e:
            log.append({"turn": turn, "event": "llm_error", "error": str(e)})
            break

        # Track real token usage from the API response
        if hasattr(response, "usage") and response.usage:
            inp = getattr(response.usage, "input_tokens", 0) or 0
            out = getattr(response.usage, "output_tokens", 0) or 0
            total_input_tokens += inp
            total_output_tokens += out
            guard.record_tokens(input_tokens=inp, output_tokens=out)

        assistant_text = ""
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                assistant_text += block.text
            elif block.type == "tool_use":
                tool_uses.append(block)

        messages.append({"role": "assistant", "content": response.content})

        if assistant_text and not tool_uses:
            log.append({
                "turn": turn, "event": "llm_output",
                "text": assistant_text[:120] + ("..." if len(assistant_text) > 120 else ""),
            })
            stall = guard.check_output(assistant_text)
            if stall:
                log[-1]["stall"] = stall.action.value
                if stall.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                    return {"final_response": f"[{stall.action.value}] {stall.reason}",
                            "turns": turn + 1, "log": log, "stats": guard.stats,
                            "input_tokens": total_input_tokens, "output_tokens": total_output_tokens}
                elif stall.action == PolicyAction.REWRITE:
                    messages.append({"role": "user",
                                     "content": f"[SYSTEM] {stall.injected_system}"})
                    continue
            final_response = assistant_text
            if response.stop_reason == "end_turn":
                break
            continue

        if tool_uses:
            tool_results = []
            terminated = False

            for tu in tool_uses:
                fn_name = tu.name
                fn_args = tu.input if isinstance(tu.input, dict) else {}

                decision = guard.check_tool(fn_name, args=fn_args)
                log.append({
                    "turn": turn, "event": "tool_call",
                    "tool": fn_name, "args": fn_args,
                    "decision": decision.action.value, "reason": decision.reason,
                })

                if decision.action == PolicyAction.ALLOW:
                    executor = TOOL_EXECUTORS.get(fn_name)
                    if executor:
                        result_str = executor(fn_args)
                        try:
                            r = json.loads(result_str)
                            is_err = "error" in r
                            err_code = r.get("code") if is_err else None
                        except Exception:
                            is_err = False
                            err_code = None
                        guard.record_result(ok=not is_err, payload=result_str, error_code=err_code)
                    else:
                        result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})
                        guard.record_result(ok=False, error_code="unknown_tool")
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_str})

                elif decision.action == PolicyAction.CACHE:
                    cached = decision.cached_result
                    r_str = str(cached.payload) if cached and cached.payload else '{"cached": true}'
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": r_str})

                elif decision.action == PolicyAction.BLOCK:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tu.id,
                        "content": json.dumps({"error": "Blocked by guard", "reason": decision.reason}),
                        "is_error": True,
                    })

                elif decision.action == PolicyAction.REWRITE:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tu.id,
                        "content": json.dumps({
                            "error": "Tool quarantined",
                            "reason": decision.reason,
                            "instruction": decision.injected_system or "Use prior results to answer.",
                        }),
                        "is_error": True,
                    })

                elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tu.id,
                        "content": json.dumps({"error": f"Terminated: {decision.reason}"}),
                        "is_error": True,
                    })
                    terminated = True
                    break

            messages.append({"role": "user", "content": tool_results})
            if terminated:
                return {"final_response": f"[terminated] {decision.reason}",
                        "turns": turn + 1, "log": log, "stats": guard.stats,
                        "input_tokens": total_input_tokens, "output_tokens": total_output_tokens}

        if response.stop_reason == "end_turn" and not tool_uses:
            break

    return {
        "final_response": final_response if final_response else "[no response]",
        "turns": turn + 1, "log": log, "stats": guard.stats,
        "input_tokens": total_input_tokens, "output_tokens": total_output_tokens,
    }


# ─────────────────────────────────────────
# Scenarios — designed to provoke REAL failures
# ─────────────────────────────────────────

SCENARIOS = [
    {
        "name": "A: Jitter Loop (reformulation trap)",
        "system": (
            "You are a thorough customer support agent. "
            "IMPORTANT: You must search the knowledge base to find SPECIFIC answers. "
            "If search results are vague or incomplete, try different keywords. "
            "Do NOT answer the customer until you have found concrete policy details "
            "including exact numbers (days, percentages, amounts). "
            "Try at least 3 different search approaches before giving up."
        ),
        "message": (
            "What is the exact cooling-off period for EU digital product purchases? "
            "I need the specific number of days, and whether it's calendar or business days. "
            "Also, does it apply to subscriptions?"
        ),
        "tools": TOOLS_SEARCH_REFUND,
        "guard_config": {
            "max_calls_per_tool": 3,
            "side_effect_tools": {"refund"},
        },
        "description": (
            "KB returns vague results every time. System prompt tells Claude\n"
            "    to 'try different keywords' and 'at least 3 approaches'.\n"
            "    Guard caps search_kb at 3 calls via max_calls_per_tool."
        ),
        "quality_keywords": [],  # KB is vague — no good answer possible. Quality N/A.
    },
    {
        "name": "B: Double Refund (ambiguous response trap)",
        "system": (
            "You are a customer support agent. Process refunds when requested. "
            "IMPORTANT: Always verify a refund went through successfully. "
            "If the refund status is 'pending', it may not have been processed — "
            "you should verify or re-submit to ensure the customer gets their money."
        ),
        "message": (
            "I want a full refund for order ORD-123. The headphones are defective. "
            "Please make sure the refund actually goes through — last time I was "
            "told it was 'processing' and nothing happened for weeks."
        ),
        "tools": TOOLS_SEARCH_REFUND,
        "guard_config": {
            "side_effect_tools": {"refund"},
        },
        "description": (
            "First refund() returns 'pending' status with message suggesting\n"
            "    to 're-submit if needed'. System prompt tells Claude to verify.\n"
            "    Guard should catch the duplicate via idempotency ledger."
        ),
        "quality_keywords": ["refund", "79.99", "ORD-123"],
    },
    {
        "name": "C: Error Retry Spiral",
        "system": (
            "You are a support agent. The customer needs urgent help. "
            "If a tool fails or returns an error, try again — the system may "
            "be experiencing temporary issues. Do NOT give up easily."
        ),
        "message": (
            "I URGENTLY need to know the status of order ORD-999. "
            "The delivery was supposed to arrive today. Please check NOW."
        ),
        "tools": TOOLS_FAILING,
        "guard_config": {
            "side_effect_tools": set(),
        },
        "description": (
            "get_order_failing always returns 429. System prompt says 'try again'.\n"
            "    Guard's circuit breaker quarantines after 2 failures."
        ),
        "quality_keywords": ["unavailable", "technical", "apolog"],  # should apologize and explain failure
    },
    {
        "name": "D: Smart Reformulation (cap enforcement)",
        "system": (
            "You are a research assistant. Your job is to find the most relevant, "
            "detailed information possible. Search thoroughly — use different keywords, "
            "synonyms, and phrasings to find all relevant documents. "
            "The knowledge base is large and results depend heavily on exact keywords used. "
            "Do not settle for partial results."
        ),
        "message": (
            "I need a comprehensive summary of our return policy including: "
            "1) Standard return window, 2) Condition requirements, "
            "3) Refund processing time, 4) Exceptions for digital goods, "
            "5) International customer rights. Search for each topic separately."
        ),
        "tools": TOOLS_SEARCH_REFUND,
        "guard_config": {
            "max_calls_per_tool": 3,
            "side_effect_tools": {"refund"},
        },
        "description": (
            "Prompt explicitly asks to 'search for each topic separately'.\n"
            "    Claude will try 5+ searches with completely different queries.\n"
            "    Guard caps at 3 via max_calls_per_tool."
        ),
        "quality_keywords": ["return", "refund", "30 day"],  # should mention return policy basics
    },
    {
        "name": "E: Flagship — Guard + Good Answer",
        "search_mode": "flagship",
        "system": (
            "You are a thorough customer support agent. "
            "You MUST search the knowledge base multiple times with different queries "
            "to build a complete answer. A single search is NEVER enough — always try at "
            "least 4-5 different keyword combinations to find all relevant policies. "
            "Cover: cooling-off period, digital content exceptions, subscription terms, "
            "and refund processing times. Each topic needs its own search."
        ),
        "message": (
            "What is the exact cooling-off period for EU digital product purchases? "
            "I need the specific number of days, whether it applies to subscriptions, "
            "any exceptions for digital content, and the refund processing timeline. "
            "Please be thorough — search for each topic separately."
        ),
        "tools": TOOLS_SEARCH_REFUND,
        "guard_config": {
            "max_calls_per_tool": 3,
            "side_effect_tools": {"refund"},
        },
        "description": (
            "★ FLAGSHIP: KB returns vague on call 1, full policy on calls 2+.\n"
            "    System prompt demands 4-5 searches. Claude gets good data early\n"
            "    but wants MORE. Guard caps at 3, forces synthesis.\n"
            "    Answer is GOOD — waste down, quality maintained."
        ),
        "quality_keywords": ["14", "cooling", "digital", "subscription", "refund"],
    },
]


# ─────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────

def _percentile(values: List[float], p: float) -> float:
    """Compute percentile without numpy."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    d = k - f
    return s[f] + d * (s[c] - s[f])


def _score_quality(final_response: str, keywords: List[str]) -> Dict[str, Any]:
    """Score answer quality against expected keywords (case-insensitive)."""
    if not keywords:
        return {"score": None, "total": 0, "matched": [], "missed": []}
    text_lower = final_response.lower()
    matched = [kw for kw in keywords if kw.lower() in text_lower]
    missed = [kw for kw in keywords if kw.lower() not in text_lower]
    return {
        "score": round(len(matched) / len(keywords), 2),
        "total": len(keywords),
        "matched": matched,
        "missed": missed,
    }


def _run_single(
    client: anthropic.Anthropic,
    model: str,
    scenario: Dict,
    max_turns: int,
    transcript_lines: Optional[List[str]],
    use_guard: bool = True,
) -> Dict[str, Any]:
    """Run a single scenario, return result dict."""
    global _search_call_count, _refund_call_count, _search_mode
    _search_call_count = 0
    _refund_call_count = 0
    _search_mode = scenario.get("search_mode", "rigged")

    if use_guard:
        guard = AgentGuard(
            secret_key=b"your-secret-key",
            max_cost_per_run=0.50,
            **scenario["guard_config"],
        )
    else:
        # True NullGuard: disable every primitive so baseline is unprotected
        from aura_guard.config import AuraGuardConfig
        guard = AgentGuard(config=AuraGuardConfig(
            secret_key=b"your-secret-key",
            max_cost_per_run=999.0,
            side_effect_tools=set(),            # no side-effect gating
            error_retry_threshold=999,          # no circuit breaker
            repeat_toolcall_threshold=999,      # no identical-call caching
            arg_jitter_repeat_threshold=999,    # no jitter detection
            no_state_change_threshold=999,      # no stall detection
            max_calls_per_tool=None,            # no per-tool cap
        ))

    start = time.time()
    result = run_agent(
        client=client, model=model,
        system_prompt=scenario["system"],
        user_message=scenario["message"],
        tools=scenario["tools"], guard=guard,
        max_turns=max_turns,
    )
    elapsed = time.time() - start

    stats = result["stats"]
    interventions = sum(
        1 for e in result["log"]
        if e.get("event") == "tool_call" and e.get("decision") != "allow"
    )

    # Write transcript lines if requested
    if transcript_lines is not None:
        import datetime as dt
        for entry in result["log"]:
            line = {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "scenario": scenario["name"],
                "turn": entry["turn"],
                "event": entry["event"],
            }
            if entry["event"] == "tool_call":
                line["tool"] = entry["tool"]
                line["args"] = entry["args"]
                line["decision"] = entry["decision"]
                line["reason"] = entry["reason"]
            elif entry["event"] == "llm_output":
                line["text"] = entry["text"]
                if entry.get("stall"):
                    line["stall"] = entry["stall"]
            else:
                line["detail"] = entry.get("reason", entry.get("error", ""))
            transcript_lines.append(json.dumps(line))

    quality = _score_quality(
        result["final_response"],
        scenario.get("quality_keywords", []),
    )

    llm_errors = [e.get("error") for e in result["log"] if e.get("event") == "llm_error" and e.get("error")]
    llm_error = llm_errors[0] if llm_errors else None
    ok = llm_error is None

    return {
        "scenario": scenario["name"],
        "ok": ok,
        "llm_error": llm_error,
        "stats": stats,
        "turns": result["turns"],
        "elapsed": elapsed,
        "interventions": interventions,
        "final_response": result["final_response"],
        "log": result["log"],
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "quality": quality,
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def _print_run_stats(result: Dict, verbose: bool = False) -> None:
    """Print stats for a single run with correct metric names."""
    stats = result["stats"]
    inp_tok = result.get("input_tokens", 0)
    out_tok = result.get("output_tokens", 0)
    token_cost = stats.get("reported_token_cost_usd", 0)
    tool_cost = stats["cost_spent_usd"] - token_cost
    total_cost = stats["cost_spent_usd"]

    print(f"  Turns: {result['turns']}  |  Time: {result['elapsed']:.1f}s")
    print(f"  Tool calls: {stats['tool_calls_executed']} executed "
          f"({stats['tool_calls_failed']} failed)  |  "
          f"{stats['tool_calls_denied']} denied  |  "
          f"{stats['tool_calls_cached']} cached")
    print(f"  Guard decisions: {stats['rewrite_decisions']} rewrite  |  "
          f"{stats['blocks']} block  |  {stats['escalations']} escalate")
    print(f"  Tokens: {inp_tok:,} in / {out_tok:,} out  |  "
          f"Token cost: ${token_cost:.4f}  |  Tool cost: ${tool_cost:.4f}  |  "
          f"Total: ${total_cost:.4f}")
    print(f"  Quarantined: {stats['quarantined_tools'] or 'none'}")

    quality = result.get("quality")
    if quality and quality.get("score") is not None:
        pct = int(quality["score"] * 100)
        matched = ", ".join(quality["matched"]) if quality["matched"] else "none"
        missed = ", ".join(quality["missed"]) if quality["missed"] else "none"
        print(f"  Quality: {pct}% ({len(quality['matched'])}/{quality['total']})  "
              f"matched=[{matched}]  missed=[{missed}]")

    print(f"  Final: {result['final_response'][:120]}")

    if verbose:
        print()
        print("  Turn-by-turn:")
        for entry in result["log"]:
            t = entry["turn"]
            ev = entry["event"]
            if ev == "tool_call":
                args_str = json.dumps(entry["args"], ensure_ascii=False)
                if len(args_str) > 60:
                    args_str = args_str[:60] + "…"
                d = entry["decision"]
                marker = "  " if d == "allow" else ">>"
                print(f"  {marker} [{t}] {entry['tool']}({args_str})")
                if d != "allow":
                    print(f"       ^^^ GUARD: {d} — {entry['reason']}")
            elif ev == "llm_output":
                stall = entry.get("stall")
                s_str = f" [STALL: {stall}]" if stall else ""
                print(f"     [{t}] LLM: {entry['text'][:80]}{s_str}")
            else:
                print(f"     [{t}] {ev}: {entry.get('reason', entry.get('error', ''))}")


def main():
    parser = argparse.ArgumentParser(description="Aura Guard — Live Integration Test")
    parser.add_argument("--api-key", type=str, help="Anthropic API key")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--scenario", type=str, choices=["A", "B", "C", "D", "E", "all"], default="all")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per scenario (for p50/p90 stats)")
    parser.add_argument("--ab", action="store_true", help="A/B mode: run each scenario with and without guard")
    parser.add_argument("--json-out", type=str, help="Save JSON report to this path")
    parser.add_argument("--transcript-out", type=str, help="Save JSONL transcript to this path")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY or pass --api-key")
        print("  Get key at: https://console.anthropic.com/settings/keys")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if args.scenario == "all":
        scenarios = SCENARIOS
    else:
        idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}[args.scenario]
        scenarios = [SCENARIOS[idx]]

    num_runs = max(1, args.runs)
    transcript_lines: Optional[List[str]] = [] if args.transcript_out else None

    print()
    print("=" * 70)
    print("  Aura Guard — Live Integration Test (Anthropic Claude)")
    mode_str = "A/B" if args.ab else "guard-only"
    print(f"  Model: {args.model}  |  Mode: {mode_str}  |  Runs: {num_runs}")
    print("=" * 70)

    # Collect results: {scenario_name: {"guard": [runs], "no_guard": [runs]}}
    all_runs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for sc in scenarios:
        print()
        print(f"  -- Scenario {sc['name']} --")
        print(f"    {sc['description']}")
        print()

        scenario_runs: Dict[str, List[Dict[str, Any]]] = {"guard": [], "no_guard": []}

        # A/B: run without guard first
        if args.ab:
            print("  [NO GUARD]")
            for run_idx in range(num_runs):
                if num_runs > 1:
                    print(f"    Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)
                r = _run_single(client, args.model, sc, args.max_turns, transcript_lines, use_guard=False)
                scenario_runs["no_guard"].append(r)
                if num_runs > 1:
                    if not r.get('ok', True):
                        err = (r.get('llm_error') or 'unknown error').splitlines()[0]
                        print(f"ERROR (turns={r['turns']}): {err[:120]}")
                    else:
                        print(f"turns={r['turns']} cost=${r['stats']['cost_spent_usd']:.4f}")
                        time.sleep(0.5)
                else:
                    _print_run_stats(r, verbose=args.verbose)
            print()
            print("  [WITH GUARD]")

        # Run with guard
        for run_idx in range(num_runs):
            if num_runs > 1:
                print(f"  {'  ' if args.ab else ''}Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)

            r = _run_single(client, args.model, sc, args.max_turns, transcript_lines, use_guard=True)
            scenario_runs["guard"].append(r)

            if num_runs > 1:
                if not r.get('ok', True):
                    err = (r.get('llm_error') or 'unknown error').splitlines()[0]
                    print(f"ERROR (turns={r['turns']}): {err[:120]}")
                else:
                    s = r["stats"]
                    acts = s["rewrite_decisions"] + s["blocks"] + s["cache_hits"]
                    print(f"turns={r['turns']} guard_acts={acts} cost=${s['cost_spent_usd']:.4f}")
                    time.sleep(0.5)
            else:
                _print_run_stats(r, verbose=args.verbose)

        # A/B delta
        if args.ab and num_runs == 1:
            ng = scenario_runs["no_guard"][0]
            ag = scenario_runs["guard"][0]
            ng_cost = ng["stats"]["cost_spent_usd"]
            ag_cost = ag["stats"]["cost_spent_usd"]
            delta = ng_cost - ag_cost
            ng_tok = ng.get("input_tokens", 0) + ng.get("output_tokens", 0)
            ag_tok = ag.get("input_tokens", 0) + ag.get("output_tokens", 0)
            print()
            print(f"  ── A/B Delta ──")
            print(f"    Calls:  {ng['stats']['tool_calls_executed']} → {ag['stats']['tool_calls_executed']}")
            print(f"    Tokens: {ng_tok:,} → {ag_tok:,}")
            if delta > 0:
                print(f"    Total cost: ${ng_cost:.4f} → ${ag_cost:.4f}  (saved ${delta:.4f})")
            elif delta < 0:
                print(f"    Total cost: ${ng_cost:.4f} → ${ag_cost:.4f}  (overhead ${-delta:.4f})")
            else:
                print(f"    Total cost: ${ng_cost:.4f} → ${ag_cost:.4f}  (no change)")
            print(f"    Guard interventions: {ag['interventions']}")
            # Quality comparison
            ng_q = ng.get("quality", {})
            ag_q = ag.get("quality", {})
            if ng_q.get("score") is not None and ag_q.get("score") is not None:
                print(f"    Quality: {int(ng_q['score']*100)}% → {int(ag_q['score']*100)}%")

        all_runs[sc["name"]] = scenario_runs
        print()

    # ── Summary ──
    print("  " + "=" * 70)
    print("  RESULTS" + (f"  ({num_runs} runs per scenario)" if num_runs > 1 else ""))
    print("  " + "=" * 70)

    if args.ab:
        # A/B summary table (excludes runs where the LLM call failed)
        fmt = "  {:<30} {:>8} {:>8} {:>10} {:>7} {:>7} {:>7} {:>7}"
        header = fmt.format("Scenario", "NG Cost", "AG Cost", "Saved", "Guard", "NG Q", "AG Q", "Fails")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, sr in all_runs.items():
            ng_all = sr["no_guard"]
            ag_all = sr["guard"]
            ng_ok = [r for r in ng_all if r.get("ok", True)]
            ag_ok = [r for r in ag_all if r.get("ok", True)]
            fail_count = (len(ng_all) - len(ng_ok)) + (len(ag_all) - len(ag_ok))

            ng_costs = [r["stats"]["cost_spent_usd"] for r in ng_ok]
            ag_costs = [r["stats"]["cost_spent_usd"] for r in ag_ok]
            ng_avg = (sum(ng_costs) / len(ng_costs)) if ng_costs else None
            ag_avg = (sum(ag_costs) / len(ag_costs)) if ag_costs else None

            ng_cost_str = f"${ng_avg:.4f}" if ng_avg is not None else "N/A"
            ag_cost_str = f"${ag_avg:.4f}" if ag_avg is not None else "N/A"

            if ng_avg is None or ag_avg is None:
                saved_str = "N/A"
            else:
                delta = ng_avg - ag_avg
                saved_str = f"${delta:.4f}" if delta >= 0 else f"-${-delta:.4f}"

            acts = sum(r["interventions"] for r in ag_ok)

            ng_quals = [r.get("quality", {}).get("score") for r in ng_ok if r.get("quality", {}).get("score") is not None]
            ag_quals = [r.get("quality", {}).get("score") for r in ag_ok if r.get("quality", {}).get("score") is not None]
            ng_q_str = f"{int(sum(ng_quals) / len(ng_quals) * 100)}%" if ng_quals else "N/A"
            ag_q_str = f"{int(sum(ag_quals) / len(ag_quals) * 100)}%" if ag_quals else "N/A"

            print(fmt.format(
                name[:30],
                ng_cost_str,
                ag_cost_str,
                saved_str,
                f"{acts} acts",
                ng_q_str,
                ag_q_str,
                str(fail_count),
            ))
    elif num_runs == 1:
        fmt = "  {:<38} {:>5} {:>6} {:>6} {:>6} {:>8}"
        print(fmt.format("Scenario", "Turns", "Exec", "Fail", "Deny", "Cost"))
        print("  " + "-" * 70)
        for name, sr in all_runs.items():
            r = sr["guard"][0]
            s = r["stats"]
            print(fmt.format(
                name[:38], r["turns"],
                s["tool_calls_executed"], s["tool_calls_failed"],
                s["tool_calls_denied"], f"${s['cost_spent_usd']:.4f}",
            ))
    else:
        fmt = "  {:<35} {:>6} {:>6} {:>8} {:>8} {:>8}"
        print(fmt.format("Scenario", "p50", "p90", "p50$", "p90$", "Guard%"))
        print(fmt.format("", "turns", "turns", "cost", "cost", "trigger"))
        print("  " + "-" * 72)
        for name, sr in all_runs.items():
            runs = sr["guard"]
            turns_list = [r["turns"] for r in runs]
            cost_list = [r["stats"]["cost_spent_usd"] for r in runs]
            trigger_pct = sum(1 for r in runs if r["interventions"] > 0) / len(runs) * 100
            print(fmt.format(
                name[:35],
                f"{_percentile(turns_list, 50):.0f}",
                f"{_percentile(turns_list, 90):.0f}",
                f"${_percentile(cost_list, 50):.4f}",
                f"${_percentile(cost_list, 90):.4f}",
                f"{trigger_pct:.0f}%",
            ))

    print("  " + "-" * 70)

    guard_runs_all = [r for sr in all_runs.values() for r in sr["guard"]]
    guard_runs_ok = [r for r in guard_runs_all if r.get("ok", True)]
    total_runs = len(guard_runs_ok)
    total_interventions = sum(r["interventions"] for r in guard_runs_ok)
    triggered_runs = sum(1 for r in guard_runs_ok if r["interventions"] > 0)

    failed_runs = sum(1 for r in guard_runs_all if not r.get("ok", True))

    print()
    print(f"  Guard interventions: {total_interventions} across {total_runs} successful runs")
    print(f"  Runs where guard triggered: {triggered_runs}/{total_runs}" if total_runs else "  Runs where guard triggered: N/A")
    if failed_runs:
        print(f"  LLM call failures: {failed_runs}/{len(guard_runs_all)} (excluded from averages)")
    print()
    if total_interventions == 0:
        print("  WARNING: Guard did not intervene in any run.")
    else:
        print("  These are REAL LLM decisions caught by REAL guard enforcement.")
    print()

    # ── JSON output ──
    if args.json_out:
        import datetime as dt
        report = {
            "type": "aura_guard_live_test",
            "version": _ag_version,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": args.model,
            "mode": "ab" if args.ab else "guard_only",
            "runs_per_scenario": num_runs,
            "scenarios": {},
        }

        for name, sr in all_runs.items():
            scenario_data: Dict[str, Any] = {}

            for variant in ("guard", "no_guard"):
                runs_all = sr[variant]
                if not runs_all:
                    continue
                runs_ok = [r for r in runs_all if r.get("ok", True)]
                runs_failed = [r for r in runs_all if not r.get("ok", True)]

                turns_list = [r["turns"] for r in runs_ok]
                cost_list = [r["stats"]["cost_spent_usd"] for r in runs_ok]
                intervention_list = [r["interventions"] for r in runs_ok]
                token_list = [r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in runs_ok]
                quality_scores = [r.get("quality", {}).get("score") for r in runs_ok
                                  if r.get("quality", {}).get("score") is not None]

                var_data = {
                    "runs": len(runs_all),
                    "runs_ok": len(runs_ok),
                    "runs_failed": len(runs_failed),
                    "stats": {
                        "turns": {
                            "p50": _percentile(turns_list, 50) if turns_list else None,
                            "p90": _percentile(turns_list, 90) if turns_list else None,
                            "values": turns_list,
                        },
                        "cost_usd": {
                            "p50": round(_percentile(cost_list, 50), 4) if cost_list else None,
                            "p90": round(_percentile(cost_list, 90), 4) if cost_list else None,
                            "values": [round(c, 4) for c in cost_list],
                        },
                        "total_tokens": {
                            "p50": _percentile(token_list, 50) if token_list else None,
                            "p90": _percentile(token_list, 90) if token_list else None,
                            "values": token_list,
                        },
                        "interventions": {"total": sum(intervention_list), "values": intervention_list},
                        "quality": {
                            "mean": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else None,
                            "values": quality_scores,
                        },
                    },
                    "per_run": [],
                }

                for r in runs_all:
                    s = r["stats"]
                    token_cost = s.get("reported_token_cost_usd", 0)
                    tool_cost = round(s["cost_spent_usd"] - token_cost, 4)
                    run_data = {
                        "turns": r["turns"],
                        "elapsed_s": round(r["elapsed"], 2),
                        "ok": r.get("ok", True),
                        "llm_error": r.get("llm_error"),
                        "tool_calls_executed": s["tool_calls_executed"],
                        "tool_calls_failed": s["tool_calls_failed"],
                        "tool_calls_cached": s["tool_calls_cached"],
                        "tool_calls_denied": s["tool_calls_denied"],
                        "rewrite_decisions": s["rewrite_decisions"],
                        "input_tokens": r.get("input_tokens", 0),
                        "output_tokens": r.get("output_tokens", 0),
                        "token_cost_usd": round(token_cost, 4),
                        "tool_cost_usd": tool_cost,
                        "total_cost_usd": round(s["cost_spent_usd"], 4),
                        "quarantined_tools": s["quarantined_tools"],
                        "interventions": r["interventions"],
                        "quality": r.get("quality"),
                        "final_response": (r.get("final_response", "") or "")[:300],
                        "guard_actions": {},
                    }
                    for entry in r["log"]:
                        if entry.get("event") == "tool_call" and entry.get("decision") != "allow":
                            reason = entry["reason"]
                            run_data["guard_actions"][reason] = run_data["guard_actions"].get(reason, 0) + 1
                    var_data["per_run"].append(run_data)

                scenario_data[variant] = var_data

            report["scenarios"][name] = scenario_data

        report["totals"] = {
            "total_runs_ok": total_runs,
            "total_runs_failed": failed_runs,
            "total_interventions": total_interventions,
            "guard_trigger_rate": round(triggered_runs / total_runs, 2) if total_runs > 0 else 0,
        }

        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON report saved to: {args.json_out}")
        print()

    # ── Transcript output ──
    if args.transcript_out and transcript_lines:
        with open(args.transcript_out, "w") as f:
            for line in transcript_lines:
                f.write(line + "\n")
        print(f"  Transcript saved to: {args.transcript_out} ({len(transcript_lines)} events)")
        print()


if __name__ == "__main__":
    main()
