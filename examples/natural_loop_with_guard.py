"""
NATURAL LOOP: AGENT RETRIES A FAILING SEARCH - WITH AURA GUARD
================================================================

Same scenario: agent has a search tool that returns no results.
Aura Guard enforces tool-call caps and detects repeated patterns.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python natural_loop_with_guard.py
"""

import anthropic
import time
import sys

try:
    from aura_guard import AgentGuard
    print("[OK] aura-guard imported")
except ImportError as e:
    print(f"[ERROR] Cannot import aura_guard: {e}")
    print("Run: pip install aura-guard")
    sys.exit(1)

MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_MILLION = 1.00
OUTPUT_COST_PER_MILLION = 5.00
MAX_ROUNDS = 40
MAX_COST = 1.00

client = anthropic.Anthropic()

guard = AgentGuard(
    max_cost_per_run=1.00,
    max_calls_per_tool=8,
    default_tool_cost=0.003,
)

print("[OK] AgentGuard initialized (max_cost=$1.00, max_calls/tool=8)")
sys.stdout.flush()

total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0
round_num = 0
start_time = time.time()

tools = [
    {
        "name": "search_web",
        "description": "Search the web for information. Returns search results as text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
]

system_prompt = (
    "You are a research assistant. The user needs specific data. "
    "Use the search_web tool to find it. If no results are found, "
    "keep trying new queries with different keywords, phrasings, or "
    "narrower/broader scope. Do not conclude the task unless you find "
    "results. Always try at least one more search."
)


def call_agent(messages):
    global total_input_tokens, total_output_tokens, total_cost
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=system_prompt,
        tools=tools,
        messages=messages,
    )
    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens
    cost = (
        response.usage.input_tokens * INPUT_COST_PER_MILLION / 1_000_000
        + response.usage.output_tokens * OUTPUT_COST_PER_MILLION / 1_000_000
    )
    total_cost += cost
    return response


print()
print("=" * 70)
print("  NATURAL LOOP: AGENT vs EMPTY SEARCH - WITH AURA GUARD")
print("  No forced prompts. search_web returns 'no results'.")
print("  Guard enforces tool-call caps + detects repeated patterns.")
print("=" * 70)
print(f"  Model:              {MODEL}")
print(f"  Guard budget:       $1.00")
print(f"  Max calls per tool: 8")
print(f"  Scenario:           Agent asked to find CEO salary data.")
print(f"                      search_web always returns empty results.")
print("=" * 70)
print()
sys.stdout.flush()

messages = [
    {
        "role": "user",
        "content": "Find me the exact average CEO salary at Fortune 500 companies in 2025, broken down by industry. I need precise numbers with sources.",
    }
]

stop_reason = ""
guard_stopped = False

try:
    while round_num < MAX_ROUNDS and total_cost < MAX_COST:
        round_num += 1

        try:
            response = call_agent(messages)

            # Collect ALL tool_use blocks
            tool_uses = []
            text = ""
            for block in response.content:
                if block.type == "tool_use":
                    tool_uses.append(block)
                elif block.type == "text":
                    text = block.text

            if tool_uses:
                blocked = False
                for tu in tool_uses:
                    query = tu.input.get("query", "")

                    decision = guard.check_tool(
                        name="search_web",
                        args={"query": query},
                    )

                    print(f"  Round {round_num:3d} | SEARCH: \"{query[:60]}\"")
                    print(f"            | Guard: {decision.action.upper()}")
                    print(f"            | Cost: ${total_cost:.4f} | Tokens: {total_input_tokens + total_output_tokens:,}")
                    sys.stdout.flush()

                    if decision.action != "allow":
                        reason_str = ""
                        if hasattr(decision, 'reason') and decision.reason:
                            reason_str = decision.reason

                        print()
                        print(f"  {'=' * 60}")
                        print(f"  >>> AURA GUARD STOPPED THE RETRY LOOP")
                        print(f"  >>> Decision: {decision.action.upper()}")
                        if reason_str:
                            print(f"  >>> Reason: {reason_str}")
                        print(f"  >>> Spend cut off.")
                        print(f"  {'=' * 60}")
                        stop_reason = f"AURA GUARD: {decision.action.upper()}"
                        if reason_str:
                            stop_reason += f" ({reason_str})"
                        guard_stopped = True
                        blocked = True
                        break

                    guard.record_result(ok=True, payload={"result": "no results"})

                if blocked:
                    break

                print(f"            | Result: No results found")
                print()

                # Append assistant response
                messages.append({"role": "assistant", "content": response.content})

                # Send tool_result for EVERY tool_use (list-of-blocks format)
                tool_results = []
                for tu in tool_uses:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": [
                            {
                                "type": "text",
                                "text": "No results found for this query. Try different keywords or a more specific search."
                            }
                        ],
                    })
                messages.append({"role": "user", "content": tool_results})

            else:
                print(f"  Round {round_num:3d} | AGENT GAVE UP")
                print(f"            | {text[:90]}")
                stop_reason = "AGENT GAVE UP"
                break

        except anthropic.BadRequestError as e:
            print(f"  Round {round_num:3d} | API ERROR: {e}")
            sys.stdout.flush()
            stop_reason = "API ERROR"
            break

        if total_cost >= MAX_COST:
            stop_reason = "BUDGET LIMIT ($1.00)"
            break

    if not stop_reason:
        stop_reason = "ROUND LIMIT (40)"

except KeyboardInterrupt:
    stop_reason = "MANUAL STOP (Ctrl+C)"
    print("\n  [Stopped manually]")

elapsed = time.time() - start_time

print()
print("=" * 70)
print("  RESULTS: NATURAL LOOP WITH AURA GUARD")
print("=" * 70)
print(f"  Rounds completed:    {round_num}")
print(f"  Time elapsed:        {elapsed:.1f} seconds")
print(f"  Input tokens:        {total_input_tokens:,}")
print(f"  Output tokens:       {total_output_tokens:,}")
print(f"  Total tokens:        {total_input_tokens + total_output_tokens:,}")
print(f"  Total cost:          ${total_cost:.4f}")
print(f"  Stopped by:          {stop_reason}")

if guard_stopped:
    print()
    print("  +---------------------------------------------------+")
    print("  |  AURA GUARD STOPPED THE RETRY LOOP.               |")
    print("  +---------------------------------------------------+")
    print(f"  |  Tokens used:      {total_input_tokens + total_output_tokens:>10,}                  |")
    print(f"  |  Money spent:      ${total_cost:>10.4f}                  |")
    print(f"  |                                                   |")
    print(f"  |  No forced prompts. No trick instructions.        |")
    print(f"  |  Just a search tool that returns empty results.   |")
    print("  +---------------------------------------------------+")

print("=" * 70)
print()
sys.stdout.flush()
