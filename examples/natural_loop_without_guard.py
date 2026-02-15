"""
NATURAL LOOP: AGENT RETRIES A FAILING SEARCH - WITHOUT AURA GUARD
===================================================================

No forced prompts. No trick instructions. Just a realistic scenario:
an agent has a search tool that always returns "no results found."

Safety cap: stops at 40 rounds or $1 budget.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python natural_loop_without_guard.py
"""

import anthropic
import time
import sys

MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_MILLION = 1.00
OUTPUT_COST_PER_MILLION = 5.00
MAX_COST = 1.00
MAX_ROUNDS = 40

client = anthropic.Anthropic()

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
print("  NATURAL LOOP: AGENT vs EMPTY SEARCH - WITHOUT AURA GUARD")
print("  No forced prompts. search_web always returns 'no results'.")
print("=" * 70)
print(f"  Model:      {MODEL}")
print(f"  Safety cap: stops at {MAX_ROUNDS} rounds or ${MAX_COST:.2f}")
print(f"  Scenario:   Agent asked to find specific salary data.")
print(f"              search_web always returns empty results.")
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
                for tu in tool_uses:
                    query = tu.input.get("query", "")
                    print(f"  Round {round_num:3d} | SEARCH: \"{query[:70]}\"")
                    print(f"            | Result: No results found")
                print(f"            | Cost: ${total_cost:.4f} | Tokens: {total_input_tokens + total_output_tokens:,}")
                print()
                sys.stdout.flush()

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
                print(f"            | Cost: ${total_cost:.4f}")
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
cost_per_minute = total_cost / (elapsed / 60) if elapsed > 0 else 0

print()
print("=" * 70)
print("  RESULTS: NATURAL LOOP WITHOUT AURA GUARD")
print("=" * 70)
print(f"  Rounds completed:    {round_num}")
print(f"  Time elapsed:        {elapsed:.1f} seconds")
print(f"  Input tokens:        {total_input_tokens:,}")
print(f"  Output tokens:       {total_output_tokens:,}")
print(f"  Total tokens:        {total_input_tokens + total_output_tokens:,}")
print(f"  Total cost:          ${total_cost:.4f}")
print(f"  Cost per minute:     ${cost_per_minute:.4f}")
print(f"  Stopped by:          {stop_reason}")
print()
if stop_reason != "AGENT GAVE UP":
    print(f"  The agent kept retrying with different keywords each time.")
    print(f"  EXTRAPOLATION (using measured ${cost_per_minute:.4f}/min from this run):")
    print(f"    1 hour:              ${cost_per_minute * 60:.2f}")
    print(f"    8 hours (overnight): ${cost_per_minute * 60 * 8:.2f}")
    print(f"    11 days:             ${cost_per_minute * 60 * 24 * 11:.2f}")
else:
    print(f"  The agent gave up after {round_num} retries.")
print("=" * 70)
print()
sys.stdout.flush()
