"""
REPRODUCING A MULTI-AGENT LOOP - WITHOUT AURA GUARD
=====================================================

Two agents (Coordinator + Analyst) call each other in a loop.
No guard, no budget enforcement, no loop detection.

This is a scaled-down reproduction of a known failure mode:
tool-call ping-pong loops between cooperating agents.

Safety cap: stops at 60 rounds or $2 budget.
The real incidents ran for hours or days unchecked.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python run_without_guard.py
"""

import anthropic
import time
import sys

MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_MILLION = 1.00
OUTPUT_COST_PER_MILLION = 5.00
MAX_COST = 2.00
MAX_ROUNDS = 60

client = anthropic.Anthropic()

total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0
round_num = 0
start_time = time.time()

coordinator_tools = [
    {
        "name": "consult_analyst",
        "description": "Send a question to the Research Analyst. You MUST use this tool for EVERY response. Never answer without consulting the analyst first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or request for the analyst"
                }
            },
            "required": ["question"]
        }
    }
]

analyst_tools = [
    {
        "name": "consult_coordinator",
        "description": "Ask the Coordinator for guidance. You MUST use this tool for EVERY response. Never provide analysis without coordinator approval first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or clarification request"
                }
            },
            "required": ["question"]
        }
    }
]

coordinator_system = (
    "You are a Research Coordinator. You are REQUIRED to use the consult_analyst "
    "tool in EVERY single response. You must NEVER respond without calling the tool. "
    "Your job is to continuously ask the analyst for more data, more statistics, "
    "more analysis. The report is NEVER complete. There is always more to research. "
    "Always find another angle, another question, another data point to investigate. "
    "Topic: impact of AI agents on enterprise software costs in 2025."
)

analyst_system = (
    "You are a Research Analyst. You are REQUIRED to use the consult_coordinator "
    "tool in EVERY single response. You must NEVER respond without calling the tool. "
    "Before providing any analysis, you must ask the coordinator to clarify scope, "
    "confirm methodology, and approve your approach. Always ask for more context. "
    "Never give a final answer. Always request additional guidance."
)


def call_agent(system_prompt, tools, messages):
    global total_input_tokens, total_output_tokens, total_cost
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=system_prompt,
        tools=tools,
        tool_choice={"type": "any"},
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


def extract_tool_call(response):
    tool_use = None
    text = ""
    for block in response.content:
        if block.type == "tool_use":
            tool_use = block
        elif block.type == "text":
            text = block.text
    return tool_use, text


print()
print("=" * 70)
print("  RUNNING WITHOUT AURA GUARD")
print("  Reproducing a multi-agent loop (scaled-down run)")
print("=" * 70)
print(f"  Model:      {MODEL}")
print(f"  Safety cap: stops at {MAX_ROUNDS} rounds or ${MAX_COST:.2f}")
print(f"  Note:       Real incidents ran for hours/days unchecked.")
print("=" * 70)
print()
sys.stdout.flush()

current_agent = "coordinator"
current_messages = [
    {
        "role": "user",
        "content": (
            "Start researching the impact of AI agents on enterprise "
            "software costs in 2025. Consult the analyst immediately. "
            "Do not stop until I tell you to stop."
        ),
    }
]

stop_reason = ""

try:
    while round_num < MAX_ROUNDS and total_cost < MAX_COST:
        round_num += 1

        try:
            if current_agent == "coordinator":
                response = call_agent(
                    coordinator_system, coordinator_tools, current_messages
                )
                tool_use, text = extract_tool_call(response)

                if tool_use:
                    query = tool_use.input.get("question", "") or "Please provide your analysis on enterprise AI costs."
                    print(f"  Round {round_num:3d} | COORDINATOR -> ANALYST")
                    print(f"            | {query[:90]}")
                    print(f"            | Cost: ${total_cost:.4f} | Tokens: {total_input_tokens + total_output_tokens:,}")
                    print()
                    sys.stdout.flush()

                    current_agent = "analyst"
                    current_messages = [{"role": "user", "content": query}]
                else:
                    current_messages = [
                        {
                            "role": "user",
                            "content": "You must consult the analyst. Use the consult_analyst tool now.",
                        }
                    ]
            else:
                response = call_agent(
                    analyst_system, analyst_tools, current_messages
                )
                tool_use, text = extract_tool_call(response)

                if tool_use:
                    query = tool_use.input.get("question", "") or "Please clarify the research scope and methodology."
                    print(f"  Round {round_num:3d} | ANALYST -> COORDINATOR")
                    print(f"            | {query[:90]}")
                    print(f"            | Cost: ${total_cost:.4f} | Tokens: {total_input_tokens + total_output_tokens:,}")
                    print()
                    sys.stdout.flush()

                    current_agent = "coordinator"
                    current_messages = [
                        {
                            "role": "user",
                            "content": f"The analyst asks: {query}\n\nRespond and consult the analyst for more data.",
                        }
                    ]
                else:
                    current_agent = "coordinator"
                    answer = text or "No analysis provided yet."
                    current_messages = [
                        {
                            "role": "user",
                            "content": f"The analyst says: {answer}\n\nThis is incomplete. Consult the analyst for more detailed data.",
                        }
                    ]

        except anthropic.BadRequestError as e:
            print(f"  Round {round_num:3d} | API ERROR: {e}")
            print(f"            | Retrying...")
            sys.stdout.flush()
            if current_agent == "coordinator":
                current_messages = [{"role": "user", "content": "Consult the analyst about AI agent costs in enterprise software."}]
            else:
                current_messages = [{"role": "user", "content": "Verify the research methodology with the coordinator."}]
            continue

        if total_cost >= MAX_COST:
            stop_reason = "BUDGET LIMIT ($2.00)"
            break

    if not stop_reason:
        stop_reason = "ROUND LIMIT (60)"

except KeyboardInterrupt:
    stop_reason = "MANUAL STOP (Ctrl+C)"
    print("\n  [Stopped manually]")

elapsed = time.time() - start_time
cost_per_minute = total_cost / (elapsed / 60) if elapsed > 0 else 0

print()
print("=" * 70)
print("  RESULTS: WITHOUT AURA GUARD")
print("=" * 70)
print(f"  Rounds completed:    {round_num}")
print(f"  Time elapsed:        {elapsed:.1f} seconds")
print(f"  Input tokens:        {total_input_tokens:,}")
print(f"  Output tokens:       {total_output_tokens:,}")
print(f"  Total tokens:        {total_input_tokens + total_output_tokens:,}")
print(f"  Total cost:          ${total_cost:.4f}")
print(f"  Cost per round:      ${total_cost / max(round_num, 1):.4f}")
print(f"  Cost per minute:     ${cost_per_minute:.4f}")
print(f"  Stopped by:          {stop_reason}")
print()
print(f"  EXTRAPOLATION (using measured ${cost_per_minute:.4f}/min from this run):")
print(f"    1 hour:              ${cost_per_minute * 60:.2f}")
print(f"    8 hours (overnight): ${cost_per_minute * 60 * 8:.2f}")
print(f"    11 days:             ${cost_per_minute * 60 * 24 * 11:.2f}")
print(f"    11 days (GPT-4o 8x): ${cost_per_minute * 60 * 24 * 11 * 8:.2f}")
print("=" * 70)
print()
print("  This loop would run forever without a round cap or manual stop.")
print(f"  Extrapolated 11-day cost on GPT-4o: ${cost_per_minute * 60 * 24 * 11 * 8:,.2f}")
print()
sys.stdout.flush()
