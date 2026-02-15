"""
REPRODUCING A MULTI-AGENT LOOP - WITH AURA GUARD
==================================================

Same setup as run_without_guard.py, but Aura Guard sits between
the agents and their tools. It detects the loop and cuts off spend.

This is a deterministic reproduction of a known failure mode:
tool-call ping-pong loops between cooperating agents.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python run_with_guard.py
"""

import anthropic
import time
import sys

try:
    from aura_guard import AgentGuard
    print("  [OK] aura-guard imported successfully")
except ImportError as e:
    print(f"  [ERROR] Cannot import aura_guard: {e}")
    print("  Run: pip install aura-guard")
    sys.exit(1)

MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_MILLION = 1.00
OUTPUT_COST_PER_MILLION = 5.00
MAX_ROUNDS = 60

client = anthropic.Anthropic()

guard = AgentGuard(
    max_cost_per_run=2.00,
    max_calls_per_tool=15,
    default_tool_cost=0.003,
)

print("  [OK] AgentGuard initialized")
print(f"        max_cost_per_run=$2.00, max_calls_per_tool=15")
sys.stdout.flush()

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
print("  RUNNING WITH AURA GUARD")
print("  Same loop, but with jitter detection + budget enforcement")
print("=" * 70)
print(f"  Model:            {MODEL}")
print(f"  Guard budget:     $2.00")
print(f"  Jitter threshold: 0.60 similarity, block after 3")
print(f"  Max calls/tool:   15")
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
guard_stopped = False

try:
    while round_num < MAX_ROUNDS:
        round_num += 1

        try:
            if current_agent == "coordinator":
                response = call_agent(
                    coordinator_system, coordinator_tools, current_messages
                )
                tool_use, text = extract_tool_call(response)

                if tool_use:
                    query = tool_use.input.get("question", "") or "Please provide your analysis on enterprise AI costs."

                    decision = guard.check_tool(
                        name="consult_analyst",
                        args={"question": query},
                    )

                    print(f"  Round {round_num:3d} | COORDINATOR -> ANALYST")
                    print(f"            | {query[:80]}")
                    print(f"            | Guard: {decision.action.upper()}")
                    print(f"            | Cost: ${total_cost:.4f} | Tokens: {total_input_tokens + total_output_tokens:,}")
                    sys.stdout.flush()

                    if decision.action != "allow":
                        print()
                        print(f"  {'=' * 60}")
                        print(f"  >>> AURA GUARD DETECTED LOOP")
                        print(f"  >>> Decision: {decision.action.upper()}")
                        if hasattr(decision, 'reason') and decision.reason:
                            print(f"  >>> Reason: {decision.reason}")
                        if decision.action == "cache":
                            print(f"  >>> The agent is repeating identical tool calls.")
                            print(f"  >>> Cached result returned. Spend cut off.")
                        else:
                            print(f"  >>> Tool call prevented. Spend cut off.")
                        print(f"  {'=' * 60}")
                        stop_reason = f"AURA GUARD: {decision.action.upper()} - loop detected, spend cut off"
                        guard_stopped = True
                        break

                    guard.record_result(ok=True, payload={"sent": True})

                    print()
                    current_agent = "analyst"
                    current_messages = [{"role": "user", "content": query}]
                else:
                    current_messages = [
                        {"role": "user", "content": "You must consult the analyst. Use the consult_analyst tool now."}
                    ]
            else:
                response = call_agent(
                    analyst_system, analyst_tools, current_messages
                )
                tool_use, text = extract_tool_call(response)

                if tool_use:
                    query = tool_use.input.get("question", "") or "Please clarify the research scope and methodology."

                    decision = guard.check_tool(
                        name="consult_coordinator",
                        args={"question": query},
                    )

                    print(f"  Round {round_num:3d} | ANALYST -> COORDINATOR")
                    print(f"            | {query[:80]}")
                    print(f"            | Guard: {decision.action.upper()}")
                    print(f"            | Cost: ${total_cost:.4f} | Tokens: {total_input_tokens + total_output_tokens:,}")
                    sys.stdout.flush()

                    if decision.action != "allow":
                        print()
                        print(f"  {'=' * 60}")
                        print(f"  >>> AURA GUARD DETECTED LOOP")
                        print(f"  >>> Decision: {decision.action.upper()}")
                        if hasattr(decision, 'reason') and decision.reason:
                            print(f"  >>> Reason: {decision.reason}")
                        if decision.action == "cache":
                            print(f"  >>> The agent is repeating identical tool calls.")
                            print(f"  >>> Cached result returned. Spend cut off.")
                        else:
                            print(f"  >>> Tool call prevented. Spend cut off.")
                        print(f"  {'=' * 60}")
                        stop_reason = f"AURA GUARD: {decision.action.upper()} - loop detected, spend cut off"
                        guard_stopped = True
                        break

                    guard.record_result(ok=True, payload={"sent": True})

                    print()
                    current_agent = "coordinator"
                    current_messages = [
                        {
                            "role": "user",
                            "content": f"The analyst asks: {query}\n\nRespond and consult the analyst for more data.",
                        }
                    ]
                else:
                    answer = text or "No analysis provided yet."
                    current_agent = "coordinator"
                    current_messages = [
                        {
                            "role": "user",
                            "content": f"The analyst says: {answer}\n\nThis is incomplete. Consult the analyst for more detailed data.",
                        }
                    ]

        except anthropic.BadRequestError as e:
            print(f"  Round {round_num:3d} | API ERROR: {e}")
            sys.stdout.flush()
            if current_agent == "coordinator":
                current_messages = [{"role": "user", "content": "Consult the analyst about AI agent costs in enterprise software."}]
            else:
                current_messages = [{"role": "user", "content": "Verify the research methodology with the coordinator."}]
            continue

except KeyboardInterrupt:
    stop_reason = "MANUAL STOP (Ctrl+C)"
    print("\n  [Stopped manually]")

elapsed = time.time() - start_time

# Use the unguarded run's measured burn rate for honest comparison
# From our tests: unguarded burns ~$0.04/min on Haiku 4.5
UNGUARDED_COST_PER_MINUTE = 0.04

print()
print("=" * 70)
print("  RESULTS: WITH AURA GUARD")
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
    print("  |  AURA GUARD DETECTED LOOP. SPEND CUT OFF.         |")
    print("  +---------------------------------------------------+")
    print(f"  |  Tokens used:      {total_input_tokens + total_output_tokens:>10,}                  |")
    print(f"  |  Money spent:      ${total_cost:>10.4f}                  |")
    print(f"  |                                                   |")
    print(f"  |  WITHOUT GUARD (using measured $0.04/min):        |")
    print(f"  |    1 hour:         ${UNGUARDED_COST_PER_MINUTE * 60:>10.2f}                  |")
    print(f"  |    8 hours:        ${UNGUARDED_COST_PER_MINUTE * 60 * 8:>10.2f}                  |")
    print(f"  |    11 days:        ${UNGUARDED_COST_PER_MINUTE * 60 * 24 * 11:>10.2f}                  |")
    print(f"  |    11 days GPT-4o: ${UNGUARDED_COST_PER_MINUTE * 60 * 24 * 11 * 8:>10.2f}                  |")
    print("  +---------------------------------------------------+")

print("=" * 70)
print()
sys.stdout.flush()
