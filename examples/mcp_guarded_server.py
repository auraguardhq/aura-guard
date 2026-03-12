"""Example: MCP server with AuraGuard protection.

Requires: pip install mcp aura-guard

This creates an MCP-compatible tool server where every tool call
is automatically protected by AuraGuard. Loop detection, duplicate
side-effect prevention, cost budgets — all enforced automatically.

Run:
    python examples/mcp_guarded_server.py

Then connect from any MCP client (Claude Desktop, Cursor, etc.):
    {
        "mcpServers": {
            "support": {
                "command": "python",
                "args": ["examples/mcp_guarded_server.py"]
            }
        }
    }
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from aura_guard.adapters.mcp_adapter import GuardedMCP
except ImportError:
    print("ERROR: pip install mcp aura-guard")
    sys.exit(1)


# Create a guarded MCP server
mcp = GuardedMCP(
    "Customer Support",
    secret_key=b"example-mcp-server-key",
    side_effect_tools={"refund", "send_email"},
    max_cost_per_run=0.50,
    max_calls_per_tool=5,
)


# ── Tools ──────────────────────────────────

_kb = {
    "refund_policy": "Full refund within 30 days. EU customers get 14-day cooling-off period.",
    "shipping": "Standard 5-7 days. Express 1-2 days. Free over $50.",
    "returns": "Return within 30 days in original packaging. Refund processed in 5-10 business days.",
}


@mcp.tool()
def search_kb(query: str) -> str:
    """Search the customer support knowledge base."""
    results = [v for k, v in _kb.items() if query.lower() in k.lower()]
    if results:
        return json.dumps({"results": results, "query": query})
    return json.dumps({"results": [], "query": query, "note": "No results found."})


@mcp.tool()
def get_order(order_id: str) -> str:
    """Look up an order by ID."""
    return json.dumps({
        "order_id": order_id,
        "status": "shipped",
        "items": ["Widget Pro"],
        "total": 49.99,
    })


@mcp.tool(side_effect=True)
def refund(order_id: str, amount: float, reason: str) -> str:
    """Issue a refund for an order. This is a side-effect tool (mutation)."""
    return json.dumps({
        "status": "refunded",
        "order_id": order_id,
        "amount": amount,
        "reason": reason,
    })


@mcp.tool(side_effect=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer. This is a side-effect tool (mutation)."""
    return json.dumps({
        "status": "sent",
        "to": to,
        "subject": subject,
    })


if __name__ == "__main__":
    import sys
    # MCP stdio uses stdout for JSON-RPC — never print() to stdout.
    print("Starting guarded MCP server (stdio)...", file=sys.stderr)
    print(f"Guard config: max_cost={mcp.guard.cost_limit}, max_calls_per_tool=5", file=sys.stderr)
    print("Side-effect tools: refund, send_email", file=sys.stderr)
    mcp.run(transport="stdio")
