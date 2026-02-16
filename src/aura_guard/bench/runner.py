"""aura_guard.bench.runner

Agent failure benchmark runner.

Loads scenarios from package data, runs them with and without Aura Guard,
and produces a cost savings report with honest math.

Usage via CLI:
    aura-guard bench --all
    aura-guard bench --scenario ag_bench_01
    aura-guard bench --json-out report.json

Usage via Python:
    from aura_guard.bench.runner import run_all, print_report
    results = run_all()
    print_report(results)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AuraGuardConfig, CostModel
from ..guard import AuraGuard, GuardState
from ..middleware import AgentGuard


def _get_version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "unknown"
from ..types import PolicyAction, ToolCall, ToolResult


# ─────────────────────────────────────────
# Scenario Loading
# ─────────────────────────────────────────

def _load_scenarios_from_package() -> List[Dict[str, Any]]:
    """Load scenarios from package data (works after pip install)."""
    try:
        # Python 3.9+
        pkg = resources.files("aura_guard.bench") / "scenarios" / "scenarios.json"
        text = pkg.read_text(encoding="utf-8")
    except (TypeError, AttributeError):
        # Fallback: locate relative to this file
        here = Path(__file__).parent / "scenarios" / "scenarios.json"
        text = here.read_text(encoding="utf-8")

    return json.loads(text)


def load_scenarios(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load scenarios from a file path or from package data."""
    if path:
        with open(path) as f:
            return json.load(f)
    return _load_scenarios_from_package()


# ─────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────

@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    failure_mode: str
    total_steps: int

    # No guard
    no_guard_calls: int
    no_guard_side_effects: int
    no_guard_cost: float

    # With guard
    guard_calls: int
    guard_side_effects: int
    guard_tool_cost: float          # actual tool execution cost
    guard_rewrite_llm_cost: float   # estimated LLM cost for rewrites
    guard_total_cost: float         # tool cost + rewrite LLM cost
    guard_blocks: int
    guard_cache_hits: int
    guard_rewrites: int
    guard_escalations: int
    guard_quarantined: List[str] = field(default_factory=list)
    terminated: Optional[str] = None

    @property
    def saved(self) -> float:
        return self.no_guard_cost - self.guard_total_cost

    @property
    def saved_pct(self) -> float:
        if self.no_guard_cost <= 0:
            return 0.0
        return (self.saved / self.no_guard_cost) * 100

    @property
    def side_effects_prevented(self) -> int:
        return max(0, self.no_guard_side_effects - self.guard_side_effects)


# ─────────────────────────────────────────
# Simulation Runners
# ─────────────────────────────────────────

def run_no_guard(scenario: Dict[str, Any]) -> tuple:
    """Run scenario with no protection. Returns (calls, side_effects, cost)."""
    tool_cost = float(scenario.get("tool_cost", 0.04))
    calls = 0
    side_effects = 0

    for step in scenario["steps"]:
        if step["type"] == "tool":
            calls += 1
            if step.get("side_effect"):
                side_effects += 1

    return calls, side_effects, calls * tool_cost


def run_with_guard(
    scenario: Dict[str, Any],
    rewrite_llm_cost: float = 0.02,
) -> Dict[str, Any]:
    """Run scenario with Aura Guard.

    Args:
        scenario: Scenario definition
        rewrite_llm_cost: Estimated LLM cost per REWRITE intervention (the
            model must be re-called with the injected system prompt). Defaults
            to $0.02 per rewrite, which is conservative for GPT-4 class models.
    """
    tool_cost = float(scenario.get("tool_cost", 0.04))
    # Generous budget so budget enforcement only triggers on the budget scenario
    budget = len(scenario["steps"]) * tool_cost

    guard = AgentGuard(
        secret_key=b"aura-guard-bench-key",
        max_cost_per_run=budget,
        default_tool_cost=tool_cost,
        side_effect_tools={"refund", "send_reply", "cancel"},
    )

    calls_executed = 0
    side_effects = 0
    rewrites = 0
    terminated = None

    for step in scenario["steps"]:
        if terminated:
            break

        if step["type"] == "tool":
            name = step["name"]
            args = step.get("args", {})
            side_effect = step.get("side_effect")

            decision = guard.check_tool(name, args=args, side_effect=side_effect)

            if decision.action == PolicyAction.ALLOW:
                result_ok = step.get("result_ok", True)
                error_code = step.get("error_code")
                guard.record_result(
                    ok=result_ok,
                    payload={"status": "ok"} if result_ok else None,
                    error_code=error_code,
                )
                calls_executed += 1
                if step.get("side_effect"):
                    side_effects += 1
            elif decision.action == PolicyAction.REWRITE:
                rewrites += 1
            elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                terminated = decision.action.value

        elif step["type"] == "llm":
            text = step["text"]
            stall = guard.check_output(text)
            if stall:
                if stall.action == PolicyAction.REWRITE:
                    rewrites += 1
                elif stall.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                    terminated = stall.action.value

    tool_cost_total = guard.cost_spent
    rewrite_cost_total = rewrites * rewrite_llm_cost

    return {
        "calls_executed": calls_executed,
        "side_effects": side_effects,
        "tool_cost": tool_cost_total,
        "rewrite_llm_cost": rewrite_cost_total,
        "total_cost": tool_cost_total + rewrite_cost_total,
        "blocks": guard.blocks,
        "cache_hits": guard.cache_hits,
        "rewrites": rewrites,
        "escalations": guard.escalations,
        "quarantined": list(guard.quarantined_tools.keys()),
        "terminated": terminated,
    }


def run_scenario(
    scenario: Dict[str, Any],
    rewrite_llm_cost: float = 0.02,
) -> ScenarioResult:
    """Run a single scenario with both variants."""
    ng_calls, ng_side, ng_cost = run_no_guard(scenario)
    g = run_with_guard(scenario, rewrite_llm_cost=rewrite_llm_cost)

    return ScenarioResult(
        scenario_id=scenario["scenario_id"],
        scenario_name=scenario["scenario_name"],
        failure_mode=scenario.get("failure_mode", "unknown"),
        total_steps=len(scenario["steps"]),
        no_guard_calls=ng_calls,
        no_guard_side_effects=ng_side,
        no_guard_cost=ng_cost,
        guard_calls=g["calls_executed"],
        guard_side_effects=g["side_effects"],
        guard_tool_cost=g["tool_cost"],
        guard_rewrite_llm_cost=g["rewrite_llm_cost"],
        guard_total_cost=g["total_cost"],
        guard_blocks=g["blocks"],
        guard_cache_hits=g["cache_hits"],
        guard_rewrites=g["rewrites"],
        guard_escalations=g["escalations"],
        guard_quarantined=g["quarantined"],
        terminated=g["terminated"],
    )


def run_all(
    scenarios_path: Optional[str] = None,
    rewrite_llm_cost: float = 0.02,
) -> List[ScenarioResult]:
    """Run all scenarios and return results."""
    scenarios = load_scenarios(scenarios_path)
    return [run_scenario(s, rewrite_llm_cost=rewrite_llm_cost) for s in scenarios]


# ─────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────

def print_report(results: List[ScenarioResult], rewrite_llm_cost: float = 0.02) -> None:
    """Print a formatted cost savings report with honest math."""
    num = len(results)
    if num == 0:
        print("  No results.")
        return

    # Split: cost-saving scenarios vs reliability-only scenarios
    # (scenarios where guard costs MORE but prevents infinite loops)
    cost_results = [r for r in results if r.saved >= 0]
    reliability_results = [r for r in results if r.saved < 0]

    print()
    print("=" * 72)
    print("  Aura Guard — Agent Failure Benchmark Report")
    print("=" * 72)
    print()

    # Main cost table
    fmt = "  {:<34} {:>10} {:>12} {:>8}"
    print(fmt.format("Scenario", "No Guard", "Aura Guard", "Saved"))
    print("  " + "─" * 68)

    total_ng = 0.0
    total_g = 0.0

    for r in results:
        print(fmt.format(
            r.scenario_name,
            f"${r.no_guard_cost:.2f}",
            f"${r.guard_total_cost:.2f}",
            f"{r.saved_pct:.0f}%",
        ))
        total_ng += r.no_guard_cost
        total_g += r.guard_total_cost

    print("  " + "─" * 68)
    total_saved = total_ng - total_g
    total_pct = (total_saved / total_ng * 100) if total_ng > 0 else 0
    print(fmt.format("TOTAL", f"${total_ng:.2f}", f"${total_g:.2f}", f"{total_pct:.0f}%"))

    # Honest per-run math
    avg_saved = total_saved / num
    avg_ng = total_ng / num
    avg_g = total_g / num

    print()
    print(f"  Suite total ({num} scenarios):       ${total_saved:.2f} saved")
    print(f"  Average per agent run:             ${avg_saved:.4f} saved "
          f"(${avg_ng:.4f} → ${avg_g:.4f})")
    print()
    print(f"  Projected monthly savings @ 1K runs/day:    ${avg_saved * 1000 * 30:,.2f}")
    print(f"  Projected monthly savings @ 10K runs/day:  ${avg_saved * 10000 * 30:,.2f}")

    # Cost breakdown
    total_tool_saved = sum(r.no_guard_cost - r.guard_tool_cost for r in results)
    total_rewrite_cost = sum(r.guard_rewrite_llm_cost for r in results)
    total_rewrites = sum(r.guard_rewrites for r in results)
    print()
    print(f"  Cost breakdown:")
    print(f"    Tool cost avoided:       ${total_tool_saved:.2f}")
    print(f"    Rewrite LLM cost added:  ${total_rewrite_cost:.2f}  "
          f"({total_rewrites} rewrites × ${rewrite_llm_cost:.2f} each)")
    print(f"    Net savings:             ${total_saved:.2f}")

    # Side-effects prevented
    total_se_prevented = sum(
        max(0, r.no_guard_side_effects - r.guard_side_effects) for r in results
    )
    if total_se_prevented > 0:
        print()
        print(f"  Side-effects prevented:    {total_se_prevented}")
        for r in results:
            prevented = r.no_guard_side_effects - r.guard_side_effects
            if prevented > 0:
                print(f"    {r.scenario_name}: {prevented} "
                      f"({r.no_guard_side_effects} → {r.guard_side_effects})")

    # Reliability containment (negative-savings scenarios where guard prevents infinite loops)
    if reliability_results:
        print()
        print(f"  Reliability containment (guard costs more but prevents infinite loops):")
        for r in reliability_results:
            print(f"    {r.scenario_name}: +${abs(r.saved):.2f} overhead, "
                  f"but escalated in {r.guard_calls} calls "
                  f"(vs unbounded without guard)")
    print()

    # Detail table
    det_fmt = "  {:<28} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}  {}"
    print("  Detail:")
    print(det_fmt.format("Scenario", "NG↓", "AG↓", "Block", "Cache", "Rw", "Esc", "Quarantined"))
    print("  " + "─" * 75)
    for r in results:
        print(det_fmt.format(
            r.scenario_name[:28],
            r.no_guard_calls, r.guard_calls,
            r.guard_blocks, r.guard_cache_hits,
            r.guard_rewrites, r.guard_escalations,
            ", ".join(r.guard_quarantined) if r.guard_quarantined else "-",
        ))
    print()


def to_json(results: List[ScenarioResult]) -> Dict[str, Any]:
    """Convert results to a JSON-serializable dict."""
    import datetime

    num = len(results)
    total_ng = sum(r.no_guard_cost for r in results)
    total_g = sum(r.guard_total_cost for r in results)
    total_saved = total_ng - total_g
    avg_saved = total_saved / num if num > 0 else 0

    return {
        "type": "aura_guard_bench",
        "benchmark": "aura_guard_agent_failure",
        "version": _get_version(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "num_scenarios": num,
        "scenarios": [
            {
                "scenario_id": r.scenario_id,
                "scenario_name": r.scenario_name,
                "failure_mode": r.failure_mode,
                "no_guard": {
                    "tool_calls": r.no_guard_calls,
                    "side_effects": r.no_guard_side_effects,
                    "cost_usd": round(r.no_guard_cost, 4),
                },
                "aura_guard": {
                    "tool_calls": r.guard_calls,
                    "side_effects": r.guard_side_effects,
                    "tool_cost_usd": round(r.guard_tool_cost, 4),
                    "rewrite_llm_cost_usd": round(r.guard_rewrite_llm_cost, 4),
                    "total_cost_usd": round(r.guard_total_cost, 4),
                    "blocks": r.guard_blocks,
                    "cache_hits": r.guard_cache_hits,
                    "rewrites": r.guard_rewrites,
                    "escalations": r.guard_escalations,
                    "quarantined": r.guard_quarantined,
                    "terminated": r.terminated,
                },
                "saved_usd": round(r.saved, 4),
                "saved_pct": round(r.saved_pct, 1),
                "side_effects_prevented": r.side_effects_prevented,
            }
            for r in results
        ],
        "totals": {
            "no_guard_cost_usd": round(total_ng, 4),
            "aura_guard_cost_usd": round(total_g, 4),
            "saved_usd": round(total_saved, 4),
            "avg_saved_per_run_usd": round(avg_saved, 4),
            "projected_monthly_1k_per_day": round(avg_saved * 1000 * 30, 2),
            "projected_monthly_10k_per_day": round(avg_saved * 10000 * 30, 2),
            "total_side_effects_prevented": sum(r.side_effects_prevented for r in results),
        },
    }


def save_json_report(results: List[ScenarioResult], path: str) -> None:
    """Save results as JSON."""
    data = to_json(results)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON report saved to: {path}")
