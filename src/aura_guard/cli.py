"""aura_guard.cli

CLI entrypoint for Aura Guard.

Usage:
    aura-guard demo                     Run the triage simulation demo
    aura-guard bench --all              Run all benchmark scenarios
    aura-guard bench --scenario ID      Run a specific scenario
    aura-guard bench --json-out F       Save JSON report to file
    aura-guard version                  Print version
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def cmd_demo(args: argparse.Namespace) -> None:
    """Run the triage simulation demo."""
    from .bench.demo import run_demo
    run_demo(json_out=getattr(args, "json_out", None))


def cmd_bench(args: argparse.Namespace) -> None:
    """Run the benchmark suite."""
    from .bench.runner import (
        load_scenarios,
        run_scenario,
        print_report,
        save_json_report,
    )

    scenarios = load_scenarios(getattr(args, "scenarios_file", None))

    if hasattr(args, "scenario") and args.scenario:
        scenarios = [s for s in scenarios if s["scenario_id"] == args.scenario]
        if not scenarios:
            print(f"  Scenario '{args.scenario}' not found.", file=sys.stderr)
            sys.exit(1)

    rewrite_cost = getattr(args, "rewrite_cost", 0.02)
    results = [run_scenario(s, rewrite_llm_cost=rewrite_cost) for s in scenarios]
    print_report(results, rewrite_llm_cost=rewrite_cost)

    json_out = getattr(args, "json_out", None)
    if json_out:
        save_json_report(results, json_out)


def cmd_version(args: argparse.Namespace) -> None:
    """Print version."""
    from . import __version__
    print(f"aura-guard {__version__}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aura-guard",
        description="Aura Guard — reliability middleware for tool-using agents (idempotency, circuit breaking, loop detection).",
    )
    subparsers = parser.add_subparsers(dest="command")

    # demo
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the triage simulation demo",
    )
    demo_parser.add_argument("--json-out", type=str, help="Save JSON report to this path")
    demo_parser.set_defaults(func=cmd_demo)

    # bench
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run the agent failure benchmark suite",
    )
    bench_parser.add_argument("--all", action="store_true", help="Run all scenarios")
    bench_parser.add_argument("--scenario", type=str, help="Run a specific scenario by ID")
    bench_parser.add_argument("--scenarios-file", type=str, help="Path to custom scenarios JSON")
    bench_parser.add_argument("--json-out", type=str, help="Save JSON report to this path")
    bench_parser.add_argument(
        "--rewrite-cost", type=float, default=0.02,
        help="Estimated LLM cost per REWRITE intervention (default: $0.02)",
    )
    bench_parser.set_defaults(func=cmd_bench)

    # version
    version_parser = subparsers.add_parser("version", help="Print version")
    version_parser.set_defaults(func=cmd_version)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "bench" and not args.all and not getattr(args, "scenario", None):
        bench_parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
