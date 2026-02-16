# Live A/B Example (Screenshots)

This page is **optional** and mainly here for people who like to *see the console output*.

If you prefer fully reproducible artifacts, use the JSON report files in `reports/`.

---

## The command that produced the report

> **Note:** This report was generated using v0.3.1 of the guard engine. Releases v0.3.2–v0.3.3 and v0.3.5–v0.3.6 were documentation and packaging fixes. v0.3.4 added secret-key enforcement, `strict_mode`, and thread-safety assertions. v0.3.7 changed error streak tracking (now resets on success) and added `side_effect_executed` support. None of these changes affect the scenarios measured in this report.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
python examples/live_test.py --ab --runs 5 --json-out ab.json
```

The JSON artifact is committed in:

- `reports/2026-02-09_claude-sonnet-4_ab.json`

---

## A/B summary (5 runs per scenario)

![A/B summary table](assets/live_ab_summary.png)

---

## Per-scenario console output

### Scenario A — Jitter Loop

![Scenario A output](assets/scenario_a.png)

### Scenario B — Double Refund

![Scenario B output](assets/scenario_b.png)

### Scenario C — Error Retry Spiral

![Scenario C output](assets/scenario_c.png)

### Scenario D — Smart Reformulation

![Scenario D output](assets/scenario_d.png)

### Scenario E — Flagship

![Scenario E output](assets/scenario_e.png)

## Results Analysis

### Cost reduction

| Scenario | No Guard | With Guard | Saved | Saved % |
|----------|----------|------------|-------|---------|
| A: Jitter Loop | $0.2778 | $0.1446 | $0.1332 | 48% |
| B: Double Refund | $0.1397 | $0.1456 | -$0.0059 | -4% *(prevented duplicate refund)* |
| C: Error Retry Spiral | $0.1345 | $0.0953 | $0.0392 | 29% |
| D: Smart Reformulation | $0.8607 | $0.1465 | $0.7142 | 83% |
| E: Flagship | $0.3494 | $0.1446 | $0.2048 | 59% |

### Task completion (quality)

| Scenario | No Guard | With Guard | Notes |
|----------|----------|------------|-------|
| A: Jitter Loop | N/A | N/A | No ground-truth answer (KB intentionally vague) |
| B: Double Refund | 100% | 100% | Guard prevented duplicate, task still completed |
| C: Error Retry | 40% | 80% | Guard stopped retry storm, escalated cleanly |
| D: Smart Reformulation | 67% | 80% | Guard capped search, forced resolution |
| E: Flagship | 100% | 100% | Full task completion with fewer calls |

Task completion was maintained or improved in scored scenarios (B–E). Scenario A quality was not scored (loop-containment test only).

### Guard interventions

Total: 64 across 25 runs (all with-guard runs triggered at least one intervention).

Breakdown by type:
- max_calls_per_tool: 15
- tool_quarantined (max_calls_per_tool): 39
- idempotent_replay: 5
- tool_quarantined (error_retry 429): 5

### False positive analysis

Of the 64 guard interventions across 25 runs, all 64 were reviewed:

- 64 true positives (correctly prevented loops or duplicate side-effects)
- No false positives observed in manual review
- False positive rate: 0%

**Caveat:** These scenarios use rigged tool implementations designed to trigger known failure modes. A 0% false positive rate here is expected and does not predict production performance. Real-world false positive rate requires shadow mode evaluation on production traffic — see EVALUATION_PLAN.md Phase 2–3 for methodology.
