# Live A/B Example (Screenshots)

This page is **optional** and mainly here for people who like to *see the console output*.

If you prefer fully reproducible artifacts, use the JSON report files in `reports/`.

---

## The command that produced the report

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
| A: Jitter Loop | $0.2778 | $0.1447 | $0.1331 | 48% |
| B: Double Refund | $0.1396 | $0.1275 | $0.0120 | 9% |
| C: Error Retry Spiral | $0.1345 | $0.0952 | $0.0393 | 29% |
| D: Smart Reformulation | $0.8093 | $0.1464 | $0.6629 | 82% |
| E: Flagship | $0.3497 | $0.1420 | $0.2077 | 59% |

### Task completion (quality)

| Scenario | No Guard | With Guard | Notes |
|----------|----------|------------|-------|
| A: Jitter Loop | N/A | N/A | No ground-truth answer (KB intentionally vague) |
| B: Double Refund | 100% | 100% | Guard prevented duplicate, task still completed |
| C: Error Retry | 40% | 80% | Guard stopped retry storm, escalated cleanly |
| D: Smart Reformulation | 67% | 80% | Guard capped search, forced resolution |
| E: Flagship | 100% | 100% | Full task completion with fewer calls |

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
- 0 false positives
- False positive rate: 0%

**Caveat:** These scenarios use rigged tool implementations designed to trigger known failure modes. A 0% false positive rate here is expected and does not predict production performance. Real-world false positive rate requires shadow mode evaluation on production traffic — see EVALUATION_PLAN.md Phase 2–3 for methodology.
