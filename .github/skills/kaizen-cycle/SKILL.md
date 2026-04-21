---
name: kaizen-cycle
description: "Run a full Kaizen improvement cycle: capture baseline, propose mutation, evaluate with A/B test, compute Delta, accept or reject. Use when: iterating on tracker parameters, running optimization loops, continuous improvement, tuning filter settings."
---

# Kaizen Improvement Cycle

A structured, repeatable workflow for incremental tracker optimization.

## When to Use

- After identifying a performance bottleneck or failure pattern
- During parameter tuning iterations
- When testing a new hypothesis about tracker behavior

## Prerequisites

- Working build: `distrobox enter tracker-dev -- bash -c "cmake --build build2 -j$(nproc)"`
- All tests passing: ctest + pytest
- Baseline FinalScore known

## Procedure

### Step 1 — Capture Baseline

Run the 20-sequence A/B test and record the baseline:

```bash
python3 scripts/ab_test.py --imm --gmc --adaptive-r 2>&1 | tee /tmp/kaizen_baseline.log
```

Extract `FinalScore` from the summary line. This is your **baseline**.

### Step 2 — Propose Mutation

Choose ONE of these mutation strategies:

| Strategy | Description | Risk |
|----------|-------------|------|
| **Single-variable** | Change one parameter by ±10-50% | Low |
| **Correlated pair** | Adjust two interacting params together | Medium |
| **Structural** | Modify decision logic or add new feature | High |

Edit the config file or source code. Document exactly what changed.

### Step 3 — Rebuild (if C++ changed)

```bash
distrobox enter tracker-dev -- bash -c "cd /home/ykula/tracker && cmake --build build2 -j$(nproc)"
distrobox enter tracker-dev -- bash -c "cd /home/ykula/tracker/build2 && ctest --output-on-failure"
```

Both must succeed. If build or tests fail, fix before proceeding.

### Step 4 — Evaluate

```bash
python3 scripts/ab_test.py --imm --gmc --adaptive-r 2>&1 | tee /tmp/kaizen_trial.log
```

### Step 5 — Compute Delta

```
Δ = trial_FinalScore − baseline_FinalScore
```

### Step 6 — Decision

| Δ | Action |
|---|--------|
| > +0.005 | **ACCEPT** — commit the change, update baseline |
| −0.001 to +0.005 | **OBSERVE** — run on full train split to confirm |
| < −0.001 | **REJECT** — revert all changes immediately |

### Step 7 — Log Result

Append to evolution journal:
```
Trial #N | [date] | [mutation description] | Δ = [value] | [ACCEPT/REJECT/OBSERVE]
```

### Step 8 — Iterate or Pivot

- **ACCEPT**: Return to Step 2 with new baseline
- **REJECT (1st-2nd)**: Try different parameter/strategy → Step 2
- **REJECT (3rd consecutive)**: Switch mutation strategy entirely
- **REJECT (5th consecutive)**: Stop. Escalate to @architect for structural review.

## Anti-Patterns

- Tuning multiple parameters simultaneously without understanding interactions
- Using cached replay results as final validation (always use live A/B)
- Accepting marginal gains (+0.001) without cross-validation on broader dataset
- Continuing to tune the same parameter after 3 rejections
