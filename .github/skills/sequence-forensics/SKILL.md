---
name: sequence-forensics
description: "Deep forensic analysis of a specific tracking sequence. Use when: investigating why a sequence fails, analyzing per-frame tracking data, diagnosing occlusion and coasting issues, understanding AUC drops on specific sequences."
argument-hint: "Sequence name, e.g. dataset3/uav1"
---

# Sequence Forensics

Deep-dive failure analysis on a single tracking sequence to identify root cause and propose targeted fixes.

## When to Use

- A sequence has Δ AUC < -0.05 (IMM hurts instead of helping)
- A sequence has AUC < 0.4 (poor absolute performance)
- Understanding why coasting failed on a specific video
- After @diagnosis identifies a problematic sequence

## Procedure

### Step 1 — Run the Sequence

```bash
python3 scripts/run_competition.py --seq <dataset>/<name> --split train 2>&1 | tee /tmp/forensics_<name>.log
```

### Step 2 — Collect Frame-Level Data

From the output, extract per-frame:
- Frame number
- AI confidence score
- Tracker state (TRACKING / COASTING / LOST)
- Predicted bbox [x, y, w, h]
- Ground truth bbox [x, y, w, h]
- IoU between prediction and GT

### Step 3 — Identify Failure Windows

A **failure window** is a contiguous range of frames where `IoU < 0.3`.

For each window, record:
- Start/end frame
- Duration (frames)
- State at entry (what was tracker doing when it entered failure?)
- Confidence trend (rising, falling, collapsed)
- Was GMC active? Did it help or hurt?

### Step 4 — Classify Failure Mode

Use the taxonomy from @diagnosis:

| Code | Mode | Key Signal |
|------|------|-----------|
| F1 | Coast-Too-Long | COASTING > 15 frames, IoU monotonically declining |
| F2 | Coast-Too-Short | LOST triggered, but GT still visible in frame |
| F3 | AI-Confidence-Collapse | conf < 0.1 for > 5 frames, no occlusion in GT |
| F4 | GMC-Warp-Error | Affine matrix has extreme values (scale > 1.5 or < 0.5) |
| F5 | KF-Drift | Distance between predict center and GT center > 50px/frame growing |
| F6 | ID-Switch | IoU with wrong object > 0.5 |
| F7 | Scale-Mismatch | Predicted w/h differs from GT w/h by > 50% |
| F8 | Edge-of-Frame | GT bbox partially outside frame bounds |

### Step 5 — Propose Fix

For each failure window, propose a specific fix:
- Parameter change: which parameter, current value → proposed value, expected effect
- Code change: which module, what logic to modify, pseudo-code
- Config variant: create a new YAML for category-specific tuning

### Step 6 — Cross-Validate

**Never** accept a fix based on single-sequence improvement. Test on:
1. The failed sequence (must improve)
2. 5 sequences from the same category (must not regress)
3. 5 sequences from different categories (must not regress)

```bash
python3 scripts/run_competition.py --seq <same_category_seq_1> --split train
python3 scripts/run_competition.py --seq <same_category_seq_2> --split train
# ... repeat for all validation sequences
```

## Output Format

```
## Forensic Report — [sequence_name]

### Summary
| Metric | Value |
|--------|-------|
| Total Frames | ... |
| AUC (raw AI) | ... |
| AUC (IMM) | ... |
| Δ AUC | ... |
| Failure Windows | N |
| Total Failure Frames | ... |

### Failure Timeline
| Window | Frames | Duration | Entry State | Mode | Confidence | IoU Range |
|--------|--------|----------|-------------|------|-----------|-----------|

### Root Cause Analysis
[Detailed narrative for each failure window]

### Proposed Fixes
| # | Target | Current | Proposed | Expected Δ AUC |
|---|--------|---------|----------|----------------|

### Cross-Validation Plan
| Sequence | Category | Purpose |
|----------|----------|---------|
```
