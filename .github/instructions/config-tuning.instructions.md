---
description: "YAML config validation and parameter tuning rules. Use when: modifying tracker YAML configs, tuning filter parameters, reviewing imm_tuned.yaml or tracker_config.yaml, creating new config variants."
---

# Config & Tuning Rules

## Parameter Documentation

Every parameter in a config YAML must have a comment explaining:
1. What it controls
2. Valid range
3. What happens at extremes

## Cached Replay vs Live Evaluation

**Cached replay is NOT ground truth.** Optuna/Bayesian tuning on cached AI outputs can produce configs that fail in live closed-loop testing. The rejected `configs/filter_tuned.yaml` is proof.

Always verify config changes with:
```bash
python3 scripts/ab_test.py --imm --gmc --adaptive-r
```

## Change Protocol

1. Never modify `configs/tracker_config.yaml` (baseline defaults) — create a new variant
2. `configs/imm_tuned.yaml` is the active tuned config — back up before modifying
3. Every config change must be paired with an A/B test result showing Delta
4. If Delta is negative, revert immediately

## Known Parameter Interactions

- `q_scale` ↔ `r_pos_scale`: increasing Q without increasing R makes KF trust AI less
- `conf_threshold` ↔ `coast_threshold`: must maintain `conf_threshold ≤ coast_threshold`
- `max_coast_frames` ↔ category: aerial/animal need shorter coast, static/structure tolerates longer
- `adaptive_r_floor` ↔ `conf_threshold`: floor must be > conf_threshold to have any effect
