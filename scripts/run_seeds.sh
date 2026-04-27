#!/bin/bash
set -e

# Default config if needed
CONFIG="configs/i12_rescue_area_gate.yaml"

for SEED in 1 2 3; do
    echo "============================================================"
    echo "==================== RUNNING SEED $SEED ===================="
    echo "============================================================"
    
    # Run competition script on the train split (since evaluate_local expects GT)
    python3 scripts/run_competition.py --split train --seed $SEED --imm-config "$CONFIG"
    
    echo "------------------------------------------------------------"
    echo "------------------ EVALUATING SEED $SEED -------------------"
    echo "------------------------------------------------------------"
    
    # Evaluate predictions
    python3 scripts/evaluate_local.py
    
    echo ""
done
