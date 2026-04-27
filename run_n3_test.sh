#!/bin/bash
cd /home/ykula/tracker
python3 scripts/ab_test.py \
    --gmc \
    --mode ai_lead \
    --imm-config configs/n4_f5_coast_only.yaml \
    --seq dataset4/car6 \
    --seq dataset5/building2 \
    --seq dataset3/truck_night \
    --seq dataset3/air_conditioning_box2
