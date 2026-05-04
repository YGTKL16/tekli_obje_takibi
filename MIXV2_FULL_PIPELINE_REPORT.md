# MixFormerV2-Small Full Pipeline Experiment

**Date:** 2025  
**Goal:** Evaluate impact of providing MixFormerV2 with the same instruments (IMM + KF + GMC + DecisionMaker + TrackerState FSM) that SGLATrack uses.

---

## TL;DR

> **MixFormerV2 + full pipeline cannot exceed MixV2 raw.**  
> Best achievable: `0.6944` (config `mixv2_c1_no_mahal_block.yaml`)  
> MixV2 raw (no KF/IMM/GMC): `0.6947`  
> SGLATrack + full pipeline (`imm_tuned.yaml`): `0.7142`
>
> The IMM/KF machinery — tuned for SGLATrack's confidence distribution — actively *hurts* MixV2 by `-0.0049` AUC when used unmodified. Disabling the Mahalanobis bypass gate (`mahalanobis.bypass_after: 0`) recovers the loss, but the pipeline contributes essentially nothing positive.

---

## Setup

- **Backend:** MixFormerV2-Small (PyTorch, ViT, `mixformerv2_small.pth.tar`)
- **Wrapper:** `python/tracker/mixformerv2_wrapper.py`
- **Split:** `train` (255 sequences, ~223k frames)
- **Hardware:** RTX 4060 Max-Q
- **Throughput:** 130 fps (full pipeline) vs 118 fps (raw)
- **Eval:** `scripts/evaluate_local.py`, FinalScore = 0.6·AUC + 0.4·NormPrecision

## Results matrix

| Run | Config | conf_bypass | mahal.bypass_after | F5 | AUC | NP | **FinalScore** | ΔvsRaw |
|-----|--------|-------------|--------------------|----|-----|-----|----------------|--------|
| Raw | `--no-kf` | – | – | – | 0.6566 | 0.7519 | **0.6947** | 0 |
| A1 | `imm_tuned.yaml` | 0.74 | 2 | off | 0.6558 | 0.7407 | 0.6898 | −0.0049 |
| B1 | `mixv2_b1_aggressive_bypass.yaml` | 0.50 | 2 | off | 0.6558 | 0.7407 | 0.6898 | −0.0049 |
| B2 | `mixv2_b2_minimal_imm.yaml` (Q×0.16, R×0.11) | 0.40 | 2 | off | 0.6491 | 0.7336 | 0.6829 | −0.0118 |
| **C1** | `mixv2_c1_no_mahal_block.yaml` | 0.50 | **0** | off | **0.6602** | **0.7457** | **0.6944** | **−0.0003** |
| C2 | `mixv2_c2_f5_coast.yaml` | 0.50 | 0 | coast-only | 0.6602 | 0.7457 | 0.6944 | −0.0003 |
| C3 | `mixv2_c3_f5_always.yaml` | 0.50 | 0 | always | 0.6602 | 0.7457 | 0.6944 | −0.0003 |
| C4 | `mixv2_c4_pos_only.yaml` | 0.85 | 0 | pos-only | 0.6600 | 0.7455 | 0.6942 | −0.0005 |

## Diagnosis

### Why imm_tuned hurts MixV2 (A1)

Per-sequence delta vs raw on A1:
- IMM helps: 12, hurts: **28**, ties: 215
- Top regressions: `person7_1` (−0.287), `person7` (−0.170), `person12_2` (−0.164), `human2` (−0.143), `Vaulting` (−0.079), `group1` (−0.065)
- **Pattern:** non-rigid pedestrian/human motion. SGLA-tuned `mahal_chi2_threshold=23.51` blocks bypass when MixV2 reports high confidence on a marginally-shifted bbox (frequent for human deformation). Once bypass is blocked, KF blend pulls bbox toward stale prediction, search anchor drifts, MixV2 then loses the target.

### Why C1 fixes it

Setting `mahalanobis.bypass_after: 0` makes the Mahalanobis check non-blocking — high-confidence MixV2 measurements always pass through to the output (and reset the KF). This restores raw-equivalent behaviour. After C1 the per-sequence delta vs raw is essentially zero (mean dAUC = −0.0003, only 1 win, 6 small losses, 248 ties).

### Why F5 changes nothing (C2/C3/C4)

When `conf_bypass_threshold` is low and the gate doesn't block, the output bbox equals `ai_bbox` (MixV2 raw). F5 then sets the next search anchor to that same `ai_bbox` — exactly what `MixFormerV2Wrapper.track()` does internally on line 266. F5 becomes a no-op. Raising the bypass threshold (C4) forces blending and immediately costs accuracy.

### Why MixV2 doesn't benefit from IMM the way SGLATrack does

| | SGLATrack | MixFormerV2 |
|---|-----------|-------------|
| Confidence distribution | flatter, mid-range possible | bimodal: very high or very low |
| Failure mode | gradual confidence drop | sudden complete loss |
| Bbox stability when high-conf | small jitter | already very stable |
| Coast benefit | high (KF stabilises borderline frames) | low (no borderline frames to stabilise) |

The IMM/KF pipeline's value is greatest when AI confidence is **noisy in the middle**. MixV2 has very little middle, so there is little signal for the KF to clean.

## Conclusions

1. **MixV2 + full pipeline is not better than MixV2 raw.** Any submission using MixV2 should pass `--no-kf`.
2. **If SGLA-tuned IMM is forced onto MixV2, set `mahalanobis.bypass_after: 0`** (config C1) to avoid catastrophic regressions on pedestrian/group sequences.
3. **SGLATrack remains the better full-pipeline backbone** for this competition: 0.7142 vs MixV2-best 0.6944, a Δ of +0.0198.
4. **Recommended action:** keep SGLATrack + `imm_tuned.yaml` as the production stack; treat MixV2 as a candidate for ensemble/fusion only.

## Artifacts

- Configs: `configs/mixv2_b1_*`, `configs/mixv2_b2_*`, `configs/mixv2_c{1,2,3,4}_*.yaml`
- Output CSVs: `outputs/submission_train_mixv2_{imm,b1,b2,c1,c2,c3,c4}.csv`
- Eval logs: `/tmp/eval_{mixv2_imm_A1,b2,c1,c2,c3,c4}.txt`
- Wrapper: `python/tracker/mixformerv2_wrapper.py`
- Entry: `scripts/run_competition.py --backend mixformerv2`
