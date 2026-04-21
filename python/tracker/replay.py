"""Offline replay of the AI-leads IMM policy on cached AI outputs.

Replays the exact `ab_test.py` policy on cached AI bboxes/confidences instead
of running TRT inference. This keeps tuning and A/B evaluation aligned while
still allowing sub-second evaluation per sequence.
"""

import numpy as np

from tracker.decision import DecisionMaker
from tracker.imm_policy import step_guided_imm
from tracker.metrics import evaluate_sequence

# ── IMM model Q base diagonals (fixed physics ratios) ────────────
MODEL_Q_BASES = {
    0: np.array([  # CV (Constant Velocity)
        1.0, 1.0, 1.0, 1.0,
        0.01, 0.01, 0.0001, 0.0001,
        1e-6, 1e-6,
    ], dtype=np.float32),
    1: np.array([  # CA (Constant Acceleration)
        1.0, 1.0, 1.0, 1.0,
        0.1, 0.1, 0.0001, 0.0001,
        1.0, 1.0,
    ], dtype=np.float32),
    2: np.array([  # Singer (High Maneuverability)
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.001, 0.001,
        100.0, 100.0,
    ], dtype=np.float32),
}

BASE_R = np.array([1.0, 1.0, 10.0, 10.0], dtype=np.float32)


def _build_model_q(model_idx: int, q_scale: float) -> np.ndarray:
    Q = np.zeros((10, 10), dtype=np.float32)
    np.fill_diagonal(Q, MODEL_Q_BASES[model_idx] * q_scale)
    return Q


def _build_measurement_r(r_pos_scale: float, r_size_scale: float) -> np.ndarray:
    R = np.zeros((4, 4), dtype=np.float32)
    R[0, 0] = BASE_R[0] * r_pos_scale
    R[1, 1] = BASE_R[1] * r_pos_scale
    R[2, 2] = BASE_R[2] * r_size_scale
    R[3, 3] = BASE_R[3] * r_size_scale
    return R


def _build_transition_matrix(pi_persist: float) -> np.ndarray:
    off = (1.0 - pi_persist) / 2.0
    return np.array([
        [pi_persist, off, off],
        [off, pi_persist, off],
        [off, off, pi_persist],
    ], dtype=np.float32)


def replay_sequence(cache_path: str, params: dict) -> list:
    """Replay the AI-leads IMM policy on cached AI outputs.

    Args:
        cache_path: Path to .npz file with keys:
            ai_bboxes (N,4), confs (N,), gt (N,4), init_bbox (4,),
            frame_w (scalar), frame_h (scalar)
        params: Dict with tunable hyperparameters.
            Hygiene/gating: conf_threshold, coast_threshold, max_coast_frames,
                reinit_after, max_area_frac, max_center_jump_frac,
                aspect_ratio_lo, aspect_ratio_hi
            KF physics: q_scale, r_pos_scale, r_size_scale, pi_persist

    Returns:
        List of predicted bboxes [x, y, w, h] per frame.
    """
    import tracker_cpp

    data = np.load(cache_path)
    ai_bboxes = data["ai_bboxes"]      # (N, 4)
    confs = data["confs"]               # (N,)
    init_bbox = data["init_bbox"]       # (4,)
    fw = int(data["frame_w"])
    fh = int(data["frame_h"])

    n_frames = len(ai_bboxes)

    # Extract params with defaults
    conf_threshold = params.get("conf_threshold", 0.25)
    coast_threshold = params.get("coast_threshold", 0.05)
    max_coast_frames = int(params.get("max_coast_frames", 30))
    reinit_after = int(params.get("reinit_after", 8))
    max_area_frac = params.get("max_area_frac", 0.25)
    max_center_jump_frac = params.get("max_center_jump_frac", 0.30)
    aspect_ratio_lo = params.get("aspect_ratio_lo", 0.2)
    aspect_ratio_hi = params.get("aspect_ratio_hi", 5.0)

    # KF physics params
    q_scale = params.get("q_scale", 1.0)
    r_pos_scale = params.get("r_pos_scale", 1.0)
    r_size_scale = params.get("r_size_scale", 1.0)
    pi_persist = params.get("pi_persist", 0.90)
    adaptive_r_enabled = bool(params.get("adaptive_r_enabled", False))
    conf_bypass_threshold = float(params.get("conf_bypass_threshold", 0.90))
    innovation_threshold = float(params.get("innovation_threshold", 2.0))
    sm_conf_threshold = float(params.get("sm_conf_threshold", conf_threshold))
    sm_max_coast = int(params.get("sm_max_coast", 30))
    r_exponent = float(params.get("r_exponent", 1.0))
    maneuver_threshold = float(params.get("maneuver_threshold", 0.0))
    maneuver_bypass_boost = float(params.get("maneuver_bypass_boost", 0.0))
    mahal_chi2_threshold = float(params.get("mahal_chi2_threshold", 18.47))
    mahal_chi2_gate = float(params.get("mahal_chi2_gate", 0.0))
    r_pos_base = float(params.get("r_pos_base", 1.0))
    r_size_base = float(params.get("r_size_base", 10.0))
    mahal_bypass_after = int(params.get("mahal_bypass_after", 5))
    alpha_gate_k_conf = float(params.get("alpha_gate_k_conf", 0.0))
    alpha_gate_lambda = float(params.get("alpha_gate_lambda", 0.0))
    reacq_r_decay = float(params.get("reacq_r_decay", 0.0))
    # Faz C1: AI skip when confident (replicated from Pipeline._should_skip_ai)
    ai_skip_when_confident = bool(params.get("ai_skip_when_confident", False))
    ai_max_consecutive_skips = int(params.get("ai_max_consecutive_skips", 2))
    singer_skip_thr = float(params.get("singer_skip_thr", 0.35))
    cv_stable_thr   = float(params.get("cv_stable_thr",   0.70))
    cv_conf_min     = float(params.get("cv_conf_min",     0.65))

    # Create filter and decision maker
    kf = tracker_cpp.IMMFilter()
    for m in range(3):
        kf.set_model_process_noise(m, _build_model_q(m, q_scale))
    kf.set_measurement_noise(_build_measurement_r(r_pos_scale, r_size_scale))
    kf.set_transition_matrix(_build_transition_matrix(pi_persist))

    adaptive_r_floor = params.get("adaptive_r_floor", 0.4)
    if adaptive_r_enabled:
        kf.set_adaptive_r_floor(float(adaptive_r_floor))

    sm = tracker_cpp.TrackerState()
    tracking_state_enum = tracker_cpp.TrackState.TRACKING
    sm.set_confidence_threshold(sm_conf_threshold)
    sm.set_max_coast_frames(sm_max_coast)
    dec = DecisionMaker(
        conf_threshold=conf_threshold,
        coast_threshold=coast_threshold,
        max_coast_frames=max_coast_frames,
        max_area_frac=max_area_frac,
        aspect_ratio_range=(aspect_ratio_lo, aspect_ratio_hi),
        max_center_jump_frac=max_center_jump_frac,
    )

    pred_bboxes = []
    last_good_bbox = init_bbox.tolist()
    reject_streak = 0
    _prev_confidence = 0.0
    _consecutive_skips = 0

    for i in range(n_frames):
        if i == 0:
            # Frame 0: init
            init_arr = np.array(init_bbox, dtype=np.float32)
            kf.init(init_arr)
            sm.force_tracking()
            pred_bboxes.append(init_bbox.tolist())
        else:
            ai_list = ai_bboxes[i].tolist()
            conf = float(confs[i])

            # Always predict once per frame (mirrors Pipeline order).
            predicted_state = np.array(kf.predict()).flatten()

            # Faz C1: IMM-guided AI skip (mirrors Pipeline._should_skip_ai).
            # When skipping: carry forward prev confidence; pass judged_bbox=None
            # so the filter coasts (predict-only), exactly as Pipeline does.
            skip_ai = False
            if ai_skip_when_confident and kf.is_initialized():
                if _consecutive_skips >= ai_max_consecutive_skips:
                    _consecutive_skips = 0
                else:
                    _mu = np.array(kf.get_model_probabilities(), dtype=np.float32)
                    mu_cv     = float(_mu[0])
                    mu_ca     = float(_mu[1])
                    mu_singer = float(_mu[2])
                    if mu_singer <= singer_skip_thr and _prev_confidence >= 0.40:
                        if mu_cv > cv_stable_thr and _prev_confidence > cv_conf_min:
                            _consecutive_skips += 1
                            skip_ai = True
                        elif mu_ca > 0.50 and _prev_confidence > 0.50 and _consecutive_skips < 1:
                            _consecutive_skips += 1
                            skip_ai = True
                    if not skip_ai:
                        _consecutive_skips = 0

            if skip_ai:
                # Carry forward previous confidence; coast (no measurement).
                conf = _prev_confidence
                judged_bbox = None
            else:
                _prev_confidence = conf
                _consecutive_skips = 0
                judged_bbox = np.asarray(ai_list, dtype=np.float32)

            track_state = sm.step(conf)
            step = step_guided_imm(
                kf,
                dec,
                predicted_state,
                judged_bbox,
                conf,
                fw,
                fh,
                is_tracking=(track_state == tracking_state_enum),
                judge_reference_bbox=predicted_state[:4],
                last_good_bbox=last_good_bbox,
                reject_streak=reject_streak,
                reinit_after=reinit_after,
                adaptive_r_enabled=adaptive_r_enabled,
                r_exponent=r_exponent,
                conf_bypass_threshold=conf_bypass_threshold,
                innovation_threshold=innovation_threshold,
                maneuver_threshold=maneuver_threshold,
                maneuver_bypass_boost=maneuver_bypass_boost,
                mahal_chi2_threshold=mahal_chi2_threshold,
                mahal_chi2_gate=mahal_chi2_gate,
                r_pos_base=r_pos_base,
                r_size_base=r_size_base,
                mahal_bypass_after=mahal_bypass_after,
                coast_count=sm.coast_count(),
                alpha_gate_k_conf=alpha_gate_k_conf,
                alpha_gate_lambda=alpha_gate_lambda,
                reacq_r_decay=reacq_r_decay,
            )
            bbox = step.bbox
            last_good_bbox = step.last_good_bbox
            reject_streak = step.reject_streak

            pred_bboxes.append(bbox)

    return pred_bboxes


def evaluate_cached(cache_path: str, params: dict) -> tuple[float, float]:
    """Replay and evaluate one cached sequence.

    Returns:
        (auc, norm_precision)
    """
    data = np.load(cache_path)
    gt = data["gt"].tolist()
    pred_bboxes = replay_sequence(cache_path, params)
    return evaluate_sequence(gt, pred_bboxes)


def raw_baseline_score(cache_path: str) -> tuple[float, float]:
    """Evaluate AI-only baseline (no filter) from cached outputs.

    Returns:
        (auc, norm_precision)
    """
    data = np.load(cache_path)
    gt = data["gt"].tolist()
    ai_bboxes = data["ai_bboxes"].tolist()
    return evaluate_sequence(gt, ai_bboxes)


# ── Default params matching ab_test.py runtime defaults ──────────

DEFAULT_PARAMS = {
    "conf_threshold": 0.18,
    "coast_threshold": 0.05,
    "max_coast_frames": 30,
    "reinit_after": 8,
    "max_area_frac": 0.25,
    "max_center_jump_frac": 0.30,
    "aspect_ratio_lo": 0.2,
    "aspect_ratio_hi": 5.0,
    # Legacy no-op fields kept for backward-compatible tuned configs.
    "alpha_conf_lo": 0.3,
    "alpha_conf_hi": 0.7,
    "q_scale": 1.0,
    "r_pos_scale": 1.0,
    "r_size_scale": 1.0,
    "pi_persist": 0.90,
    "adaptive_r_enabled": False,
    "r_exponent": 1.0,
    # Smart intervention params
    "conf_bypass_threshold": 0.15,
    "innovation_threshold": 2.0,
    "sm_conf_threshold": 0.18,
    "sm_max_coast": 30,
    "maneuver_threshold": 0.0,
    "maneuver_bypass_boost": 0.0,
    # Mahalanobis gate params
    "mahal_chi2_threshold": 23.51,
    "mahal_chi2_gate": 0.0,
    "r_pos_base": 1.0,
    "r_size_base": 10.0,
    "mahal_bypass_after": 2,
}
