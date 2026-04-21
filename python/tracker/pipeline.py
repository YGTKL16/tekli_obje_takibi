"""Main tracking pipeline: AI -> Decision -> Kalman Filter -> Output."""

import time
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import tracker_cpp  # pyright: ignore[reportMissingImports]
    HAS_CPP = True
except ImportError:
    tracker_cpp = None
    HAS_CPP = False
    print("[WARN] tracker_cpp module not found. Build with: cmake -B build && cmake --build build")

from .decision import DecisionMaker
from .gmc import GMCEstimator
from .imm_policy import observe_with_guidance, step_guided_imm
from .sglatrack_wrapper import DEFAULT_CHECKPOINT_PATH, SGLATrackWrapper
from .visualizer import visualize


class Pipeline:
    """End-to-end tracking pipeline.

    Flow per frame:
        1. Read frame             (~1-3 ms)
        2. GMC estimate           (~3-6 ms, optional)
        3. AI inference           (~10-15 ms, skippable when confident)
        4. Decision (update/coast) (~0.01 ms)
        5. KF predict/update      (~0.05 ms)
        6. Visualize              (~1-2 ms)
        Total budget: ≤ 30 ms / frame
    """

    def __init__(
        self,
        video_path: str,
        initial_bbox: np.ndarray,
        model_path: str | None = None,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.2,
        max_coast_frames: int = 60,
        show: bool = True,
        association_enabled: bool = True,
        association_top_k: int = 5,
        association_iou_threshold: float = 0.3,
        association_score_weight: float = 0.0,
        gmc_enabled: bool = False,
        gmc_fail_q_boost: float = 4.0,
        gmc_downsample: float = 0.5,
        gmc_n_features: int = 200,
        gmc_inlier_ratio_threshold: float = 0.3,
        gmc_foreground_dilate_factor: float = 1.4,
        adaptive_r_enabled: bool = False,
        adaptive_r_floor: float = 0.4,
        r_exponent: float = 1.0,
        mahal_chi2_gate: float = 0.0,
        r_pos_base: float = 1.0,
        r_size_base: float = 10.0,
        maneuver_threshold: float = 0.0,
        maneuver_bypass_boost: float = 0.0,
        ai_skip_when_confident: bool = False,
        ai_skip_conf_threshold: float = 0.8,
        ai_max_consecutive_skips: int = 2,
        singer_skip_thr: float = 0.35,
        cv_stable_thr: float = 0.70,
        cv_conf_min: float = 0.65,
        search_scale_boost: float = 0.0,
        alpha_gate_k_conf: float = 0.0,
        alpha_gate_lambda: float = 0.0,
        reacq_r_decay: float = 0.0,
        conf_refresh_low: float = 0.35,
        conf_refresh_high: float = 0.65,
        refresh_patience: int = 4,
    ):
        self.video_path = video_path
        self.initial_bbox = initial_bbox.astype(np.float32)
        self.show = show
        self.association_enabled = association_enabled
        self.association_top_k = association_top_k

        self.adaptive_r_enabled = adaptive_r_enabled
        self.r_exponent = r_exponent
        self.mahal_chi2_gate = mahal_chi2_gate
        self.r_pos_base = r_pos_base
        self.r_size_base = r_size_base
        self.maneuver_threshold = maneuver_threshold
        self.maneuver_bypass_boost = maneuver_bypass_boost
        self.ai_skip_when_confident = ai_skip_when_confident
        self.ai_skip_conf_threshold = ai_skip_conf_threshold
        self.ai_max_consecutive_skips = ai_max_consecutive_skips
        self.singer_skip_thr = singer_skip_thr
        self.cv_stable_thr = cv_stable_thr
        self.cv_conf_min = cv_conf_min
        self.search_scale_boost = search_scale_boost
        self.alpha_gate_k_conf = alpha_gate_k_conf
        self.alpha_gate_lambda = alpha_gate_lambda
        self.reacq_r_decay = reacq_r_decay
        self.conf_refresh_low = conf_refresh_low
        self.conf_refresh_high = conf_refresh_high
        self.refresh_patience = refresh_patience

        # Components
        checkpoint_path = model_path if model_path is not None else DEFAULT_CHECKPOINT_PATH
        self.ai = SGLATrackWrapper(
            checkpoint_path,
            association_enabled=association_enabled,
            association_top_k=association_top_k,
            association_iou_threshold=association_iou_threshold,
            association_score_weight=association_score_weight,
        )
        self.decision = DecisionMaker(conf_threshold, iou_threshold)

        self.gmc_enabled = bool(gmc_enabled and HAS_CV2)
        self.gmc = (
            GMCEstimator(
                n_features=gmc_n_features,
                inlier_ratio_threshold=gmc_inlier_ratio_threshold,
                foreground_dilate_factor=gmc_foreground_dilate_factor,
                downsample=gmc_downsample,
            )
            if self.gmc_enabled
            else None
        )

        self.kf = None
        self.state_machine = None

        if HAS_CPP:
            assert tracker_cpp is not None
            self.kf = tracker_cpp.IMMFilter()
            self.state_machine = tracker_cpp.TrackerState()
            self.state_machine.set_confidence_threshold(conf_threshold)
            self.state_machine.set_max_coast_frames(max_coast_frames)
            if self.gmc_enabled and hasattr(self.kf, "set_gmc_q_boost"):
                self.kf.set_gmc_q_boost(float(gmc_fail_q_boost))
            if self.adaptive_r_enabled and hasattr(self.kf, "set_adaptive_r_floor"):
                self.kf.set_adaptive_r_floor(float(adaptive_r_floor))

        # Frame-skipping state: tracks previous SGLATrack confidence and
        # consecutive-skip count so one coast can't chain forever.
        self._prev_confidence = 0.0
        self._consecutive_skips = 0

        # IMM policy state persisted across frames.
        self._reject_streak: int = 0
        self._last_good_bbox: list[float] | None = None

        # Proactive template refresh: counts consecutive frames where
        # AI confidence is in the moderate zone [conf_refresh_low, conf_refresh_high].
        self._moderate_conf_streak: int = 0

        # GMC state: the previous raw frame (BGR) used to pair with the current
        # frame for homography estimation.
        self._prev_frame_bgr: "np.ndarray | None" = None

        # Timing
        self.timings = {"read": [], "gmc": [], "ai": [], "decision": [], "kf": [], "viz": []}

    def _maybe_refresh_template(
        self,
        frame_rgb: "np.ndarray",
        confidence: float,
        step: "object",
    ) -> None:
        """Proactive template refresh for the moderate-confidence staleness spiral.

        When AI confidence lingers in the moderate zone [conf_refresh_low,
        conf_refresh_high] for refresh_patience consecutive accepted-measurement
        frames, the template has drifted but not yet triggered the deep-loss
        rescue.  Re-initialising at the KF-fused location resets entropy and
        breaks the downward confidence spiral.

        No-op when alpha_gate_k_conf=0 (default) or when refresh_patience=0.
        Only triggers on accepted-measurement frames to avoid reiniting at a
        wrong location while coasting.
        """
        if self.refresh_patience <= 0:
            return
        if not getattr(step, "accepted_measurement", False):
            self._moderate_conf_streak = 0
            return
        if self.conf_refresh_low <= confidence <= self.conf_refresh_high:
            self._moderate_conf_streak += 1
            if self._moderate_conf_streak >= self.refresh_patience:
                self.ai.init(frame_rgb, np.array(step.bbox, dtype=np.float32))
                self._moderate_conf_streak = 0
        else:
            self._moderate_conf_streak = 0

    def _should_skip_ai(self) -> bool:
        """IMM model probabilities drive AI inference scheduling.

        Policy:
          Singer > 0.35 OR conf < 0.40  → NEVER skip (maneuver / uncertainty)
          CV > 0.70 AND conf > 0.65     → skip up to max_consecutive_skips (stable)
          CA > 0.50 AND conf > 0.50     → skip 1 frame (transitional)
          else                           → never skip

        Consecutive skip counter reset on any non-skip frame to prevent runaway drift.
        """
        if not self.ai_skip_when_confident:
            return False
        if self.kf is None or not self.kf.is_initialized():
            return False
        if self._consecutive_skips >= self.ai_max_consecutive_skips:
            self._consecutive_skips = 0
            return False

        mu = np.asarray(self.kf.get_model_probabilities(), dtype=np.float32).flatten()
        mu_cv    = float(mu[0])
        mu_ca    = float(mu[1])
        mu_singer= float(mu[2])
        prev_conf = self._prev_confidence

        # Singer dominance or low confidence → always run AI
        if mu_singer > self.singer_skip_thr or prev_conf < 0.40:
            self._consecutive_skips = 0
            return False

        # CV stable: allow consecutive skips up to ai_max_consecutive_skips
        if mu_cv > self.cv_stable_thr and prev_conf > self.cv_conf_min:
            self._consecutive_skips += 1
            return True

        # CA transitional: allow 1 skip max
        if mu_ca > 0.50 and prev_conf > 0.50 and self._consecutive_skips < 1:
            self._consecutive_skips += 1
            return True

        self._consecutive_skips = 0
        return False

    def run(self):
        """Run the tracking loop."""
        if not HAS_CV2:
            print("[ERROR] OpenCV not available. Install with: pip install opencv-python")
            return
        if not HAS_CPP:
            print("[ERROR] tracker_cpp not built. Run: cmake -B build && cmake --build build")
            return

        assert cv2 is not None
        assert tracker_cpp is not None
        assert self.kf is not None
        assert self.state_machine is not None

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {self.video_path}")
            return

        # Read first frame and initialize
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read first frame")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ai.init(frame_rgb, self.initial_bbox)
        self.kf.init(self.initial_bbox)
        self.state_machine.force_tracking()
        self._prev_frame_bgr = frame.copy() if self.gmc_enabled else None

        fps = 0.0
        frame_count = 0

        print(f"[INFO] Tracking started on: {self.video_path}")
        print(f"[INFO] Initial bbox: {self.initial_bbox}")
        print("[INFO] Press 'q' to quit")

        while True:
            t_total_start = time.perf_counter_ns()

            # 1. Read frame
            t0 = time.perf_counter_ns()
            ret, frame = cap.read()
            if not ret:
                break
            t_read = (time.perf_counter_ns() - t0) / 1e6
            self.timings["read"].append(t_read)

            # 2. GMC: estimate homography against prev frame, warp state or
            #    flag failure so next predict inflates Q.
            t0 = time.perf_counter_ns()
            if self.gmc_enabled and self._prev_frame_bgr is not None:
                assert self.gmc is not None
                kf_bbox = np.array(self.kf.get_state()).flatten()[:4].astype(np.float32)
                H, ok = self.gmc.estimate(self._prev_frame_bgr, frame, kf_bbox)
                if ok and hasattr(self.kf, "apply_gmc"):
                    self.kf.apply_gmc(H.astype(np.float32))
                elif hasattr(self.kf, "set_gmc_failed"):
                    self.kf.set_gmc_failed(True)
            t_gmc = (time.perf_counter_ns() - t0) / 1e6
            self.timings["gmc"].append(t_gmc)

            # 3. AI inference (skippable when previous frame was confident).
            t0 = time.perf_counter_ns()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ai_bbox = np.zeros(4, dtype=np.float32)
            confidence = 0.0
            matched_bbox = None
            observed_bbox = None

            # Always predict once at the start of each frame so the
            # KF state is at current time.  update() will skip its
            # internal predict because predicted_ is now True.
            predicted_state = np.array(self.kf.predict()).flatten()

            # Phase 4: IMM manoeuvre probability — used to lower bypass threshold
            # when CA/Singer mode becomes dominant (target manoeuvring).
            p_maneuver = 0.0
            mu_singer = 0.0
            if hasattr(self.kf, "get_model_probabilities"):
                _mu = np.array(self.kf.get_model_probabilities())
                p_maneuver = float(_mu[1] + _mu[2])
                mu_singer = float(_mu[2])

            skip_ai = self._should_skip_ai()

            if skip_ai:
                ai_bbox = predicted_state[:4].astype(np.float32)
                confidence = self._prev_confidence  # carry forward
                # _consecutive_skips already incremented by _should_skip_ai()
            elif self.association_enabled:
                observation = observe_with_guidance(
                    self.ai,
                    frame_rgb,
                    predicted_state,
                    mode="ai_lead",
                    last_output_bbox=predicted_state[:4],
                    singer_prob=mu_singer,
                    search_scale_boost=self.search_scale_boost,
                )
                if observation.bbox is not None:
                    matched_bbox = observation.bbox
                    ai_bbox = matched_bbox
                    confidence = observation.confidence
                    observed_bbox = matched_bbox
                self._consecutive_skips = 0
            else:
                observation = observe_with_guidance(
                    self.ai,
                    frame_rgb,
                    predicted_state,
                    mode="ai_lead",
                    last_output_bbox=predicted_state[:4],
                    singer_prob=mu_singer,
                    search_scale_boost=self.search_scale_boost,
                )
                if observation.bbox is not None:
                    ai_bbox = observation.bbox
                    confidence = observation.confidence
                    observed_bbox = ai_bbox
                self._consecutive_skips = 0
            t_ai = (time.perf_counter_ns() - t0) / 1e6
            self.timings["ai"].append(t_ai)

            # 3. Decision
            t0 = time.perf_counter_ns()
            track_state = self.state_machine.step(confidence)
            t_decision = (time.perf_counter_ns() - t0) / 1e6
            self.timings["decision"].append(t_decision)

            # 4. Kalman filter
            t0 = time.perf_counter_ns()
            judged_bbox = None if observed_bbox is None else np.asarray(observed_bbox, dtype=np.float32)
            coast_count = self.state_machine.coast_count() if self.state_machine is not None else 0
            step = step_guided_imm(
                self.kf,
                self.decision,
                predicted_state,
                judged_bbox,
                confidence,
                frame.shape[1],
                frame.shape[0],
                is_tracking=(track_state == tracker_cpp.TrackState.TRACKING),
                judge_reference_bbox=predicted_state[:4],
                adaptive_r_enabled=self.adaptive_r_enabled,
                r_exponent=self.r_exponent,
                reject_streak=self._reject_streak,
                last_good_bbox=self._last_good_bbox,
                mahal_chi2_gate=self.mahal_chi2_gate,
                r_pos_base=self.r_pos_base,
                r_size_base=self.r_size_base,
                maneuver_probability=p_maneuver,
                maneuver_threshold=self.maneuver_threshold,
                maneuver_bypass_boost=self.maneuver_bypass_boost,
                coast_count=coast_count,
                alpha_gate_k_conf=self.alpha_gate_k_conf,
                alpha_gate_lambda=self.alpha_gate_lambda,
                reacq_r_decay=self.reacq_r_decay,
            )
            self._reject_streak = step.reject_streak
            self._last_good_bbox = step.last_good_bbox
            state = step.state
            # Phase 1 — The Great Rescue: after deep loss (streak ≥ 5 during
            # coasting), reinit AI template at the KF-predicted location.
            # init() renews BOTH the search centre AND the template (unlike
            # set_state which only moves the centre, leaving stale template).
            if self._reject_streak >= 5 and step.should_coast:
                self.ai.init(frame_rgb, np.array(step.bbox, dtype=np.float32))
                self._moderate_conf_streak = 0
            else:
                self._maybe_refresh_template(frame_rgb, confidence, step)
            t_kf = (time.perf_counter_ns() - t0) / 1e6
            self.timings["kf"].append(t_kf)

            # 5. Visualize
            t0 = time.perf_counter_ns()
            state_name = track_state.name
            coast_count = self.state_machine.coast_count()

            t_total = (time.perf_counter_ns() - t_total_start) / 1e6
            fps = 1000.0 / t_total if t_total > 0 else 0.0

            if self.show:
                vis = visualize(frame, ai_bbox, state, confidence,
                                state_name, fps, coast_count)
                cv2.imshow("Tracker", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            t_viz = (time.perf_counter_ns() - t0) / 1e6
            self.timings["viz"].append(t_viz)

            frame_count += 1
            self._prev_confidence = confidence
            if self.gmc_enabled:
                self._prev_frame_bgr = frame.copy()

            # Budget check (every 100 frames)
            if frame_count % 100 == 0:
                total_ms = t_read + t_ai + t_decision + t_kf + t_viz
                status = "OK" if total_ms < 30 else "OVER"
                print(f"[PERF] Frame {frame_count}: "
                      f"total={total_ms:.1f}ms "
                      f"(read={t_read:.1f} ai={t_ai:.1f} kf={t_kf:.2f} viz={t_viz:.1f}) "
                      f"[{status}]")

        cap.release()
        if self.show:
            cv2.destroyAllWindows()

        self._print_summary(frame_count)

    def _print_summary(self, frame_count):
        """Print timing summary."""
        print(f"\n{'='*50}")
        print(f"Tracking complete: {frame_count} frames")
        print(f"{'='*50}")
        for name, times in self.timings.items():
            if times:
                arr = np.array(times)
                print(f"  {name:>10}: avg={arr.mean():.2f}ms  "
                      f"p95={np.percentile(arr, 95):.2f}ms  "
                      f"p99={np.percentile(arr, 99):.2f}ms")
        all_times = np.array([
            sum(x) for x in zip(*self.timings.values())
        ]) if all(self.timings.values()) else np.array([0])
        print(f"  {'TOTAL':>10}: avg={all_times.mean():.2f}ms  "
              f"p95={np.percentile(all_times, 95):.2f}ms  "
              f"p99={np.percentile(all_times, 99):.2f}ms")
        budget_ok = np.percentile(all_times, 99) < 30
        print(f"  30ms budget: {'PASS' if budget_ok else 'FAIL'}")
