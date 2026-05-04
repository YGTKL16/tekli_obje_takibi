"""Tests for tracking pipeline (real C++ filters, only I/O mocked)."""

import numpy as np
import pytest
from tracker.gmc import GMCQualityReport
from tracker.imm_policy import IMMObservation, IMMPolicyStep
from unittest.mock import patch, MagicMock

import tracker_cpp  # pyright: ignore[reportMissingImports]
from tracker.decision import DecisionMaker


def _fake_cap(frames):
    """Build a mock cv2.VideoCapture that yields *frames* then stops."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    return cap


class TestPipeline:
    """Pipeline tests using real tracker_cpp.IMMFilter / TrackerState.

    Only cv2 (video I/O) and SGLATrackWrapper (TRT inference) are mocked —
    those require hardware that isn't available in CI.
    """

    @patch("tracker.pipeline.HAS_CV2", True)
    def test_pipeline_init_components(self):
        from tracker.pipeline import Pipeline
        bbox = np.array([100, 200, 50, 60], dtype=np.float32)

        with patch("tracker.pipeline.SGLATrackWrapper"):
            p = Pipeline(
                video_path="/fake/video.mp4",
                initial_bbox=bbox,
                conf_threshold=0.3,
                iou_threshold=0.2,
                max_coast_frames=60,
                show=False,
            )
            assert isinstance(p.kf, tracker_cpp.IMMFilter)
            assert isinstance(p.state_machine, tracker_cpp.TrackerState)
            assert isinstance(p.decision, DecisionMaker)

    @patch("tracker.pipeline.HAS_CPP", False)
    @patch("tracker.pipeline.HAS_CV2", True)
    def test_pipeline_without_cpp(self):
        from tracker.pipeline import Pipeline
        bbox = np.array([100, 200, 50, 60], dtype=np.float32)

        with patch("tracker.pipeline.SGLATrackWrapper"):
            p = Pipeline(video_path="/fake/video.mp4", initial_bbox=bbox, show=False)
            assert p.kf is None
            assert p.state_machine is None

    @patch("tracker.pipeline.HAS_CV2", True)
    def test_pipeline_uses_default_checkpoint_when_model_path_missing(self):
        from tracker.pipeline import DEFAULT_CHECKPOINT_PATH, Pipeline
        bbox = np.array([100, 200, 50, 60], dtype=np.float32)

        with patch("tracker.pipeline.SGLATrackWrapper") as mock_wrapper:
            Pipeline(video_path="/fake/video.mp4", initial_bbox=bbox, show=False)

        assert mock_wrapper.call_args is not None
        assert mock_wrapper.call_args.args[0] == DEFAULT_CHECKPOINT_PATH

    @patch("tracker.pipeline.HAS_CV2", False)
    def test_run_without_cv2_exits_gracefully(self, capsys):
        from tracker.pipeline import Pipeline
        bbox = np.array([100, 200, 50, 60], dtype=np.float32)

        with patch("tracker.pipeline.SGLATrackWrapper"):
            p = Pipeline(video_path="/fake/video.mp4", initial_bbox=bbox, show=False)
            p.run()
            captured = capsys.readouterr()
            assert "OpenCV" in captured.out or "not available" in captured.out

    @patch("tracker.pipeline.HAS_CV2", True)
    @patch("tracker.pipeline.cv2")
    def test_run_no_video_exits_gracefully(self, mock_cv2, capsys):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        from tracker.pipeline import Pipeline
        bbox = np.array([100, 200, 50, 60], dtype=np.float32)

        with patch("tracker.pipeline.SGLATrackWrapper"):
            p = Pipeline(video_path="/nonexistent.mp4", initial_bbox=bbox, show=False)
            p.run()
            captured = capsys.readouterr()
            assert "Cannot open" in captured.out or "ERROR" in captured.out

    @patch("tracker.pipeline.HAS_CV2", True)
    @patch("tracker.pipeline.cv2")
    def test_association_updates_with_matched_candidate(self, mock_cv2):
        """Good association match → real IMMFilter update pulls state toward detection."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.VideoCapture.return_value = _fake_cap([frame, frame])
        mock_cv2.cvtColor.return_value = frame
        mock_cv2.COLOR_BGR2RGB = 4

        init_bbox = np.array([100, 200, 50, 60], dtype=np.float32)
        matched_det = np.array([102, 202, 51, 61], dtype=np.float32)

        ai = MagicMock()

        from tracker.pipeline import Pipeline

        with patch("tracker.pipeline.SGLATrackWrapper", return_value=ai), \
             patch(
                 "tracker.pipeline.observe_with_guidance",
                 return_value=IMMObservation(
                     bbox=matched_det,
                     confidence=0.9,
                     search_bbox=init_bbox.tolist(),
                     candidate_bboxes=np.array([[400, 400, 20, 20], matched_det], dtype=np.float32),
                     candidate_scores=np.array([0.5, 0.9], dtype=np.float32),
                 ),
             ):
            p = Pipeline(
                video_path="/fake/video.mp4",
                initial_bbox=init_bbox,
                conf_threshold=0.1,
                iou_threshold=0.05,
                show=False,
            )
            p.run()

        # F5 closed-loop feedback: set_state is called every frame to feed
        # the KF-blended bbox back to the SGLATrack search window.
        ai.set_state.assert_called()

    @patch("tracker.pipeline.HAS_CV2", True)
    @patch("tracker.pipeline.cv2")
    def test_association_coasts_when_no_match(self, mock_cv2):
        """No association match → real IMMFilter predict-only keeps bbox near init."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.VideoCapture.return_value = _fake_cap([frame, frame])
        mock_cv2.cvtColor.return_value = frame
        mock_cv2.COLOR_BGR2RGB = 4

        init_bbox = np.array([100, 200, 50, 60], dtype=np.float32)

        ai = MagicMock()

        from tracker.pipeline import Pipeline

        with patch("tracker.pipeline.SGLATrackWrapper", return_value=ai), \
             patch(
                 "tracker.pipeline.observe_with_guidance",
                 return_value=IMMObservation(
                     bbox=None,
                     confidence=0.0,
                     search_bbox=init_bbox.tolist(),
                     candidate_bboxes=np.array([[400, 400, 20, 20]], dtype=np.float32),
                     candidate_scores=np.array([0.95], dtype=np.float32),
                 ),
             ):
            p = Pipeline(
                video_path="/fake/video.mp4",
                initial_bbox=init_bbox,
                conf_threshold=0.1,
                iou_threshold=0.05,
                show=False,
            )
            p.run()

        # F5 closed-loop feedback: set_state is called every frame.
        ai.set_state.assert_called()

    @patch("tracker.pipeline.HAS_CV2", True)
    @patch("tracker.pipeline.cv2")
    def test_reinit_after_deep_loss(self, mock_cv2):
        """6+ coasting frames → reject_streak >= 5 + should_coast → Great Rescue fires ai.init."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 7 frames: init + 6 tracking frames all with zero confidence.
        # Frame 5 → streak=5 + should_coast=True → rescue triggers ai.init.
        mock_cv2.VideoCapture.return_value = _fake_cap([frame] * 7)
        mock_cv2.cvtColor.return_value = frame
        mock_cv2.COLOR_BGR2RGB = 4

        init_bbox = np.array([100, 200, 50, 60], dtype=np.float32)

        ai = MagicMock()

        from tracker.pipeline import Pipeline

        with patch("tracker.pipeline.SGLATrackWrapper", return_value=ai), \
             patch(
                 "tracker.pipeline.observe_with_guidance",
                 return_value=IMMObservation(
                     bbox=None,
                     confidence=0.0,
                     search_bbox=init_bbox.tolist(),
                     candidate_bboxes=np.array([[400, 400, 20, 20]], dtype=np.float32),
                     candidate_scores=np.array([0.95], dtype=np.float32),
                 ),
             ):
            p = Pipeline(
                video_path="/fake/video.mp4",
                initial_bbox=init_bbox,
                conf_threshold=0.1,
                iou_threshold=0.05,
                show=False,
            )
            p.run()

        # Phase 1 — Great Rescue: after streak>=5 + should_coast, ai.init is called
        # (at least once for the rescue, plus the initial init at frame 0).
        assert ai.init.call_count >= 2
        # F5 closed-loop feedback: set_state is called every frame (even during coasting).
        ai.set_state.assert_called()

    @patch("tracker.pipeline.HAS_CV2", True)
    @patch("tracker.pipeline.cv2")
    def test_good_gmc_applies_warp(self, mock_cv2):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.VideoCapture.return_value = _fake_cap([frame, frame])
        mock_cv2.cvtColor.side_effect = lambda img, code: img
        mock_cv2.COLOR_BGR2RGB = 4

        init_bbox = np.array([100, 200, 50, 60], dtype=np.float32)
        ai = MagicMock()
        gmc = MagicMock()
        gmc.estimate_with_quality.return_value = (
            np.eye(3, dtype=np.float64),
            GMCQualityReport(
                match_count=40,
                inlier_count=30,
                inlier_ratio=0.75,
                quality_state="good",
                raw_ok=True,
            ),
        )

        dummy_step = IMMPolicyStep(
            bbox=init_bbox.tolist(),
            state=np.zeros(10, dtype=np.float32),
            predicted_state=np.zeros(10, dtype=np.float32),
            accepted_measurement=False,
            measurement_sane=False,
            should_coast=False,
            reject_streak=0,
            last_good_bbox=init_bbox.tolist(),
        )

        from tracker.pipeline import Pipeline

        with patch("tracker.pipeline.SGLATrackWrapper", return_value=ai), \
             patch("tracker.pipeline.GMCEstimator", return_value=gmc), \
             patch(
                 "tracker.pipeline.observe_with_guidance",
                 return_value=IMMObservation(
                     bbox=None,
                     confidence=0.0,
                     search_bbox=init_bbox.tolist(),
                     candidate_bboxes=np.zeros((0, 4), dtype=np.float32),
                     candidate_scores=np.zeros((0,), dtype=np.float32),
                 ),
             ), \
             patch("tracker.pipeline.step_guided_imm", return_value=dummy_step):
            p = Pipeline(
                video_path="/fake/video.mp4",
                initial_bbox=init_bbox,
                gmc_enabled=True,
                show=False,
            )
            p.kf = MagicMock()
            p.kf.get_state.return_value = np.zeros(10, dtype=np.float32)
            p.kf.predict.return_value = np.zeros(10, dtype=np.float32)
            p.kf.get_model_probabilities.return_value = np.array([0.2, 0.3, 0.5], dtype=np.float32)
            p.state_machine = MagicMock()
            p.state_machine.step.return_value = tracker_cpp.TrackState.TRACKING
            p.state_machine.coast_count.return_value = 0
            p.run()

        p.kf.apply_gmc.assert_called_once()
        p.kf.set_gmc_failed.assert_not_called()

    @patch("tracker.pipeline.HAS_CV2", True)
    @patch("tracker.pipeline.cv2")
    def test_vetoed_gmc_suppresses_maneuver_and_freezes_next_frame(self, mock_cv2):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.VideoCapture.return_value = _fake_cap([frame, frame, frame])
        mock_cv2.cvtColor.side_effect = lambda img, code: img
        mock_cv2.COLOR_BGR2RGB = 4

        init_bbox = np.array([100, 200, 50, 60], dtype=np.float32)
        ai = MagicMock()
        gmc = MagicMock()
        gmc.estimate_with_quality.side_effect = [
            (
                np.eye(3, dtype=np.float64),
                GMCQualityReport(
                    match_count=12,
                    inlier_count=3,
                    inlier_ratio=0.25,
                    quality_state="veto",
                    raw_ok=False,
                    reason="history_veto",
                ),
            ),
            (
                np.eye(3, dtype=np.float64),
                GMCQualityReport(
                    match_count=40,
                    inlier_count=30,
                    inlier_ratio=0.75,
                    quality_state="good",
                    raw_ok=True,
                ),
            ),
        ]

        dummy_step = IMMPolicyStep(
            bbox=init_bbox.tolist(),
            state=np.zeros(10, dtype=np.float32),
            predicted_state=np.zeros(10, dtype=np.float32),
            accepted_measurement=False,
            measurement_sane=False,
            should_coast=False,
            reject_streak=0,
            last_good_bbox=init_bbox.tolist(),
        )

        from tracker.pipeline import Pipeline

        with patch("tracker.pipeline.SGLATrackWrapper", return_value=ai), \
             patch("tracker.pipeline.GMCEstimator", return_value=gmc), \
             patch(
                 "tracker.pipeline.observe_with_guidance",
                 return_value=IMMObservation(
                     bbox=None,
                     confidence=0.0,
                     search_bbox=init_bbox.tolist(),
                     candidate_bboxes=np.zeros((0, 4), dtype=np.float32),
                     candidate_scores=np.zeros((0,), dtype=np.float32),
                 ),
             ), \
             patch("tracker.pipeline.step_guided_imm", return_value=dummy_step) as step_mock:
            p = Pipeline(
                video_path="/fake/video.mp4",
                initial_bbox=init_bbox,
                gmc_enabled=True,
                maneuver_pi_enabled=True,
                maneuver_bypass_boost=0.10,
                gmc_freeze_frames_after_veto=1,
                show=False,
            )
            p.kf = MagicMock()
            p.kf.get_state.return_value = np.zeros(10, dtype=np.float32)
            p.kf.predict.return_value = np.zeros(10, dtype=np.float32)
            p.kf.get_model_probabilities.return_value = np.array([0.1, 0.4, 0.5], dtype=np.float32)
            p.state_machine = MagicMock()
            p.state_machine.step.return_value = tracker_cpp.TrackState.TRACKING
            p.state_machine.coast_count.return_value = 0
            p.run()

        assert p.kf.apply_gmc.call_count == 1
        assert p.kf.set_gmc_failed.call_count == 1
        assert step_mock.call_count == 2
        first_kwargs = step_mock.call_args_list[0].kwargs
        second_kwargs = step_mock.call_args_list[1].kwargs
        assert first_kwargs["maneuver_probability"] == 0.0
        assert first_kwargs["maneuver_bypass_boost"] == 0.0
        assert first_kwargs["maneuver_pi_enabled"] is False
        assert second_kwargs["maneuver_probability"] == 0.0
        assert second_kwargs["maneuver_bypass_boost"] == 0.0
        assert second_kwargs["maneuver_pi_enabled"] is False
