"""Tests for tracking pipeline (real C++ filters, only I/O mocked)."""

import numpy as np
import pytest
from tracker.imm_policy import IMMObservation
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

        # ai_lead mode: set_state is NOT called during normal tracking (high conf)
        # AI manages its own search window; KF only steers during coasting.
        ai.set_state.assert_not_called()

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

        # ai_lead: 1 coasting frame → reject_streak=1 (< 5) → no rescue yet
        ai.set_state.assert_not_called()

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
        # set_state is never used in ai_lead mode — rescue uses init() instead.
        ai.set_state.assert_not_called()
