import os
import sys

import numpy as np

# Project root (tracker/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SGLA_ROOT = os.path.join(_PROJECT_ROOT, "models", "SGLATrack")
DEFAULT_CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT, "models", "SGLATrack", "checkpoints", "sglatrack_ep0297.pth.tar"
)


class SGLATrackWrapper:
    """Wraps SGLATrack model for single-object tracking.

    Usage:
        wrapper = SGLATrackWrapper(checkpoint_path="models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar")
        wrapper.init(first_frame_rgb, initial_bbox)  # [x, y, w, h] top-left
        for frame in video:
            bbox, confidence = wrapper.track(frame_rgb)
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        association_enabled: bool = True,
        association_top_k: int = 5,
        association_iou_threshold: float = 0.3,
        association_score_weight: float = 0.0,
    ):
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT_PATH
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.network = None
        self.initialized = False
        self._device = None
        self._cfg = None
        self._torch = None
        self._sample_target = None
        self._clip_box = None
        self._state = None
        self._z_dict = None
        self._box_mask_z = None
        self.preprocessor = None
        self.template_factor = None
        self.template_size = None
        self.search_factor = None
        self.search_size = None
        self.feat_sz = None
        self.output_window = None
        self._use_ce = False
        self.association_enabled = association_enabled
        self.association_top_k = association_top_k
        self.association_iou_threshold = association_iou_threshold
        self.association_score_weight = association_score_weight

    def _load_model(self):
        """Load SGLATrack model (deferred until first use)."""
        import torch

        # Add SGLATrack to path
        if _SGLA_ROOT not in sys.path:
            sys.path.insert(0, _SGLA_ROOT)

        from lib.config.sglatrack.config import cfg, update_config_from_file  # pyright: ignore[reportMissingImports]
        from lib.models.sglatrack import build_sglatrack  # pyright: ignore[reportMissingImports]
        from lib.test.tracker.data_utils import Preprocessor  # pyright: ignore[reportMissingImports]
        from lib.train.data.processing_utils import sample_target  # pyright: ignore[reportMissingImports]
        from lib.test.utils.hann import hann2d  # pyright: ignore[reportMissingImports]
        from lib.utils.box_ops import clip_box  # pyright: ignore[reportMissingImports]

        # Store imports for later use
        self._sample_target = sample_target
        self._clip_box = clip_box
        self._torch = torch

        # Load config
        yaml_file = os.path.join(_SGLA_ROOT, "experiments", "sglatrack", "deit_distilled.yaml")
        update_config_from_file(yaml_file)
        self._cfg = cfg

        # Build model
        network = build_sglatrack(cfg, training=False)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        network.load_state_dict(ckpt["net"], strict=True)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network.to(self._device).eval()

        # Preprocessor
        self.preprocessor = Preprocessor()

        # Config values
        self.template_factor = cfg.TEST.TEMPLATE_FACTOR
        self.template_size = cfg.TEST.TEMPLATE_SIZE
        self.search_factor = cfg.TEST.SEARCH_FACTOR
        self.search_size = cfg.TEST.SEARCH_SIZE

        self.feat_sz = self.search_size // cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True
        ).to(self._device)

        # CE masking (empty list for DeiT config = no masking)
        self._use_ce = bool(cfg.MODEL.BACKBONE.CE_LOC)

        print(f"[INFO] SGLATrack loaded: {self.checkpoint_path}")
        print(f"[INFO] Device: {self._device}, Search: {self.search_size}, Template: {self.template_size}")

    def init(self, frame: np.ndarray, bbox: np.ndarray):
        """Initialize tracker with first frame and bounding box.

        Args:
            frame: RGB image (H, W, 3) uint8 numpy array
            bbox: [x, y, w, h] top-left format
        """
        if self.network is None:
            self._load_model()

        self.set_state(bbox)

        # Extract and process template
        z_patch, resize_factor, z_amask = self._sample_target(
            frame, self._state, self.template_factor, output_sz=self.template_size
        )
        with self._torch.no_grad():
            self._z_dict = self.preprocessor.process(z_patch, z_amask)

        # CE template mask
        self._box_mask_z = None
        if self._use_ce:
            from lib.utils.ce_utils import generate_mask_cond  # pyright: ignore[reportMissingImports]
            from lib.train.data.processing_utils import transform_image_to_crop  # pyright: ignore[reportMissingImports]

            crop_sz = self._torch.Tensor([self.template_size, self.template_size])
            bbox_tensor = self._torch.tensor(self._state)
            template_bbox = transform_image_to_crop(
                bbox_tensor, bbox_tensor, resize_factor, crop_sz, normalize=True
            )
            template_bbox = template_bbox.view(1, 1, 4).to(self._device)
            self._box_mask_z = generate_mask_cond(self._cfg, 1, self._device, template_bbox)

        self.initialized = True

    def set_state(self, bbox):
        """Synchronize the internal search state with the chosen bbox."""
        bbox_list = bbox.tolist() if isinstance(bbox, np.ndarray) else list(bbox)
        self._state = [float(x) for x in bbox_list]

    def _run_search(self, frame: np.ndarray):
        """Run one search pass without mutating the current state."""
        H, W, _ = frame.shape
        x_patch, resize_factor, x_amask = self._sample_target(
            frame, self._state, self.search_factor, output_sz=self.search_size
        )

        with self._torch.no_grad():
            x_dict = self.preprocessor.process(x_patch, x_amask)
            out_dict = self.network(
                template=self._z_dict.tensors,
                search=x_dict.tensors,
                ce_template_mask=self._box_mask_z,
            )

        response = self.output_window * out_dict["score_map"]
        return response, out_dict["size_map"], out_dict["offset_map"], resize_factor, H, W

    def _decode_bbox_from_index(
        self,
        idx: int,
        size_map: np.ndarray,
        offset_map: np.ndarray,
        resize_factor: float,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Decode one candidate bbox from a flattened response index."""
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        offset_x = float(offset_map[0, idx_y, idx_x])
        offset_y = float(offset_map[1, idx_y, idx_x])
        w_pred = float(size_map[0, idx_y, idx_x])
        h_pred = float(size_map[1, idx_y, idx_x])

        cx_norm = (idx_x + offset_x) / self.feat_sz
        cy_norm = (idx_y + offset_y) / self.feat_sz
        cx_patch = cx_norm * self.search_size
        cy_patch = cy_norm * self.search_size
        w_px = w_pred * self.search_size
        h_px = h_pred * self.search_size

        cx_patch /= resize_factor
        cy_patch /= resize_factor
        w_px /= resize_factor
        h_px /= resize_factor

        cx_prev = self._state[0] + 0.5 * self._state[2]
        cy_prev = self._state[1] + 0.5 * self._state[3]
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx_patch + (cx_prev - half_side)
        cy_real = cy_patch + (cy_prev - half_side)
        new_bbox = [cx_real - 0.5 * w_px, cy_real - 0.5 * h_px, w_px, h_px]
        return np.array(self._clip_box(new_bbox, H, W, margin=10), dtype=np.float32)

    def track_candidates(self, frame: np.ndarray, top_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-K candidate boxes and their scores for one frame."""
        if not self.initialized:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        if top_k is None:
            top_k = self.association_top_k
        if top_k <= 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        response, size_map, offset_map, resize_factor, H, W = self._run_search(frame)
        flat = response.reshape(-1)
        flat_size = int(flat.numel()) if hasattr(flat, "numel") else int(np.asarray(flat).size)
        k = min(int(top_k), flat_size)
        scores, indices = self._torch.topk(flat, k=k)

        candidate_scores = scores.detach().cpu().to(self._torch.float32).numpy()
        candidate_indices = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        size_map_np = size_map[0].detach().cpu().numpy()
        offset_map_np = offset_map[0].detach().cpu().numpy()

        bboxes = np.empty((k, 4), dtype=np.float32)
        for out_idx, flat_idx in enumerate(candidate_indices.tolist()):
            bboxes[out_idx] = self._decode_bbox_from_index(
                int(flat_idx), size_map_np, offset_map_np, resize_factor, H, W
            )

        return bboxes, candidate_scores

    def track(self, frame: np.ndarray) -> tuple:
        """Track target in frame.

        Args:
            frame: RGB image (H, W, 3) uint8 numpy array

        Returns:
            (bbox, confidence): bbox is [x, y, w, h] top-left format, confidence 0-1
        """
        candidate_bboxes, candidate_scores = self.track_candidates(frame, top_k=1)
        if candidate_bboxes.shape[0] == 0:
            return np.zeros(4, dtype=np.float32), 0.0

        bbox = candidate_bboxes[0]
        confidence = float(candidate_scores[0])
        self.set_state(bbox)
        return bbox, confidence
