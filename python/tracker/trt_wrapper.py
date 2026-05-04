# pyright: reportAttributeAccessIssue=false
"""TensorRT-accelerated SGLATrack inference wrapper.

Runs the neural network via TensorRT engine. Preprocessing and postprocessing
are done in NumPy — no PyTorch required at inference time.

Usage:
    wrapper = TRTTrackWrapper(engine_path="models/sglatrack_fp16.engine")
    wrapper.init(first_frame_rgb, initial_bbox)  # [x, y, w, h] top-left
    for frame in video:
        bbox, confidence = wrapper.track(frame_rgb)
"""

import os
import numpy as np  # pyright: ignore[reportMissingImports]
import cv2  # pyright: ignore[reportMissingImports]

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _hann1d(sz):
    """1D Hann window."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, sz + 1) / (sz + 1)))


def _hann2d(h, w):
    """2D Hann window, shape (1, 1, h, w)."""
    return (_hann1d(h).reshape(1, 1, -1, 1) * _hann1d(w).reshape(1, 1, 1, -1)).astype(np.float32)


def _sample_target(image, bbox, factor, output_sz):
    """Crop and resize a target-centered patch from the image.

    Args:
        image: (H, W, 3) uint8 numpy array
        bbox: [x, y, w, h] top-left format
        factor: context factor (how much area around bbox to include)
        output_sz: output patch size (square)

    Returns:
        patch: (output_sz, output_sz, 3) uint8
        resize_factor: scale factor applied
        att_mask: (output_sz, output_sz) attention mask (1=padded, 0=valid)
    """
    x, y, w, h = bbox
    w = max(float(w), 1.0)
    h = max(float(h), 1.0)
    cx, cy = x + w / 2, y + h / 2

    # Context region
    crop_sz = np.ceil(np.sqrt(w * h) * factor)
    crop_sz = max(crop_sz, 1)

    x1 = int(round(cx - crop_sz / 2))
    y1 = int(round(cy - crop_sz / 2))
    x2 = int(round(cx + crop_sz / 2))
    y2 = int(round(cy + crop_sz / 2))

    H, W = image.shape[:2]
    # Padding for out-of-bounds
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)

    crop = image[y1c:y2c, x1c:x2c]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")

    resize_factor = output_sz / crop.shape[0]

    patch = cv2.resize(crop, (output_sz, output_sz), interpolation=cv2.INTER_LINEAR)

    # Attention mask (1 where padded)
    att_mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
    row_end = att_mask.shape[0] - pad_bottom if pad_bottom > 0 else att_mask.shape[0]
    col_end = att_mask.shape[1] - pad_right if pad_right > 0 else att_mask.shape[1]
    att_mask[pad_top:row_end, pad_left:col_end] = 0
    att_mask = cv2.resize(att_mask, (output_sz, output_sz), interpolation=cv2.INTER_NEAREST)

    return patch, resize_factor, att_mask


def _preprocess(patch):
    """ImageNet normalize: (H,W,3) uint8 → (1,3,H,W) float32."""
    x = patch.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[np.newaxis]  # (1, 3, H, W)
    x = (x - _MEAN) / _STD
    return np.ascontiguousarray(x)


def _clip_box(bbox, H, W, margin=10):
    """Clip bbox to image bounds with margin."""
    x, y, w, h = bbox
    x = max(0, min(x, W - margin))
    y = max(0, min(y, H - margin))
    w = max(margin, min(w, W - x))
    h = max(margin, min(h, H - y))
    return [x, y, w, h]


class TRTTrackWrapper:
    """TensorRT-accelerated SGLATrack tracker.

    Same interface as SGLATrackWrapper but uses TensorRT engine.
    """

    def __init__(
        self,
        engine_path=None,
        association_enabled: bool = True,
        association_top_k: int = 5,
        association_iou_threshold: float = 0.3,
        association_score_weight: float = 0.0,
    ):
        if engine_path is None:
            engine_path = os.path.join(_PROJECT_ROOT, "models", "sglatrack_ep32_fp16.engine")
        self.engine_path = os.path.abspath(engine_path)
        self.initialized = False
        self._engine_loaded = False
        self.association_enabled = association_enabled
        self.association_top_k = association_top_k
        self.association_iou_threshold = association_iou_threshold
        self.association_score_weight = association_score_weight

        # SGLATrack DeiT-tiny config
        self.template_factor = 2.0
        self.template_size = 128
        self.search_factor = 4.0
        self.search_size = 256
        self.feat_sz = 16  # 256 / 16

        # Hann window for score weighting
        self.output_window = _hann2d(self.feat_sz, self.feat_sz)
        self._ema_patch_f32: np.ndarray | None = None

    def __del__(self):
        """Release TensorRT resources and CUDA memory."""
        if not getattr(self, "_engine_loaded", False):
            return
        try:
            del self._io
            del self._context
            del self._engine
            del self._stream
            if hasattr(self, "_out_bufs"):
                del self._out_bufs
            self._torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_engine(self):
        """Load TensorRT engine and allocate buffers using PyTorch CUDA."""
        import tensorrt as trt  # pyright: ignore[reportMissingImports]
        import torch  # pyright: ignore[reportMissingImports]

        self._torch = torch

        _TRT_TO_TORCH = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
        }
        _TRT_TO_NP = {
            trt.float32: np.float32,
            trt.float16: np.float16,
            trt.int32: np.int32,
            trt.int8: np.int8,
        }

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream()

        # Discover I/O tensors and allocate torch CUDA buffers
        self._io = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            trt_dtype = self._engine.get_tensor_dtype(name)
            torch_dtype = _TRT_TO_TORCH.get(trt_dtype, torch.float32)
            np_dtype = _TRT_TO_NP.get(trt_dtype, np.float32)

            mode = self._engine.get_tensor_mode(name)
            is_input = mode == trt.TensorIOMode.INPUT

            gpu_tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")

            self._io[name] = {
                "gpu": gpu_tensor,
                "shape": shape,
                "np_dtype": np_dtype,
                "is_input": is_input,
            }
            self._context.set_tensor_address(name, gpu_tensor.data_ptr())

        self._engine_loaded = True
        print(f"[INFO] TensorRT engine loaded: {self.engine_path}")
        for name, info in self._io.items():
            tag = "INPUT" if info["is_input"] else "OUTPUT"
            print(f"  {tag} {name}: {info['shape']} {info['np_dtype'].__name__}")

    def reset_context(self):
        """Recreate the TRT execution context to clear any residual engine state.

        TRT FP16 inference has mild non-determinism: the first pass through a
        sequence can produce slightly different score maps depending on GPU
        workspace initialization.  When ab_test.py runs raw THEN imm for the
        same sequence, the imm run sees a "primed" workspace state from the raw
        run, causing reproducible but biased results.
        Calling reset_context() between the two runs restores a clean context so
        both passes start from the same (uninitialized) workspace state.
        """
        if not self._engine_loaded:
            return
        del self._context
        self._context = self._engine.create_execution_context()
        for name, info in self._io.items():
            self._context.set_tensor_address(name, info["gpu"].data_ptr())
        if hasattr(self, "_out_bufs"):
            del self._out_bufs
        self.initialized = False  # require re-init before next track()

    def _infer(self, template_np, search_np):
        """Run TensorRT inference.

        Args:
            template_np: (1, 3, 128, 128) float32
            search_np: (1, 3, 256, 256) float32

        Returns:
            score_map: (1, 1, 16, 16)
            size_map: (1, 2, 16, 16)
            offset_map: (1, 2, 16, 16)
        """
        torch = self._torch

        # Zero-copy CPU tensor from numpy, then direct copy to pre-allocated GPU buffer
        # NEVER use .cuda() here — it creates a temporary GPU tensor each frame
        self._io["template"]["gpu"].copy_(torch.from_numpy(template_np), non_blocking=True)
        self._io["search"]["gpu"].copy_(torch.from_numpy(search_np), non_blocking=True)

        # Execute
        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()

        # Copy outputs into pre-allocated numpy buffers (lazy init)
        if not hasattr(self, "_out_bufs"):
            self._out_bufs = {}
            for name, info in self._io.items():
                if not info["is_input"]:
                    self._out_bufs[name] = np.empty(info["shape"], dtype=info["np_dtype"])

        for name, buf in self._out_bufs.items():
            buf[:] = self._io[name]["gpu"].cpu().numpy()

        return self._out_bufs["score_map"], self._out_bufs["size_map"], self._out_bufs["offset_map"]

    def init(self, frame, bbox):
        """Initialize tracker with first frame and bounding box.

        Args:
            frame: RGB image (H, W, 3) uint8 numpy array
            bbox: [x, y, w, h] top-left format
        """
        if not self._engine_loaded:
            self._load_engine()

        self.set_state(bbox)

        # Extract and preprocess template
        z_patch, _, _ = _sample_target(frame, self._state, self.template_factor, self.template_size)
        self._z_tensor = _preprocess(z_patch)
        self._ema_patch_f32 = None  # reset EMA buffer on fresh init

        self.initialized = True

    def set_state(self, bbox):
        """Synchronize the internal search state with the chosen bbox."""
        bbox_list = bbox.tolist() if isinstance(bbox, np.ndarray) else list(bbox)
        self._state = [float(x) for x in bbox_list]

    def update_template_ema(
        self, frame_rgb: np.ndarray, bbox, alpha: float = 0.05
    ) -> None:
        """Exponential moving average template update.

        Blends the current frame patch into the TRT z_tensor buffer with
        weight *alpha* (new frame weight).  Low alpha = slow/stable update.
        Called on accepted high-confidence frames so the template gradually
        adapts without a hard reinit.
        """
        if not self.initialized:
            return
        bbox_list = bbox.tolist() if isinstance(bbox, np.ndarray) else list(bbox)
        z_patch, _, _ = _sample_target(
            frame_rgb, bbox_list, self.template_factor, self.template_size
        )
        patch_f32 = z_patch.astype(np.float32)
        if self._ema_patch_f32 is None:
            self._ema_patch_f32 = patch_f32.copy()
        else:
            self._ema_patch_f32 = alpha * patch_f32 + (1.0 - alpha) * self._ema_patch_f32
        blended = np.clip(self._ema_patch_f32, 0.0, 255.0).astype(np.uint8)
        self._z_tensor = _preprocess(blended)

    def _run_search(self, frame):
        """Run one search pass without mutating the current state."""
        H, W = frame.shape[:2]

        x_patch, resize_factor, _ = _sample_target(
            frame, self._state, self.search_factor, self.search_size
        )
        x_tensor = _preprocess(x_patch)
        score_map, size_map, offset_map = self._infer(self._z_tensor, x_tensor)
        response = self.output_window * score_map
        return response, size_map, offset_map, resize_factor, H, W

    def _decode_bbox_from_index(self, idx, size_map, offset_map, resize_factor, H, W):
        """Decode one candidate bbox from a flattened response index."""
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        offset_x = float(offset_map[0, 0, idx_y, idx_x])
        offset_y = float(offset_map[0, 1, idx_y, idx_x])
        w_pred = float(size_map[0, 0, idx_y, idx_x])
        h_pred = float(size_map[0, 1, idx_y, idx_x])

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
        return np.array(_clip_box(new_bbox, H, W), dtype=np.float32)

    def track_candidates(self, frame, top_k: int | None = None):
        """Return the top-K candidate boxes and their scores for one frame."""
        if not self.initialized:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        if top_k is None:
            top_k = self.association_top_k
        if top_k <= 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        response, size_map, offset_map, resize_factor, H, W = self._run_search(frame)
        flat = response.reshape(-1)
        k = min(int(top_k), flat.size)

        if k == flat.size:
            top_indices = np.arange(flat.size, dtype=np.int64)
        else:
            top_indices = np.argpartition(flat, -k)[-k:]

        top_scores = flat[top_indices]
        order = np.argsort(top_scores)[::-1]
        top_indices = top_indices[order]
        top_scores = top_scores[order].astype(np.float32, copy=False)

        bboxes = np.empty((k, 4), dtype=np.float32)
        for out_idx, flat_idx in enumerate(top_indices.tolist()):
            bboxes[out_idx] = self._decode_bbox_from_index(
                int(flat_idx), size_map, offset_map, resize_factor, H, W
            )

        return bboxes, top_scores

    def track(self, frame):
        """Track target in frame.

        Args:
            frame: RGB image (H, W, 3) uint8 numpy array

        Returns:
            (bbox, confidence): bbox is [x, y, w, h] top-left, confidence 0-1
        """
        candidate_bboxes, candidate_scores = self.track_candidates(frame, top_k=1)
        if candidate_bboxes.shape[0] == 0:
            return np.zeros(4, dtype=np.float32), 0.0

        bbox = candidate_bboxes[0]
        confidence = float(candidate_scores[0])
        self.set_state(bbox)
        return bbox, confidence
