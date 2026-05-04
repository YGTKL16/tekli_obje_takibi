"""MixFormerV2 wrapper with the same runtime contract as SGLATrackWrapper."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import os
import sys
import types
from typing import Any, Iterator

import numpy as np


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MIXFORMER_ROOT = os.path.join(_PROJECT_ROOT, "models", "MixFormerV2")

DEFAULT_REPO_ROOT = _MIXFORMER_ROOT
DEFAULT_CONFIG_PATH = os.path.join(
    _MIXFORMER_ROOT,
    "experiments",
    "mixformer2_vit_online",
    "224_depth4_mlp1_score.yaml",
)
DEFAULT_CHECKPOINT_PATH = os.path.join(
    _MIXFORMER_ROOT,
    "models",
    "mixformerv2_small.pth.tar",
)
DEFAULT_MAE_BASE_PATH = os.path.join(
    _MIXFORMER_ROOT,
    "models",
    "mae_pretrain_vit_base.pth",
)


def _resolve_project_path(path: str | None, default: str) -> str:
    chosen = default if path is None else path
    if os.path.isabs(chosen):
        return os.path.abspath(chosen)
    return os.path.abspath(os.path.join(_PROJECT_ROOT, chosen))


def _is_lib_module(name: str) -> bool:
    return name == "lib" or name.startswith("lib.")


@contextmanager
def _isolated_lib_import(repo_root: str) -> Iterator[None]:
    """Import an upstream tracker whose package root is named ``lib``.

    Both SGLATrack and MixFormerV2 use absolute imports under a top-level
    ``lib`` package.  Keeping both in ``sys.modules`` at the same time makes
    whichever tracker imported first win.  This context gives MixFormerV2 a
    clean ``lib`` namespace for construction, then restores the caller's
    previous namespace so SGLATrack can keep running.
    """
    saved_modules = {
        name: module for name, module in sys.modules.items() if _is_lib_module(name)
    }
    old_path = list(sys.path)
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, repo_root)
    try:
        yield
    finally:
        for name in [name for name in sys.modules if _is_lib_module(name)]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = old_path


def _load_processing_utils(repo_root: str):
    module_name = "_mixformerv2_processing_utils"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached.sample_target

    module_path = os.path.join(repo_root, "lib", "train", "data", "processing_utils.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"MixFormerV2 processing_utils.py not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MixFormerV2 processing utils from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.sample_target


def _checkpoint_safe_globals() -> list[type]:
    """Allow the authors' training metadata while loading tensor-only weights."""
    for package_name in ("lib.train", "lib.train.admin"):
        sys.modules.setdefault(package_name, types.ModuleType(package_name))

    safe_classes: list[type] = []
    module_classes = {
        "lib.train.admin.stats": ("AverageMeter", "StatValue"),
        "lib.train.admin.settings": ("Settings",),
        "lib.train.admin.local": ("EnvironmentSettings",),
    }
    for module_name, class_names in module_classes.items():
        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is None:
                cls = type(class_name, (), {"__module__": module_name})
                setattr(module, class_name, cls)
            safe_classes.append(cls)
    return safe_classes


class MixFormerV2Wrapper:
    """Wrap MixFormerV2-Small as ``init(frame, bbox)`` / ``track(frame)``.

    The public shape mirrors ``SGLATrackWrapper``:
      - ``init(frame_rgb, bbox_xywh)`` initializes template and tracker state.
      - ``track(frame_rgb)`` returns ``(bbox_xywh_float32, confidence_float)``.
      - ``track_candidates(frame_rgb, top_k=1)`` returns one candidate for
        association-compatible callers.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        repo_root: str | None = None,
        device: str | None = None,
        search_factor: float | None = None,
        online_size: int | None = None,
        update_interval: int | None = None,
        max_score_decay: float = 1.0,
    ):
        self.repo_root = _resolve_project_path(repo_root, DEFAULT_REPO_ROOT)
        self.checkpoint_path = _resolve_project_path(checkpoint_path, DEFAULT_CHECKPOINT_PATH)
        self.config_path = _resolve_project_path(config_path, DEFAULT_CONFIG_PATH)
        self.device_name = device
        self.search_factor_override = search_factor
        self.online_size_override = online_size
        self.update_interval_override = update_interval
        self.max_score_decay = float(max_score_decay)

        self.network = None
        self.initialized = False
        self._torch: Any = None
        self._device = None
        self._cfg = None
        self._sample_target = None
        self._clip_box = None
        self._mean = None
        self._std = None

        self.template = None
        self.online_template = None
        self.online_max_template = None
        self.online_forget_id = 0
        self.max_pred_score = -1.0
        self.frame_id = 0
        self._state: list[float] | None = None

        self.template_factor = None
        self.template_size = None
        self.search_factor = None
        self.search_size = None
        self.online_size = 1
        self.update_interval = 1

        self.association_enabled = False
        self.association_top_k = 1
        self.association_iou_threshold = 0.0
        self.association_score_weight = 0.0

    def _load_model(self) -> None:
        if not os.path.isdir(self.repo_root):
            raise FileNotFoundError(f"MixFormerV2 repo not found: {self.repo_root}")
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"MixFormerV2 config not found: {self.config_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                "MixFormerV2-Small checkpoint not found: "
                f"{self.checkpoint_path}. Expected the authors' "
                "mixformerv2_small.pth.tar under models/MixFormerV2/models/."
            )

        import torch

        self._torch = torch
        device_name = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_name)
        self._sample_target = _load_processing_utils(self.repo_root)

        try:
            with _isolated_lib_import(self.repo_root):
                from lib.config.mixformer2_vit_online.config import (  # pyright: ignore[reportMissingImports]
                    update_new_config_from_file,
                )
                from lib.models.mixformer2_vit import build_mixformer2_vit_online  # pyright: ignore[reportMissingImports]
                from lib.utils.box_ops import clip_box  # pyright: ignore[reportMissingImports]

                cfg = update_new_config_from_file(self.config_path)
                network = build_mixformer2_vit_online(cfg, train=False)
                safe_globals = _checkpoint_safe_globals()
                with torch.serialization.safe_globals(safe_globals):
                    ckpt = torch.load(
                        self.checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
                network.load_state_dict(state_dict, strict=True)
        except ModuleNotFoundError as exc:
            raise ImportError(
                "MixFormerV2 dependencies are missing. Install the tracker "
                "runtime requirements, including timm, easydict, and einops."
            ) from exc

        self._cfg = cfg
        self._clip_box = clip_box
        self.network = network.to(self._device).eval()
        if hasattr(self.network, "box_head") and hasattr(self.network.box_head, "indice"):
            self.network.box_head.indice = self.network.box_head.indice.to(self._device)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)

        self.template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
        self.template_size = int(cfg.TEST.TEMPLATE_SIZE)
        self.search_factor = (
            float(self.search_factor_override)
            if self.search_factor_override is not None
            else float(cfg.TEST.SEARCH_FACTOR)
        )
        self.search_size = int(cfg.TEST.SEARCH_SIZE)

        update_intervals = getattr(cfg.TEST.UPDATE_INTERVALS, "GOT10K_TEST", [1])
        online_sizes = getattr(cfg.TEST.ONLINE_SIZES, "GOT10K_TEST", [1])
        self.update_interval = int(
            self.update_interval_override
            if self.update_interval_override is not None
            else update_intervals[0]
        )
        self.online_size = int(
            self.online_size_override
            if self.online_size_override is not None
            else online_sizes[0]
        )

        print(f"[INFO] MixFormerV2 loaded: {self.checkpoint_path}")
        print(f"[INFO] Device: {self._device}, Search: {self.search_size}, Template: {self.template_size}")

    def _process_patch(self, img_arr: np.ndarray):
        img_arr = np.ascontiguousarray(img_arr)
        img_tensor = self._torch.from_numpy(img_arr).to(self._device).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        return ((img_tensor / 255.0) - self._mean) / self._std

    def init(self, frame: np.ndarray, bbox: np.ndarray) -> None:
        if self.network is None:
            self._load_model()

        self.set_state(bbox)
        z_patch, _, _ = self._sample_target(
            frame,
            self._state,
            self.template_factor,
            output_sz=self.template_size,
        )
        self.template = self._process_patch(z_patch)
        self.online_template = self.template
        self.online_max_template = self.template
        self.online_forget_id = 0
        self.max_pred_score = -1.0
        self.frame_id = 0

        if self.online_size > 1:
            with self._torch.no_grad():
                self.network.set_online(self.template, self.online_template)

        self.initialized = True

    def set_state(self, bbox) -> None:
        bbox_list = bbox.tolist() if isinstance(bbox, np.ndarray) else list(bbox)
        self._state = [float(x) for x in bbox_list]

    def _map_box_back(self, pred_box: list[float], resize_factor: float) -> list[float]:
        assert self._state is not None
        cx_prev = self._state[0] + 0.5 * self._state[2]
        cy_prev = self._state[1] + 0.5 * self._state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def _infer(self, frame: np.ndarray, *, update_state: bool) -> tuple[np.ndarray, float]:
        if not self.initialized or self._state is None:
            return np.zeros(4, dtype=np.float32), 0.0

        H, W, _ = frame.shape
        x_patch, resize_factor, _ = self._sample_target(
            frame,
            self._state,
            self.search_factor,
            output_sz=self.search_size,
        )
        search = self._process_patch(x_patch)

        with self._torch.no_grad():
            out_dict = self.network(
                self.template,
                self.online_template,
                search,
                softmax=True,
                run_score_head=True,
            )

        pred_boxes = out_dict["pred_boxes"].view(-1, 4)
        pred_box = (
            pred_boxes.mean(dim=0) * float(self.search_size) / float(resize_factor)
        ).detach().cpu().tolist()
        bbox = np.array(
            self._clip_box(self._map_box_back(pred_box, resize_factor), H, W, margin=10),
            dtype=np.float32,
        )
        score_tensor = out_dict.get("pred_scores")
        confidence = (
            float(score_tensor.view(1).sigmoid().item())
            if score_tensor is not None
            else 1.0
        )

        if update_state:
            self.set_state(bbox)
            self._update_online_template(frame, confidence)

        return bbox, confidence

    def _update_online_template(self, frame: np.ndarray, confidence: float) -> None:
        self.frame_id += 1
        # F2: floor at 0.5 (matches hard gate below) to prevent decay underflow
        # silently freezing all template updates after ~hundreds of frames.
        self.max_pred_score = max(self.max_pred_score * self.max_score_decay, 0.5)
        if confidence > 0.5 and confidence > self.max_pred_score:
            z_patch, _, _ = self._sample_target(
                frame,
                self._state,
                self.template_factor,
                output_sz=self.template_size,
            )
            self.online_max_template = self._process_patch(z_patch)
            self.max_pred_score = confidence

        if self.update_interval <= 0 or self.frame_id % self.update_interval != 0:
            return

        if self.online_size <= 1:
            self.online_template = self.online_max_template
        elif self.online_template.shape[0] < self.online_size:
            self.online_template = self._torch.cat([self.online_template, self.online_max_template])
        else:
            self.online_template[self.online_forget_id:self.online_forget_id + 1] = (
                self.online_max_template
            )
            self.online_forget_id = (self.online_forget_id + 1) % self.online_size

        if self.online_size > 1:
            with self._torch.no_grad():
                self.network.set_online(self.template, self.online_template)

        self.max_pred_score = -1.0
        self.online_max_template = self.template

    def track_candidates(
        self,
        frame: np.ndarray,
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if top_k is not None and top_k <= 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        bbox, confidence = self._infer(frame, update_state=False)
        if confidence <= 0.0 and np.allclose(bbox, 0.0):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return bbox.reshape(1, 4), np.array([confidence], dtype=np.float32)

    def track(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        return self._infer(frame, update_state=True)
