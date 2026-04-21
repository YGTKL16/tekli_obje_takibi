# pyright: reportAttributeAccessIssue=false
"""Export SGLATrack model to ONNX and optionally build a TensorRT engine.

Usage:
    # Step 1: Export to ONNX
    python scripts/export_tensorrt.py --checkpoint models/SGLATrack/checkpoints/sglatrack_ep0297.pth.tar

    # Step 2 (auto): If tensorrt is installed, builds .engine automatically.
    #         (manual): trtexec --onnx=models/sglatrack.onnx --saveEngine=models/sglatrack_fp16.engine --fp16

    # Verify:
    python scripts/export_tensorrt.py --verify --engine models/sglatrack_fp16.engine
"""

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SGLA_ROOT = os.path.join(_PROJECT_ROOT, "models", "SGLATrack")


def _build_export_module(model):
    """Create an nn.Module with clean forward(template, search) -> (score_map, size_map, offset_map)."""
    import torch  # pyright: ignore[reportMissingImports]
    import torch.nn as nn  # pyright: ignore[reportMissingImports]

    # SGLA routing constants (from base_backbone.py)
    start_layer = 5
    enabled_layer_num = 1

    class ExportableForward(nn.Module):
        def __init__(self, backbone, box_head, feat_sz_s, feat_len_s):
            super().__init__()
            self.patch_embed = backbone.patch_embed
            self.pos_embed_z = backbone.pos_embed_z
            self.pos_embed_x = backbone.pos_embed_x
            self.pos_drop = backbone.pos_drop
            self.blocks = backbone.blocks
            self.norm = backbone.norm
            self.mlp = backbone.MLP  # SGLA routing MLP
            self.box_head = box_head
            self.feat_sz_s = feat_sz_s
            self.feat_len_s = feat_len_s

        def forward(self, template, search):
            # Patch embedding
            z = self.patch_embed(template)
            x = self.patch_embed(search)

            # Add position embeddings
            z = z + self.pos_embed_z
            x = x + self.pos_embed_x

            # Concatenate template + search tokens (CAT_MODE='direct')
            x = torch.cat((z, x), dim=1)
            x = self.pos_drop(x)

            # Run blocks 0..start_layer (inclusive)
            for i in range(start_layer + 1):
                x = self.blocks[i](x)

            # SGLA routing: MLP scores which later layer to use
            # x[:,:,0] is the first feature dim across all tokens (CLS-like)
            pro = self.mlp(x[:, :, 0].clone())  # (B, 6)
            # Select the best layer (top-1)
            selected_idx = torch.argmax(pro, dim=1)  # (B,)

            # For ONNX: compute all candidate outputs and select via one-hot
            # Each candidate is blocks[start_layer+1+j](x) for j in 0..5
            num_candidates = len(self.blocks) - start_layer - 1
            one_hot = torch.nn.functional.one_hot(
                selected_idx, num_classes=num_candidates
            ).float()  # (B, 6)

            # Start from base features after block 5
            result = torch.zeros_like(x)
            for j in range(num_candidates):
                candidate = self.blocks[start_layer + 1 + j](x)
                # Weight by one-hot mask: only selected layer contributes
                weight = one_hot[:, j].view(-1, 1, 1)  # (B, 1, 1)
                result = result + weight * candidate

            x = result
            x = self.norm(x)

            # Extract search region features
            enc_opt = x[:, -self.feat_len_s:]
            opt = enc_opt.unsqueeze(-1).permute(0, 3, 2, 1).contiguous()
            _, _, C, _ = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

            # Run center prediction head
            score_map, _, size_map, offset_map = self.box_head(opt_feat, None)

            return score_map, size_map, offset_map

    return ExportableForward(
        model.backbone, model.box_head, model.feat_sz_s, model.feat_len_s
    )


def load_sglatrack(checkpoint_path):
    """Load SGLATrack model from checkpoint."""
    import torch  # pyright: ignore[reportMissingImports]

    if _SGLA_ROOT not in sys.path:
        sys.path.insert(0, _SGLA_ROOT)

    from lib.config.sglatrack.config import cfg, update_config_from_file  # pyright: ignore[reportMissingImports]
    from lib.models.sglatrack import build_sglatrack  # pyright: ignore[reportMissingImports]

    yaml_file = os.path.join(_SGLA_ROOT, "experiments", "sglatrack", "deit_distilled.yaml")
    update_config_from_file(yaml_file)

    network = build_sglatrack(cfg, training=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network.load_state_dict(ckpt["net"], strict=True)
    network.eval()

    print(f"[INFO] Loaded SGLATrack from {checkpoint_path}")
    n_params = sum(p.numel() for p in network.parameters()) / 1e6
    print(f"[INFO] Parameters: {n_params:.2f}M")

    return network, cfg


def export_onnx(checkpoint_path, output_path, opset=17):
    """Export SGLATrack to ONNX format."""
    import torch  # pyright: ignore[reportMissingImports]

    network, cfg = load_sglatrack(checkpoint_path)
    export_model = _build_export_module(network)
    export_model.eval()

    template_size = cfg.TEST.TEMPLATE_SIZE  # 128
    search_size = cfg.TEST.SEARCH_SIZE      # 256

    dummy_template = torch.randn(1, 3, template_size, template_size)
    dummy_search = torch.randn(1, 3, search_size, search_size)

    # Verify forward pass works
    with torch.no_grad():
        score_map, size_map, offset_map = export_model(dummy_template, dummy_search)
    print(f"[INFO] Forward pass OK: score_map={score_map.shape}, size_map={size_map.shape}, offset_map={offset_map.shape}")

    # Export
    torch.onnx.export(
        export_model,
        (dummy_template, dummy_search),
        output_path,
        input_names=["template", "search"],
        output_names=["score_map", "size_map", "offset_map"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # Fixed shapes for TRT optimization
    )

    # Re-save with embedded weights (no external .data file) for TensorRT compat
    try:
        import onnx  # pyright: ignore[reportMissingImports]
        model_onnx = onnx.load(output_path, load_external_data=True)
        for tensor in model_onnx.graph.initializer:
            tensor.ClearField("data_location")
        onnx.save(model_onnx, output_path, save_as_external_data=False)
        # Remove leftover .data file if present
        data_file = output_path + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)
        onnx.checker.check_model(model_onnx)
        print("[INFO] ONNX model verification passed")
    except ImportError:
        print("[WARN] onnx package not installed, skipping re-embed/verification")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] ONNX exported: {output_path} ({file_size_mb:.1f} MB)")

    return output_path


# Head layer name patterns — these layers lose precision in FP16.
# Identified from TRT network: layers 1699-1729 are the CenterPredictor head.
# Conv layers: node_Conv_1024..1035, node_conv2d_6/11/16
# Activation layers: node_relu..relu_11, node_sigmoid, node_sigmoid_1, node_clamp, node_clamp_1
_HEAD_LAYER_NAMES = {
    # CTR branch (score_map)
    "node_Conv_1024", "node_relu", "node_Conv_1025", "node_relu_1",
    "node_Conv_1026", "node_relu_2", "node_Conv_1027", "node_relu_3",
    "node_conv2d_6", "node_sigmoid", "node_clamp",
    # Offset branch (offset_map)
    "node_Conv_1028", "node_relu_4", "node_Conv_1029", "node_relu_5",
    "node_Conv_1030", "node_relu_6", "node_Conv_1031", "node_relu_7",
    "node_conv2d_11",
    # Size branch (size_map)
    "node_Conv_1032", "node_relu_8", "node_Conv_1033", "node_relu_9",
    "node_Conv_1034", "node_relu_10", "node_Conv_1035", "node_relu_11",
    "node_conv2d_16", "node_sigmoid_1", "node_clamp_1",
}


def _is_head_layer(layer_name, layer_inputs):
    """Check if a TRT layer belongs to the prediction head."""
    return layer_name in _HEAD_LAYER_NAMES


class _CalibrationDataset:
    """Provides template+search pairs from train sequences for INT8 calibration."""

    def __init__(self, n_samples=500, template_size=128, search_size=256):
        import cv2  # pyright: ignore[reportMissingImports]
        import numpy as np  # pyright: ignore[reportMissingImports]
        import json
        import random

        self.template_size = template_size
        self.search_size = search_size
        self.samples = []  # list of (template_np, search_np) float32 arrays

        data_root = os.path.join(_PROJECT_ROOT, "data", "contest_release")
        manifest_path = os.path.join(data_root, "metadata", "contestant_manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        seq_ids = list(manifest["train"].keys())
        random.seed(42)
        random.shuffle(seq_ids)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        collected = 0
        for seq_id in seq_ids:
            if collected >= n_samples:
                break
            info = manifest["train"][seq_id]
            video_path = os.path.join(data_root, info["video_path"])
            cap = cv2.VideoCapture(video_path)
            n_frames = info["n_frames"]
            # Sample up to 3 frames per sequence
            frame_indices = sorted(random.sample(range(n_frames), min(3, n_frames)))
            for fi in frame_indices:
                if collected >= n_samples:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, bgr = cap.read()
                if not ret:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                # Center crop for template and search
                t = self._center_crop(rgb, h, w, template_size)
                s = self._center_crop(rgb, h, w, search_size)
                # Normalize to match model preprocessing
                t = ((t.astype(np.float32) / 255.0) - mean) / std
                s = ((s.astype(np.float32) / 255.0) - mean) / std
                # HWC -> CHW
                t = np.ascontiguousarray(t.transpose(2, 0, 1)[np.newaxis])  # (1,3,128,128)
                s = np.ascontiguousarray(s.transpose(2, 0, 1)[np.newaxis])  # (1,3,256,256)
                self.samples.append((t, s))
                collected += 1
            cap.release()

        print(f"[INFO] Calibration dataset: {len(self.samples)} samples from {len(seq_ids)} sequences")

    @staticmethod
    def _center_crop(img, h, w, size):
        import cv2  # pyright: ignore[reportMissingImports]
        if h >= size and w >= size:
            y0 = (h - size) // 2
            x0 = (w - size) // 2
            return img[y0:y0+size, x0:x0+size]
        return cv2.resize(img, (size, size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Int8Calibrator:
    """TensorRT INT8 entropy calibrator using real video frames."""

    def __init__(self, dataset, cache_file="int8_calibration.cache"):
        import tensorrt as trt  # pyright: ignore[reportMissingImports]
        self._dataset = dataset
        self._cache_file = os.path.join(_PROJECT_ROOT, "models", cache_file)
        self._current_idx = 0
        self._device_buffers = None
        # Inherit from the correct base at runtime
        self.__class__.__bases__ = (trt.IInt8EntropyCalibrator2,)
        trt.IInt8EntropyCalibrator2.__init__(self)

    def get_batch_size(self):
        return 1

    def get_batch(self, names, p_str=None):
        import numpy as np  # pyright: ignore[reportMissingImports]
        if self._current_idx >= len(self._dataset):
            return None
        t, s = self._dataset[self._current_idx]
        self._current_idx += 1
        if self._device_buffers is None:
            import cuda  # noqa
            try:
                import pycuda.driver as cuda_drv  # pyright: ignore[reportMissingImports]
                import pycuda.autoinit  # pyright: ignore[reportMissingImports, reportUnusedImport]
                self._alloc = cuda_drv.mem_alloc
                self._memcpy = cuda_drv.memcpy_htod
            except ImportError:
                # Fallback to torch CUDA
                import torch  # pyright: ignore[reportMissingImports]
                self._template_buf = torch.empty(1, 3, 128, 128, dtype=torch.float32, device="cuda")
                self._search_buf = torch.empty(1, 3, 256, 256, dtype=torch.float32, device="cuda")
                self._device_buffers = [
                    int(self._template_buf.data_ptr()),
                    int(self._search_buf.data_ptr()),
                ]
                self._use_torch = True
                self._memcpy_torch(t, s)
                return self._device_buffers
            self._device_buffers = [
                cuda_drv.mem_alloc(t.nbytes),
                cuda_drv.mem_alloc(s.nbytes),
            ]
            self._use_torch = False

        if getattr(self, "_use_torch", False):
            self._memcpy_torch(t, s)
        else:
            self._memcpy(self._device_buffers[0], np.ascontiguousarray(t))
            self._memcpy(self._device_buffers[1], np.ascontiguousarray(s))
        return [int(b) for b in self._device_buffers]

    def _memcpy_torch(self, t, s):
        import torch  # pyright: ignore[reportMissingImports]
        self._template_buf.copy_(torch.from_numpy(t))
        self._search_buf.copy_(torch.from_numpy(s))

    def read_calibration_cache(self):
        if os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self._cache_file, "wb") as f:
            f.write(cache)
        print(f"[INFO] INT8 calibration cache saved: {self._cache_file}")


def build_trt_engine(onnx_path, engine_path, fp16=True, workspace_gb=2,
                     mixed_precision=False, int8_backbone=False, calib_samples=500):
    """Build TensorRT engine from ONNX model.

    Args:
        mixed_precision: Force head layers (score/offset/size) to FP32.
        int8_backbone: Enable INT8 for backbone with calibration data.
    """
    try:
        import tensorrt as trt  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("[WARN] tensorrt not installed. Build engine manually with trtexec:")
        precision = "--fp16" if fp16 else ""
        print(f"  trtexec --onnx={onnx_path} --saveEngine={engine_path} {precision}")
        return None

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 precision enabled")

    # INT8 calibration setup
    calibrator = None
    if int8_backbone:
        config.set_flag(trt.BuilderFlag.INT8)
        dataset = _CalibrationDataset(n_samples=calib_samples)
        calibrator = Int8Calibrator(dataset)
        config.int8_calibrator = calibrator
        print(f"[INFO] INT8 enabled with {len(dataset)} calibration samples")

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    # Per-layer precision override for mixed precision
    if mixed_precision or int8_backbone:
        n_forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_inputs = []
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is not None:
                    layer_inputs.append(inp.name)

            if _is_head_layer(layer.name, layer_inputs):
                layer.precision = trt.float32
                for k in range(layer.num_outputs):
                    layer.set_output_type(k, trt.float32)
                n_forced += 1
        print(f"[INFO] Mixed precision: {n_forced} head layers forced to FP32")

    print("[INFO] Building TensorRT engine (this may take a few minutes)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"[INFO] TensorRT engine saved: {engine_path} ({file_size_mb:.1f} MB, built in {elapsed:.1f}s)")
    return engine_path


def verify_engine(engine_path):
    """Verify TensorRT engine produces valid output using PyTorch CUDA tensors."""
    try:
        import tensorrt as trt  # pyright: ignore[reportMissingImports]
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("[ERROR] tensorrt and torch required for verification")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    # Allocate buffers using PyTorch CUDA tensors
    _TRT_TO_TORCH = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
    }

    tensors = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        trt_dtype = engine.get_tensor_dtype(name)
        torch_dtype = _TRT_TO_TORCH.get(trt_dtype, torch.float32)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

        if is_input:
            t = torch.randn(shape, dtype=torch_dtype, device="cuda")
        else:
            t = torch.empty(shape, dtype=torch_dtype, device="cuda")

        tensors[name] = {"tensor": t, "is_input": is_input, "shape": shape}
        context.set_tensor_address(name, t.data_ptr())

    # Run inference
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    # Print outputs
    for name, info in tensors.items():
        if not info["is_input"]:
            arr = info["tensor"].cpu().numpy()
            print(f"  {name}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")

    # Benchmark latency
    n_warmup, n_runs = 50, 200
    for _ in range(n_warmup):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    t0 = time.time()
    for _ in range(n_runs):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    latency_ms = (time.time() - t0) / n_runs * 1000

    print(f"[INFO] TensorRT latency: {latency_ms:.2f} ms ({1000/latency_ms:.0f} FPS)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export SGLATrack to ONNX/TensorRT")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(_PROJECT_ROOT, "models", "SGLATrack", "checkpoints", "sglatrack_ep0297.pth.tar"))
    parser.add_argument("--onnx-output", type=str,
                        default=os.path.join(_PROJECT_ROOT, "models", "sglatrack.onnx"))
    parser.add_argument("--engine-output", type=str,
                        default=os.path.join(_PROJECT_ROOT, "models", "sglatrack_fp16.engine"))
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--mixed", action="store_true", help="Mixed precision: head layers FP32, backbone FP16")
    parser.add_argument("--int8", action="store_true", help="INT8 backbone + FP32 head (requires calibration)")
    parser.add_argument("--calib-samples", type=int, default=500, help="Number of calibration samples for INT8")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true", help="Only verify existing engine")
    parser.add_argument("--engine", type=str, help="Engine path for --verify")
    parser.add_argument("--skip-trt", action="store_true", help="Only export ONNX, skip TRT build")
    args = parser.parse_args()

    if args.verify:
        engine = args.engine or args.engine_output
        print(f"[INFO] Verifying engine: {engine}")
        verify_engine(engine)
        return

    # Step 1: ONNX export
    os.makedirs(os.path.dirname(args.onnx_output), exist_ok=True)
    export_onnx(args.checkpoint, args.onnx_output, opset=args.opset)

    if args.skip_trt:
        return

    # Step 2: TensorRT engine build
    os.makedirs(os.path.dirname(args.engine_output), exist_ok=True)
    build_trt_engine(
        args.onnx_output, args.engine_output,
        fp16=args.fp16,
        mixed_precision=args.mixed or args.int8,
        int8_backbone=args.int8,
        calib_samples=args.calib_samples,
    )


if __name__ == "__main__":
    main()
