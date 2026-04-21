"""Quick TRT engine benchmark inside distrobox."""
import tensorrt as trt
import torch
import numpy as np
import time

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("models/sglatrack_fp16.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
}

# Discover and bind all I/O tensors
io = {}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = tuple(engine.get_tensor_shape(name))
    dtype = _TRT_TO_TORCH.get(engine.get_tensor_dtype(name), torch.float32)
    mode = engine.get_tensor_mode(name)
    is_input = mode == trt.TensorIOMode.INPUT
    buf = torch.randn(shape, device="cuda", dtype=dtype) if is_input else torch.empty(shape, device="cuda", dtype=dtype)
    context.set_tensor_address(name, buf.data_ptr())
    io[name] = {"buf": buf, "is_input": is_input}
    tag = "INPUT" if is_input else "OUTPUT"
    print(f"  {tag} {name}: {shape} {dtype}")

stream = torch.cuda.Stream()

# Warmup
for _ in range(10):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()

# Benchmark
N = 100
t0 = time.perf_counter()
for _ in range(N):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()
t1 = time.perf_counter()
ms = (t1 - t0) / N * 1000
print(f"\nTRT latency: {ms:.2f} ms ({1000/ms:.0f} FPS)")

# Show output shapes
for name, info in io.items():
    if not info["is_input"]:
        print(f"  {name}: {info['buf'].shape}, sample={info['buf'].flatten()[:4].cpu().numpy()}")
