import torch
import torch.nn.functional as F
import time
import sys
from pathlib import Path

# Load our SYCL extension
build_dir = Path(__file__).parent.parent / "build"
sys.path.append(str(build_dir))
import bnn_pytorch_ext

def run_bench():
    print("🏁 THE BEAST vs PyTorch: BNN CONV2D 🏁")
    device = "cpu"

    # Typical deep ResNet layer parameters
    batch_size = 128
    in_channels = 256
    out_channels = 256
    in_h, in_w = 32, 32
    kernel_size = 3
    stride = 1
    padding = 1
    iters = 50
    
    in_c_int64 = in_channels // 64
    out_c_int64 = out_channels // 64
    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1

    print(f"Shape: [{batch_size}, {in_channels}, {in_h}, {in_w}]")
    print(f"Params: {in_channels * out_channels * kernel_size * kernel_size} Bitwise Weights")

    # ==========================================
    # 1. BNN Memory Allocations (Explicit NHWC)
    # ==========================================
    # We allocate as [Batch, Height, Width, Channels]
    inputs_bnn = torch.randint(-100, 100, (batch_size, in_h, in_w, in_c_int64), dtype=torch.int64, device=device)
    weights_bnn = torch.randint(-100, 100, (out_channels, kernel_size, kernel_size, in_c_int64), dtype=torch.int64, device=device)
    thresholds = torch.randint(0, 2304, (out_channels,), dtype=torch.int32, device=device)
    out_bnn = torch.zeros((batch_size, out_h, out_w, out_c_int64), dtype=torch.int64, device=device)

    # ==========================================
    # 2. PyTorch FP32 Allocations (Standard NCHW)
    # ==========================================
    x_fp32 = torch.randn(batch_size, in_channels, in_h, in_w, dtype=torch.float32, device=device)
    w_fp32 = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32, device=device)
    b_fp32 = torch.randn(out_channels, dtype=torch.float32, device=device)

    print("\nWarming up silicon...")
    for _ in range(5):
        F.conv2d(x_fp32, w_fp32, b_fp32, stride=stride, padding=padding)
        bnn_pytorch_ext.conv2d_nhwc_device_out(inputs_bnn, weights_bnn, thresholds, out_bnn, stride, padding)

    print("\n🟢 GREEN LIGHT 🟢\n")

    # --- PyTorch FP32 Baseline ---
    start = time.perf_counter()
    for _ in range(iters):
        # PyTorch uses highly optimized MKL/OneDNN under the hood for this
        out_fp32 = F.conv2d(x_fp32, w_fp32, b_fp32, stride=stride, padding=padding)
    fp32_time = (time.perf_counter() - start) / iters

    # --- BNN Device Engine (1x4 AVX-512 Unrolled) ---
    start = time.perf_counter()
    for _ in range(iters):
        bnn_pytorch_ext.conv2d_nhwc_device_out(inputs_bnn, weights_bnn, thresholds, out_bnn, stride, padding)
    bnn_time = (time.perf_counter() - start) / iters

    print(f"PyTorch FP32 (MKL) : {fp32_time * 1000:.3f} ms")
    print(f"BNN AVX-512 (NHWC) : {bnn_time * 1000:.3f} ms")
    print(f"\n====== SPEEDUP MULTIPLIER ======")
    print(f"{fp32_time / bnn_time:.2f}x faster than PyTorch FP32")

if __name__ == "__main__":
    torch.set_num_threads(torch.get_num_threads())
    run_bench()
