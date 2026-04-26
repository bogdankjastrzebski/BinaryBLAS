import torch
import time
import sys
from pathlib import Path

# Load our SYCL extension
build_dir = Path(__file__).parent.parent / "build"
sys.path.append(str(build_dir))
import bnn_pytorch_ext

def run_bench():
    print("🏁 THE BNN BENCHMARK (Python/PyTorch) 🏁")
    device = "cpu" # Force CPU to benchmark our AVX-512 vs PyTorch CPU MKL

    batch_size = 128
    in_features = 4096
    out_features = 4096
    in_int64s = in_features // 64
    out_int64s = out_features // 64
    iters = 100

    print(f"Matrix Size: {batch_size}x{in_features} -> {out_features}")

    # 1. PyTorch FP32 Baseline (F.linear equivalent)
    x_fp32 = torch.randn(batch_size, in_features, dtype=torch.float32, device=device)
    w_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device) 
    t_fp32 = torch.randn(out_features, dtype=torch.float32, device=device) # Bias/Threshold
    out_fp32 = torch.zeros(batch_size, out_features, dtype=torch.float32, device=device)

    # 2. BNN Memory (Packed 64x)
    x_bnn = torch.randint(-100, 100, (batch_size, in_int64s), dtype=torch.int64, device=device)
    w_bnn = torch.randint(-100, 100, (out_features, in_int64s), dtype=torch.int64, device=device)
    t_bnn = torch.zeros((out_features,), dtype=torch.int32, device=device)
    out_bnn = torch.zeros((batch_size, out_int64s), dtype=torch.int64, device=device)

    # 3. BNN Memory (UNPACKED 1x - Simulating no bitwise compression)
    # We pass 'in_features' instead of 'in_int64s' to force 64x more loads/popcounts
    x_unpacked = torch.randint(-100, 100, (batch_size, in_features), dtype=torch.int64, device=device)
    w_unpacked = torch.randint(-100, 100, (out_features, in_features), dtype=torch.int64, device=device)

    # 4. PyTorch FP32 Small (Memory bandwidth limit test)
    x_fp32_small = torch.randn(batch_size, in_int64s, dtype=torch.float32, device=device)
    w_fp32_small = torch.randn(out_features, in_int64s, dtype=torch.float32, device=device) 
    out_fp32_small = torch.zeros(batch_size, out_features, dtype=torch.float32, device=device) # Fixed shape to avoid warning

    print("\nWarming up silicon...")
    for _ in range(10):
        # addmm is the strict in-place equivalent of F.linear(x, w, bias)
        torch.addmm(t_fp32, x_fp32, w_fp32.T, out=out_fp32)
        bnn_pytorch_ext.linear_forward_device_out(x_bnn, w_bnn, t_bnn, out_bnn)
        bnn_pytorch_ext.linear_forward_device_out(x_unpacked, w_unpacked, t_bnn, out_bnn)
        torch.matmul(x_fp32_small, w_fp32_small.T, out=out_fp32_small)

    print("\n🟢 GREEN LIGHT 🟢")

    # --- PyTorch FP32 (F.linear) ---
    start = time.perf_counter()
    for _ in range(iters):
        torch.addmm(t_fp32, x_fp32, w_fp32.T, out=out_fp32)
    fp32_time = (time.perf_counter() - start) / iters

    # --- BNN Device Engine (Packed 64x) ---
    start = time.perf_counter()
    for _ in range(iters):
        bnn_pytorch_ext.linear_forward_device_out(x_bnn, w_bnn, t_bnn, out_bnn)
    dev_time = (time.perf_counter() - start) / iters

    # --- BNN Device Engine (Unpacked 1x) ---
    start = time.perf_counter()
    for _ in range(iters):
        bnn_pytorch_ext.linear_forward_device_out(x_unpacked, w_unpacked, t_bnn, out_bnn)
    unpacked_time = (time.perf_counter() - start) / iters

    # --- PyTorch FP32 Small ---
    start = time.perf_counter()
    for _ in range(iters):
        torch.matmul(x_fp32_small, w_fp32_small.T, out=out_fp32_small)
    fp32s_time = (time.perf_counter() - start) / iters

    print("\n====== TIMING (Per Inference) ======")
    print(f"PT FP32 (F.linear) : {fp32_time * 1000:.3f} ms")
    print(f"BNN Packed (64x)   : {dev_time * 1000:.3f} ms")
    print(f"BNN Unpacked (1x)  : {unpacked_time * 1000:.3f} ms")
    print(f"PT FP32 SMALL      : {fp32s_time * 1000:.3f} ms")

    print("\n====== SPEEDUP MULTIPLIER ======")
    print(f"BNN vs PT FP32     : {fp32_time / dev_time:.2f}x faster")
    print(f"Packing Advantage  : {unpacked_time / dev_time:.2f}x faster than naive bitwise")

if __name__ == "__main__":
    torch.set_num_threads(torch.get_num_threads())
    run_bench()
