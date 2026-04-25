import torch
import sys
import time
from pathlib import Path

# Load our SYCL extension
build_dir = Path(__file__).parent / "build"
sys.path.append(str(build_dir))
import bnn_pytorch_ext

def main():
    # 1. Define the Logical Network Shape
    batch_size = 1024
    in_features = 1024 
    out_features = 1024
    
    print(f"--- BNN vs PyTorch Head-to-Head ---")
    print(f"Matrix: {batch_size}x{in_features} -> {out_features}")
    
    # 2. Setup PyTorch FP32 Data (The Standard)
    print("\nAllocating PyTorch FP32 Tensors (Standard Neural Net)...")
    # Using torch.matmul directly to avoid nn.Module overhead
    x_fp32 = torch.randn(batch_size, in_features, dtype=torch.float32)
    w_fp32 = torch.randn(in_features, out_features, dtype=torch.float32) # Note: matmul expects (in, out)
    
    # 3. Setup SYCL BNN Data (Packed Bits)
    print("Allocating SYCL BNN Int64 Tensors...")
    in_int64s = in_features // 64
    x_bnn = torch.randint(-1000, 1000, (batch_size, in_int64s), dtype=torch.int64)
    w_bnn = torch.randint(-1000, 1000, (out_features, in_int64s), dtype=torch.int64)
    thresh = torch.zeros((out_features,), dtype=torch.int32)
    
    iters = 100
    
    # --- WARMUP ---
    for _ in range(5):
        _ = torch.matmul(x_fp32, w_fp32)
        _ = bnn_pytorch_ext.linear_forward(x_bnn, w_bnn, thresh)
        
    # --- BENCHMARK 1: PYTORCH FP32 ---
    print(f"\nRunning PyTorch FP32 ({iters} iterations)...")
    start_pt = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(x_fp32, w_fp32)
    end_pt = time.perf_counter()
    time_pt = end_pt - start_pt
    
    # --- BENCHMARK 2: OUR SYCL BNN ---
    print(f"Running SYCL BNN ({iters} iterations)...")
    start_bnn = time.perf_counter()
    for _ in range(iters):
        _ = bnn_pytorch_ext.linear_forward(x_bnn, w_bnn, thresh)
    end_bnn = time.perf_counter()
    time_bnn = end_bnn - start_bnn
    
    # --- RESULTS ---
    print("\n====== FINAL SCORECARD ======")
    print(f"PyTorch FP32 Throughput : {iters / time_pt:.2f} inferences/sec")
    print(f"SYCL BNN Throughput     : {iters / time_bnn:.2f} inferences/sec")
    
    if time_bnn < time_pt:
        speedup = time_pt / time_bnn
        print(f"\n🏆 BNN Wins! It is {speedup:.2f}x FASTER than PyTorch FP32.")
    else:
        gap = time_bnn / time_pt
        print(f"\n🛑 PyTorch Wins. BNN is currently {gap:.2f}x SLOWER than PyTorch FP32.")

if __name__ == "__main__":
    # Ensure PyTorch uses all CPU cores for its OneMKL backend
    torch.set_num_threads(torch.get_num_threads())
    main()
