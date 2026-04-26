import torch
import sys
import time
from pathlib import Path

build_dir = Path(__file__).parent / "build"
sys.path.append(str(build_dir))
import bnn_pytorch_ext

def main():
    # A realistic server-side batched workload
    batch_size = 1024
    in_features = 4096 
    out_features = 4096
    in_int64s = in_features // 64
    
    inputs = torch.randint(-1000, 1000, (batch_size, in_int64s), dtype=torch.int64)
    weights = torch.randint(-1000, 1000, (out_features, in_int64s), dtype=torch.int64)
    thresholds = torch.zeros((out_features,), dtype=torch.int32)
    
    print(f"Warming up... Matrix size: {batch_size}x{in_features} -> {out_features}")
    
    # Warmup
    for _ in range(10):
        _ = bnn_pytorch_ext.linear_forward(inputs, weights, thresholds)
        
    # The actual timed loop for the profiler
    iters = 1000
    start = time.perf_counter()
    
    for _ in range(iters):
        _ = bnn_pytorch_ext.linear_forward(inputs, weights, thresholds)
        
    end = time.perf_counter()
    
    total_time = end - start
    print(f"Total time for {iters} iterations: {total_time:.4f} seconds")
    print(f"Throughput: {iters / total_time:.2f} inferences/sec")

if __name__ == "__main__":
    main()
