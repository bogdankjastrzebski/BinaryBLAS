import torch
import sys
from pathlib import Path

# Add the build directory to the path so Python can find the .so file
build_dir = Path(__file__).parent / "build"
sys.path.append(str(build_dir))

# Import your custom SYCL C++ engine!
import bnn_pytorch_ext


# --- SCENARIO SELECTION ---
# Switch this to "xpu" or "cuda" to instantly test the GPU -> GPU scenario.
# The C++ code doesn't change!
device = "xpu" 

batch_size = 4096
in_features = 4096
out_features = 4096
in_int64s = in_features // 64
out_int64s = out_features // 64

print("1. Allocating Memory (Cold Path)...")
# We allocate everything ONCE before the loop, explicitly on the target device.
x = torch.randint(-100, 100, (batch_size, in_int64s), dtype=torch.int64, device=device)
w = torch.randint(-100, 100, (out_features, in_int64s), dtype=torch.int64, device=device)
t = torch.zeros((out_features,), dtype=torch.int32, device=device)

# PRE-ALLOCATE THE OUTPUT BUFFER!
out_buffer = torch.zeros((batch_size, out_int64s), dtype=torch.int64, device=device)

print("2. Starting Hot Loop...")
for epoch in range(100):
    # ZERO allocations. ZERO hardware probing. ZERO copies.
    # Python just passes four 64-bit memory addresses to C++.
    bnn_pytorch_ext.linear_forward_server_out(x, w, t, out_buffer)

print("Done!")
# The result is safely sitting in 'out_buffer'
