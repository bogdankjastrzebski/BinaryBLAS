import torch
import sys
from pathlib import Path

# Add the build directory to the path so Python can find the .so file
build_dir = Path(__file__).parent / "build"
sys.path.append(str(build_dir))

# Import your custom SYCL C++ engine!
import bnn_pytorch_ext

def main():
    print("--- BNN SYCL Engine Test ---")
    
    # 1. Define the network shape
    batch_size = 4
    in_features = 256  # Must be a multiple of 64 for this test
    out_features = 128
    
    in_int64s = in_features // 64
    
    # 2. Create dummy data directly in PyTorch
    # We use random integers to simulate packed bit-vectors
    print("Allocating memory...")
    inputs = torch.randint(-1000, 1000, (batch_size, in_int64s), dtype=torch.int64)
    weights = torch.randint(-1000, 1000, (out_features, in_int64s), dtype=torch.int64)
    
    # Thresholds for the Batch Norm fusion (int32)
    thresholds = torch.zeros((out_features,), dtype=torch.int32)
    
    print(f"Inputs Shape:  {inputs.shape} (Int64)")
    print(f"Weights Shape: {weights.shape} (Int64)")

    # 3. Fire the SYCL Kernel
    print("\nLaunching SYCL AVX-512 Kernel...")
    try:
        outputs = bnn_pytorch_ext.linear_forward(inputs, weights, thresholds)
        
        print("Kernel execution successful!")
        print(f"Outputs Shape: {outputs.shape} (Int64)")
        print("\nFirst batch output snippet (Packed bits):")
        print(outputs[0])
        
    except Exception as e:
        print("\nKERNEL CRASHED:")
        print(e)

if __name__ == "__main__":
    main()
