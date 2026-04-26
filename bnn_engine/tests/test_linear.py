import torch
import sys
from pathlib import Path

# Load our SYCL extension
build_dir = Path(__file__).parent.parent / "build"
sys.path.append(str(build_dir))
import bnn_pytorch_ext

def count_ones(n):
    # Mask to strict 64-bit unsigned integer to cleanly handle PyTorch negative numbers
    return bin(n & 0xFFFFFFFFFFFFFFFF).count('1')

def test_ground_truth_loop():
    print("--- BNN Python Ground Truth Test (Scalar Loop) ---")
    device = "cpu"
    batch_size, in_features, out_features = 17, 256, 128 
    in_int64s = in_features // 64
    out_int64s = out_features // 64

    # 1. Generate Raw Data
    inputs = torch.randint(-100000, 100000, (batch_size, in_int64s), dtype=torch.int64, device=device)
    weights = torch.randint(-100000, 100000, (out_features, in_int64s), dtype=torch.int64, device=device)
    thresholds = torch.randint(0, in_features, (out_features,), dtype=torch.int32, device=device)
    
    # 2. Run BOTH C++ SYCL Engines
    out_dev = torch.zeros((batch_size, out_int64s), dtype=torch.int64, device=device)
    out_srv = torch.zeros((batch_size, out_int64s), dtype=torch.int64, device=device)
    
    bnn_pytorch_ext.linear_forward_device_out(inputs, weights, thresholds, out_dev)
    bnn_pytorch_ext.linear_forward_server_out(inputs, weights, thresholds, out_srv)

    # 3. Native Python Scalar Loop (Foolproof Ground Truth)
    print("Calculating native Python scalar ground truth (this may take a second)...")
    out_list = [[0] * out_int64s for _ in range(batch_size)]
    
    # Extract to pure Python lists to bypass PyTorch tensor quirks
    in_list = inputs.tolist()
    w_list = weights.tolist()
    t_list = thresholds.tolist()

    for b in range(batch_size):
        for o in range(out_features):
            pop_sum = 0
            for i in range(in_int64s):
                # Bitwise XNOR: ~(in ^ w)
                xnor_val = ~(in_list[b][i] ^ w_list[o][i])
                pop_sum += count_ones(xnor_val)
            
            # Thresholding and Packing
            if pop_sum > t_list[o]:
                out_int_idx = o // 64
                bit_pos = o % 64
                out_list[b][out_int_idx] |= (1 << bit_pos)

    # Convert infinite-precision Python Ints back to signed 64-bit bounds
    for b in range(batch_size):
        for i in range(out_int64s):
            if out_list[b][i] >= (1 << 63):
                out_list[b][i] -= (1 << 64)

    out_expected = torch.tensor(out_list, dtype=torch.int64, device=device)

    # 4. The Final Verdict
    print("\n====== SCORECARD ======")
    print(out_dev)
    print(out_expected)
    mismatches_dev = (out_dev != out_expected).sum().item()
    if mismatches_dev > 0:
        print(f"❌ DEVICE Mismatches: {mismatches_dev} elements differ.")
    else:
        print("✅ DEVICE Engine Perfectly Matches!")

    mismatches_srv = (out_srv != out_expected).sum().item()
    if mismatches_srv > 0:
        print(f"❌ SERVER Mismatches: {mismatches_srv} elements differ.")
    else:
        print("✅ SERVER Engine Perfectly Matches!")

    torch.testing.assert_close(out_dev, out_expected, msg="Device Engine mismatch!")
    torch.testing.assert_close(out_srv, out_expected, msg="Server Engine mismatch!")

if __name__ == "__main__":
    test_ground_truth_loop()
