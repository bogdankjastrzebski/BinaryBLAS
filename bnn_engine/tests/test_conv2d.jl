using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "julia", "BinaryBLAS"))

using Test
using Random
using BinaryBLAS

@testset "BNN Conv2D NHWC Correctness" begin
    println("--- BNN Convolution Ground Truth Test ---")
    print_hardware_info()

    # We use 256 channels to trigger the "Beast" 4x4x4 unrolled code path
    batch_size = 2
    in_channels = 256
    out_channels = 256
    in_h, in_w = 8, 8
    kernel_size = 3
    stride = 1
    padding = 1
    
    in_c_int64 = div(in_channels, 64)
    out_c_int64 = div(out_channels, 64)
    
    out_h = div(in_h + 2*padding - kernel_size, stride) + 1
    out_w = div(in_w + 2*padding - kernel_size, stride) + 1

    # Allocations (C, W, H, N)
    inputs = rand(UInt64, in_c_int64, in_w, in_h, batch_size)
    weights = rand(UInt64, in_c_int64, kernel_size, kernel_size, out_channels)
    thresholds = rand(Int32, out_channels)
    out_sycl = zeros(UInt64, out_c_int64, out_w, out_h, batch_size)
    
    # 1. Run C++ Engine
    conv2d_nhwc_device!(out_sycl, inputs, weights, thresholds; stride=stride, padding=padding)
    
    # 2. Pure Julia Ground Truth (Simulating Implicit GEMM)
    println("Calculating native Julia spatial sliding window...")
    out_expected = zeros(UInt64, out_c_int64, out_w, out_h, batch_size)
    
    for b in 1:batch_size
        for oh in 1:out_h
            for ow in 1:out_w
                for oc in 1:out_channels
                    pop_sum = 0
                    
                    for kh in 1:kernel_size
                        for kw in 1:kernel_size
                            ih = (oh - 1) * stride - padding + kh
                            iw = (ow - 1) * stride - padding + kw
                            
                            # Bounds check (Padding behaves as 0 bits)
                            if ih >= 1 && ih <= in_h && iw >= 1 && iw <= in_w
                                for ic in 1:in_c_int64
                                    in_val = inputs[ic, iw, ih, b]
                                    w_val = weights[ic, kw, kh, oc]
                                    pop_sum += count_ones(~(in_val ⊻ w_val))
                                end
                            end
                        end
                    end
                    
                    if pop_sum > thresholds[oc]
                        out_idx = div(oc - 1, 64) + 1
                        bit_pos = (oc - 1) % 64
                        out_expected[out_idx, ow, oh, b] |= (UInt64(1) << bit_pos)
                    end
                end
            end
        end
    end
    
    @test out_sycl == out_expected
    println("✅ Spatial Padding, Strides, and AVX Unrolling match perfectly!")
end
