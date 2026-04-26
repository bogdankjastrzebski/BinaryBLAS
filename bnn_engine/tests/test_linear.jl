using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "julia", "BinaryBLAS")) # Points to the folder with Project.toml

using Test
using Random
using BinaryBLAS

@testset "BNN Absolute Correctness" begin
    println("--- BNN Julia Ground Truth Test ---")
    print_hardware_info()

    batch_size = 17
    in_features = 256
    out_features = 128
    
    in_int64s = div(in_features, 64)
    out_int64s = div(out_features, 64)
    
    # Pure standard Julia memory allocation
    inputs = rand(UInt64, in_int64s, batch_size)
    weights = rand(UInt64, in_int64s, out_features)
    thresholds = rand(Int32, out_features)
    out_sycl = zeros(UInt64, out_int64s, batch_size)
    
    # 1. Run our C++ SYCL Engine via the elegant wrapper
    linear_forward_device!(out_sycl, inputs, weights, thresholds)
    
    # 2. Calculate Pure Native Julia Ground Truth
    println("Calculating native Julia scalar ground truth...")
    out_expected = zeros(UInt64, out_int64s, batch_size)
    
    for b in 1:batch_size
        for o in 1:out_features
            pop_sum = 0
            for i in 1:in_int64s
                xnor_val = ~(inputs[i, b] ⊻ weights[i, o])
                pop_sum += count_ones(xnor_val)
            end
            if pop_sum > thresholds[o]
                out_int_idx = div(o - 1, 64) + 1
                bit_pos = (o - 1) % 64
                out_expected[out_int_idx, b] |= (UInt64(1) << bit_pos)
            end
        end
    end
    
    @test out_sycl == out_expected
    println("✅ Julia Native Math Perfectly Matches SYCL Execution!")
end
