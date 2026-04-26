using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "julia", "BinaryBLAS"))

using BenchmarkTools
using Random
using LinearAlgebra
using BinaryBLAS

function run_benchmark()
    println("🏁 THE BNN BENCHMARK (Julia) 🏁")
    print_hardware_info()

    batch_size = 128
    in_features = 4096
    out_features = 4096
    
    in_int64s = div(in_features, 64)
    out_int64s = div(out_features, 64)

    println("Matrix Size: $(batch_size)x$(in_features) -> $(out_features)")

    # 1. Native Julia FP32 Baseline
    x_fp32 = rand(Float32, in_features, batch_size)
    w_fp32 = rand(Float32, in_features, out_features)
    out_fp32 = zeros(Float32, out_features, batch_size)

    # 2. BNN Memory (Packed 64x)
    x_bnn = rand(UInt64, in_int64s, batch_size)
    w_bnn = rand(UInt64, in_int64s, out_features)
    t_bnn = zeros(Int32, out_features)
    out_bnn = zeros(UInt64, out_int64s, batch_size)

    # 3. BNN Memory (UNPACKED 1x - Simulating no bitwise compression)
    x_unpacked = rand(UInt64, in_features, batch_size)
    w_unpacked = rand(UInt64, in_features, out_features)
    out_unpacked = zeros(UInt64, out_int64s, batch_size) 

    # 4. FP32 Small (Memory Limit Test)
    x_fp32_small = rand(Float32, in_int64s, batch_size)
    w_fp32_small = rand(Float32, in_int64s, out_features)
    out_fp32_small = zeros(Float32, out_features, batch_size)

    println("\n🟢 GREEN LIGHT 🟢\n")

    println("--- Native Julia FP32 ---")
    fp32_bench = @benchmark begin 
        mul!($out_fp32, transpose($w_fp32), $x_fp32) 
    end samples=50
    display(fp32_bench)

    println("\n--- BNN Device Engine (Packed 64x) ---")
    dev_bench = @benchmark begin 
        linear_forward_device!($out_bnn, $x_bnn, $w_bnn, $t_bnn)
    end samples=50
    display(dev_bench)

    println("\n--- BNN Device Engine (Unpacked 1x) ---")
    unp_bench = @benchmark begin
        linear_forward_device!($out_unpacked, $x_unpacked, $w_unpacked, $t_bnn)
    end samples=10
    display(unp_bench)

    println("\n--- Native Julia FP32 SMALL ---")
    small_bench = @benchmark begin 
        mul!($out_fp32_small, transpose($w_fp32_small), $x_fp32_small) 
    end samples=50
    display(small_bench)
end

run_benchmark()
