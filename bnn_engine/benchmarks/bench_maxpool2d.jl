using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "julia", "BinaryBLAS"))

using BenchmarkTools
using Random
using NNlib
using BinaryBLAS

function run_benchmark()
    println("🏁 THE BEAST vs NNlib: BNN MAXPOOL (Julia) 🏁")
    print_hardware_info()

    batch_size = 256
    channels = 256
    in_h, in_w = 32, 32
    kernel_size = 2
    stride = 2
    
    channels_int64 = div(channels, 64)
    out_h = div(in_h, stride)
    out_w = div(in_w, stride)

    # 1. BNN Memory Allocations
    inputs_bnn = rand(UInt64, channels_int64, in_w, in_h, batch_size)
    out_bnn = zeros(UInt64, channels_int64, out_w, out_h, batch_size)

    # 2. NNlib FP32 Allocations (W, H, C, N)
    x_fp32 = rand(Float32, in_w, in_h, channels, batch_size)
    out_fp32 = zeros(Float32, out_w, out_h, channels, batch_size)
    pdims = PoolDims(x_fp32, (kernel_size, kernel_size); stride=(stride, stride))

    println("\n🟢 GREEN LIGHT 🟢\n")

    println("--- NNlib FP32 Baseline ---")
    fp32_bench = @benchmark begin 
        maxpool!($out_fp32, $x_fp32, $pdims)
    end samples=50
    display(fp32_bench)

    println("\n--- BNN Device Engine (Bitwise OR) ---")
    dev_bench = @benchmark begin 
        maxpool2d_nhwc_device!($out_bnn, $inputs_bnn; kernel_size=$kernel_size, stride=$stride)
    end samples=50
    display(dev_bench)
end

run_benchmark()
