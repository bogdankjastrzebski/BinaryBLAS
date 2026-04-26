using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "julia", "BinaryBLAS"))

using BenchmarkTools
using Random
using NNlib       # <--- Add this!
using BinaryBLAS

function run_benchmark()
    println("🏁 THE BEAST vs NNlib: BNN CONV2D (Julia) 🏁")
    print_hardware_info()

    # Typical deep ResNet layer parameters
    batch_size = 128
    in_channels = 256
    out_channels = 256
    in_h, in_w = 32, 32
    kernel_size = 3
    stride = 1
    padding = 1
    
    in_c_int64 = div(in_channels, 64)
    out_c_int64 = div(out_channels, 64)
    out_h = div(in_h + 2*padding - kernel_size, stride) + 1
    out_w = div(in_w + 2*padding - kernel_size, stride) + 1

    println("Shape: [$batch_size, $in_channels, $in_h, $in_w]")
    println("Params: $(in_channels * out_channels * kernel_size * kernel_size) Bitwise Weights")

    # ==========================================
    # 1. BNN Memory Allocations (C, W, H, N)
    # ==========================================
    inputs_bnn = rand(UInt64, in_c_int64, in_w, in_h, batch_size)
    weights_bnn = rand(UInt64, in_c_int64, kernel_size, kernel_size, out_channels)
    thresholds = rand(Int32(0):Int32(2304), out_channels) # Max popcount is 3*3*256=2304
    out_bnn = zeros(UInt64, out_c_int64, out_w, out_h, batch_size)

    # ==========================================
    # 2. NNlib FP32 Allocations (W, H, C, N)
    # ==========================================
    x_fp32 = rand(Float32, in_w, in_h, in_channels, batch_size)
    w_fp32 = rand(Float32, kernel_size, kernel_size, in_channels, out_channels)
    out_fp32 = zeros(Float32, out_w, out_h, out_channels, batch_size)
    
    # NNlib requires pre-compiling the dimensions for optimal in-place speed
    cdims = DenseConvDims(x_fp32, w_fp32; stride=stride, padding=padding)

    println("\n🟢 GREEN LIGHT 🟢\n")

    println("--- NNlib FP32 Baseline ---")
    fp32_bench = @benchmark begin 
        conv!($out_fp32, $x_fp32, $w_fp32, $cdims)
    end samples=50
    display(fp32_bench)

    println("\n--- BNN Device Engine (1x4 AVX-512 Unrolled) ---")
    dev_bench = @benchmark begin 
        conv2d_nhwc_device!($out_bnn, $inputs_bnn, $weights_bnn, $thresholds; stride=$stride, padding=$padding)
    end samples=50
    display(dev_bench)
end

run_benchmark()
