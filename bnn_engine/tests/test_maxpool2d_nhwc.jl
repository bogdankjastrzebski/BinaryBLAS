using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "julia", "BinaryBLAS"))

using Test
using Random
using BinaryBLAS

@testset "BNN MaxPool2D NHWC Correctness" begin
    println("--- BNN MaxPool2D Ground Truth Test ---")
    print_hardware_info()

    batch_size = 4
    channels = 256
    in_h, in_w = 32, 32
    kernel_size = 2
    stride = 2
    
    channels_int64 = div(channels, 64)
    out_h = div(in_h, stride)
    out_w = div(in_w, stride)

    inputs = rand(UInt64, channels_int64, in_w, in_h, batch_size)
    out_sycl = zeros(UInt64, channels_int64, out_w, out_h, batch_size)
    
    # 1. Run C++ Engine
    maxpool2d_nhwc_device!(out_sycl, inputs; kernel_size=kernel_size, stride=stride)
    
    # 2. Pure Julia Ground Truth
    println("Calculating native Julia bitwise OR ground truth...")
    out_expected = zeros(UInt64, channels_int64, out_w, out_h, batch_size)
    
    for b in 1:batch_size
        for oh in 1:out_h
            for ow in 1:out_w
                for c in 1:channels_int64
                    max_val = UInt64(0)
                    for kh in 0:(kernel_size - 1)
                        for kw in 0:(kernel_size - 1)
                            ih = (oh - 1) * stride + 1 + kh
                            iw = (ow - 1) * stride + 1 + kw
                            max_val |= inputs[c, iw, ih, b]
                        end
                    end
                    out_expected[c, ow, oh, b] = max_val
                end
            end
        end
    end
    
    @test out_sycl == out_expected
    println("✅ Bitwise Spatial Pooling matches perfectly!")
end
