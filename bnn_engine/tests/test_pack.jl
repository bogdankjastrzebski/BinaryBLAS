using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "julia", "BinaryBLAS"))

using Test
using Random
using BinaryBLAS

@testset "FP32 to UInt64 Binarize and Pack" begin
    println("--- BNN Data Importer Ground Truth Test ---")
    print_hardware_info()

    # Typical parameters for the output of Layer 1 (e.g., ResNet)
    batch_size = 4
    channels = 256
    w, h = 32, 32
    
    channels_int64 = div(channels, 64)

    # 1. Allocate PyTorch-style FP32 data
    # randn generates a normal distribution centered around 0 (roughly 50% positive, 50% negative)
    input_fp32 = randn(Float32, channels, w, h, batch_size)
    
    # 2. Allocate the BNN Target Buffer
    out_sycl = zeros(UInt64, channels_int64, w, h, batch_size)

    # 3. Run the AVX-512 Engine
    pack_fp32_to_uint64!(out_sycl, input_fp32)

    # 4. Pure Native Julia Ground Truth
    println("Calculating native Julia bit-packing ground truth...")
    out_expected = zeros(UInt64, channels_int64, w, h, batch_size)

    for b in 1:batch_size
        for y in 1:h
            for x in 1:w
                for c in 1:channels
                    # The Mathematical Sign Function
                    if input_fp32[c, x, y, b] > 0.0f0
                        
                        # Find which UInt64 block this channel belongs to
                        c_blk = div(c - 1, 64) + 1
                        
                        # Find the exact bit position inside that 64-bit block
                        bit_pos = (c - 1) % 64
                        
                        # Flip the bit to 1
                        out_expected[c_blk, x, y, b] |= (UInt64(1) << bit_pos)
                    end
                end
            end
        end
    end

    @test out_sycl == out_expected
    println("✅ Sign function and 64-bit packing perfectly match native Julia logic!")
end
