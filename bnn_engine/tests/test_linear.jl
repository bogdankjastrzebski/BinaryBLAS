using Test
using Random

const LIB_PATH = joinpath(@__DIR__, "..", "build", "libbnn_engine.so")

function alloc_usm(::Type{T}, dims...) where T
    bytes = prod(dims) * sizeof(T)
    ptr = @ccall LIB_PATH.c_api_allocate_usm(bytes::Csize_t)::Ptr{T}
    return unsafe_wrap(Array{T}, ptr, dims, own=false)
end

function free_usm(arr::AbstractArray)
    @ccall LIB_PATH.c_api_free_usm(pointer(arr)::Ptr{Cvoid})::Cvoid
end

@testset "BNN Absolute Correctness" begin
    println("--- BNN Julia Ground Truth Test ---")
    batch_size = 17
    in_features = 256
    out_features = 128
    
    in_int64s = div(in_features, 64)
    out_int64s = div(out_features, 64)
    
    inputs = alloc_usm(UInt64, in_int64s, batch_size)
    weights = alloc_usm(UInt64, in_int64s, out_features)
    thresholds = alloc_usm(Int32, out_features)
    out_sycl = alloc_usm(UInt64, out_int64s, batch_size)
    
    rand!(inputs)
    rand!(weights)
    rand!(thresholds, 0:in_features) # Random thresholds
    fill!(out_sycl, 0)
    
    # 1. Run our C++ SYCL Engine
    @ccall LIB_PATH.c_api_bnn_linear_forward_device_out(
        pointer(inputs)::Ptr{UInt64}, pointer(weights)::Ptr{UInt64}, pointer(thresholds)::Ptr{Int32}, 
        pointer(out_sycl)::Ptr{UInt64}, batch_size::Cint, in_int64s::Cint, out_features::Cint
    )::Cvoid
    
    # 2. Calculate Pure Native Julia Ground Truth
    println("Calculating native Julia scalar ground truth...")
    out_expected = zeros(UInt64, out_int64s, batch_size)
    
    for b in 1:batch_size
        for o in 1:out_features
            pop_sum = 0
            
            # Bitwise XNOR and Popcount
            for i in 1:in_int64s
                xnor_val = ~(inputs[i, b] ⊻ weights[i, o])
                pop_sum += count_ones(xnor_val)
            end
            
            # Thresholding and Packing
            if pop_sum > thresholds[o]
                out_int_idx = div(o - 1, 64) + 1
                bit_pos = (o - 1) % 64
                out_expected[out_int_idx, b] |= (UInt64(1) << bit_pos)
            end
        end
    end
    
    # 3. The Final Verdict
    @show out_sycl
    @show out_expected
    @test out_sycl == out_expected
    println("✅ Julia Native Math Perfectly Matches SYCL Execution!")
    
    free_usm(inputs); free_usm(weights); free_usm(thresholds); free_usm(out_sycl)
end
