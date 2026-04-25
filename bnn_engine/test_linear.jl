# test_linear.jl
using Printf

# Dynamically find the .so file in the build directory
const LIB_PATH = joinpath(@__DIR__, "build", "libbnn_engine.so")

function bnn_linear_forward(inputs::Matrix{UInt64}, weights::Matrix{UInt64}, thresholds::Vector{Int32})
    # Because Julia is Column-Major, the FIRST dimension is contiguous in memory.
    # We allocate as (features, batch) so C++ sees it as contiguous [batch][features].
    in_int64s, batch_size = size(inputs)
    _, out_features = size(weights)
    
    out_int64s = cld(out_features, 64) # Ceiling division
    
    # Allocate output matrix (Column-Major: out_int64s x batch_size)
    outputs = Matrix{UInt64}(undef, out_int64s, batch_size)
    
    # The magical zero-overhead C-call
    @ccall LIB_PATH.c_api_bnn_linear_forward(
        inputs::Ptr{UInt64}, 
        weights::Ptr{UInt64}, 
        thresholds::Ptr{Int32}, 
        outputs::Ptr{UInt64}, 
        batch_size::Cint, 
        in_int64s::Cint, 
        out_features::Cint
    )::Cvoid
    
    return outputs
end

function main()
    println("--- BNN SYCL Engine Test (Julia) ---")
    
    batch_size = 4
    in_features = 256
    out_features = 128
    
    in_int64s = div(in_features, 64)
    
    println("Allocating memory...")
    # Generate random 64-bit integers natively
    inputs = rand(UInt64, in_int64s, batch_size)
    weights = rand(UInt64, in_int64s, out_features)
    thresholds = zeros(Int32, out_features)
    
    println("Inputs Shape (Julia layout):  $(size(inputs))")
    println("Weights Shape (Julia layout): $(size(weights))")
    
    println("\nLaunching SYCL AVX-512 Kernel via @ccall...")
    
    # Fire the kernel
    outputs = bnn_linear_forward(inputs, weights, thresholds)
    
    println("Kernel execution successful!")
    println("Outputs Shape (Julia layout): $(size(outputs))")
    
    println("\nFirst batch output snippet (Packed bits in Hex):")
    # Print the first block of the first batch in hex so we can see the bits
    @printf("0x%016X\n", outputs[1, 1])
end

main()
