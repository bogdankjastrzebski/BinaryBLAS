# test_linear.jl
using Printf

const LIB_PATH = joinpath(@__DIR__, "build", "libbnn_engine.so")

# Device API
function bnn_linear_forward_device(inputs::Matrix{UInt64}, weights::Matrix{UInt64}, thresholds::Vector{Int32})
    in_int64s, batch_size = size(inputs)
    _, out_features = size(weights)
    out_int64s = cld(out_features, 64) 
    outputs = Matrix{UInt64}(undef, out_int64s, batch_size)
    
    @ccall LIB_PATH.c_api_bnn_linear_forward_device(
        inputs::Ptr{UInt64}, weights::Ptr{UInt64}, thresholds::Ptr{Int32}, 
        outputs::Ptr{UInt64}, batch_size::Cint, in_int64s::Cint, out_features::Cint
    )::Cvoid
    
    return outputs
end

# Server API
function bnn_linear_forward_server(inputs::Matrix{UInt64}, weights::Matrix{UInt64}, thresholds::Vector{Int32})
    in_int64s, batch_size = size(inputs)
    _, out_features = size(weights)
    out_int64s = cld(out_features, 64) 
    outputs = Matrix{UInt64}(undef, out_int64s, batch_size)
    
    @ccall LIB_PATH.c_api_bnn_linear_forward_server(
        inputs::Ptr{UInt64}, weights::Ptr{UInt64}, thresholds::Ptr{Int32}, 
        outputs::Ptr{UInt64}, batch_size::Cint, in_int64s::Cint, out_features::Cint
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
    outputs = bnn_linear_forward_device(inputs, weights, thresholds)
    println(sum(inputs))
    println(sum(weights))
    println(sum(outputs))
    println(sum(thresholds))
    
    println("Kernel execution successful!")
    println("Outputs Shape (Julia layout): $(size(outputs))")

    println("\nFirst batch output snippet (Packed bits in Hex):")
    # Print the first block of the first batch in hex so we can see the bits
    @printf("0x%016X\n", outputs[1, 1])
    println()

    outputs = bnn_linear_forward_server(inputs, weights, thresholds)
    println(sum(inputs))
    println(sum(weights))
    println(sum(outputs))
    println(sum(thresholds))

    println("Kernel execution successful!")
    println("Outputs Shape (Julia layout): $(size(outputs))")
    
    println("\nFirst batch output snippet (Packed bits in Hex):")
    # Print the first block of the first batch in hex so we can see the bits
    @printf("0x%016X\n", outputs[1, 1])
end

main()
