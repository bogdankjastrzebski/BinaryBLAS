module BinaryBLAS

export print_hardware_info, linear_forward_device!, conv2d_nhwc_device!, pack_fp32_to_uint64!

const LIB_PATH = joinpath(@__DIR__,"..", "..", "..", "bnn_engine", "build", "libbnn_engine.so")

function print_hardware_info()
    @ccall LIB_PATH.c_api_print_hardware_info()::Cvoid
end

"""
    linear_forward_device!(outputs, inputs, weights, thresholds)

Performs an in-place Binarized Neural Network linear layer forward pass using AVX-512.
"""
function linear_forward_device!(
    outputs::Matrix{UInt64}, 
    inputs::Matrix{UInt64}, 
    weights::Matrix{UInt64}, 
    thresholds::Vector{Int32}
)
    # Julia matrices are Column-Major: size is (Rows, Columns)
    in_int64s = size(inputs, 1)
    batch_size = size(inputs, 2)
    out_features = size(weights, 2)

    # Idiomatic Julia bounds checking
    @assert size(weights, 1) == in_int64s "Weight input dimension mismatch"
    @assert size(outputs, 2) == batch_size "Output batch size mismatch"
    @assert size(outputs, 1) == cld(out_features, 64) "Output feature dimension mismatch"
    @assert length(thresholds) == out_features "Threshold length mismatch"

    @ccall LIB_PATH.c_api_bnn_linear_forward_device_out(
        pointer(inputs)::Ptr{UInt64}, 
        pointer(weights)::Ptr{UInt64}, 
        pointer(thresholds)::Ptr{Int32}, 
        pointer(outputs)::Ptr{UInt64}, 
        batch_size::Cint, 
        in_int64s::Cint, 
        out_features::Cint
    )::Cvoid
    
    return outputs
end

"""
    conv2d_nhwc_device!(outputs, inputs, weights, thresholds; stride=1, padding=0)

Performs in-place BNN 2D Convolution using AVX-512 Implicit GEMM.
Julia Layout -> C++ NHWC mapping:
Inputs:  (C_int64s, W, H, Batch)
Weights: (C_int64s, KW, KH, Out_C)
Outputs: (Out_C_int64s, OW, OH, Batch)
"""
function conv2d_nhwc_device!(
    outputs::Array{UInt64, 4}, inputs::Array{UInt64, 4}, 
    weights::Array{UInt64, 4}, thresholds::Vector{Int32};
    stride::Int = 1, padding::Int = 0
)
    in_channels_int64 = size(inputs, 1)
    in_w = size(inputs, 2)
    in_h = size(inputs, 3)
    batch_size = size(inputs, 4)

    kernel_w = size(weights, 2)
    kernel_h = size(weights, 3)
    out_channels = length(thresholds)

    @assert size(weights, 1) == in_channels_int64 "Weight input channel mismatch"
    @assert kernel_w == kernel_h "Only square kernels supported currently"

    @ccall LIB_PATH.c_api_bnn_conv2d_nhwc_device_out(
        pointer(inputs)::Ptr{UInt64}, pointer(weights)::Ptr{UInt64}, pointer(thresholds)::Ptr{Int32}, 
        pointer(outputs)::Ptr{UInt64}, batch_size::Cint, in_channels_int64::Cint, out_channels::Cint,
        in_h::Cint, in_w::Cint, kernel_h::Cint, stride::Cint, padding::Cint
    )::Cvoid
    
    return outputs
end

"""
    pack_fp32_to_uint64!(output::Array{UInt64, 4}, input::Array{Float32, 4})

Binarizes (Sign > 0) and packs standard FP32 features into the UInt64 BNN layout.
Input Layout: (Channels, Width, Height, Batch)
Output Layout: (Channels_int64s, Width, Height, Batch)
"""
function pack_fp32_to_uint64!(output::Array{UInt64, 4}, input::Array{Float32, 4})
    channels = size(input, 1)
    in_w = size(input, 2)
    in_h = size(input, 3)
    batch_size = size(input, 4)

    spatial_size = in_w * in_h
    channels_int64 = cld(channels, 64)

    # Safety checks
    @assert size(output, 1) == channels_int64 "Output channel dimension mismatch"
    @assert size(output, 2) == in_w "Output width mismatch"
    @assert size(output, 3) == in_h "Output height mismatch"
    @assert size(output, 4) == batch_size "Output batch size mismatch"

    @ccall LIB_PATH.c_api_bnn_pack_fp32_to_uint64(
        pointer(input)::Ptr{Float32},
        pointer(output)::Ptr{UInt64},
        batch_size::Cint,
        spatial_size::Cint,
        channels::Cint
    )::Cvoid
    
    return output
end

end # module
