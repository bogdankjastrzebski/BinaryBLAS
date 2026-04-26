#include "kernel_linear.hpp"
#include "kernel_linear_server.hpp" 
#include "kernel_maxpool2d_nhwc.hpp"
#include "kernel_pack.hpp"
#include "kernel_conv2d_nhwc.hpp"
#include <torch/extension.h>

// 1. DUAL HARDWARE SINGLETONS
sycl::queue& get_cpu_queue() {
    static sycl::queue q{sycl::cpu_selector_v}; 
    return q;
}

sycl::queue& get_gpu_queue() {
    // Falls back to CPU if no GPU is found to prevent crashing
    static sycl::queue q{sycl::default_selector_v}; 
    return q;
}

// 2. INTELLIGENT DISPATCHER
sycl::queue& get_queue_for_tensor(const at::Tensor& t) {
    if (t.device().is_cpu()) {
        return get_cpu_queue();
    } else {
        return get_gpu_queue();
    }
}

void bnn_linear_forward_device_out(at::Tensor inputs, at::Tensor weights, at::Tensor thresholds, at::Tensor outputs) {
    TORCH_CHECK(inputs.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(weights.dim() == 2, "Weights must be 2D");
    TORCH_CHECK(outputs.is_contiguous(), "Outputs must be contiguous");
    
    int batch_size = inputs.size(0);
    int in_int64s = inputs.size(1); 
    int out_features = weights.size(0);

    // Automatically route to CPU or GPU based on the PyTorch Tensor!
    sycl::queue& q = get_queue_for_tensor(inputs);

    launch_binary_linear_fused(
        q,
        reinterpret_cast<const uint64_t*>(inputs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(weights.data_ptr<int64_t>()),
        reinterpret_cast<const int32_t*>(thresholds.data_ptr<int32_t>()),
        reinterpret_cast<uint64_t*>(outputs.data_ptr<int64_t>()),
        batch_size, in_int64s, out_features
    );
    
    q.wait(); 
}

void bnn_linear_forward_server_out(at::Tensor inputs, at::Tensor weights, at::Tensor thresholds, at::Tensor outputs) {
    TORCH_CHECK(inputs.dim() == 2, "Inputs must be 2D");
    
    int batch_size = inputs.size(0);
    int in_int64s = inputs.size(1); 
    int out_features = weights.size(0);

    sycl::queue& q = get_queue_for_tensor(inputs);

    launch_binary_linear_server(
        q,
        reinterpret_cast<const uint64_t*>(inputs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(weights.data_ptr<int64_t>()),
        reinterpret_cast<const int32_t*>(thresholds.data_ptr<int32_t>()),
        reinterpret_cast<uint64_t*>(outputs.data_ptr<int64_t>()),
        batch_size, in_int64s, out_features
    );
    
    q.wait();
}

void bnn_maxpool2d_device_out(at::Tensor inputs, at::Tensor outputs, int kernel_size, int stride) {
    TORCH_CHECK(inputs.dim() == 4, "Inputs must be 4D: [Batch, Channels/64, Height, Width]");
    
    int batch_size = inputs.size(0);
    int channels_in_int64s = inputs.size(1);
    int height = inputs.size(2);
    int width = inputs.size(3);

    sycl::queue& q = get_queue_for_tensor(inputs);

    launch_binary_maxpool2d_nhwc(
        q,
        reinterpret_cast<const uint64_t*>(inputs.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(outputs.data_ptr<int64_t>()),
        batch_size, channels_in_int64s, height, width, kernel_size, stride
    );
    
    q.wait();
}

void bnn_conv2d_nhwc_device_out(
    at::Tensor inputs, at::Tensor weights, at::Tensor thresholds, 
    at::Tensor outputs, int stride, int padding) 
{
    // Enforcing NHWC tensor shapes from Python:
    // inputs:  [Batch, In_H, In_W, In_C_int64s]
    // weights: [Out_C, Kernel_H, Kernel_W, In_C_int64s]
    // outputs: [Batch, Out_H, Out_W, Out_C_int64s]
    
    TORCH_CHECK(inputs.dim() == 4, "Inputs must be 4D NHWC");
    TORCH_CHECK(weights.dim() == 4, "Weights must be 4D NHWC");
    
    int batch_size = inputs.size(0);
    int in_h = inputs.size(1);
    int in_w = inputs.size(2);
    int in_channels_int64 = inputs.size(3);

    int out_channels = thresholds.size(0);
    int kernel_size = weights.size(1);

    sycl::queue& q = get_queue_for_tensor(inputs);

    launch_binary_conv2d_nhwc(
        q,
        reinterpret_cast<const uint64_t*>(inputs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(weights.data_ptr<int64_t>()),
        reinterpret_cast<const int32_t*>(thresholds.data_ptr<int32_t>()),
        reinterpret_cast<uint64_t*>(outputs.data_ptr<int64_t>()),
        batch_size, in_channels_int64, out_channels,
        in_h, in_w, kernel_size, stride, padding
    );
    
    q.wait();
}


void bnn_pack_fp32_to_uint64(at::Tensor input_fp32, at::Tensor output_bnn) {
    // Expected Layout: NHWC
    TORCH_CHECK(input_fp32.dim() == 4, "Input must be 4D NHWC");
    TORCH_CHECK(input_fp32.scalar_type() == at::kFloat, "Input must be Float32");
    
    int batch_size = input_fp32.size(0);
    int h = input_fp32.size(1);
    int w = input_fp32.size(2);
    int channels = input_fp32.size(3);
    
    int spatial_size = h * w;

    sycl::queue& q = get_queue_for_tensor(input_fp32);

    launch_binarize_pack_nhwc(
        q,
        input_fp32.data_ptr<float>(),
        reinterpret_cast<uint64_t*>(output_bnn.data_ptr<int64_t>()),
        batch_size, spatial_size, channels
    );
    
    q.wait();
}


PYBIND11_MODULE(bnn_pytorch_ext, m) {
    m.def("linear_forward_device_out", &bnn_linear_forward_device_out, "BNN Linear");
    m.def("linear_forward_server_out", &bnn_linear_forward_server_out, "BNN Linear SLM");
    m.def("maxpool2d_device_out", &bnn_maxpool2d_device_out, "BNN MaxPool2d");
    m.def("conv2d_nhwc_device_out", &bnn_conv2d_nhwc_device_out, "BNN Conv2D NHWC Implicit GEMM");
    m.def("pack_fp32_to_uint64", &bnn_pack_fp32_to_uint64, "Binarize and Pack FP32 to UInt64");
}
