#include "kernel_linear.hpp"
#include "kernel_linear_server.hpp" 
#include <torch/extension.h>

// 1. THE HARDWARE SINGLETON
// This guarantees the hardware is only probed ONCE, and the queue stays hot forever.
sycl::queue& get_queue() {
    // You can swap this to sycl::cpu_selector_v or sycl::gpu_selector_v to force a specific device
    static sycl::queue q{sycl::default_selector_v}; 
    return q;
}

// 2. IN-PLACE DEVICE ENGINE
// Notice we don't return a Tensor. We pass 'outputs' IN as an argument.
void bnn_linear_forward_device_out(at::Tensor inputs, at::Tensor weights, at::Tensor thresholds, at::Tensor outputs) {
    TORCH_CHECK(inputs.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(weights.dim() == 2, "Weights must be 2D");
    TORCH_CHECK(outputs.is_contiguous(), "Outputs must be contiguous");
    
    int batch_size = inputs.size(0);
    int in_int64s = inputs.size(1); 
    int out_features = weights.size(0);

    launch_binary_linear_fused(
        get_queue(),
        reinterpret_cast<const uint64_t*>(inputs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(weights.data_ptr<int64_t>()),
        reinterpret_cast<const int32_t*>(thresholds.data_ptr<int32_t>()),
        reinterpret_cast<uint64_t*>(outputs.data_ptr<int64_t>()),
        batch_size, in_int64s, out_features
    );
    
    get_queue().wait(); // Synchronize before returning to Python
}

// 3. IN-PLACE SERVER ENGINE
void bnn_linear_forward_server_out(at::Tensor inputs, at::Tensor weights, at::Tensor thresholds, at::Tensor outputs) {
    TORCH_CHECK(inputs.dim() == 2, "Inputs must be 2D");
    
    int batch_size = inputs.size(0);
    int in_int64s = inputs.size(1); 
    int out_features = weights.size(0);

    launch_binary_linear_server(
        get_queue(),
        reinterpret_cast<const uint64_t*>(inputs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(weights.data_ptr<int64_t>()),
        reinterpret_cast<const int32_t*>(thresholds.data_ptr<int32_t>()),
        reinterpret_cast<uint64_t*>(outputs.data_ptr<int64_t>()),
        batch_size, in_int64s, out_features
    );
    
    get_queue().wait();
}

PYBIND11_MODULE(bnn_pytorch_ext, m) {
    m.def("linear_forward_device_out", &bnn_linear_forward_device_out, "BNN Linear - SIMD Broadcast (In-Place)");
    m.def("linear_forward_server_out", &bnn_linear_forward_server_out, "BNN Linear - SLM Tiled (In-Place)");
}
