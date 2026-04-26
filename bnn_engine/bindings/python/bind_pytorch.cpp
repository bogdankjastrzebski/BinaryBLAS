#include "kernel_linear.hpp"
#include "kernel_linear_server.hpp" 
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

PYBIND11_MODULE(bnn_pytorch_ext, m) {
    m.def("linear_forward_device_out", &bnn_linear_forward_device_out, "BNN Linear - SIMD Broadcast (In-Place)");
    m.def("linear_forward_server_out", &bnn_linear_forward_server_out, "BNN Linear - SLM Tiled (In-Place)");
}
