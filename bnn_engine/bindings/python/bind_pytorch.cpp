// bindings/python/bind_pytorch.cpp
#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include "kernel_linear.hpp"

// Global SYCL Queue (Initialize once)
static sycl::queue global_q(sycl::cpu_selector_v);

// The Python-facing function
at::Tensor bnn_linear_forward(
    at::Tensor inputs,      // Int64 tensor
    at::Tensor weights,     // Int64 tensor
    at::Tensor thresholds)  // Int32 tensor
{
    int batch_size = inputs.size(0);
    int in_int64s = inputs.size(1);
    int out_features = thresholds.size(0);
    int out_int64s = (out_features + 63) / 64;

    // Allocate output tensor in PyTorch
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    at::Tensor outputs = torch::empty({batch_size, out_int64s}, options);

    // Extract raw pointers and launch SYCL
    launch_binary_linear_fused(
        global_q,
        reinterpret_cast<const uint64_t*>(inputs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(weights.data_ptr<int64_t>()),
        thresholds.data_ptr<int>(),
        reinterpret_cast<uint64_t*>(outputs.data_ptr<int64_t>()),
        batch_size,
        in_int64s,
        out_features
    );

    return outputs;
}

PYBIND11_MODULE(bnn_pytorch_ext, m) {
    m.def("linear_forward", &bnn_linear_forward, "BNN Fused Linear Forward (SYCL)");
}
