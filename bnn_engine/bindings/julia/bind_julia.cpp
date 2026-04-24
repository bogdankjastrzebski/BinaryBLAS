// bindings/julia/bind_julia.cpp
#include <sycl/sycl.hpp>
#include "kernel_linear.hpp"

static sycl::queue global_q(sycl::cpu_selector_v);

extern "C" {
    // A flat C-API that Julia can call via @ccall
    void c_api_bnn_linear_forward(
        const uint64_t* inputs,
        const uint64_t* weights,
        const int* thresholds,
        uint64_t* outputs,
        int batch_size,
        int in_int64s,
        int out_features) 
    {
        launch_binary_linear_fused(global_q, inputs, weights, thresholds, outputs, 
                                   batch_size, in_int64s, out_features);
    }
}
