// bindings/julia/bind_julia.cpp
#include "kernel_linear.hpp"
#include "kernel_linear_server.hpp"
#include <sycl/sycl.hpp>

// Create a global queue for the library to use
sycl::queue global_q;

extern "C" {

    // 1. THE ENTERPRISE ALLOCATOR
    // Julia calls this to get hardware-safe memory
    void* c_api_allocate_usm(size_t bytes) {
        // malloc_shared creates memory that BOTH the CPU and GPU can read/write simultaneously
        return sycl::malloc_shared(bytes, global_q);
    }

    // 2. THE DEALLOCATOR
    void c_api_free_usm(void* ptr) {
        sycl::free(ptr, global_q);
    }

    // 3. ZERO-COPY DEVICE INFERENCE
    void c_api_bnn_linear_forward_device(
        const uint64_t* inputs, const uint64_t* weights, const int32_t* thresholds, 
        uint64_t* outputs, int batch_size, int in_int64s, int out_features) 
    {
        // NO MALLOC. NO MEMCPY. Bare metal pointer execution.
        launch_binary_linear_fused(global_q, inputs, weights, thresholds, outputs, batch_size, in_int64s, out_features);
        global_q.wait(); // Only wait for the math to finish
    }

    // 4. ZERO-COPY SERVER INFERENCE
    void c_api_bnn_linear_forward_server(
        const uint64_t* inputs, const uint64_t* weights, const int32_t* thresholds, 
        uint64_t* outputs, int batch_size, int in_int64s, int out_features) 
    {
        launch_binary_linear_server(global_q, inputs, weights, thresholds, outputs, batch_size, in_int64s, out_features);
        global_q.wait();
    }
}
