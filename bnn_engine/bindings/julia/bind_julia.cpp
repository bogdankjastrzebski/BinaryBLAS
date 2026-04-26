#include "kernel_linear.hpp"
#include "kernel_linear_server.hpp"
#include <sycl/sycl.hpp>

// Hardware Singleton
sycl::queue& get_queue() {
    static sycl::queue q{sycl::default_selector_v}; 
    return q;
}

extern "C" {
    // USM Allocators for Julia
    void* c_api_allocate_usm(size_t bytes) {
        return sycl::malloc_shared(bytes, get_queue());
    }

    void c_api_free_usm(void* ptr) {
        sycl::free(ptr, get_queue());
    }

    // IN-PLACE DEVICE ENGINE
    void c_api_bnn_linear_forward_device_out(
        const uint64_t* inputs, const uint64_t* weights, const int32_t* thresholds, 
        uint64_t* outputs, int batch_size, int in_int64s, int out_features) 
    {
        launch_binary_linear_fused(get_queue(), inputs, weights, thresholds, outputs, batch_size, in_int64s, out_features);
        get_queue().wait(); 
    }

    // IN-PLACE SERVER ENGINE
    void c_api_bnn_linear_forward_server_out(
        const uint64_t* inputs, const uint64_t* weights, const int32_t* thresholds, 
        uint64_t* outputs, int batch_size, int in_int64s, int out_features) 
    {
        launch_binary_linear_server(get_queue(), inputs, weights, thresholds, outputs, batch_size, in_int64s, out_features);
        get_queue().wait();
    }
}
