#include "kernel_linear.hpp"
#include "kernel_linear_server.hpp"
#include "kernel_conv2d_nhwc.hpp"
#include "kernel_pack.hpp"
#include <sycl/sycl.hpp>
#include <iostream>


// Hardware Singleton - FORCED TO CPU for AVX-512 Micro-Kernel
sycl::queue& get_queue() {
    static sycl::queue q{sycl::cpu_selector_v}; 
    return q;
}

extern "C" {
    // Hardware Introspection
    void c_api_print_hardware_info() {
        std::cout << "[SYCL Julia Backend] Executing on: " 
                  << get_queue().get_device().get_info<sycl::info::device::name>() 
                  << std::endl;
    }

    // USM Allocators
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

    // IN-PLACE NHWC CONVOLUTION
    void c_api_bnn_conv2d_nhwc_device_out(
        const uint64_t* inputs, const uint64_t* weights, const int32_t* thresholds, 
        uint64_t* outputs, int batch_size, int in_channels_int64, int out_channels,
        int in_h, int in_w, int kernel_size, int stride, int padding) 
    {
        launch_binary_conv2d_nhwc(get_queue(), inputs, weights, thresholds, outputs, 
                                  batch_size, in_channels_int64, out_channels, 
                                  in_h, in_w, kernel_size, stride, padding);
        get_queue().wait();
    }

    // FP32 to UInt64 Binarize and Pack
    void c_api_bnn_pack_fp32_to_uint64(
        const float* input, uint64_t* output, 
        int batch_size, int spatial_size, int channels) 
    {
        launch_binarize_pack_nhwc(get_queue(), input, output, batch_size, spatial_size, channels);
        get_queue().wait();
    }

}
