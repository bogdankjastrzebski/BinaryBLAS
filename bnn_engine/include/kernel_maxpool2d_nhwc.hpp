#pragma once
#include <sycl/sycl.hpp>

class BinaryMaxPool2d_NHWC;

// STRICT NHWC LAYOUT: [Batch, Height, Width, Channels_int64]
inline void launch_binary_maxpool2d_nhwc(
    sycl::queue& q,
    const uint64_t* inputs, 
    uint64_t* outputs, 
    int batch_size,
    int channels_int64, 
    int in_h, int in_w,
    int kernel_size = 2,
    int stride = 2)
{
    int out_h = in_h / stride;
    int out_w = in_w / stride;
    
    // Grid: [Batch * Out_H, Out_W, Channels_int64]
    sycl::range<3> global_range(batch_size * out_h, out_w, channels_int64);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<BinaryMaxPool2d_NHWC>(global_range, [=](sycl::id<3> idx) {
            int b_oh = idx[0];
            int ow = idx[1];
            int c = idx[2];

            int b = b_oh / out_h;
            int oh = b_oh % out_h;

            int in_h_start = oh * stride;
            int in_w_start = ow * stride;

            uint64_t max_val = 0; 

            // 2x2 spatial window bitwise OR
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = in_h_start + kh;
                    int iw = in_w_start + kw;
                    
                    if (ih < in_h && iw < in_w) {
                        int in_idx = b * (in_h * in_w * channels_int64) +
                                     ih * (in_w * channels_int64) + 
                                     iw * channels_int64 + c;
                        
                        max_val |= inputs[in_idx];
                    }
                }
            }

            int out_idx = b * (out_h * out_w * channels_int64) +
                          oh * (out_w * channels_int64) + 
                          ow * channels_int64 + c;
            
            outputs[out_idx] = max_val;
        });
    });
}
