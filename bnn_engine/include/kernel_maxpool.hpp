#pragma once
#include <sycl/sycl.hpp>

class BinaryMaxPool2d;

// Memory Format assumed: NCHW where C is packed into uint64_t
// Shape: [Batch, Channels_in_int64s, Height, Width]

inline void launch_binary_maxpool2d(
    sycl::queue& q,
    const uint64_t* inputs, 
    uint64_t* outputs, 
    int batch_size,
    int channels_in_int64s, 
    int height,
    int width,
    int kernel_size = 2,
    int stride = 2)
{
    int out_height = height / stride;
    int out_width = width / stride;
    
    // Total parallel threads: 1 thread per output spatial pixel per packed channel block
    sycl::range<3> global_range(batch_size * channels_in_int64s, out_height, out_width);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<BinaryMaxPool2d>(global_range, [=](sycl::id<3> idx) {
            int bc = idx[0]; // Fused Batch and Channel Block
            int oh = idx[1]; // Output Y
            int ow = idx[2]; // Output X

            int b = bc / channels_in_int64s;
            int c_blk = bc % channels_in_int64s;

            int in_h_start = oh * stride;
            int in_w_start = ow * stride;

            // 0 is the mathematical minimum in bitwise OR logic (all bits 0)
            uint64_t max_val = 0; 

            // 2x2 spatial window pooling using pure Bitwise OR
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = in_h_start + kh;
                    int iw = in_w_start + kw;
                    
                    if (ih < height && iw < width) {
                        // Standard NCHW flat index calculation
                        int in_idx = b * (channels_in_int64s * height * width) +
                                     c_blk * (height * width) +
                                     ih * width + iw;
                        
                        max_val |= inputs[in_idx];
                    }
                }
            }

            // Write to output tensor
            int out_idx = b * (channels_in_int64s * out_height * out_width) +
                          c_blk * (out_height * out_width) +
                          oh * out_width + ow;
            
            outputs[out_idx] = max_val;
        });
    });
}
