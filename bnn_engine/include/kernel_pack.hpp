#pragma once
#include <sycl/sycl.hpp>

class BinarizeAndPackNHWC;

// Input:  [Batch, H, W, Channels] (Standard FP32)
// Output: [Batch, H, W, Channels_int64] (Packed BNN format)

inline void launch_binarize_pack_nhwc(
    sycl::queue& q,
    const float* input,
    uint64_t* output,
    int batch_size,
    int spatial_size, // Height * Width
    int channels)
{
    int channels_int64 = (channels + 63) / 64;
    int total_pixels = batch_size * spatial_size;

    // Grid: [Total Spatial Pixels, Channel Blocks (uint64_t chunks)]
    sycl::range<2> global_range(total_pixels, channels_int64);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<BinarizeAndPackNHWC>(global_range, [=](sycl::id<2> idx) {
            int pixel_idx = idx[0];
            int c_blk = idx[1];

            int start_c = c_blk * 64;
            int end_c = (start_c + 64 > channels) ? channels : (start_c + 64);

            uint64_t packed = 0;
            
            // Pointer to the exact pixel in the FP32 NHWC tensor
            const float* pixel_in = &input[pixel_idx * channels];

            // Compress 64 contiguous floats into 1 integer
            for (int c = start_c; c < end_c; ++c) {
                // Binarization: Sign function (val > 0 becomes 1, else 0)
                if (pixel_in[c] > 0.0f) {
                    int bit_pos = c - start_c;
                    packed |= (1ULL << bit_pos);
                }
            }

            // Write to the BNN tensor
            output[pixel_idx * channels_int64 + c_blk] = packed;
        });
    });
}
