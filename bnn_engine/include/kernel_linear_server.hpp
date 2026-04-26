// include/kernel_linear_server.hpp
#pragma once
#include <sycl/sycl.hpp>

class FusedBinaryLinearServer;

void launch_binary_linear_server(
    sycl::queue& q,
    const uint64_t* inputs,
    const uint64_t* weights,
    const int32_t* thresholds,
    uint64_t* outputs,
    int batch_size,
    int in_int64s,
    int out_features)
{
    int out_int64s = (out_features + 63) / 64; 

    // Work-Group Size: 64 Threads.
    int local_b = 64;
    int global_b = ((batch_size + local_b - 1) / local_b) * local_b; // Pad to multiple of 64

    sycl::range<2> global_range(out_int64s, global_b);
    sycl::range<2> local_range(1, local_b);
    sycl::nd_range<2> nd_range(global_range, local_range);

    q.submit([&](sycl::handler& cgh) {
        // Allocate 32KB of Ultra-Fast Shared Local Memory (SLM)
        sycl::local_accessor<uint64_t, 1> slm_weights(sycl::range<1>(64 * in_int64s), cgh);

        cgh.parallel_for<FusedBinaryLinearServer>(nd_range, [=](sycl::nd_item<2> item) {
            int out_block = item.get_global_id(0);
            int b = item.get_global_id(1);
            int local_id = item.get_local_id(1);

            int start_out_idx = out_block * 64;

            // ==========================================
            // STEP 1: COOPERATIVE MEMORY PREFETCH
            // ==========================================
            int out_idx_to_load = start_out_idx + local_id;
            if (out_idx_to_load < out_features) {
                for (int i = 0; i < in_int64s; ++i) {
                    slm_weights[local_id * in_int64s + i] = weights[out_idx_to_load * in_int64s + i];
                }
            }

            // ==========================================
            // STEP 2: THE HARDWARE BARRIER
            // ==========================================
            sycl::group_barrier(item.get_group());

            // ==========================================
            // STEP 3: HIGH-SPEED MATH (Auto-Vectorized)
            // ==========================================
            if (b < batch_size) {
                uint64_t packed_out = 0;
                int end_out_idx = (start_out_idx + 64 > out_features) ? out_features : (start_out_idx + 64);
                
                const uint64_t* my_input = &inputs[b * in_int64s];

                for (int out_idx = start_out_idx; out_idx < end_out_idx; ++out_idx) {
                    int popcount_sum = 0;
                    int slm_row = out_idx - start_out_idx;
                    
                    // We drop the explicit sycl::vec and multi_ptr cast here. 
                    // The Intel compiler will flawlessly AVX2 auto-vectorize this loop.
                    for (int i = 0; i < in_int64s; ++i) {
                        uint64_t in_val = my_input[i];
                        uint64_t w_val = slm_weights[slm_row * in_int64s + i];
                        popcount_sum += sycl::popcount(~(in_val ^ w_val));
                    }

                    if (popcount_sum > thresholds[out_idx]) {
                        int bit_pos = out_idx % 64;
                        packed_out |= (1ULL << bit_pos);
                    }
                }

                outputs[b * out_int64s + out_block] = packed_out;
            }
        });
    }).wait(); 
}
