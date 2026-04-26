// include/kernel_linear.hpp
#pragma once
#include <sycl/sycl.hpp>

constexpr int VEC_SIZE = 4;

class FusedBinaryLinearTransposed; // Renamed for clarity

void launch_binary_linear_fused(
    sycl::queue& q,
    const uint64_t* inputs,     // Packed input features [Batch, In_Int64s]
    const uint64_t* weights,    // Packed weights [OutFeatures, In_Int64s]
    const int* thresholds,      // Pre-computed Batch Norm thresholds [OutFeatures]
    uint64_t* outputs,          // Packed outputs [Batch, Out_Int64s]
    int batch_size,
    int in_int64s,              // Number of uint64_t per input feature
    int out_features)           // Total number of output neurons
{
    int out_int64s = (out_features + 63) / 64; 

    // --- THE FIX: The Transposed Grid ---
    // Dimension 0: Output Blocks (Slow moving)
    // Dimension 1: Batch Size (Fast moving -> Executes simultaneously in SIMD lanes)
    sycl::range<2> global_range(out_int64s, batch_size);

    q.parallel_for<FusedBinaryLinearTransposed>(global_range, [=](sycl::id<2> idx) {
        int out_block = idx[0];
        int b = idx[1];

        uint64_t packed_out = 0;
        
        int start_out_idx = out_block * 64;
        int end_out_idx = (start_out_idx + 64 > out_features) ? out_features : (start_out_idx + 64);

        // All threads in this SIMD lane will loop over this exact same chunk of weights
        for (int out_idx = start_out_idx; out_idx < end_out_idx; ++out_idx) {
            
            int popcount_sum = 0;
            const uint64_t* my_input = &inputs[b * in_int64s];
            const uint64_t* my_weights = &weights[out_idx * in_int64s]; // <--- Broadcast Read!

            int i = 0;
            for (; i <= in_int64s - VEC_SIZE; i += VEC_SIZE) {
                sycl::vec<uint64_t, VEC_SIZE> in_v, w_v;
                
                in_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(my_input + i));
                w_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(my_weights + i));

                sycl::vec<uint64_t, VEC_SIZE> xnor_v = ~(in_v ^ w_v);
                sycl::vec<uint64_t, VEC_SIZE> pops = sycl::popcount(xnor_v);
                popcount_sum += pops.s0() + pops.s1() + pops.s2() + pops.s3();
            }

            for (; i < in_int64s; ++i) {
                popcount_sum += sycl::popcount(~(my_input[i] ^ my_weights[i]));
            }

            if (popcount_sum > thresholds[out_idx]) {
                int bit_pos = out_idx % 64;
                packed_out |= (1ULL << bit_pos);
            }
        }

        // Write safely to separate cache lines (No False Sharing)
        outputs[b * out_int64s + out_block] = packed_out;

    }).wait(); 
}
