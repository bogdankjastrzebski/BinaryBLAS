// include/kernel_linear.hpp
#pragma once
#include <sycl/sycl.hpp>

constexpr int VEC_SIZE = 4;

class FusedBinaryLinear; // Forward declaration for kernel naming

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
    int out_int64s = (out_features + 63) / 64; // Ceiling division

    q.parallel_for<FusedBinaryLinear>(sycl::range<1>(batch_size), [=](sycl::id<1> idx) {
        int batch_idx = idx[0];
        
        // Pointers for this specific batch
        const uint64_t* my_input = &inputs[batch_idx * in_int64s];
        uint64_t* my_output = &outputs[batch_idx * out_int64s];

        // Iterate over output features (neurons)
        for (int out_idx = 0; out_idx < out_features; ++out_idx) {
            const uint64_t* my_weights = &weights[out_idx * in_int64s];
            int popcount_sum = 0;

            // --- The AVX-512 Core Loop ---
            // Process 512 bits (VEC_SIZE * 64) per iteration
            int i = 0;
            for (; i <= in_int64s - VEC_SIZE; i += VEC_SIZE) {
                sycl::vec<uint64_t, VEC_SIZE> in_v;
                sycl::vec<uint64_t, VEC_SIZE> w_v;
                
                in_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(my_input + i));
                w_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(my_weights + i));

                // XNOR
                sycl::vec<uint64_t, VEC_SIZE> xnor_v = ~(in_v ^ w_v);
                
                // Vectorized Popcount
                sycl::vec<uint64_t, VEC_SIZE> pops = sycl::popcount(xnor_v);
                
                // Horizontal reduction (Sum the 8 popcounts)
                popcount_sum += pops.s0() + pops.s1() + pops.s2() + pops.s3();
            }

            // Scalar tail loop (if in_int64s is not a multiple of 8)
            for (; i < in_int64s; ++i) {
                popcount_sum += sycl::popcount(~(my_input[i] ^ my_weights[i]));
            }

            // --- Fusion: Threshold & Pack ---
            uint64_t bit = (popcount_sum > thresholds[out_idx]) ? 1ULL : 0ULL;
            
            int out_pack_idx = out_idx / 64;
            int bit_pos = out_idx % 64;

            // Write the bit to the correct position
            if (bit_pos == 0) my_output[out_pack_idx] = 0; // Initialize new block
            my_output[out_pack_idx] |= (bit << bit_pos);
        }
    }).wait(); // For profiling, we block. In production, remove .wait() for async.
}
