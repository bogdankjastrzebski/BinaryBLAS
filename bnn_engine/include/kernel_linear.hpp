// // include/kernel_linear.hpp
// #pragma once
// #include <sycl/sycl.hpp>
// 
// constexpr int VEC_SIZE = 4;
// 
// class FusedBinaryLinearTransposed; // Renamed for clarity
// 
// void launch_binary_linear_fused(
//     sycl::queue& q,
//     const uint64_t* inputs,     // Packed input features [Batch, In_Int64s]
//     const uint64_t* weights,    // Packed weights [OutFeatures, In_Int64s]
//     const int* thresholds,      // Pre-computed Batch Norm thresholds [OutFeatures]
//     uint64_t* outputs,          // Packed outputs [Batch, Out_Int64s]
//     int batch_size,
//     int in_int64s,              // Number of uint64_t per input feature
//     int out_features)           // Total number of output neurons
// {
//     int out_int64s = (out_features + 63) / 64; 
// 
//     // --- THE FIX: The Transposed Grid ---
//     // Dimension 0: Output Blocks (Slow moving)
//     // Dimension 1: Batch Size (Fast moving -> Executes simultaneously in SIMD lanes)
//     sycl::range<2> global_range(out_int64s, batch_size);
// 
//     q.parallel_for<FusedBinaryLinearTransposed>(global_range, [=](sycl::id<2> idx) {
//         int out_block = idx[0];
//         int b = idx[1];
// 
//         uint64_t packed_out = 0;
//         
//         int start_out_idx = out_block * 64;
//         int end_out_idx = (start_out_idx + 64 > out_features) ? out_features : (start_out_idx + 64);
// 
//         // All threads in this SIMD lane will loop over this exact same chunk of weights
//         for (int out_idx = start_out_idx; out_idx < end_out_idx; ++out_idx) {
//             
//             int popcount_sum = 0;
//             const uint64_t* my_input = &inputs[b * in_int64s];
//             const uint64_t* my_weights = &weights[out_idx * in_int64s]; // <--- Broadcast Read!
// 
//             int i = 0;
//             for (; i <= in_int64s - VEC_SIZE; i += VEC_SIZE) {
//                 sycl::vec<uint64_t, VEC_SIZE> in_v, w_v;
//                 
//                 in_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(my_input + i));
//                 w_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(my_weights + i));
// 
//                 sycl::vec<uint64_t, VEC_SIZE> xnor_v = ~(in_v ^ w_v);
//                 sycl::vec<uint64_t, VEC_SIZE> pops = sycl::popcount(xnor_v);
//                 popcount_sum += pops.s0() + pops.s1() + pops.s2() + pops.s3();
//             }
// 
//             for (; i < in_int64s; ++i) {
//                 popcount_sum += sycl::popcount(~(my_input[i] ^ my_weights[i]));
//             }
// 
//             if (popcount_sum > thresholds[out_idx]) {
//                 int bit_pos = out_idx % 64;
//                 packed_out |= (1ULL << bit_pos);
//             }
//         }
// 
//         // Write safely to separate cache lines (No False Sharing)
//         outputs[b * out_int64s + out_block] = packed_out;
// 
//     }).wait(); 
// }
// 
// include/kernel_linear.hpp
// #pragma once
// #include <sycl/sycl.hpp>
// 
// constexpr int VEC_SIZE = 4; // 4 * 64 bits = 256-bit AVX2 vector
// 
// class FusedBinaryLinearTransposed;
// 
// void launch_binary_linear_fused(
//     sycl::queue& q,
//     const uint64_t* inputs, 
//     const uint64_t* weights, 
//     const int32_t* thresholds, 
//     uint64_t* outputs, 
//     int batch_size,
//     int in_int64s, 
//     int out_features)
// {
//     int out_int64s = (out_features + 63) / 64; 
//     sycl::range<2> global_range(out_int64s, batch_size);
// 
//     q.submit([&](sycl::handler& cgh) {
//         cgh.parallel_for<FusedBinaryLinearTransposed>(global_range, [=](sycl::id<2> idx) {
//             int out_block = idx[0];
//             int b = idx[1];
// 
//             int start_out_idx = out_block * 64;
//             int end_out_idx = (start_out_idx + 64 > out_features) ? out_features : (start_out_idx + 64);
// 
//             uint64_t packed_out = 0;
//             const uint64_t* my_input = &inputs[b * in_int64s];
// 
//             for (int out_idx = start_out_idx; out_idx < end_out_idx; ++out_idx) {
//                 int popcount_sum = 0;
//                 const uint64_t* my_weight = &weights[out_idx * in_int64s];
//                 
//                 int i = 0;
//                 // ====================================================
//                 // THE EXPLICIT AVX2 VECTOR LOOP
//                 // We force the CPU to load 256 bits at a time.
//                 // ====================================================
//                 for (; i <= in_int64s - VEC_SIZE; i += VEC_SIZE) {
//                     sycl::vec<uint64_t, VEC_SIZE> in_v, w_v;
//                     
//                     // Aligned explicit 256-bit load from RAM
//                     in_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(my_input + i));
//                     w_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(my_weight + i));
// 
//                     // 256-bit simultaneous XNOR
//                     sycl::vec<uint64_t, VEC_SIZE> xnor_v = ~(in_v ^ w_v);
//                     
//                     // Hardware vector popcount
//                     sycl::vec<uint64_t, VEC_SIZE> pops = sycl::popcount(xnor_v);
//                     
//                     // Horizontal sum of the vector
//                     popcount_sum += pops.s0() + pops.s1() + pops.s2() + pops.s3();
//                 }
// 
//                 // Handle the fringe (if in_int64s isn't perfectly divisible by 4)
//                 for (; i < in_int64s; ++i) {
//                     popcount_sum += sycl::popcount(~(my_input[i] ^ my_weight[i]));
//                 }
// 
//                 if (popcount_sum > thresholds[out_idx]) {
//                     int bit_pos = out_idx % 64;
//                     packed_out |= (1ULL << bit_pos);
//                 }
//             }
// 
//             outputs[b * out_int64s + out_block] = packed_out;
//         });
//     });
// }

// include/kernel_linear.hpp
// #pragma once
// #include <sycl/sycl.hpp>
// 
// constexpr int VEC_SIZE = 4; // 256-bit AVX2 vectors
// constexpr int UNROLL_FACTOR = 4; // The Micro-Kernel Block Size
// 
// class FusedBinaryLinearTransposed;
// 
// void launch_binary_linear_fused(
//     sycl::queue& q,
//     const uint64_t* inputs, 
//     const uint64_t* weights, 
//     const int32_t* thresholds, 
//     uint64_t* outputs, 
//     int batch_size,
//     int in_int64s, 
//     int out_features)
// {
//     int out_int64s = (out_features + 63) / 64; 
//     sycl::range<2> global_range(out_int64s, batch_size);
// 
//     q.submit([&](sycl::handler& cgh) {
//         cgh.parallel_for<FusedBinaryLinearTransposed>(global_range, [=](sycl::id<2> idx) {
//             int out_block = idx[0];
//             int b = idx[1];
// 
//             int start_out_idx = out_block * 64;
//             int end_out_idx = (start_out_idx + 64 > out_features) ? out_features : (start_out_idx + 64);
// 
//             uint64_t packed_out = 0;
//             const uint64_t* my_input = &inputs[b * in_int64s];
// 
//             // ====================================================
//             // THE MICRO-KERNEL (1x4 Block)
//             // We step by 4 output features at a time.
//             // ====================================================
//             int out_idx = start_out_idx;
//             for (; out_idx <= end_out_idx - UNROLL_FACTOR; out_idx += UNROLL_FACTOR) {
//                 
//                 // 4 Independent Accumulators mapped directly to CPU Registers
//                 int pop_sum0 = 0, pop_sum1 = 0, pop_sum2 = 0, pop_sum3 = 0;
//                 
//                 const uint64_t* w0 = &weights[(out_idx + 0) * in_int64s];
//                 const uint64_t* w1 = &weights[(out_idx + 1) * in_int64s];
//                 const uint64_t* w2 = &weights[(out_idx + 2) * in_int64s];
//                 const uint64_t* w3 = &weights[(out_idx + 3) * in_int64s];
//                 
//                 int i = 0;
//                 for (; i <= in_int64s - VEC_SIZE; i += VEC_SIZE) {
//                     sycl::vec<uint64_t, VEC_SIZE> in_v, w0_v, w1_v, w2_v, w3_v;
//                     
//                     // LOAD ONCE: Read 256 bits of input
//                     in_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(my_input + i));
//                     
//                     // LOAD FOUR: Read 256 bits for 4 different weights
//                     w0_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(w0 + i));
//                     w1_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(w1 + i));
//                     w2_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(w2 + i));
//                     w3_v.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(w3 + i));
// 
//                     // INTERLEAVED EXECUTION: The compiler will overlap these bitwise ops
//                     sycl::vec<uint64_t, VEC_SIZE> x0 = ~(in_v ^ w0_v);
//                     sycl::vec<uint64_t, VEC_SIZE> x1 = ~(in_v ^ w1_v);
//                     sycl::vec<uint64_t, VEC_SIZE> x2 = ~(in_v ^ w2_v);
//                     sycl::vec<uint64_t, VEC_SIZE> x3 = ~(in_v ^ w3_v);
//                     
//                     sycl::vec<uint64_t, VEC_SIZE> p0 = sycl::popcount(x0);
//                     sycl::vec<uint64_t, VEC_SIZE> p1 = sycl::popcount(x1);
//                     sycl::vec<uint64_t, VEC_SIZE> p2 = sycl::popcount(x2);
//                     sycl::vec<uint64_t, VEC_SIZE> p3 = sycl::popcount(x3);
// 
//                     // ACCUMULATE
//                     pop_sum0 += p0.s0() + p0.s1() + p0.s2() + p0.s3();
//                     pop_sum1 += p1.s0() + p1.s1() + p1.s2() + p1.s3();
//                     pop_sum2 += p2.s0() + p2.s1() + p2.s2() + p2.s3();
//                     pop_sum3 += p3.s0() + p3.s1() + p3.s2() + p3.s3();
//                 }
// 
//                 // Handle Matrix Width Fringe
//                 for (; i < in_int64s; ++i) {
//                     uint64_t in_scalar = my_input[i];
//                     pop_sum0 += sycl::popcount(~(in_scalar ^ w0[i]));
//                     pop_sum1 += sycl::popcount(~(in_scalar ^ w1[i]));
//                     pop_sum2 += sycl::popcount(~(in_scalar ^ w2[i]));
//                     pop_sum3 += sycl::popcount(~(in_scalar ^ w3[i]));
//                 }
// 
//                 // Thresholding & Bit Packing
//                 if (pop_sum0 > thresholds[out_idx + 0]) packed_out |= (1ULL << ((out_idx + 0) % 64));
//                 if (pop_sum1 > thresholds[out_idx + 1]) packed_out |= (1ULL << ((out_idx + 1) % 64));
//                 if (pop_sum2 > thresholds[out_idx + 2]) packed_out |= (1ULL << ((out_idx + 2) % 64));
//                 if (pop_sum3 > thresholds[out_idx + 3]) packed_out |= (1ULL << ((out_idx + 3) % 64));
//             }
// 
//             // Handle Matrix Height Fringe (If output blocks aren't perfectly div by 4)
//             for (; out_idx < end_out_idx; ++out_idx) {
//                 int pop_sum = 0;
//                 const uint64_t* w_fringe = &weights[out_idx * in_int64s];
//                 for (int i = 0; i < in_int64s; ++i) {
//                     pop_sum += sycl::popcount(~(my_input[i] ^ w_fringe[i]));
//                 }
//                 if (pop_sum > thresholds[out_idx]) {
//                     packed_out |= (1ULL << (out_idx % 64));
//                 }
//             }
// 
//             outputs[b * out_int64s + out_block] = packed_out;
//         });
//     });
// }

// include/kernel_linear.hpp
#pragma once
#include <sycl/sycl.hpp>

                            
constexpr int LINEAR_VEC_SIZE = 4; 
constexpr int LINEAR_BLOCK_B = 4;  
constexpr int LINEAR_BLOCK_O = 4;

class FusedBinaryLinearTransposed_4x4;

void launch_binary_linear_fused(
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
    
    // We now divide the batch dimension by 4, because each thread handles 4 batches simultaneously!
    int batch_blocks = (batch_size + LINEAR_BLOCK_B - 1) / LINEAR_BLOCK_B;
    sycl::range<2> global_range(out_int64s, batch_blocks);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<FusedBinaryLinearTransposed_4x4>(global_range, [=](sycl::id<2> idx) {
            int out_block = idx[0];
            int b_block = idx[1];

            int start_out_idx = out_block * 64;
            int end_out_idx = (start_out_idx + 64 > out_features) ? out_features : (start_out_idx + 64);
            
            int b_start = b_block * LINEAR_BLOCK_B;
            int b_end = (b_start + LINEAR_BLOCK_B > batch_size) ? batch_size : (b_start + LINEAR_BLOCK_B);

            // If we are at the very edge of the batch dimension, fall back to safe scalar logic
            if (b_end - b_start != LINEAR_BLOCK_B) {
                for (int b = b_start; b < b_end; ++b) {
                    uint64_t packed_out = 0;
                    for (int o = start_out_idx; o < end_out_idx; ++o) {
                        int pop_sum = 0;
                        for (int i = 0; i < in_int64s; ++i) {
                            pop_sum += sycl::popcount(~(inputs[b * in_int64s + i] ^ weights[o * in_int64s + i]));
                        }
                        if (pop_sum > thresholds[o]) packed_out |= (1ULL << (o % 64));
                    }
                    outputs[b * out_int64s + out_block] = packed_out;
                }
                return;
            }

            // Pointers for our 4 independent Batch rows
            const uint64_t* in0 = &inputs[(b_start + 0) * in_int64s];
            const uint64_t* in1 = &inputs[(b_start + 1) * in_int64s];
            const uint64_t* in2 = &inputs[(b_start + 2) * in_int64s];
            const uint64_t* in3 = &inputs[(b_start + 3) * in_int64s];

            // 4 Output accumulators for 4 Batch rows (16 variables total)
            uint64_t packed_out0 = 0, packed_out1 = 0, packed_out2 = 0, packed_out3 = 0;

            int out_idx = start_out_idx;
            for (; out_idx <= end_out_idx - LINEAR_BLOCK_O; out_idx += LINEAR_BLOCK_O) {
                
                // 16 Independent Popcount Accumulators mapped to CPU GPRs
                int p00=0, p01=0, p02=0, p03=0; // Batch 0 vs Weights 0,1,2,3
                int p10=0, p11=0, p12=0, p13=0; // Batch 1 vs Weights 0,1,2,3
                int p20=0, p21=0, p22=0, p23=0; // Batch 2 vs Weights 0,1,2,3
                int p30=0, p31=0, p32=0, p33=0; // Batch 3 vs Weights 0,1,2,3
                
                const uint64_t* w0 = &weights[(out_idx + 0) * in_int64s];
                const uint64_t* w1 = &weights[(out_idx + 1) * in_int64s];
                const uint64_t* w2 = &weights[(out_idx + 2) * in_int64s];
                const uint64_t* w3 = &weights[(out_idx + 3) * in_int64s];
                
                int i = 0;
                for (; i <= in_int64s - LINEAR_VEC_SIZE; i += LINEAR_VEC_SIZE) {
                    // 8 Loads total
                    sycl::vec<uint64_t, LINEAR_VEC_SIZE> v_in0, v_in1, v_in2, v_in3;
                    sycl::vec<uint64_t, LINEAR_VEC_SIZE> v_w0, v_w1, v_w2, v_w3;
                    
                    v_in0.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(in0 + i));
                    v_in1.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(in1 + i));
                    v_in2.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(in2 + i));
                    v_in3.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(in3 + i));
                    
                    v_w0.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(w0 + i));
                    v_w1.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(w1 + i));
                    v_w2.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(w2 + i));
                    v_w3.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(w3 + i));

                    // 16 Math Operations perfectly unrolled
                    #define ACCUM_POP(BIN, BW, ACCUM) { \
                        auto pops = sycl::popcount(~(BIN ^ BW)); \
                        ACCUM += pops.s0() + pops.s1() + pops.s2() + pops.s3(); \
                    }
                    
                    ACCUM_POP(v_in0, v_w0, p00); ACCUM_POP(v_in0, v_w1, p01); ACCUM_POP(v_in0, v_w2, p02); ACCUM_POP(v_in0, v_w3, p03);
                    ACCUM_POP(v_in1, v_w0, p10); ACCUM_POP(v_in1, v_w1, p11); ACCUM_POP(v_in1, v_w2, p12); ACCUM_POP(v_in1, v_w3, p13);
                    ACCUM_POP(v_in2, v_w0, p20); ACCUM_POP(v_in2, v_w1, p21); ACCUM_POP(v_in2, v_w2, p22); ACCUM_POP(v_in2, v_w3, p23);
                    ACCUM_POP(v_in3, v_w0, p30); ACCUM_POP(v_in3, v_w1, p31); ACCUM_POP(v_in3, v_w2, p32); ACCUM_POP(v_in3, v_w3, p33);
                    #undef ACCUM_POP
                }

                // Inner loop fringe
                for (; i < in_int64s; ++i) {
                    uint64_t i0 = in0[i], i1 = in1[i], i2 = in2[i], i3 = in3[i];
                    uint64_t weight0 = w0[i], weight1 = w1[i], weight2 = w2[i], weight3 = w3[i];
                    
                    p00 += sycl::popcount(~(i0 ^ weight0)); p01 += sycl::popcount(~(i0 ^ weight1)); p02 += sycl::popcount(~(i0 ^ weight2)); p03 += sycl::popcount(~(i0 ^ weight3));
                    p10 += sycl::popcount(~(i1 ^ weight0)); p11 += sycl::popcount(~(i1 ^ weight1)); p12 += sycl::popcount(~(i1 ^ weight2)); p13 += sycl::popcount(~(i1 ^ weight3));
                    p20 += sycl::popcount(~(i2 ^ weight0)); p21 += sycl::popcount(~(i2 ^ weight1)); p22 += sycl::popcount(~(i2 ^ weight2)); p23 += sycl::popcount(~(i2 ^ weight3));
                    p30 += sycl::popcount(~(i3 ^ weight0)); p31 += sycl::popcount(~(i3 ^ weight1)); p32 += sycl::popcount(~(i3 ^ weight2)); p33 += sycl::popcount(~(i3 ^ weight3));
                }

                // Thresholding & Packing
                int t0 = thresholds[out_idx+0], t1 = thresholds[out_idx+1], t2 = thresholds[out_idx+2], t3 = thresholds[out_idx+3];
                uint64_t b0 = 1ULL << ((out_idx + 0) % 64);
                uint64_t b1 = 1ULL << ((out_idx + 1) % 64);
                uint64_t b2 = 1ULL << ((out_idx + 2) % 64);
                uint64_t b3 = 1ULL << ((out_idx + 3) % 64);

                if (p00 > t0) packed_out0 |= b0; if (p01 > t1) packed_out0 |= b1; if (p02 > t2) packed_out0 |= b2; if (p03 > t3) packed_out0 |= b3;
                if (p10 > t0) packed_out1 |= b0; if (p11 > t1) packed_out1 |= b1; if (p12 > t2) packed_out1 |= b2; if (p13 > t3) packed_out1 |= b3;
                if (p20 > t0) packed_out2 |= b0; if (p21 > t1) packed_out2 |= b1; if (p22 > t2) packed_out2 |= b2; if (p23 > t3) packed_out2 |= b3;
                if (p30 > t0) packed_out3 |= b0; if (p31 > t1) packed_out3 |= b1; if (p32 > t2) packed_out3 |= b2; if (p33 > t3) packed_out3 |= b3;
            }

            // Write 4 batches simultaneously
            outputs[(b_start + 0) * out_int64s + out_block] = packed_out0;
            outputs[(b_start + 1) * out_int64s + out_block] = packed_out1;
            outputs[(b_start + 2) * out_int64s + out_block] = packed_out2;
            outputs[(b_start + 3) * out_int64s + out_block] = packed_out3;
        });
    });
}
