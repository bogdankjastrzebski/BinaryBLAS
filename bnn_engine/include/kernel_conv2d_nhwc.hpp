#pragma once
#include <sycl/sycl.hpp>

constexpr int CONV_VEC_SIZE = 4; // AVX-512 256-bit loads
constexpr int CONV_BLOCK_OW = 4; // Unroll 4 Spatial Width Pixels

class BinaryConv2d_NHWC_ImplicitGemm;

inline void launch_binary_conv2d_nhwc(
    sycl::queue& q,
    const uint64_t* inputs, 
    const uint64_t* weights, 
    const int32_t* thresholds, 
    uint64_t* outputs, 
    int batch_size,
    int in_channels_int64,
    int out_channels,
    int in_h, int in_w,
    int kernel_size, int stride, int padding)
{
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int out_channels_int64 = (out_channels + 63) / 64;

    int ow_blocks = (out_w + CONV_BLOCK_OW - 1) / CONV_BLOCK_OW;

    // Grid: [Batch * Out_H, Out_W / 4, Out_C_int64 (Blocks of 64)]
    sycl::range<3> global_range(batch_size * out_h, ow_blocks, out_channels_int64);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<BinaryConv2d_NHWC_ImplicitGemm>(global_range, [=](sycl::id<3> idx) {
            int b_oh = idx[0];
            int ow_blk = idx[1];
            int oc_blk = idx[2];

            int b = b_oh / out_h;
            int oh = b_oh % out_h;

            int ow_start = ow_blk * CONV_BLOCK_OW;
            int ow_end = (ow_start + CONV_BLOCK_OW > out_w) ? out_w : (ow_start + CONV_BLOCK_OW);

            int true_oc_start = oc_blk * 64;
            int true_oc_end = (true_oc_start + 64 > out_channels) ? out_channels : (true_oc_start + 64);

            // ==========================================
            // FRINGE FALLBACK (The safety net for edges)
            // ==========================================
            if (ow_end - ow_start != CONV_BLOCK_OW) {
                for (int ow = ow_start; ow < ow_end; ++ow) {
                    uint64_t packed_out = 0;
                    for (int toc = true_oc_start; toc < true_oc_end; ++toc) {
                        int pop_sum = 0;
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    const uint64_t* in_ptr = &inputs[b*(in_h*in_w*in_channels_int64) + ih*(in_w*in_channels_int64) + iw*in_channels_int64];
                                    const uint64_t* w_ptr = &weights[toc*(kernel_size*kernel_size*in_channels_int64) + kh*(kernel_size*in_channels_int64) + kw*in_channels_int64];
                                    for (int ic = 0; ic < in_channels_int64; ++ic) {
                                        pop_sum += sycl::popcount(~(in_ptr[ic] ^ w_ptr[ic]));
                                    }
                                }
                            }
                        }
                        if (pop_sum > thresholds[toc]) packed_out |= (1ULL << (toc % 64));
                    }
                    outputs[b*(out_h*out_w*out_channels_int64) + oh*(out_w*out_channels_int64) + ow*out_channels_int64 + oc_blk] = packed_out;
                }
                return;
            }

            // ==========================================
            // THE BEAST: 1x4 SPATIAL UNROLLED MICRO-KERNEL
            // ==========================================
            uint64_t packed_out0 = 0, packed_out1 = 0, packed_out2 = 0, packed_out3 = 0;

            // Loop over the 64 output channels to build the uint64_t blocks
            for (int toc = true_oc_start; toc < true_oc_end; ++toc) {
                int p0 = 0, p1 = 0, p2 = 0, p3 = 0;

                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int ih = oh * stride - padding + kh;
                        int iw0 = (ow_start + 0) * stride - padding + kw;
                        int iw1 = (ow_start + 1) * stride - padding + kw;
                        int iw2 = (ow_start + 2) * stride - padding + kw;
                        int iw3 = (ow_start + 3) * stride - padding + kw;

                        // Bounds checking for Implicit GEMM Padding
                        bool v0 = (ih >= 0 && ih < in_h && iw0 >= 0 && iw0 < in_w);
                        bool v1 = (ih >= 0 && ih < in_h && iw1 >= 0 && iw1 < in_w);
                        bool v2 = (ih >= 0 && ih < in_h && iw2 >= 0 && iw2 < in_w);
                        bool v3 = (ih >= 0 && ih < in_h && iw3 >= 0 && iw3 < in_w);

                        const uint64_t* in0 = v0 ? &inputs[b*(in_h*in_w*in_channels_int64) + ih*(in_w*in_channels_int64) + iw0*in_channels_int64] : nullptr;
                        const uint64_t* in1 = v1 ? &inputs[b*(in_h*in_w*in_channels_int64) + ih*(in_w*in_channels_int64) + iw1*in_channels_int64] : nullptr;
                        const uint64_t* in2 = v2 ? &inputs[b*(in_h*in_w*in_channels_int64) + ih*(in_w*in_channels_int64) + iw2*in_channels_int64] : nullptr;
                        const uint64_t* in3 = v3 ? &inputs[b*(in_h*in_w*in_channels_int64) + ih*(in_w*in_channels_int64) + iw3*in_channels_int64] : nullptr;

                        const uint64_t* w_ptr = &weights[toc*(kernel_size*kernel_size*in_channels_int64) + kh*(kernel_size*in_channels_int64) + kw*in_channels_int64];

                        int ic = 0;
                        for (; ic <= in_channels_int64 - CONV_VEC_SIZE; ic += CONV_VEC_SIZE) {
                            sycl::vec<uint64_t, CONV_VEC_SIZE> vw;
                            vw.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(w_ptr + ic));

                            #define COMPUTE_SPATIAL(IN_PTR, ACCUM) \
                                if (IN_PTR) { \
                                    sycl::vec<uint64_t, CONV_VEC_SIZE> vi; \
                                    vi.load(0, sycl::multi_ptr<const uint64_t, sycl::access::address_space::global_space>(IN_PTR + ic)); \
                                    auto p = sycl::popcount(~(vi ^ vw)); \
                                    ACCUM += p.s0() + p.s1() + p.s2() + p.s3(); \
                                }

                            COMPUTE_SPATIAL(in0, p0);
                            COMPUTE_SPATIAL(in1, p1);
                            COMPUTE_SPATIAL(in2, p2);
                            COMPUTE_SPATIAL(in3, p3);
                            #undef COMPUTE_SPATIAL
                        }

                        for (; ic < in_channels_int64; ++ic) {
                            uint64_t w_val = w_ptr[ic];
                            if (v0) p0 += sycl::popcount(~(in0[ic] ^ w_val));
                            if (v1) p1 += sycl::popcount(~(in1[ic] ^ w_val));
                            if (v2) p2 += sycl::popcount(~(in2[ic] ^ w_val));
                            if (v3) p3 += sycl::popcount(~(in3[ic] ^ w_val));
                        }
                    }
                }

                // Threshold and Pack!
                int t = thresholds[toc];
                uint64_t bit = (1ULL << (toc % 64));
                if (p0 > t) packed_out0 |= bit;
                if (p1 > t) packed_out1 |= bit;
                if (p2 > t) packed_out2 |= bit;
                if (p3 > t) packed_out3 |= bit;
            }

            // WE ACTUALLY WRITE TO MEMORY THIS TIME
            int out_base = b*(out_h*out_w*out_channels_int64) + oh*(out_w*out_channels_int64) + oc_blk;
            outputs[out_base + (ow_start + 0)*out_channels_int64] = packed_out0;
            outputs[out_base + (ow_start + 1)*out_channels_int64] = packed_out1;
            outputs[out_base + (ow_start + 2)*out_channels_int64] = packed_out2;
            outputs[out_base + (ow_start + 3)*out_channels_int64] = packed_out3;
        });
    });
}
