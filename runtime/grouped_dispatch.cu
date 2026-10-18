// Adapted from ExLlamaV3 exl3_gemv_kernel.cuh at 954a8ca6e59d48c3e3462068ecf083fe9990f4dc.
// Copyright (c) 2025 Turboderp. MIT; see vendor/THIRD_PARTY_NOTICES.md.
// Scheduling only: spread independent expert/column groups over one cooperative
// grid. Preserve CFG=0 MMA, fp16 folding, fp32 reduction and both Hadamard stages.
// The worklist design also appears in vcruz305/vllm-exl3 p2b_batched.cu.
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemv_kernel.cuh"
#include "grouped_dispatch.h"

template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256)
void grouped_gemv_kernel(const half* A, float* output, half* A_had,
    int size_k, int size_n, int input_stride, GroupedPointers pointers)
{
    constexpr int size_m = 1;
    static_assert(bits == 2 || bits == 3 || bits == 4, "exl3_gemv_kernel supports 2, 3 and 4 bpw");
    constexpr int WK   = CFG == 0 ? 16 : 8;     // k-split (warps per block)
    constexpr int WNT  = CFG == 0 ? 2 : 4;      // adjacent n-tiles per warp
    constexpr int PF   = CFG == 0 ? 4 : 2;      // prefetch ring depth
    constexpr int FOLD = CFG == 0 ? 4 : 2;      // fp16->fp32 fold cadence (divides PF)
    constexpr int THREADS = WK * 32;
    constexpr int ROWS = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    constexpr int COLS = WNT * 16;

    constexpr int TWORDS = 8 * bits;                        // uint32 per 16x16 tile
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;        // warp loads per k-slice
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;            // uint32 per load
    static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");

    auto grid = cooperative_groups::this_grid();

    // Input scales and Hadamard transform, same as exl3_gemm_kernel
    {
        int total_warps = pointers.count * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            const int e = this_warp / (size_k / 128);
            const int w = this_warp % (size_k / 128);
            had_hf_r_128_inner<true, false>(
                A + e * input_stride + w * 128,
                A_had + e * size_k + w * 128,
                pointers.suh[e] + w * 128,
                0.088388347648f);
        }

        grid.sync();

    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const size_t slice_stride = (size_t) ntiles * TWORDS;   // uint32 per k-slice row
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // A fragment row indices for this lane
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

    // Per-lane extraction constants (see dq8_aligned_2bits / dq8<3, cb, 4> in exl3_dq.cuh)
    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    __shared__ float sh_red[WK][ROWS][COLS];
    [[maybe_unused]] __shared__ uint32_t sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];

    for (int item = blockIdx.x; item < pointers.count * num_groups; item += gridDim.x)
    {
        const int e = item / num_groups;
        const int group = item % num_groups;
        const uint32_t* B32 = (const uint32_t*) pointers.trellis[e];
        const half2* A2 = (const half2*) (A_had + e * size_k);
        void* C = output + e * size_n;
        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        // Prefetch ring (indices must be compile-time or pf lands in local memory)
        auto ld_b = [&] (int i, int l) -> uint32_t
        {
            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        FragC_h ch[WNT][2] = {};
        float2 acc0[WNT][2] = {};

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (SMEM_STAGE)
            {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (bits != 3 || lane < 24)
                        sh_stage[warp][l * LSTRIDE + lane] = bw[l];
                __syncwarp();
            }

            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (SMEM_STAGE)
                {
                    const uint32_t* tp = &sh_stage[warp][t * TWORDS];
                    if constexpr (bits == 4)
                        exl3_gemv_ns::dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
                    else if constexpr (bits == 2)
                        exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                    else
                        exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    // Two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
        }

        // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
        // tile*16 + frag*8 + 2*(l%4) (+1)
        {
            const int c0 = 2 * (lane & 3);
            const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
            const int sr0 = MMODE == 0 ? 0 : r0;
            if (store0)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        const int col = t * 16 + f * 8 + c0;
                        sh_red[warp][sr0][col + 0] = acc0[t][f].x;
                        sh_red[warp][sr0][col + 1] = acc0[t][f].y;
                    }
            }
        }
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
        {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
            else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }

    // Output scales and Hadamard transform, same semantics as the inner GEMM epilogue
    {
        grid.sync();

        int total_warps = pointers.count * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            const int e = this_warp / (size_n / 128);
            const int w = this_warp % (size_n / 128);
            had_ff_r_128_inner<false, true>(
                output + e * size_n + w * 128,
                output + e * size_n + w * 128,
                pointers.svh[e] + w * 128,
                0.088388347648f);
        }
    }
}

void launch_grouped(const void* input, void* output, void* had, int k, int n,
                    int input_stride, int bits, const GroupedPointers& ptrs)
{
    void* kernel = bits == 2
        ? (void*) grouped_gemv_kernel<2, true, 2, 0, 0, false>
        : (void*) grouped_gemv_kernel<3, true, 2, 0, 0, false>;
    int device, sms, resident;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, 0));
    const int blocks = std::min(ptrs.count * n / 32, resident * sms);
    void* args[] = {&input, &output, &had, &k, &n, &input_stride, (void*)&ptrs};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, dim3(blocks), dim3(512), args, 0,
        at::cuda::getCurrentCUDAStream().stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Match the eager sequence of separate fp32 multiply and add kernels. Explicit
// PTX prevents FMA contraction and preserves subnormals despite fast-math used
// by the unchanged upstream GEMV code in this translation unit.
__global__ void ordered_expert_sum(const float* down, const float* weights,
                                  float* output, int n, ExpertOrder order)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float sum = 0.0f;
    for (int i = 0; i < order.count; ++i) {
        float product;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(down[i*n+j]), "f"(weights[order.positions[i]]));
        asm("add.rn.f32 %0, %1, %2;" : "=f"(sum) : "f"(sum), "f"(product));
    }
    output[j] = sum;
}

void launch_expert_sum(const float* down, const float* weights, float* output,
                       int n, ExpertOrder order)
{
    ordered_expert_sum<<<(n+255)/256, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(down, weights, output, n, order);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
