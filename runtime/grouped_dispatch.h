#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// Pass pointers as launch parameters, avoiding a separate metadata DMA.
struct GroupedPointers {
    const uint16_t* trellis[12];
    const half* suh[12];
    const half* svh[12];
    int count;
};

void launch_grouped(const void* input, void* output, void* had, int k, int n,
                    int input_stride, int bits, const GroupedPointers& ptrs);

struct ExpertOrder { int positions[6]; int count; };
void launch_expert_sum(const float* down, const float* weights, float* output,
                       int n, ExpertOrder order);

struct SplitExpertOrder { const float* sources[6]; int positions[6]; int count; };
void launch_split_sum(const float* weights, float* output, int n, SplitExpertOrder order);
