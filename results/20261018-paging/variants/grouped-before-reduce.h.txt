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
