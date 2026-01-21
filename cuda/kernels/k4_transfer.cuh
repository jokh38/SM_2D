#pragma once
#include "core/buckets.hpp"

__global__ void K4_BucketTransfer(
    const OutflowBucket* __restrict__ OutflowBuckets,
    float* __restrict__ values_out,
    uint32_t* __restrict__ block_ids_out,
    int Nx, int Nz
);

// CPU wrapper
void run_K4_BucketTransfer(
    const OutflowBucket* buckets,
    PsiC& psi_out,
    int cell,
    int face
);
