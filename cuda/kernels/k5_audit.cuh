#pragma once
#include <cuda_runtime.h>

struct AuditReport {
    float W_error;
    bool W_pass;
};

__global__ void K5_WeightAudit(
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ block_ids_out,
    const float* __restrict__ values_out,
    const float* __restrict__ AbsorbedWeight_cutoff,
    const float* __restrict__ AbsorbedWeight_nuclear,
    AuditReport* __restrict__ report,
    int N_cells
);
