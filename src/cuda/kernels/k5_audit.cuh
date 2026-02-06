#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct AuditReport {
    float W_error;
    int W_pass;
    float W_in_total;
    float W_out_total;
    float W_cutoff_total;
    float W_nuclear_total;
    float W_boundary_total;
    int processed_cells;
};

__global__ void K5_WeightAudit(
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ block_ids_out,
    const float* __restrict__ values_out,
    const float* __restrict__ AbsorbedWeight_cutoff,
    const float* __restrict__ AbsorbedWeight_nuclear,
    const float* __restrict__ BoundaryLoss_weight,
    const float* __restrict__ PrevAbsorbedWeight_cutoff,
    const float* __restrict__ PrevAbsorbedWeight_nuclear,
    const float* __restrict__ PrevBoundaryLoss_weight,
    AuditReport* __restrict__ report,
    int N_cells
);
