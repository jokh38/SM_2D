#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct AuditReport {
    float W_error;
    int W_pass;
    float E_error;
    int E_pass;
    float W_in_total;
    float W_out_total;
    float W_cutoff_total;
    float W_nuclear_total;
    float W_boundary_total;
    double E_in_total;
    double E_out_total;
    double E_dep_total;
    double E_cutoff_total;
    double E_nuclear_total;
    double E_boundary_total;
    int processed_cells;
};

__global__ void K5_ConservationAudit(
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ block_ids_out,
    const float* __restrict__ values_out,
    const double* __restrict__ EdepC,
    const double* __restrict__ AbsorbedEnergy_nuclear,
    const double* __restrict__ BoundaryLoss_energy,
    const double* __restrict__ PrevEdepC,
    const double* __restrict__ PrevAbsorbedEnergy_nuclear,
    const double* __restrict__ PrevBoundaryLoss_energy,
    const float* __restrict__ AbsorbedWeight_cutoff,
    const float* __restrict__ AbsorbedWeight_nuclear,
    const float* __restrict__ BoundaryLoss_weight,
    const float* __restrict__ PrevAbsorbedWeight_cutoff,
    const float* __restrict__ PrevAbsorbedWeight_nuclear,
    const float* __restrict__ PrevBoundaryLoss_weight,
    const float* __restrict__ E_edges,
    int N_E,
    int N_E_local,
    AuditReport* __restrict__ report,
    int N_cells
);
