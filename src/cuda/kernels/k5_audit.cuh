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
    float W_source_out_of_grid_total;
    float W_source_slot_drop_total;
    float W_transport_drop_total;
    double E_in_total;
    double E_out_total;
    double E_dep_total;
    double E_cutoff_total;
    double E_nuclear_total;
    double E_boundary_total;
    double E_source_out_of_grid_total;
    double E_source_slot_drop_total;
    double E_transport_drop_total;
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
    const double* __restrict__ AbsorbedEnergy_cutoff,
    const double* __restrict__ PrevEdepC,
    const double* __restrict__ PrevAbsorbedEnergy_nuclear,
    const double* __restrict__ PrevBoundaryLoss_energy,
    const double* __restrict__ PrevAbsorbedEnergy_cutoff,
    const float* __restrict__ AbsorbedWeight_cutoff,
    const float* __restrict__ AbsorbedWeight_nuclear,
    const float* __restrict__ BoundaryLoss_weight,
    const float* __restrict__ PrevAbsorbedWeight_cutoff,
    const float* __restrict__ PrevAbsorbedWeight_nuclear,
    const float* __restrict__ PrevBoundaryLoss_weight,
    float source_out_of_grid_weight,
    float source_slot_dropped_weight,
    double source_out_of_grid_energy,
    double source_slot_dropped_energy,
    float transport_dropped_weight,
    double transport_dropped_energy,
    int include_source_terms,
    const float* __restrict__ E_edges,
    int N_E,
    int N_E_local,
    AuditReport* __restrict__ report,
    int N_cells
);
