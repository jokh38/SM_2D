#include "kernels/k5_audit.cuh"
#include "device/device_psic.cuh"  // For DEVICE_Kb
#include "core/local_bins.hpp"  // For LOCAL_BINS
#include <cmath>
#include <cstdint>

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
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N_cells) return;

    float W_in = 0, W_out = 0;
    // FIX: Use actual constants instead of hardcoded 32
    constexpr int LOCAL_BINS_val = LOCAL_BINS;  // = 128
    constexpr int Kb = DEVICE_Kb;               // = 8

    for (int slot = 0; slot < Kb; ++slot) {
        if (block_ids_in[cell * Kb + slot] != 0xFFFFFFFFu) {
            for (int lidx = 0; lidx < LOCAL_BINS_val; ++lidx) {
                W_in += values_in[(cell * Kb + slot) * LOCAL_BINS_val + lidx];
            }
        }
        if (block_ids_out[cell * Kb + slot] != 0xFFFFFFFFu) {
            for (int lidx = 0; lidx < LOCAL_BINS_val; ++lidx) {
                W_out += values_out[(cell * Kb + slot) * LOCAL_BINS_val + lidx];
            }
        }
    }

    // Use iteration deltas (current cumulative - previous cumulative).
    // This keeps K5 local to the current iteration while preserving cumulative tallies.
    float W_cutoff = fmaxf(0.0f, AbsorbedWeight_cutoff[cell] - PrevAbsorbedWeight_cutoff[cell]);
    float W_nuclear = fmaxf(0.0f, AbsorbedWeight_nuclear[cell] - PrevAbsorbedWeight_nuclear[cell]);
    float W_boundary = fmaxf(0.0f, BoundaryLoss_weight[cell] - PrevBoundaryLoss_weight[cell]);

    // Global conservation audit:
    //   Sum(W_in) = Sum(W_out) + Sum(W_cutoff) + Sum(W_nuclear) + Sum(W_boundary)
    // Per-cell equality is not valid under lateral transfer, so reduce globally.
    atomicAdd(&report[0].W_in_total, W_in);
    atomicAdd(&report[0].W_out_total, W_out);
    atomicAdd(&report[0].W_cutoff_total, W_cutoff);
    atomicAdd(&report[0].W_nuclear_total, W_nuclear);
    atomicAdd(&report[0].W_boundary_total, W_boundary);

    int prev_count = atomicAdd(&report[0].processed_cells, 1);
    if (prev_count == N_cells - 1) {
        float rhs = report[0].W_out_total +
                    report[0].W_cutoff_total +
                    report[0].W_nuclear_total +
                    report[0].W_boundary_total;
        float W_error_abs = fabsf(report[0].W_in_total - rhs);
        float W_rel_error = W_error_abs / fmaxf(report[0].W_in_total, 1e-20f);

        report[0].W_error = W_rel_error;
        report[0].W_pass = (isfinite(W_rel_error) && W_rel_error < 1e-6f) ? 1 : 0;
    }
}
