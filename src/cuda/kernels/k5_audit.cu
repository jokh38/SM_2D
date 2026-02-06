#include "kernels/k5_audit.cuh"
#include "device/device_psic.cuh"  // For DEVICE_Kb
#include "core/local_bins.hpp"  // For LOCAL_BINS
#include <cmath>
#include <cstdint>

__device__ inline float k5_energy_from_bin(
    const float* __restrict__ E_edges,
    int N_E,
    uint32_t b_E,
    int E_local,
    int N_E_local
) {
    int E_bin = static_cast<int>(b_E) * N_E_local + E_local;
    E_bin = max(0, min(E_bin, N_E - 1));
    float E_lower = E_edges[E_bin];
    float E_upper = E_edges[E_bin + 1];
    float E_half_width = 0.5f * (E_upper - E_lower);
    return E_lower + 0.50f * E_half_width;
}

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
    float source_out_of_grid_weight,
    float source_slot_dropped_weight,
    double source_out_of_grid_energy,
    double source_slot_dropped_energy,
    int include_source_terms,
    const float* __restrict__ E_edges,
    int N_E,
    int N_E_local,
    AuditReport* __restrict__ report,
    int N_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N_cells) return;

    double W_in = 0.0;
    double W_out = 0.0;
    double E_in = 0.0;
    double E_out = 0.0;
    constexpr int LOCAL_BINS_val = LOCAL_BINS;
    constexpr int Kb = DEVICE_Kb;

    for (int slot = 0; slot < Kb; ++slot) {
        uint32_t bid_in = block_ids_in[cell * Kb + slot];
        if (bid_in != 0xFFFFFFFFu) {
            uint32_t b_E = (bid_in >> 12) & 0xFFF;
            for (int lidx = 0; lidx < LOCAL_BINS_val; ++lidx) {
                float w = values_in[(cell * Kb + slot) * LOCAL_BINS_val + lidx];
                if (w <= 0.0f) continue;
                W_in += static_cast<double>(w);
                int E_local = (lidx / N_theta_local) % N_E_local;
                float E = k5_energy_from_bin(E_edges, N_E, b_E, E_local, N_E_local);
                E_in += static_cast<double>(w) * static_cast<double>(E);
            }
        }
        uint32_t bid_out = block_ids_out[cell * Kb + slot];
        if (bid_out != 0xFFFFFFFFu) {
            uint32_t b_E = (bid_out >> 12) & 0xFFF;
            for (int lidx = 0; lidx < LOCAL_BINS_val; ++lidx) {
                float w = values_out[(cell * Kb + slot) * LOCAL_BINS_val + lidx];
                if (w <= 0.0f) continue;
                W_out += static_cast<double>(w);
                int E_local = (lidx / N_theta_local) % N_E_local;
                float E = k5_energy_from_bin(E_edges, N_E, b_E, E_local, N_E_local);
                E_out += static_cast<double>(w) * static_cast<double>(E);
            }
        }
    }

    // Use iteration deltas (current cumulative - previous cumulative).
    // This keeps K5 local to the current iteration while preserving cumulative tallies.
    float W_cutoff = fmaxf(0.0f, AbsorbedWeight_cutoff[cell] - PrevAbsorbedWeight_cutoff[cell]);
    float W_nuclear = fmaxf(0.0f, AbsorbedWeight_nuclear[cell] - PrevAbsorbedWeight_nuclear[cell]);
    float W_boundary = fmaxf(0.0f, BoundaryLoss_weight[cell] - PrevBoundaryLoss_weight[cell]);
    double E_dep = fmax(0.0, EdepC[cell] - PrevEdepC[cell]);
    double E_nuclear = fmax(0.0, AbsorbedEnergy_nuclear[cell] - PrevAbsorbedEnergy_nuclear[cell]);
    double E_boundary = fmax(0.0, BoundaryLoss_energy[cell] - PrevBoundaryLoss_energy[cell]);
    double E_cutoff = 0.0;

    // Global conservation audit:
    //   Sum(W_in) = Sum(W_out) + Sum(W_cutoff) + Sum(W_nuclear) + Sum(W_boundary)
    // Per-cell equality is not valid under lateral transfer, so reduce globally.
    atomicAdd(&report[0].W_in_total, static_cast<float>(W_in));
    atomicAdd(&report[0].W_out_total, static_cast<float>(W_out));
    atomicAdd(&report[0].W_cutoff_total, W_cutoff);
    atomicAdd(&report[0].W_nuclear_total, W_nuclear);
    atomicAdd(&report[0].W_boundary_total, W_boundary);
    atomicAdd(&report[0].E_in_total, E_in);
    atomicAdd(&report[0].E_out_total, E_out);
    atomicAdd(&report[0].E_dep_total, E_dep);
    atomicAdd(&report[0].E_cutoff_total, E_cutoff);
    atomicAdd(&report[0].E_nuclear_total, E_nuclear);
    atomicAdd(&report[0].E_boundary_total, E_boundary);

    if (include_source_terms && cell == 0) {
        float W_source_out = fmaxf(0.0f, source_out_of_grid_weight);
        float W_source_slot = fmaxf(0.0f, source_slot_dropped_weight);
        double E_source_out = fmax(0.0, source_out_of_grid_energy);
        double E_source_slot = fmax(0.0, source_slot_dropped_energy);

        // Lift iteration-1 audit from in-grid-only accounting to source-total accounting.
        atomicAdd(&report[0].W_in_total, W_source_out + W_source_slot);
        atomicAdd(&report[0].E_in_total, E_source_out + E_source_slot);

        atomicAdd(&report[0].W_source_out_of_grid_total, W_source_out);
        atomicAdd(&report[0].W_source_slot_drop_total, W_source_slot);
        atomicAdd(&report[0].E_source_out_of_grid_total, E_source_out);
        atomicAdd(&report[0].E_source_slot_drop_total, E_source_slot);
    }

    int prev_count = atomicAdd(&report[0].processed_cells, 1);
    if (prev_count == N_cells - 1) {
        float W_rhs = report[0].W_out_total +
                      report[0].W_cutoff_total +
                      report[0].W_nuclear_total +
                      report[0].W_boundary_total +
                      report[0].W_source_out_of_grid_total +
                      report[0].W_source_slot_drop_total;
        float W_error_abs = fabsf(report[0].W_in_total - W_rhs);
        float W_rel_error = W_error_abs / fmaxf(report[0].W_in_total, 1e-20f);

        report[0].W_error = W_rel_error;
        report[0].W_pass = (isfinite(W_rel_error) && W_rel_error < 1e-6f) ? 1 : 0;

        double E_rhs = report[0].E_out_total +
                       report[0].E_dep_total +
                       report[0].E_cutoff_total +
                       report[0].E_nuclear_total +
                       report[0].E_boundary_total +
                       report[0].E_source_out_of_grid_total +
                       report[0].E_source_slot_drop_total;
        double E_error_abs = fabs(report[0].E_in_total - E_rhs);
        float E_rel_error = static_cast<float>(E_error_abs / fmax(report[0].E_in_total, 1e-20));
        report[0].E_error = E_rel_error;
        report[0].E_pass = (isfinite(E_rel_error) && E_rel_error < 1e-5f) ? 1 : 0;
    }
}
