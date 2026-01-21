#include "kernels/k5_audit.cuh"
#include <cmath>

__global__ void K5_WeightAudit(
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ block_ids_out,
    const float* __restrict__ values_out,
    const float* __restrict__ AbsorbedWeight_cutoff,
    const float* __restrict__ AbsorbedWeight_nuclear,
    AuditReport* __restrict__ report,
    int N_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N_cells) return;

    float W_in = 0, W_out = 0;
    constexpr int LOCAL_BINS = 32;
    constexpr int Kb = 32;

    for (int slot = 0; slot < Kb; ++slot) {
        if (block_ids_in[cell * Kb + slot] != 0xFFFFFFFF) {
            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                W_in += values_in[(cell * Kb + slot) * LOCAL_BINS + lidx];
            }
        }
        if (block_ids_out[cell * Kb + slot] != 0xFFFFFFFF) {
            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                W_out += values_out[(cell * Kb + slot) * LOCAL_BINS + lidx];
            }
        }
    }

    float W_cutoff = AbsorbedWeight_cutoff[cell];
    float W_nuclear = AbsorbedWeight_nuclear[cell];

    float W_error = fabsf(W_in - W_out - W_cutoff - W_nuclear);
    float W_rel_error = W_error / fmaxf(W_in, 1e-20f);

    report[cell].W_error = W_rel_error;
    report[cell].W_pass = (W_rel_error < 1e-6f);
}
