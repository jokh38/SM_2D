#include "kernels/k1_activemask.cuh"
#include "core/block_encoding.hpp"

__global__ void K1_ActiveMask(
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,
    uint8_t* __restrict__ ActiveMask,
    int Nx, int Nz,
    float E_trigger,
    float weight_active_min
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    float W_cell = 0;
    bool needs_fine_transport = false;

    // Check all slots in cell
    // IC-1: Fine transport activates at LOW energy (Bragg peak region)
    // where stopping power is LARGE, not at high energy
    for (int slot = 0; slot < 32; ++slot) {
        uint32_t bid = block_ids[cell * 32 + slot];
        if (bid == 0xFFFFFFFF) continue;

        uint32_t b_E = (bid >> 12) & 0xFFF;
        // FIXED: Changed from >= (HIGH energy) to < (LOW energy)
        // b_E is energy bin index; lower values = lower energy
        // Activate fine transport when energy is below threshold (LOW energy region)
        if (b_E < 20) needs_fine_transport = true;

        for (int lidx = 0; lidx < 32; ++lidx) {
            W_cell += values[(cell * 32 + slot) * 32 + lidx];
        }
    }

    ActiveMask[cell] = (needs_fine_transport && W_cell > weight_active_min) ? 1 : 0;
}

// CPU wrapper implementation
void run_K1_ActiveMask(
    const PsiC& psi,
    uint8_t* ActiveMask,
    int Nx, int Nz,
    float E_trigger,
    float weight_active_min
) {
    for (int cell = 0; cell < Nx * Nz; ++cell) {
        float W_cell = 0;
        bool needs_fine_transport = false;

        for (int slot = 0; slot < 32; ++slot) {
            uint32_t bid = psi.block_id[cell][slot];
            if (bid == 0xFFFFFFFF) continue;

            uint32_t b_E = (bid >> 12) & 0xFFF;
            // FIXED: Activate fine transport at LOW energy, not HIGH energy
            if (b_E < 20) needs_fine_transport = true;

            for (int lidx = 0; lidx < 32; ++lidx) {
                W_cell += psi.value[cell][slot][lidx];
            }
        }

        ActiveMask[cell] = (needs_fine_transport && W_cell > weight_active_min) ? 1 : 0;
    }
}
