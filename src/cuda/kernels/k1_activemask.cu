#include "kernels/k1_activemask.cuh"
#include "core/block_encoding.hpp"  // For EMPTY_BLOCK_ID
#include "core/local_bins.hpp"  // For LOCAL_BINS
#include "device/device_psic.cuh"  // For DEVICE_Kb, DEVICE_EMPTY_BLOCK_ID

__global__ void K1_ActiveMask(
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,
    uint8_t* __restrict__ ActiveMask,
    const uint8_t* __restrict__ ActiveMaskPrev,
    int Nx, int Nz,
    int b_E_fine_on,
    int b_E_fine_off,
    float weight_active_min
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    float W_cell = 0;
    bool below_fine_on = false;
    bool below_fine_off = false;

    // Check all slots in cell
    // IC-1: Fine transport activates at LOW energy (Bragg peak region)
    // where stopping power is LARGE, not at high energy
    // FIX: Use actual Kb (device slot count) instead of hardcoded 32
    constexpr int Kb = DEVICE_Kb;  // Must match DevicePsiC::Kb
    constexpr int LOCAL_BINS_val = LOCAL_BINS;  // Must match compile-time constant

    for (int slot = 0; slot < Kb; ++slot) {
        uint32_t bid = block_ids[cell * Kb + slot];
        if (bid == EMPTY_BLOCK_ID) continue;

        uint32_t b_E = (bid >> 12) & 0xFFF;
        if (b_E < static_cast<uint32_t>(b_E_fine_on)) below_fine_on = true;
        if (b_E < static_cast<uint32_t>(b_E_fine_off)) below_fine_off = true;

        for (int lidx = 0; lidx < LOCAL_BINS_val; ++lidx) {
            W_cell += values[(cell * Kb + slot) * LOCAL_BINS_val + lidx];
        }
    }

    const bool prev_active = (ActiveMaskPrev != nullptr) && (ActiveMaskPrev[cell] != 0);
    const bool needs_fine_transport = below_fine_on || (prev_active && below_fine_off);
    ActiveMask[cell] = (needs_fine_transport && W_cell > weight_active_min) ? 1 : 0;
}

// CPU wrapper implementation
void run_K1_ActiveMask(
    const PsiC& psi,
    uint8_t* ActiveMask,
    const uint8_t* ActiveMaskPrev,
    int Nx, int Nz,
    int b_E_fine_on,
    int b_E_fine_off,
    float weight_active_min
) {
    for (int cell = 0; cell < Nx * Nz; ++cell) {
        float W_cell = 0;
        bool below_fine_on = false;
        bool below_fine_off = false;

        // Use psi.Kb (host-side slot count) instead of hardcoded value
        for (int slot = 0; slot < psi.Kb; ++slot) {
            uint32_t bid = psi.block_id[cell][slot];
            if (bid == EMPTY_BLOCK_ID) continue;

            uint32_t b_E = (bid >> 12) & 0xFFF;
            if (b_E < static_cast<uint32_t>(b_E_fine_on)) below_fine_on = true;
            if (b_E < static_cast<uint32_t>(b_E_fine_off)) below_fine_off = true;

            // Use LOCAL_BINS constant instead of hardcoded value
            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                W_cell += psi.value[cell][slot][lidx];
            }
        }

        const bool prev_active = (ActiveMaskPrev != nullptr) && (ActiveMaskPrev[cell] != 0);
        const bool needs_fine_transport = below_fine_on || (prev_active && below_fine_off);
        ActiveMask[cell] = (needs_fine_transport && W_cell > weight_active_min) ? 1 : 0;
    }
}
