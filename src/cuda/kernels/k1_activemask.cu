#include "kernels/k1_activemask.cuh"
#include "core/block_encoding.hpp"  // For EMPTY_BLOCK_ID
#include "core/local_bins.hpp"  // For LOCAL_BINS
#include "device/device_psic.cuh"  // For DEVICE_Kb, DEVICE_EMPTY_BLOCK_ID

__global__ void K1_ActiveMask(
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,
    uint8_t* __restrict__ ActiveMask,
    int Nx, int Nz,
    int b_E_trigger,  // P4 FIX: Now uses block index directly (pre-computed from E_trigger)
    float weight_active_min
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    float W_cell = 0;
    bool needs_fine_transport = false;

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
        // P4 FIX: Now uses the b_E_trigger parameter instead of hardcoded 20
        // b_E is coarse energy block index; lower values = lower energy
        // Activate fine transport when energy is below threshold (LOW energy region)
        if (b_E < static_cast<uint32_t>(b_E_trigger)) needs_fine_transport = true;

        for (int lidx = 0; lidx < LOCAL_BINS_val; ++lidx) {
            W_cell += values[(cell * Kb + slot) * LOCAL_BINS_val + lidx];
        }
    }

    ActiveMask[cell] = (needs_fine_transport && W_cell > weight_active_min) ? 1 : 0;
}

// CPU wrapper implementation
void run_K1_ActiveMask(
    const PsiC& psi,
    uint8_t* ActiveMask,
    int Nx, int Nz,
    int b_E_trigger,  // P4 FIX: Now uses block index directly (pre-computed from E_trigger)
    float weight_active_min
) {
    for (int cell = 0; cell < Nx * Nz; ++cell) {
        float W_cell = 0;
        bool needs_fine_transport = false;

        // Use psi.Kb (host-side slot count) instead of hardcoded value
        for (int slot = 0; slot < psi.Kb; ++slot) {
            uint32_t bid = psi.block_id[cell][slot];
            if (bid == EMPTY_BLOCK_ID) continue;

            uint32_t b_E = (bid >> 12) & 0xFFF;
            // P4 FIX: Now uses the b_E_trigger parameter instead of hardcoded 20
            if (b_E < static_cast<uint32_t>(b_E_trigger)) needs_fine_transport = true;

            // Use LOCAL_BINS constant instead of hardcoded value
            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                W_cell += psi.value[cell][slot][lidx];
            }
        }

        ActiveMask[cell] = (needs_fine_transport && W_cell > weight_active_min) ? 1 : 0;
    }
}
