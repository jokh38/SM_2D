#pragma once
#include "core/psi_storage.hpp"
#include "core/grids.hpp"

// Active cell identification kernel.
// Fine classification uses E_fine_on/E_fine_off block thresholds with optional hysteresis.
__global__ void K1_ActiveMask(
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,
    uint8_t* __restrict__ ActiveMask,
    const uint8_t* __restrict__ ActiveMaskPrev,
    int Nx, int Nz,
    int b_E_fine_on,   // Coarse energy block index for E_fine_on (not MeV!)
    int b_E_fine_off,  // Coarse energy block index for E_fine_off (not MeV!)
    float weight_active_min
);

// CPU wrapper for testing
void run_K1_ActiveMask(
    const PsiC& psi,
    uint8_t* ActiveMask,
    const uint8_t* ActiveMaskPrev,
    int Nx, int Nz,
    int b_E_fine_on,   // Coarse energy block index (not MeV!)
    int b_E_fine_off,  // Coarse energy block index (not MeV!)
    float weight_active_min
);

// Legacy wrapper (no hysteresis): b_E_fine_off == b_E_fine_on and no previous mask.
inline void run_K1_ActiveMask(
    const PsiC& psi,
    uint8_t* ActiveMask,
    int Nx, int Nz,
    int b_E_trigger,
    float weight_active_min
) {
    run_K1_ActiveMask(
        psi,
        ActiveMask,
        nullptr,
        Nx,
        Nz,
        b_E_trigger,
        b_E_trigger,
        weight_active_min
    );
}
