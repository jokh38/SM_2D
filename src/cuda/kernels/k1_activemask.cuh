#pragma once
#include "core/psi_storage.hpp"
#include "core/grids.hpp"

// Active cell identification kernel
// P4 FIX: b_E_trigger is the coarse energy block index threshold (pre-computed from E_trigger)
__global__ void K1_ActiveMask(
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,
    uint8_t* __restrict__ ActiveMask,
    int Nx, int Nz,
    int b_E_trigger,  // Coarse energy block index (not MeV!)
    float weight_active_min
);

// CPU wrapper for testing
void run_K1_ActiveMask(
    const PsiC& psi,
    uint8_t* ActiveMask,
    int Nx, int Nz,
    int b_E_trigger,  // Coarse energy block index (not MeV!)
    float weight_active_min
);
