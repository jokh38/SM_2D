#pragma once
#include "core/psi_storage.hpp"

// Active cell identification kernel
__global__ void K1_ActiveMask(
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,
    uint8_t* __restrict__ ActiveMask,
    int Nx, int Nz,
    float E_trigger,
    float weight_active_min
);

// CPU wrapper for testing
void run_K1_ActiveMask(
    const PsiC& psi,
    uint8_t* ActiveMask,
    int Nx, int Nz,
    float E_trigger,
    float weight_active_min
);
