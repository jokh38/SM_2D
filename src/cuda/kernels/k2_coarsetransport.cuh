#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// K2_CoarseTransport Kernel Header
// ============================================================================
// FIX Problem 4: Implement coarse transport for high-energy particles
//
// Purpose: Handle transport for cells where ActiveMask=0 (high energy region)
// - Uses larger step sizes (fewer MCS calculations)
// - Simplified physics: energy loss + simplified angular spreading
// - Transfers to fine transport when energy drops below threshold
// ============================================================================

struct K2Config {
    float E_coarse_max;      // Maximum energy for coarse transport [MeV]
    float step_coarse;       // Coarse step size [mm]
    int n_steps_per_cell;    // Number of sub-steps per cell
};

// Forward declaration of DeviceRLUT
struct DeviceRLUT;

// GPU Kernel Declaration
__global__ void K2_CoarseTransport(
    // Inputs
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint8_t* __restrict__ ActiveMask,  // 0 = needs coarse transport
    const uint32_t* __restrict__ CoarseList,  // CRITICAL FIX: List of cells needing coarse transport
    // Grid
    int Nx, int Nz, float dx, float dz,
    int n_coarse,  // Number of cells needing coarse transport
    // Device LUT
    DeviceRLUT dlut,
    // Grid edges for bin finding
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    // Coarse transport config
    K2Config config,
    // Outputs
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,
    // Outflow buckets (same structure as K3)
    struct DeviceOutflowBucket* __restrict__ OutflowBuckets,
    // CRITICAL FIX: Output phase space for particles remaining in cell
    uint32_t* __restrict__ block_ids_out,
    float* __restrict__ values_out
);

// CPU wrapper declaration
void run_K2_CoarseTransport(
    const uint32_t* block_ids_in,
    const float* values_in,
    const uint8_t* ActiveMask,
    const uint32_t* CoarseList,  // CRITICAL FIX: List of cells needing coarse transport
    int Nx, int Nz, float dx, float dz,
    int n_coarse,
    DeviceRLUT dlut,
    const float* theta_edges,
    const float* E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    K2Config config,
    double* EdepC,
    float* AbsorbedWeight_cutoff,
    float* AbsorbedWeight_nuclear,
    double* AbsorbedEnergy_nuclear,
    float* BoundaryLoss_weight,
    double* BoundaryLoss_energy,
    struct DeviceOutflowBucket* OutflowBuckets,
    uint32_t* block_ids_out,
    float* values_out
);
