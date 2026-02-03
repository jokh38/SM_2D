#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// K2_CoarseTransport Kernel Header
// ============================================================================
// K2: Simple Energy Loss for High-Energy Particles (No Lateral Spreading)
//
// Purpose: Handle transport for cells where ActiveMask=0 (high energy region)
// - Uses larger step sizes (fewer calculations)
// - Simplified physics: energy loss only
// - NO lateral scattering (delegated to K3 fine transport)
//
// Reason for not implementing lateral spreading in K2:
// - K2 uses binned phase space which cannot track per-particle state
// - Attempting to track Fermi-Eyges moments fails because moments are
//   reset to 0 at each iteration (no per-particle accumulation)
// - K3 handles lateral scattering properly using Monte Carlo methods
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
    const uint32_t* __restrict__ CoarseList,  // List of cells needing coarse transport
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
    // Outflow buckets
    struct DeviceOutflowBucket* __restrict__ OutflowBuckets,
    // Output phase space for particles remaining in cell
    uint32_t* __restrict__ block_ids_out,
    float* __restrict__ values_out
);

// CPU wrapper declaration
void run_K2_CoarseTransport(
    const uint32_t* block_ids_in,
    const float* values_in,
    const uint8_t* ActiveMask,
    const uint32_t* CoarseList,  // List of cells needing coarse transport
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
