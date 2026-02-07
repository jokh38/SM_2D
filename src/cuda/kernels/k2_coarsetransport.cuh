#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// K2_CoarseTransport Kernel Header
// ============================================================================
// K2: Energy Loss with Fermi-Eyges Moment-Based Lateral Spreading
//
// Purpose: Handle transport for cells where ActiveMask=0 (high energy region)
// - Uses larger step sizes (fewer calculations)
// - Energy loss with stopping power (dE/dx)
// - Fermi-Eyges moment tracking (A=⟨θ²⟩, B=⟨xθ⟩, C=⟨x²⟩)
// - Deterministic Gaussian weight distribution for lateral spreading
// - O(z^(3/2)) scaling from accumulated moments
//
// Phase B Implementation (PLAN_MCS.md):
// - Moments tracked per-(theta, E) bin during transport
// - sigma_x = sqrt(C) from accumulated C moment
// - Scattering power T calculated at mid-step energy
// - K2->K3 transition uses moment-based criteria
// ============================================================================

struct K2Config {
    float E_coarse_max;      // Maximum energy for coarse transport [MeV]
    float step_coarse;       // Coarse step size [mm]
    int n_steps_per_cell;    // Number of sub-steps per cell
    float E_fine_on;         // Fine transport activation threshold [MeV]
    float sigma_x_initial;   // Initial beam width [mm] (FIX C: from input config)
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
    // FIX C: Initial beam width (now from config instead of hardcoded)
    float sigma_x_initial,
    // Outputs
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    double* __restrict__ AbsorbedEnergy_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,
    // Outflow buckets
    struct DeviceOutflowBucket* __restrict__ OutflowBuckets,
    const int* __restrict__ CellToBucketBase,
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
    float sigma_x_initial,  // FIX C: Initial beam width from config
    double* EdepC,
    float* AbsorbedWeight_cutoff,
    double* AbsorbedEnergy_cutoff,
    float* AbsorbedWeight_nuclear,
    double* AbsorbedEnergy_nuclear,
    float* BoundaryLoss_weight,
    double* BoundaryLoss_energy,
    struct DeviceOutflowBucket* OutflowBuckets,
    const int* CellToBucketBase,
    uint32_t* block_ids_out,
    float* values_out
);

// Debug counters for slot allocation failures in K2 output PsiC.
void k2_reset_debug_counters();
void k2_get_debug_counters(
    unsigned long long& slot_drop_count,
    double& slot_drop_weight,
    double& slot_drop_energy,
    unsigned long long& bucket_drop_count,
    double& bucket_drop_weight,
    double& bucket_drop_energy,
    unsigned long long& pruned_weight_count,
    double& pruned_weight_sum,
    double& pruned_energy_sum
);

// ============================================================================
// Profiling Functions (ITERATION 3)
// ============================================================================
// These functions are only available when ENABLE_MCS_PROFILING is defined
// during compilation. They provide runtime statistics on moment-based
// enhancement behavior.
// ============================================================================

#ifdef ENABLE_MCS_PROFILING

/**
 * @brief Reset profiling counters to zero
 *
 * Call this before starting a simulation to clear previous profiling data.
 */
void k2_reset_profiling_counters();

/**
 * @brief Retrieve profiling counters from device
 *
 * @param enhancement_count Number of enhancements applied
 * @param total_evaluations Total moment evaluations
 * @param sqrt_A_exceeds Count: sqrt(A) >= 0.02
 * @param sqrt_C_exceeds Count: sqrt(C)/dx >= 3.0
 * @param avg_sqrt_A Average sqrt(A) value
 * @param avg_sqrt_C_dx Average sqrt(C)/dx value
 */
void k2_get_profiling_counters(
    unsigned long long& enhancement_count,
    unsigned long long& total_evaluations,
    unsigned long long& sqrt_A_exceeds,
    unsigned long long& sqrt_C_exceeds,
    double& avg_sqrt_A,
    double& avg_sqrt_C_dx
);

/**
 * @brief Print profiling summary to stdout
 *
 * Call this after simulation completes to see moment-based enhancement statistics.
 */
void k2_print_profiling_summary();

#endif // ENABLE_MCS_PROFILING
